"""
architecture.py
===============

Ce fichier contient les modules principaux du pipeline hybride :

(1) Encodeur graphe : MPNN (Message Passing Neural Network)
    - Motivation: mieux adapté à la chimie qu'un GCN/GIN vanilla
      car on incorpore explicitement les attributs des arêtes (liaisons).

(2) Alignement graphe-texte : objectif contrastif "style CLIP"
    - Motivation: construire un espace latent commun pour faire du retrieval
      (baseline Kaggle) + stabiliser l'apprentissage en multi-étapes.

(3) Génération : T5-base + LoRA + Soft Prompt
    - Motivation: T5 est seq2seq -> excellent pour "rewrite" (réécriture)
    - LoRA = Parameter Efficient Fine Tuning (cours LLM/PFT)
    - Soft prompt = injection continue des infos du graphe (cours prompting / multimodal)

Liens conceptuels avec le cours:
--------------------------------
- Representation learning for graphs (MPNN / message passing)
- Contrastive learning (InfoNCE / CLIP-like)
- RAG / Retrieve-and-Generate (pipeline modulaire)
- PEFT (LoRA) et prompting (soft prompt)
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_add_pool

from transformers import T5ForConditionalGeneration, T5EncoderModel
from peft import LoraConfig, get_peft_model


# =========================================================
# 1) Encodage des features catégorielles (atomes / liaisons)
# =========================================================

class AtomEncoder(nn.Module):
    """
    Encodeur des features atomiques.

    Contexte dataset:
    -----------------
    data.x est une matrice [N, 9] d'entiers catégoriels.
    Baseline Kaggle: "une embedding par feature, puis somme".

    Pourquoi une somme et pas concat ?
    ---------------------------------
    - somme = dimension fixe H
    - évite d'exploser la dimension (sinon 9*H)
    - proche des pratiques OGB/chemistry baselines
    """

    def __init__(self, hidden_dim: int, vocab_sizes: List[int]):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, hidden_dim) for v in vocab_sizes])
        for emb in self.embs:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: [N, F] entiers
        out = 0
        for i, emb in enumerate(self.embs):
            out = out + emb(x_cat[:, i])
        return out  # [N, H]


class BondEncoder(nn.Module):
    """
    Encodeur des features de liaisons (edge_attr).

    data.edge_attr est [E, 3] d'entiers catégoriels.
    Même logique que AtomEncoder : somme des embeddings de chaque champ.
    """

    def __init__(self, hidden_dim: int, vocab_sizes: List[int]):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, hidden_dim) for v in vocab_sizes])
        for emb in self.embs:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, e_cat: torch.Tensor) -> torch.Tensor:
        out = 0
        for i, emb in enumerate(self.embs):
            out = out + emb(e_cat[:, i])
        return out  # [E, H]


# =========================================================
# 2) MPNN Layer (Gilmer-style) + stabilisation (BN + residual)
# =========================================================

class MPNNLayer(MessagePassing):
    """
    MPNN layer inspirée de Gilmer et al. (2017):
        m_ij = f(h_j, e_ij)
        m_i  = sum_j m_ij
        h_i' = g(h_i, m_i)

    Différence cruciale vs GCN:
    ---------------------------
    - Ici, les messages dépendent explicitement des attributs d'arêtes e_ij
      (bond type, stereo, conjugation), très importants en chimie.

    Stabilisation (engineering choices):
    ------------------------------------
    - BatchNorm : stabilise l'entraînement quand on empile plusieurs couches GNN.
    - Residual connection : limite l'oversmoothing et préserve l'identité des noeuds.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__(aggr="add")

        # Message function: MLP([h_j, e_ij]) -> R^H
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        # Update function: MLP([h_i, m_i]) -> R^H
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, H]
            edge_index: [2, E]
            edge_attr: [E, edge_dim] (déjà embedded, donc continuous)
        """
        m_i = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)  # [N, H]
        x_upd = self.upd_mlp(torch.cat([x, m_i], dim=-1))                      # [N, H]
        x_upd = self.norm(x_upd)
        x_upd = F.relu(x_upd)
        x_upd = F.dropout(x_upd, p=self.dropout, training=self.training)

        # Residual: important en GNN profond
        return x + x_upd

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # message sur chaque arête j->i
        return self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))


class MPNNEncoder(nn.Module):
    """
    Encodeur graphe complet = AtomEncoder + BondEncoder + L couches MPNN + pooling global.

    Sorties:
    --------
    - graph_emb: [B, H] (représentation globale du graphe)
    - node_emb : [N_total, H] (utile si on voulait un cross-attention plus riche plus tard)

    Pourquoi un pooling add ?
    -------------------------
    - sum pooling est classique en chimie : approx "compte" les motifs
    - mean pooling serait aussi possible ; on peut ablater.
    """

    def __init__(
        self,
        node_vocab_sizes: List[int],
        edge_vocab_sizes: List[int],
        hidden_dim: int = 300,
        num_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.atom_encoder = AtomEncoder(hidden_dim, node_vocab_sizes)
        self.bond_encoder = BondEncoder(hidden_dim, edge_vocab_sizes)

        self.layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, data):
        """
        data: PyG Batch
          - data.x         : [N, F_node] categorical ints
          - data.edge_attr : [E, F_edge] categorical ints
          - data.edge_index
          - data.batch     : [N] graph assignment
        """
        x = self.atom_encoder(data.x)                  # [N, H]
        e = self.bond_encoder(data.edge_attr)          # [E, H]

        for layer in self.layers:
            x = layer(x, data.edge_index, e)           # [N, H]

        graph_emb = global_add_pool(x, data.batch)     # [B, H]
        return graph_emb, x


# =========================================================
# 3) Text encoder (retrieval) = T5 encoder (frozen)
# =========================================================

class FrozenT5TextEncoder(nn.Module):
    """
    Encodeur texte pour le retrieval.

    Choix:
    ------
    On utilise l'encodeur de T5 (seq2seq) comme encodeur de phrases.
    - Avantage: cohérence avec le générateur (même tokenizer/modèle)
    - Avantage: embeddings contextualisés puissants
    - Freeze: on le gèle pour réduire VRAM et instabilité

    Pooling:
    --------
    T5 n'a pas [CLS], donc on fait un masked mean pooling sur la séquence.
    """

    def __init__(self, model_name: str = "t5-base", freeze: bool = True):
        super().__init__()
        self.enc = T5EncoderModel.from_pretrained(model_name)
        if freeze:
            for p in self.enc.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,L,d]
        mask = attention_mask.unsqueeze(-1).float()                                          # [B,L,1]
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)                      # [B,d]
        return pooled


# =========================================================
# 4) CLIP-like alignment module (graph-text)
# =========================================================

class GraphTextCLIP(nn.Module):
    """
    Module d'alignement contrastif graphe <-> texte.

    Inspiration:
    -----------
    CLIP (image-text) : on apprend deux encodeurs + projections
    vers un espace commun, puis loss contrastive symétrique.

    Pourquoi c'est utile ici ?
    --------------------------
    - Permet de faire du retrieval de captions (baseline Kaggle)
    - Sert de "pretraining" stable avant de brancher le LLM
    - Rend le pipeline hybride (RAG) plus robuste
    """

    def __init__(
        self,
        graph_encoder: nn.Module,
        text_encoder: nn.Module,
        graph_dim: int,
        text_dim: int,
        shared_dim: int = 256,
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder

        self.graph_proj = nn.Linear(graph_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)

        # logit_scale comme CLIP: apprend la température
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_graph(self, batch_graph) -> torch.Tensor:
        g, _ = self.graph_encoder(batch_graph)     # [B, graph_dim]
        g = self.graph_proj(g)                     # [B, shared_dim]
        return F.normalize(g, dim=1)

    def encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        t = self.text_encoder(input_ids, attention_mask)  # [B, text_dim]
        t = self.text_proj(t)                              # [B, shared_dim]
        return F.normalize(t, dim=1)

    def forward(self, batch_graph, input_ids, attention_mask):
        return self.encode_graph(batch_graph), self.encode_text(input_ids, attention_mask)


# =========================================================
# 5) Generator: SoftPrompt + T5-base + LoRA
# =========================================================

class GraphSoftPromptT5(nn.Module):
    """
    Générateur hybride (rewrite) : T5 + LoRA + soft prompt.

    Pipeline:
    ---------
    input_text = FIXED_PROMPT + retrieved_caption
    soft_prompt = MLP(graph_emb) -> k embeddings de dimension d_model
    inputs_embeds = [soft_prompt ; token_embeddings(input_text)]
    T5 génère la caption finale.

    Pourquoi du soft prompt ?
    -------------------------
    - On veut injecter l'information structurée du graphe dans le LLM
      sans modifier le vocab/tokenizer.
    - Soft prompt = conditionnement continu, plus expressif qu'un prompt texte "brut".

    Pourquoi LoRA ?
    --------------
    - Fine-tune efficace (PEFT) : on entraîne une faible fraction des paramètres
      (typiquement les projections Q/V en attention).
    - Compatible RTX 4070 Ti (12 Go) et itérations rapides.
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        graph_emb_dim: int = 300,
        prompt_len: int = 8,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Tuple[str, ...] = ("q", "v"),
    ):
        super().__init__()
        self.prompt_len = prompt_len

        # Base model
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

        # LoRA adapters (PEFT)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            target_modules=list(target_modules),
        )
        self.t5 = get_peft_model(self.t5, lora_cfg)

        d_model = self.t5.config.d_model

        # Soft prompt MLP: graph_emb -> (prompt_len * d_model)
        # Choix MLP > linear : capacité non-linéaire pour encoder des patterns de graphe
        self.softprompt = nn.Sequential(
            nn.Linear(graph_emb_dim, 2 * graph_emb_dim),
            nn.ReLU(),
            nn.Linear(2 * graph_emb_dim, prompt_len * d_model),
        )

    def _build_inputs_embeds(self, graph_emb: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Construit inputs_embeds en concaténant k soft tokens et les embeddings texte.
        """
        B = graph_emb.size(0)
        d_model = self.t5.config.d_model

        sp = self.softprompt(graph_emb).view(B, self.prompt_len, d_model)      # [B,k,d]
        tok = self.t5.get_input_embeddings()(input_ids)                         # [B,L,d]
        return torch.cat([sp, tok], dim=1)                                      # [B,k+L,d]

    def forward(self, graph_emb, input_ids, attention_mask=None, labels=None):
        """
        En training SFT: labels != None => loss cross-entropy disponible via outputs.loss
        """
        inputs_embeds = self._build_inputs_embeds(graph_emb, input_ids)

        # Important: attention_mask doit être étendu avec des 1 pour les soft tokens
        if attention_mask is not None:
            B = attention_mask.size(0)
            sp_mask = torch.ones(B, self.prompt_len, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([sp_mask, attention_mask], dim=1)

        return self.t5(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, graph_emb, input_ids, attention_mask=None, **gen_kwargs):
        """
        Inference: génération texte.
        On utilise inputs_embeds pour intégrer soft prompt.
        """
        inputs_embeds = self._build_inputs_embeds(graph_emb, input_ids)

        if attention_mask is not None:
            B = attention_mask.size(0)
            sp_mask = torch.ones(B, self.prompt_len, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([sp_mask, attention_mask], dim=1)

        return self.t5.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kwargs)
