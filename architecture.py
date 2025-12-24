"""
architecture.py
===============

FICHIER CENTRAL DU PROJET (PARTIE 1 / 3)
---------------------------------------

Ce fichier implémente l’architecture du modèle hybride pour le challenge
ALTEGRAD "Molecular Graph Captioning".

⚠️ Cette version est VOLONTAIREMENT SUR-ANNOTÉE.
Elle sert de :
- support pédagogique
- base pour le rapport
- référence pour les autres membres du groupe

La verbosité pourra être réduite à la toute fin du projet.

-----------------------------------------------------------------------
VUE D'ENSEMBLE DU PIPELINE (conceptuelle)
-----------------------------------------------------------------------

        Molecular Graph (PyG Data)
              │
              ▼
        AtomEncoder + BondEncoder
              │
              ▼
            MPNN
              │
              ▼
      Graph Embedding (H)
              │
              ├───────────────┐
              │               │
              ▼               ▼
        Projection         Projection
        (graph)             (text)
              │               │
              └───── Shared Latent Space ─────┐
                                              │
                                   Retrieval (NN)
                                              │
                                              ▼
                                   Retrieved Caption
                                              │
                                   Soft Prompt (MLP)
                                              │
                                              ▼
                                           T5-base
                                              │
                                              ▼
                                     Generated Caption

-----------------------------------------------------------------------
LIENS EXPLICITES AVEC LE COURS ALTEGRAD / MVA
-----------------------------------------------------------------------

- Graph Representation Learning :
    * Message Passing Neural Networks (Gilmer et al., 2017)
- Multimodal Representation Learning :
    * Contrastive Learning (InfoNCE, CLIP)
- Large Language Models :
    * T5 (Text-to-Text Transformer)
- Prompting :
    * Soft / Continuous Prompts
- Parameter-Efficient Fine-Tuning :
    * LoRA (Low-Rank Adaptation)
- RAG (Retrieval-Augmented Generation) :
    * Retrieve → Rewrite → Generate

-----------------------------------------------------------------------
PHILOSOPHIE DE DESIGN GLOBALE
-----------------------------------------------------------------------

1) Séparer clairement les responsabilités :
   - Encoder le graphe
   - Aligner graphe et texte
   - Générer du texte

2) Ne PAS générer from-scratch :
   - On exploite un prior lexical via retrieval
   - On corrige / adapte avec un LLM

3) Privilégier la stabilité et la justifiabilité :
   - Chaque choix peut être défendu à l’oral
   - Chaque module peut être ablaté indépendamment
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_add_pool

from transformers import T5ForConditionalGeneration, T5EncoderModel
from peft import LoraConfig, get_peft_model

class AtomEncoder(nn.Module):
    """
    Encodeur des features atomiques.

    CONTEXTE DATASET
    ----------------
    Dans le dataset ALTEGRAD, chaque atome est décrit par 9 features
    catégorielles (entiers) :
        - atomic number
        - chirality
        - degree
        - formal charge
        - num hydrogens
        - num radical electrons
        - hybridization
        - is aromatic
        - is in ring

    Ces features sont stockées dans data.x ∈ ℕ^{N×9}.

    STRATÉGIE D'ENCODAGE
    -------------------
    - Une embedding par feature
    - Les embeddings sont SOMMÉES

    POURQUOI LA SOMME ?
    ------------------
    - La somme conserve une dimension fixe H
    - Elle évite la concaténation (9×H paramètres)
    - Elle est largement utilisée dans la littérature chimique (OGB, MoleculeNet)

    INTUITION
    ---------
    Chaque atome est représenté comme une superposition continue
    de ses attributs discrets.

    ANTI-CHOIX :
    ------------
    ❌ Nous n'utilisons PAS de one-hot encoding.
    → trop sparse
    → inefficace en mémoire
    → pas de partage statistique entre catégories proches

    ❌ Nous n'utilisons PAS de concaténation des embeddings.
    → explosion dimensionnelle
    → plus difficile à régulariser

    ❌ Nous n'utilisons PAS de features continues RDKit (ex: charges partielles).
    → non fournies dans le dataset
    → hors scope du challenge
    """

    def __init__(self, hidden_dim: int, vocab_sizes: List[int]):
        super().__init__()

        # Une embedding par feature catégorielle
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, hidden_dim) for vocab_size in vocab_sizes]
        )

        # Initialisation standard pour embeddings
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cat: Tensor [N, F] d'entiers catégoriels

        Returns:
            Tensor [N, H] : embeddings atomiques continus
        """
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(x_cat[:, i])
        return out

class BondEncoder(nn.Module):
    """
    Encodeur des liaisons chimiques (edge attributes).

    CONTEXTE DATASET
    ----------------
    Chaque liaison est décrite par 3 features catégorielles :
        - bond type
        - stereochemistry
        - is conjugated

    Ces informations sont CRUCIALES en chimie.

    POURQUOI UN ENCODAGE SPÉCIFIQUE DES ARÊTES ?
    -------------------------------------------
    - Un GCN classique ne voit que la connectivité.
    - En chimie, deux graphes isomorphes mais avec des liaisons différentes
      correspondent à des molécules totalement différentes.

    STRATÉGIE
    ---------
    - Une embedding par feature de liaison
    - Somme des embeddings (cohérence avec AtomEncoder)

    ANTI-CHOIX :
    ------------
    ❌ Nous n'ignorons PAS les edge attributes.
    → ce serait équivalent à un GCN
    → perte massive d'information chimique

    ❌ Nous n'utilisons PAS un simple scaling des messages par bond type.
    → trop rigide
    → incapacité à modéliser des interactions complexes
    """

    def __init__(self, hidden_dim: int, vocab_sizes: List[int]):
        super().__init__()

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, hidden_dim) for vocab_size in vocab_sizes]
        )

        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, edge_attr_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_attr_cat: Tensor [E, F_edge] d'entiers catégoriels

        Returns:
            Tensor [E, H] : embeddings de liaisons
        """
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(edge_attr_cat[:, i])
        return out

class MPNNLayer(MessagePassing):
    """
    Message Passing Neural Network (MPNN) — Gilmer et al., 2017.

    FORMALISME THÉORIQUE
    -------------------
        m_ij = f(h_j, e_ij)
        m_i  = Σ_j m_ij
        h_i' = g(h_i, m_i)

    Ici :
    - f et g sont implémentées comme des MLP
    - l'agrégation est une somme

    POURQUOI MPNN ET PAS GCN ?
    -------------------------
    - Le GCN ne dépend que de la structure du graphe
    - Le MPNN incorpore explicitement les liaisons chimiques

    AJOUTS D'INGÉNIERIE
    ------------------
    - BatchNorm : stabilise les gradients
    - Résidu     : limite l'oversmoothing

    ANTI-CHOIX :
    ------------
    ❌ Nous n'utilisons PAS une simple somme pondérée par le degré.
    → perte de dépendance aux types de liaisons

    ❌ Nous n'utilisons PAS une attention explicite (GAT).
    → plus coûteux
    → bénéfice empirique limité sur ce dataset

    ❌ Nous n'utilisons PAS de très grande profondeur (>6 couches).
    → oversmoothing rapide sur graphes moléculaires
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__(aggr="add")

        # Fonction message m_ij = f(h_j, e_ij)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        # Fonction update h_i' = g(h_i, m_i)
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x          : [N, H] embeddings atomiques
            edge_index : [2, E]
            edge_attr  : [E, H] embeddings de liaisons
        """
        m_i = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        x_upd = self.upd_mlp(torch.cat([x, m_i], dim=-1))

        x_upd = self.norm(x_upd)
        x_upd = F.relu(x_upd)
        x_upd = F.dropout(x_upd, p=self.dropout, training=self.training)

        # Connexion résiduelle (clé en chimie)
        return x + x_upd

    def message(self, x_j, edge_attr):
        """
        Message envoyé de l'atome j vers l'atome i.
        """
        return self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))

class MPNNEncoder(nn.Module):
    """
    Encodeur complet de graphes moléculaires basé sur MPNN.

    RÔLE DANS LE PIPELINE
    --------------------
    Ce module transforme un graphe moléculaire (atomes + liaisons)
    en une représentation vectorielle globale (graph embedding).

    Cette représentation est utilisée pour :
    - le retrieval (alignement graphe–texte)
    - le conditionnement du LLM via soft prompt

    COMPOSITION
    -----------
    1) AtomEncoder : embeddings atomiques initiaux
    2) BondEncoder : embeddings des liaisons
    3) L couches MPNN : propagation de l'information locale
    4) Pooling global : agrégation des atomes → graphe

    SORTIES
    -------
    - graph_emb : Tensor [B, H]
        représentation globale du graphe
    - node_emb  : Tensor [N_total, H]
        embeddings atomiques finaux (non utilisés directement ici,
        mais conservés pour extensions futures)

    ANTI-CHOIX :
    ------------
    ❌ Nous n'utilisons PAS un encodeur hiérarchique multi-échelle.
    → trop complexe pour le dataset
    → gain empirique incertain

    ❌ Nous n'utilisons PAS un pooling attentionnel global.
    → plus coûteux
    → souvent instable sur petits graphes moléculaires

    ❌ Nous n'utilisons PAS les node embeddings directement dans le LLM
    (ex: cross-attention graphe–texte).
    → modification lourde de l'architecture T5
    → soft prompt est un compromis plus simple et robuste
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

        # Encodage initial des atomes et liaisons
        self.atom_encoder = AtomEncoder(hidden_dim, node_vocab_sizes)
        self.bond_encoder = BondEncoder(hidden_dim, edge_vocab_sizes)

        # Empilement de couches MPNN
        self.layers = nn.ModuleList(
            [
                MPNNLayer(hidden_dim, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, data):
        """
        Args:
            data : torch_geometric.data.Data ou Batch
                - data.x         : [N, F_node] entiers catégoriels
                - data.edge_attr : [E, F_edge] entiers catégoriels
                - data.edge_index
                - data.batch     : [N] assignment graphe

        Returns:
            graph_emb : [B, H]
            node_emb  : [N, H]
        """

        # Étape 1 — encodage atomique et des liaisons
        x = self.atom_encoder(data.x)            # [N, H]
        e = self.bond_encoder(data.edge_attr)    # [E, H]

        # Étape 2 — message passing
        for layer in self.layers:
            x = layer(x, data.edge_index, e)

        # Étape 3 — pooling global (permutation-invariant)
        graph_emb = global_add_pool(x, data.batch)

        return graph_emb, x

class FrozenT5TextEncoder(nn.Module):
    """
    Encodeur texte utilisé pour l'alignement graphe–texte (retrieval).

    CHOIX DU MODÈLE
    ---------------
    Nous utilisons l'encodeur de T5-base (sans le décodeur).

    POURQUOI T5 ?
    -------------
    - Même backbone que le générateur → cohérence sémantique
    - Très bon compromis performance / taille
    - Bien adapté aux tâches de reformulation

    POURQUOI FROZEN ?
    -----------------
    - Dataset relativement petit (~32k)
    - Fine-tuner un encodeur texte complet est instable
    - Réduction drastique de la VRAM et du temps d'entraînement

    RÔLE EXACT
    ----------
    Ce module produit des embeddings de phrases
    utilisés UNIQUEMENT pour :
        - l'objectif contrastif (CLIP-like)
        - le retrieval de captions

    ANTI-CHOIX :
    ------------
    ❌ Nous n'utilisons PAS BERT / RoBERTa comme encodeur texte.
    → incohérence avec le générateur T5
    → deux tokenizers différents

    ❌ Nous n'utilisons PAS un pooling basé sur le premier token.
    → T5 n'a pas de token [CLS]
    → sémantiquement non justifié

    ❌ Nous ne fine-tunons PAS l'encodeur texte.
    → trop coûteux
    → pas nécessaire pour un retrieval efficace
    """

    def __init__(self, model_name: str = "t5-base", freeze: bool = True):
        super().__init__()

        self.encoder = T5EncoderModel.from_pretrained(model_name)

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids     : [B, L]
            attention_mask: [B, L]

        Returns:
            pooled_emb : [B, d_model]
                embedding de phrase
        """

        # Sortie du T5 encoder : [B, L, d_model]
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        # Mean pooling masqué (T5 n'a pas de token [CLS])
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return pooled

class GraphTextCLIP(nn.Module):
    """
    Module d'alignement contrastif graphe ↔ texte (inspiré de CLIP).

    OBJECTIF
    --------
    Apprendre un espace latent commun où :
        sim(graph_i, text_i) >> sim(graph_i, text_j)

    Cet espace est utilisé pour :
    - le retrieval de captions
    - fournir un prior lexical au générateur

    LIEN AVEC CLIP
    --------------
    - Deux encodeurs indépendants
    - Projections linéaires vers un espace partagé
    - Normalisation L2
    - Loss contrastive symétrique (InfoNCE)

    AVANTAGE CLÉ
    ------------
    Le retrieval devient une simple recherche de plus proches voisins
    dans l'espace latent.

    ANTI-CHOIX :
    ------------
    ❌ Nous n'utilisons PAS une loss unidirectionnelle (graph → text seulement).
    → symétrie améliore la qualité de l'espace latent

    ❌ Nous n'utilisons PAS une mémoire externe type FAISS pendant l'entraînement.
    → trop lourd
    → torch.topk suffit à cette échelle (~32k)

    ❌ Nous n'utilisons PAS un entraînement joint avec le générateur.
    → instabilité forte
    → séparation claire des étapes = meilleur contrôle

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

        # Projections vers l'espace latent commun
        self.graph_proj = nn.Linear(graph_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)

        # Température apprenable (comme dans CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_graph(self, batch_graph):
        """
        Encode un batch de graphes en embeddings normalisés.
        """
        g, _ = self.graph_encoder(batch_graph)   # [B, graph_dim]
        g = self.graph_proj(g)                   # [B, shared_dim]
        return F.normalize(g, dim=1)

    def encode_text(self, input_ids, attention_mask):
        """
        Encode un batch de textes en embeddings normalisés.
        """
        t = self.text_encoder(input_ids, attention_mask)  # [B, text_dim]
        t = self.text_proj(t)                              # [B, shared_dim]
        return F.normalize(t, dim=1)

class GraphSoftPromptT5(nn.Module):
    """
    Générateur final du pipeline hybride.

    RÔLE DANS LE PIPELINE
    --------------------
    Ce module est responsable de la génération de la description finale.

    Il NE génère PAS une description "from scratch".
    Il prend en entrée :
        - une description candidate (retrieved caption)
        - un prompt fixe (instruction)
        - une information continue issue du graphe (soft prompt)

    et apprend à :
        → réécrire / corriger / adapter la description
          pour qu'elle corresponde exactement au graphe moléculaire.

    FORMULATION CONCEPTUELLE
    ------------------------
    On peut voir ce module comme :
        - un modèle de réécriture conditionnelle
        - avec un conditionnement continu (soft prompt)
        - et un fine-tuning léger (LoRA)

    Cette formulation est volontaire :
        - BLEU favorise la similarité lexicale
        - BERTScore favorise la similarité sémantique
        - le retrieval fournit un "prior" lexical fort

    ANTI-CHOIX :
    ------------

    ❌ Nous ne générons PAS les descriptions "from scratch".
    → BLEU chute drastiquement
    → style du dataset très spécifique
    → retrieval fournit un prior lexical crucial

    ❌ Nous n'utilisons PAS de prompt uniquement textuel.
    → le graphe contient une information structurée
    → le soft prompt permet un conditionnement continu plus riche

    ❌ Nous n'utilisons PAS de cross-attention explicite graphe → texte.
    → modification lourde de l'architecture T5
    → coût mémoire élevé
    → soft prompt est un compromis robuste et simple

    ❌ Nous ne fine-tunons PAS tous les paramètres de T5.
    → VRAM insuffisante
    → risque d'overfitting
    → LoRA suffit largement dans ce contexte

    ❌ Nous n'optimisons PAS directement BLEU / BERTScore au début.
    → non différentiables
    → instabilité forte
    → cross-entropy d'abord, MRT en option en fin
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        graph_emb_dim: int = 300,
        prompt_len: int = 8,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: tuple = ("q", "v"),
    ):
        """
        Args:
            model_name:
                Nom du modèle T5 (t5-base par défaut)

            graph_emb_dim:
                Dimension de l'embedding du graphe (sortie du MPNN)

            prompt_len:
                Nombre de tokens continus (soft prompt)

            lora_r, lora_alpha, lora_dropout:
                Hyperparamètres LoRA (PEFT)

            target_modules:
                Sous-modules de T5 à adapter avec LoRA
                (par défaut projections Q/V de l'attention)
        """
        super().__init__()

        self.prompt_len = prompt_len

        # ====================================================
        # Chargement du modèle T5
        # ====================================================
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

        # ====================================================
        # Injection de LoRA (Parameter-Efficient Fine-Tuning)
        # ====================================================
        """
        POURQUOI LoRA ?
        ---------------
        - Fine-tuner tous les paramètres de T5 est :
            * trop coûteux (VRAM)
            * inutile sur ~32k exemples
        - LoRA permet :
            * d'adapter le modèle
            * avec <1% des paramètres entraînés
            * tout en conservant la capacité du LLM

        CIBLAGE DES MODULES :
        --------------------
        - On adapte uniquement les projections Q et V
        - Choix standard, bon compromis capacité / stabilité
        """
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

        # ====================================================
        # Soft Prompt (conditionnement graphe → texte)
        # ====================================================
        """
        Le soft prompt est une projection continue du graphe
        vers k tokens de dimension d_model.

        Conceptuellement :
            graph_emb ∈ R^H
                ↓
            MLP
                ↓
            soft_prompt ∈ R^{k × d_model}

        Ces tokens sont CONCATÉNÉS avant les tokens texte.
        """
        self.softprompt = nn.Sequential(
            nn.Linear(graph_emb_dim, 2 * graph_emb_dim),
            nn.ReLU(),
            nn.Linear(2 * graph_emb_dim, prompt_len * d_model),
        )

    # --------------------------------------------------------
    # Construction des embeddings d'entrée pour T5
    # --------------------------------------------------------
    def _build_inputs_embeds(self, graph_emb: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Construit les embeddings d'entrée pour T5 en concaténant :
            [soft prompt tokens] + [embeddings texte]

        Args:
            graph_emb : Tensor [B, H]
            input_ids : Tensor [B, L]

        Returns:
            inputs_embeds : Tensor [B, (k+L), d_model]
        """
        B = graph_emb.size(0)
        d_model = self.t5.config.d_model

        # Projection du graphe vers k tokens continus
        soft_tokens = self.softprompt(graph_emb).view(B, self.prompt_len, d_model)

        # Embeddings des tokens texte standards
        token_embeds = self.t5.get_input_embeddings()(input_ids)

        # Concaténation soft prompt + texte
        return torch.cat([soft_tokens, token_embeds], dim=1)

    # --------------------------------------------------------
    # Forward (training)
    # --------------------------------------------------------
    def forward(self, graph_emb, input_ids, attention_mask=None, labels=None):
        """
        Forward pass utilisé en entraînement supervisé (SFT).

        Args:
            graph_emb      : [B, H]
            input_ids      : [B, L]
            attention_mask : [B, L]
            labels         : [B, L_out]

        Returns:
            outputs de T5 (inclut outputs.loss si labels fournis)
        """
        inputs_embeds = self._build_inputs_embeds(graph_emb, input_ids)

        # Extension du attention_mask pour les soft tokens
        if attention_mask is not None:
            B = attention_mask.size(0)
            soft_mask = torch.ones(
                B, self.prompt_len,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([soft_mask, attention_mask], dim=1)

        return self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    # --------------------------------------------------------
    # Génération (inference)
    # --------------------------------------------------------
    @torch.no_grad()
    def generate(self, graph_emb, input_ids, attention_mask=None, **gen_kwargs):
        """
        Génération de texte (inference).

        On utilise inputs_embeds pour intégrer le soft prompt.
        """
        inputs_embeds = self._build_inputs_embeds(graph_emb, input_ids)

        if attention_mask is not None:
            B = attention_mask.size(0)
            soft_mask = torch.ones(
                B, self.prompt_len,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([soft_mask, attention_mask], dim=1)

        return self.t5.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

"""
Ce fichier implémente une architecture :

- Scientifiquement fondée
- Ingéniérie-compatible (VRAM, stabilité)
- Parfaitement justifiable à l'oral
- Facilement ablatable (chaque module est indépendant)

Il constitue à lui seul :
- un support pédagogique
- une base de rapport
- un blueprint de recherche

"""