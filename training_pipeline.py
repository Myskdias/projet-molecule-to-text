from __future__ import annotations
"""
training_pipeline.py
====================

FICHIER LE PLUS IMPORTANT APRÈS architecture.py
------------------------------------------------

Ce fichier orchestre *toute la logique d'entraînement* du modèle hybride
pour le challenge ALTEGRAD "Molecular Graph Captioning".

Il implémente un entraînement en PLUSIEURS ÉTAPES, volontairement séparées,
afin de :
- stabiliser l’apprentissage
- faciliter le debugging
- rendre chaque choix justifiable scientifiquement

⚠️ Ce fichier est VOLONTAIREMENT SUR-ANNOTÉ.
Il sert de :
- support pédagogique
- trace des choix de design
- squelette direct pour le rapport

-----------------------------------------------------------------------
PIPELINE D'ENTRAÎNEMENT (vue conceptuelle)
-----------------------------------------------------------------------

STAGE 0 (implicite) :
    Prétraitement des graphes (fourni)

STAGE 1 :
    Graph–Text Alignment (Contrastive Learning, CLIP-style)
        - encodeur graphe (MPNN)
        - encodeur texte (T5 encoder, gelé)
        - objectif : espace latent commun
        - sortie : modèle de retrieval robuste

STAGE 2 :
    Retrieval-Augmented Generation (Rewrite)
        - retrieval top-k captions
        - soft prompt (graphe → tokens continus)
        - T5-base + LoRA
        - objectif : cross-entropy (teacher forcing)

STAGE 3 (OPTIONNEL, avancé) :
    Metric-aware fine-tuning
        - BLEU / MRT / BERTScore
        - uniquement en fin de projet

-----------------------------------------------------------------------
LIENS EXPLICITES AVEC LE COURS ALTEGRAD / MVA
-----------------------------------------------------------------------

- Contrastive Learning : InfoNCE, CLIP
- Graph Representation Learning : MPNN
- Retrieval-Augmented Generation (RAG)
- Large Language Models : SFT, PEFT (LoRA)
- Evaluation Metrics : BLEU, BERTScore

-----------------------------------------------------------------------
PHILOSOPHIE DE DESIGN
-----------------------------------------------------------------------

1) Séparer les étapes = réduire l’instabilité
2) Geler tôt, dégeler tard
3) Retrieval comme "prior lexical"
4) Génération = correction, pas création from scratch
"""
"""
ANTI-CHOIX GÉNÉRAUX :
--------------------

❌ Nous n'entraînons PAS tout le modèle de bout en bout dès le début.
   → gradients instables
   → retrieval qui dérive
   → génération incohérente

❌ Nous n'utilisons PAS une seule loss globale.
   → objectifs incompatibles (retrieval vs génération)

❌ Nous n'optimisons PAS directement la métrique Kaggle dès le départ.
   → BLEU / BERTScore non différentiables
   → fort risque d'effondrement
"""


import os
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

from transformers import T5TokenizerFast

# Modules du projet
from data_utils import PreprocessedGraphTextDataset, collate_graph_text
from architecture import (
    MPNNEncoder,
    FrozenT5TextEncoder,
    GraphTextCLIP,
    GraphSoftPromptT5,
)
from retrieval import RetrievalIndex

def set_seed(seed: int = 42):
    """
    Fixe toutes les sources de hasard pertinentes.

    POURQUOI C'EST IMPORTANT :
    --------------------------
    - Le retrieval est sensible aux poids initiaux
    - Les LLM ont une variance non négligeable
    - En MVA, la reproductibilité est un critère implicite
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class Config:
    """
    Conteneur centralisé pour tous les hyperparamètres.

    POURQUOI une dataclass ?
    ------------------------
    - lisibilité
    - traçabilité
    - modification simple pour ablations

    ANTI-CHOIX :
    ------------
    ❌ Nous n'utilisons PAS argparse ici.
       → bruit inutile à ce stade
       → les expériences sont encore exploratoires
    """

    # --------------------
    # Hardware / device
    # --------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------
    # Data
    # --------------------
    train_graphs: str = "data/train_graphs.pkl"
    val_graphs: str = "data/validation_graphs.pkl"
    use_val: bool = False  # activable plus tard

    # --------------------
    # Checkpoints
    # --------------------
    stage1_path: str = "weights_stage1_clip.pt"
    stage2_path: str = "weights_stage2_t5.pt"

    # --------------------
    # Graph encoder
    # --------------------
    node_vocab_sizes: list = None
    edge_vocab_sizes: list = None
    hidden_graph: int = 300
    shared_dim: int = 256

    # --------------------
    # LLM / T5
    # --------------------
    t5_name: str = "t5-base"
    prompt_len: int = 8

    # --------------------
    # Tokenisation
    # --------------------
    max_in_len: int = 256
    max_out_len: int = 256

    # --------------------
    # Retrieval
    # --------------------
    topk: int = 5

    # --------------------
    # Stage 1 (contrastive)
    # --------------------
    stage1_epochs: int = 5
    stage1_lr: float = 2e-4

    # --------------------
    # Stage 2 (generation)
    # --------------------
    stage2_epochs: int = 5
    stage2_lr: float = 5e-5
    freeze_warmup_epochs: int = 1

    # --------------------
    # Adaptive LR (Stage 2)
    # --------------------
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    min_lr: float = 1e-6

    # --------------------
    # DataLoader
    # --------------------
    batch_size: int = 16
    num_workers: int = 0

    # --------------------
    # Checkpointing
    # --------------------
    stage2_ckpt_dir: str = "checkpoints_stage2"
    resume_stage2: bool = True

"""
ANTI-CHOIX :
------------
❌ Nous n'utilisons PAS un prompt très long ou verbeux.
   → bruit inutile
   → le soft prompt porte l'information structurée

❌ Nous n'utilisons PAS un prompt dépendant du graphe en texte.
   → redondant avec le soft prompt
   → risque d'hallucinations
"""

FIXED_PROMPT = (
    "Rephrase the following molecular description so that it accurately "
    "reflects the structure and roles of the given molecule.\n"
    "Description:\n"
)

def infer_vocab_sizes_from_dataset(dataset):
    """
    Infère les tailles de vocabulaire des features catégorielles
    sur L'ENSEMBLE du dataset, de manière ROBUSTE.

    Gère correctement :
    - graphes sans arêtes
    - dataset[i] = (graph, description, idx)
    """
    max_node = None
    max_edge = None

    for i in range(len(dataset)):
        g, _, _ = dataset[i]

        # --------
        # Nodes
        # --------
        x = g.x
        node_max = x.max(dim=0).values

        if max_node is None:
            max_node = node_max
        else:
            max_node = torch.max(max_node, node_max)

        # --------
        # Edges (attention : peut être vide)
        # --------
        if g.edge_attr is not None and g.edge_attr.size(0) > 0:
            e = g.edge_attr
            edge_max = e.max(dim=0).values

            if max_edge is None:
                max_edge = edge_max
            else:
                max_edge = torch.max(max_edge, edge_max)

    # Finalisation
    node_vocab_sizes = (max_node + 1).long().tolist()

    if max_edge is None:
        # Cas pathologique mais possible : AUCUNE arête dans tout le dataset
        raise RuntimeError("Dataset contains no edges at all.")

    edge_vocab_sizes = (max_edge + 1).long().tolist()

    return node_vocab_sizes, edge_vocab_sizes

def load_stage1_clip(cfg):
    device = cfg.device

    tokenizer = T5TokenizerFast.from_pretrained(cfg.t5_name)

    graph_encoder = MPNNEncoder(
        node_vocab_sizes=cfg.node_vocab_sizes,
        edge_vocab_sizes=cfg.edge_vocab_sizes,
        hidden_dim=cfg.hidden_graph,
    ).to(device)

    text_encoder = FrozenT5TextEncoder(cfg.t5_name, freeze=True)

    clip = GraphTextCLIP(
        graph_encoder=graph_encoder,
        text_encoder=text_encoder,
        graph_dim=cfg.hidden_graph,
        text_dim=768,
    ).to(device)

    ckpt = torch.load(cfg.stage1_path, map_location=device)
    clip.graph_encoder.load_state_dict(ckpt["graph_encoder"])
    clip.graph_proj.load_state_dict(ckpt["graph_proj"])
    clip.text_proj.load_state_dict(ckpt["text_proj"])

    clip.eval()

    return clip, tokenizer


def set_requires_grad(module: torch.nn.Module, value: bool):
    """
    Active ou désactive le calcul de gradients sur un module.

    UTILISATION :
    -------------
    - Stage 1 : on gèle l'encodeur texte
    - Stage 2 : on gèle le retrieval au début
    - Stage 3 : dégel progressif (optionnel)
    """
    for p in module.parameters():
        p.requires_grad = value


# ============================================================
# STAGE 1 : GRAPH–TEXT ALIGNMENT (CLIP-STYLE)
# ============================================================

def train_stage1_clip(train_dl, cfg: Config):
    """
    STAGE 1 : apprendre un espace latent commun graphe–texte.

    OBJECTIF
    --------
    Nous voulons que :
        sim(graph_i, text_i) >> sim(graph_i, text_j)

    Cela correspond au paradigme CLIP (image-text), adapté ici au couple :
        (molecular graph, caption).

    POURQUOI FAIRE CE STAGE AVANT LA GÉNÉRATION ?
    --------------------------------------------
    - C'est stable : loss contrastive simple, dense, bien conditionnée.
    - Cela donne un retrieval déjà performant (baseline Kaggle++),
      même sans LLM.
    - Cela fournit ensuite un prior lexical de haute qualité au générateur
      (essentiel pour BLEU).

    LIEN COURS
    ----------
    - Contrastive learning / InfoNCE
    - CLIP : température apprenable + loss symétrique

    PARAMÈTRES ENTRAÎNÉS
    --------------------
    On entraîne :
        - l’encodeur graphe (MPNN)
        - les projections (graph_proj, text_proj)
    On gèle :
        - l’encodeur texte (T5 encoder) par design

    Pourquoi geler l’encodeur texte ?
    ---------------------------------
    - Réduit la variance : le texte sert d’ancre sémantique stable
    - Empêche l’espace latent de dériver dans deux directions en même temps
    - VRAM / temps : fine-tuner un encodeur complet est coûteux et souvent inutile

    Returns
    -------
    clip : GraphTextCLIP entraîné sur l’alignement
    tokenizer : tokenizer T5 (utilisé partout ensuite)
    """

    print("\n" + "=" * 70)
    print("STAGE 1: CLIP-like graph-text alignment (retrieval pretraining)")
    print("=" * 70)

    # --------------------------------------------------------
    # Tokenizer : on prend celui de T5 pour cohérence
    # --------------------------------------------------------
    tokenizer = T5TokenizerFast.from_pretrained(cfg.t5_name)

    # --------------------------------------------------------
    # Construction du modèle Stage 1
    # --------------------------------------------------------
    """
    On instancie :
    - MPNNEncoder : encodeur graphe
    - FrozenT5TextEncoder : encodeur texte gelé
    - GraphTextCLIP : projections + logit_scale + normalisation

    Note :
    ------
    text_dim=768 car t5-base a d_model=768.
    (Attention : t5-small = 512, t5-large = 1024, etc.)
    """
    graph_enc = MPNNEncoder(
        node_vocab_sizes=cfg.node_vocab_sizes,
        edge_vocab_sizes=cfg.edge_vocab_sizes,
        hidden_dim=cfg.hidden_graph,
        num_layers=5,
        dropout=0.1,
    )

    text_enc = FrozenT5TextEncoder(cfg.t5_name, freeze=True)

    clip = GraphTextCLIP(
        graph_encoder=graph_enc,
        text_encoder=text_enc,
        graph_dim=cfg.hidden_graph,
        text_dim=768,
        shared_dim=cfg.shared_dim,
    ).to(cfg.device)

    # --------------------------------------------------------
    # Optimisation : on entraîne seulement certaines parties
    # --------------------------------------------------------
    """
    PARAMS ENTRAÎNÉS :
    - graph_encoder : apprend la structure moléculaire
    - graph_proj / text_proj : alignement dans l'espace shared

    PARAMS GELÉS :
    - text_encoder : frozen (p.requires_grad=False)

    Pourquoi entraîner text_proj si text_encoder est gelé ?
    -------------------------------------------------------
    - text_proj sert d'adaptation légère au domaine (captions chimie)
    - on ne modifie pas l'encodeur lourd, mais on permet une calibration
    """
    """
    ANTI-CHOIX STAGE 1 :
    -------------------

    ❌ Nous n'entraînons PAS le générateur (T5) en même temps que le retrieval.
    → couplage très instable
    → retrieval devient une cible mouvante ("moving target")

    ❌ Nous ne fine-tunons PAS l'encodeur texte complet.
    → coûteux
    → risque d'overfitting
    → pas nécessaire pour un retrieval performant

    ❌ Nous n'utilisons PAS de loss triplet ou margin ranking à la place.
    → InfoNCE/CLIP est plus dense (tous les négatifs du batch)
    → converge plus vite et plus stable

    ❌ Nous n'utilisons PAS d'augmentation de graphes ici.
    → potentiellement utile, mais ajoute de la variance
    → à introduire seulement une fois le pipeline stable
    """
    params = (
        list(clip.graph_encoder.parameters())
        + list(clip.graph_proj.parameters())
        + list(clip.text_proj.parameters())
    )
    opt = torch.optim.AdamW(params, lr=cfg.stage1_lr)

    # --------------------------------------------------------
    # Boucle d'entraînement
    # --------------------------------------------------------
    clip.train()

    for ep in range(cfg.stage1_epochs):
        total_loss = 0.0
        pbar = tqdm(train_dl, desc=f"[Stage1] epoch {ep+1}/{cfg.stage1_epochs}")

        for batch_graph, descs, _idxs in pbar:
            batch_graph = batch_graph.to(cfg.device)

            # Tokenisation batch de descriptions (texte brut → input_ids)
            tok = tokenizer(
                descs,
                padding=True,
                truncation=True,
                max_length=cfg.max_in_len,
                return_tensors="pt",
            ).to(cfg.device)

            # Encodage dans l'espace latent
            g = clip.encode_graph(batch_graph)                         # [B, D]
            t = clip.encode_text(tok["input_ids"], tok["attention_mask"])  # [B, D]

            # --------------------------------------------------------
            # Loss contrastive type CLIP (symétrique)
            # --------------------------------------------------------
            """
            Logits:
                logits_ij = s * <g_i, t_j>
            avec g_i, t_j normalisés => dot product = cosine similarity

            s = exp(logit_scale) = 1/temperature
            → la température est apprenable comme dans CLIP

            Labels:
                labels = [0, 1, 2, ..., B-1]
            signifiant : g_i doit matcher t_i

            Loss symétrique:
                CE(logits, labels) + CE(logits.T, labels)
            => encourage alignement dans les deux sens.
            """
            logit_scale = clip.logit_scale.exp()
            logits = logit_scale * (g @ t.t())        # [B, B]
            labels = torch.arange(g.size(0), device=cfg.device)

            loss_g2t = F.cross_entropy(logits, labels)
            loss_t2g = F.cross_entropy(logits.t(), labels)
            loss = 0.5 * (loss_g2t + loss_t2g)

            # Optimisation standard
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"[Stage1] epoch {ep+1} avg loss: {total_loss/len(train_dl):.4f}")

    # --------------------------------------------------------
    # Sauvegarde des poids Stage 1
    # --------------------------------------------------------
    """
    Pourquoi sauvegarder séparément ?
    ---------------------------------
    - Stage 2 dépend de Stage 1
    - possibilité de faire ablations :
        * retrieval-only
        * retrieval + generator
    - reproductibilité : on fige un point de référence

    On sauvegarde uniquement :
    - graph_encoder
    - graph_proj
    - text_proj
    (text_encoder est gelé, donc pas nécessaire)
    """
    torch.save(
        {
            "graph_encoder": clip.graph_encoder.state_dict(),
            "graph_proj": clip.graph_proj.state_dict(),
            "text_proj": clip.text_proj.state_dict(),
        },
        cfg.stage1_path,
    )
    print(f"[Stage1] Saved weights -> {cfg.stage1_path}")

    return clip, tokenizer


def train_stage2_t5(train_dl, clip: GraphTextCLIP, tokenizer, cfg: Config):
    """
    STAGE 2 : Retrieval-Augmented Generation (Rewrite).

    OBJECTIF
    --------
    Pour chaque graphe :
        1) encoder le graphe
        2) récupérer une ou plusieurs captions proches (retrieval)
        3) conditionner T5 via :
            - un prompt fixe (instruction)
            - un soft prompt (embedding du graphe)
            - la caption récupérée
        4) apprendre à produire la description GT

    Cette étape est RESPONSABLE de :
    - la qualité finale Kaggle
    - l'équilibre BLEU / BERTScore

    CONTRAINTES CLÉS
    ----------------
    - le retrieval doit être STABLE
    - le graphe ne doit PAS "fuir" vers la sortie
    - le LLM doit corriger, pas halluciner

    ANTI-CHOIX STAGE 2 :
    -------------------

    ❌ Nous n'entraînons PAS le retrieval end-to-end dès le départ.
    → sinon le retrieval s'adapte au bruit du générateur

    ❌ Nous ne backpropagons PAS à travers le retrieval.
    → conceptuellement faux (mémoire externe)
    → instable

    ❌ Nous n'utilisons PAS beam search pendant l'entraînement.
    → teacher forcing suffit
    → beam = uniquement inference

    ❌ Nous n'utilisons PAS plusieurs captions concaténées.
    → trop long
    → bruit sémantique

    """

    print("\n" + "=" * 70)
    print("STAGE 2: Retrieval-Augmented Generation (rewrite)")
    print("=" * 70)
    os.makedirs(cfg.stage2_ckpt_dir, exist_ok=True)
    device = cfg.device

    start_epoch = 0
    best_loss = float("inf")

    if cfg.resume_stage2:
        ckpts = sorted(glob.glob(os.path.join(cfg.stage2_ckpt_dir, "stage2_epoch_*.pt")))
        if len(ckpts) > 0:
            latest_ckpt = ckpts[-1]
            print(f"[Stage2] Resuming from {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location=device)
            generator.load_state_dict(ckpt["generator"])
            opt.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"]
            best_loss = ckpt.get("best_loss", best_loss)
            if "graph_encoder" in ckpt:
                clip.graph_encoder.load_state_dict(ckpt["graph_encoder"])
            if "graph_proj" in ckpt:
                clip.graph_proj.load_state_dict(ckpt["graph_proj"])


    # --------------------------------------------------------
    # Construction de l'index de retrieval
    # --------------------------------------------------------
    """
    On construit un index simple en mémoire :
    - embeddings texte normalisés
    - recherche top-k par similarité cosinus

    POURQUOI PAS FAISS ?
    --------------------
    - ~32k captions → torch.topk suffit
    - simplicité > micro-optimisation
    """
    clip.eval()
    retriever = RetrievalIndex(clip, tokenizer, device)
    retriever.build(train_dl)
    # --------------------------------------------------------
    # Générateur : T5 + Soft Prompt + LoRA
    # --------------------------------------------------------
    generator = GraphSoftPromptT5(
        model_name=cfg.t5_name,
        graph_emb_dim=cfg.hidden_graph,
        prompt_len=cfg.prompt_len,
    ).to(device)

    # --------------------------------------------------------
    # Chargement des poids Stage 1
    # --------------------------------------------------------
    """
    Le générateur dépend de :
    - graph_encoder entraîné en Stage 1
    - espace latent de retrieval

    IMPORTANT :
    -----------
    - On partage le graph_encoder entre clip et generator
    - Cela garantit la cohérence retrieval ↔ génération
    """
    ckpt = torch.load(cfg.stage1_path, map_location=device)
    clip.graph_encoder.load_state_dict(ckpt["graph_encoder"])
    clip.graph_proj.load_state_dict(ckpt["graph_proj"])
    clip.text_proj.load_state_dict(ckpt["text_proj"])

    # --------------------------------------------------------
    # Gel initial : stabilisation
    # --------------------------------------------------------
    """
    STRATÉGIE DE GEL :
    -----------------
    Epochs 0 → freeze_warmup_epochs:
        - graph_encoder : GELÉ
        - retrieval     : GELÉ
        - T5 (LoRA)     : entraîné

    Motivation :
    ------------
    - le retrieval fournit un prior fixe
    - le LLM apprend d'abord à exploiter ce prior
    """
    set_requires_grad(clip.graph_encoder, False)
    set_requires_grad(clip.graph_proj, False)
    # IMPORTANT : on ne touche pas aux requires_grad du générateur,
    # on laisse PEFT décider (LoRA + soft prompt uniquement).
    opt = torch.optim.AdamW(
        [p for p in generator.parameters() if p.requires_grad],
        lr=cfg.stage2_lr,
    )

    # --------------------------------------------------------
    # Boucle d'entraînement
    # --------------------------------------------------------
    generator.train()

    print(
        "text_proj requires_grad:",
        any(p.requires_grad for p in clip.text_proj.parameters())
    )
    for ep in range(start_epoch, cfg.stage2_epochs):
        print(f"\n[Stage2] epoch {ep+1}/{cfg.stage2_epochs}")
        epoch_loss = 0.0
        num_batches = 0
        # ----------------------------------------------------
        # Dégel progressif
        # ----------------------------------------------------
        if ep == cfg.freeze_warmup_epochs:
            """
            À partir de maintenant :
            - on autorise une adaptation fine du graphe
            - mais avec un LR indirectement plus faible
            """
            set_requires_grad(clip.graph_encoder, True)
            set_requires_grad(clip.graph_proj, True)
            clip.graph_encoder.train()
            clip.graph_proj.train()
            opt = torch.optim.AdamW(
                [
                    {"params": generator.parameters(), "lr": cfg.stage2_lr},
                    {"params": clip.graph_encoder.parameters(), "lr": cfg.stage2_lr * 0.2},
                    {"params": clip.graph_proj.parameters(), "lr": cfg.stage2_lr * 0.2},
                ]
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=cfg.scheduler_factor,
                patience=cfg.scheduler_patience,
                min_lr=cfg.min_lr,
                verbose=True,
            )

        pbar = tqdm(train_dl, desc=f"[Stage2] epoch {ep+1}")

        for batch_graph, descs, idxs in pbar:
            batch_graph = batch_graph.to(device)

            # ------------------------------------------------
            # Retrieval (sans gradient)
            # ------------------------------------------------
            """
            IMPORTANT :
            - retrieval = mémoire externe NON différentiable
            - aucun gradient ne doit passer ici
            """
            with torch.no_grad():
                retrieved_descs = retriever.retrieve(
                    batch_graph,
                    idxs=idxs,
                    topk=cfg.topk,
                )

            # ------------------------------------------------
            # Construction des entrées texte
            # ------------------------------------------------
            """
            Input format :
                FIXED_PROMPT + retrieved_description
            """
            inputs = [
                FIXED_PROMPT + rd
                for rd in retrieved_descs
            ]

            tok_in = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=cfg.max_in_len,
                return_tensors="pt",
            ).to(device)

            tok_out = tokenizer(
                descs,
                padding=True,
                truncation=True,
                max_length=cfg.max_out_len,
                return_tensors="pt",
            ).to(device)

            # ------------------------------------------------
            # Encodage graphe → soft prompt
            # ------------------------------------------------
            graph_emb, _ = clip.graph_encoder(batch_graph)

            # ------------------------------------------------
            # Forward T5 (teacher forcing)
            # ------------------------------------------------
            out = generator(
                graph_emb=graph_emb,
                input_ids=tok_in["input_ids"],
                attention_mask=tok_in["attention_mask"],
                labels=tok_out["input_ids"],
            )

            loss = out.loss
            epoch_loss += loss.item()
            num_batches += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        if ep >= cfg.freeze_warmup_epochs:
            scheduler.step(avg_epoch_loss)
        print(f"[Stage2] epoch {ep+1} avg loss: {avg_epoch_loss:.4f}")
        ckpt_path = os.path.join(
            cfg.stage2_ckpt_dir,
            f"stage2_epoch_{ep+1}.pt"
        )

        torch.save(
            {
            "epoch": ep + 1,
            "generator": generator.state_dict(),
            "optimizer": opt.state_dict(),
            "avg_loss": avg_epoch_loss,
            "best_loss": best_loss,

            # IMPORTANT: si tu dégèles le graphe en Stage2
            "graph_encoder": clip.graph_encoder.state_dict(),
            "graph_proj": clip.graph_proj.state_dict(),
            },
            ckpt_path,
        )

        print(f"[Stage2] Saved checkpoint: {ckpt_path}")
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(
                generator.state_dict(),
                "best_generator.pt"
            )
    # --------------------------------------------------------
    # Sauvegarde Stage 2
    # --------------------------------------------------------
    torch.save(
        {
            "generator": generator.state_dict(),
            "graph_encoder": clip.graph_encoder.state_dict(),
            "graph_proj": clip.graph_proj.state_dict(),
        },
        cfg.stage2_path,
    )
    print(f"[Stage2] Saved weights -> {cfg.stage2_path}")
    

    return generator

def train_stage3_metric_aware():
    """
    STAGE 3 (OPTIONNEL, AVANCÉ).

    Idée :
    ------
    Optimiser directement :
        - BLEU-4
        - BERTScore

    Méthodes possibles :
    --------------------
    - Minimum Risk Training (MRT)
    - REINFORCE

    POURQUOI CE N'EST PAS FAIT ICI ?
    --------------------------------
    - Implémentation longue et délicate
    - Facile à casser
    - À tenter UNIQUEMENT quand le pipeline est stable
    """
    raise NotImplementedError

def main():
    set_seed(42)
    cfg = Config()

    # --------------------------------------------------------
    # Dataset & DataLoader
    # --------------------------------------------------------
    train_ds = PreprocessedGraphTextDataset(cfg.train_graphs)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_graph_text,
        num_workers=cfg.num_workers,
    )

    # --------------------------------------------------------
    # Inférer les tailles de vocabulaire (UNE FOIS)
    # --------------------------------------------------------
    node_vocab_sizes, edge_vocab_sizes = infer_vocab_sizes_from_dataset(train_ds)

    cfg.node_vocab_sizes = node_vocab_sizes
    cfg.edge_vocab_sizes = edge_vocab_sizes
    print("Node vocab sizes:", node_vocab_sizes)
    print("Edge vocab sizes:", edge_vocab_sizes)
    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------
    if os.path.exists(cfg.stage1_path):
        print("[Main] Found pretrained Stage 1 weights. Skipping Stage 1.")

        clip, tokenizer = load_stage1_clip(cfg)
    else:
        clip, tokenizer = train_stage1_clip(train_dl, cfg)


    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------
    train_stage2_t5(train_dl, clip, tokenizer, cfg)

    print("\nTraining completed successfully.")

if __name__ == "__main__":
    main()