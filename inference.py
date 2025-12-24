"""
inference.py
============

RÔLE DANS LE PROJET
------------------
Ce fichier implémente l'inférence finale du modèle pour le challenge Kaggle
"Molecular Graph Captioning".

Il est utilisé pour :
    - charger les modèles entraînés (Stage 1 + Stage 2)
    - générer une description pour chaque molécule du test set
    - produire un fichier submission.csv conforme aux attentes Kaggle

⚠️ C'EST LE FICHIER LE PLUS SENSIBLE AUX ERREURS SILENCIEUSES ⚠️

Un modèle excellent peut obtenir un score nul si :
    - les IDs sont mal alignés
    - l'ordre des prédictions est incorrect
    - le format CSV est invalide

-----------------------------------------------------------------------
PIPELINE D'INFERENCE (vue conceptuelle)
-----------------------------------------------------------------------

Test Graph
    │
    ▼
MPNNEncoder  ──► Graph Embedding
    │
    ├─────────────┐
    │             ▼
    │       Retrieval (CLIP space)
    │             │
    │     Retrieved Caption
    │             │
    ▼             ▼
Soft Prompt    Prompt Texte
        │       │
        └───────┘
             ▼
         T5 + LoRA
             ▼
     Generated Caption
             ▼
       submission.csv

-----------------------------------------------------------------------
LIENS AVEC LE COURS / CONCEPTS
-----------------------------------------------------------------------

- Retrieval-Augmented Generation (RAG)
- Inference LLM
- Kaggle Submission Protocol
- Engineering pitfalls (data alignment)

-----------------------------------------------------------------------
PHILOSOPHIE DE DESIGN
-----------------------------------------------------------------------

1) Reproductibilité
2) Simplicité
3) Zéro hypothèse implicite
4) Séparation stricte train / test

ANTI-CHOIX :
------------

❌ Nous n'utilisons PAS le test set pour construire l'index de retrieval.
   → fuite d'information
   → score Kaggle invalidé

❌ Nous n'utilisons PAS shuffle=True sur le test DataLoader.
   → ordre des IDs cassé
   → soumission invalide

❌ Nous n'utilisons PAS batch_size > 1 par défaut.
   → plus simple pour éviter les bugs d'alignement
   → batch possible mais risqué

❌ Nous n'écrivons PAS directement une liste brute dans un CSV.
   → DictWriter garantit l'ordre des colonnes

Ce fichier :
- protège contre les erreurs Kaggle classiques
- sépare strictement train / test
- garantit l'alignement ID ↔ prédiction
- rend l'inférence reproductible et auditable
"""

from __future__ import annotations

import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import T5TokenizerFast

# Modules du projet
from data_utils import PreprocessedGraphTestDataset, collate_graph_only
from architecture import (
    MPNNEncoder,
    FrozenT5TextEncoder,
    GraphTextCLIP,
    GraphSoftPromptT5,
)
from retrieval import RetrievalIndex


# ============================================================
# CONFIGURATION INFERENCE
# ============================================================

class InferenceConfig:
    """
    Configuration spécifique à l'inférence.

    POURQUOI UNE CLASSE SÉPARÉE ?
    ------------------------------
    - éviter toute confusion avec les hyperparamètres d'entraînement
    - protéger contre les erreurs de chemin / device
    """

    # --------------------
    # Paths
    # --------------------
    test_graphs: str = "data/test_graphs.pkl"
    stage1_path: str = "weights_stage1_clip.pt"
    stage2_path: str = "weights_stage2_t5.pt"
    submission_path: str = "submission.csv"

    # --------------------
    # Model
    # --------------------
    t5_name: str = "t5-base"
    prompt_len: int = 8
    hidden_graph: int = 300

    # --------------------
    # Tokenisation
    # --------------------
    max_in_len: int = 256
    max_out_len: int = 256

    # --------------------
    # Retrieval
    # --------------------
    topk: int = 1

    # --------------------
    # Generation
    # --------------------
    num_beams: int = 4
    temperature: float = 1.0

    # --------------------
    # Hardware
    # --------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# PROMPT FIXE (DOIT ÊTRE IDENTIQUE AU TRAIN)
# ============================================================

FIXED_PROMPT = (
    "Rephrase the following molecular description so that it accurately "
    "reflects the structure and roles of the given molecule.\n"
    "Description:\n"
)


# ============================================================
# FONCTION PRINCIPALE D'INFERENCE
# ============================================================

@torch.no_grad()
def run_inference(cfg: InferenceConfig):
    """
    Exécute l'inférence complète sur le test set et génère submission.csv.

    ÉTAPES
    ------
    1) Charger les modèles entraînés
    2) Construire l'index de retrieval (à partir du train)
    3) Parcourir le test set
    4) Générer une description par molécule
    5) Sauvegarder le CSV Kaggle

    IMPORTANT
    ---------
    - AUCUN gradient
    - AUCUNE donnée du test utilisée pour l'entraînement
    """

    device = cfg.device
    print(f"[Inference] Using device: {device}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------
    tokenizer = T5TokenizerFast.from_pretrained(cfg.t5_name)

    # --------------------------------------------------------
    # Chargement des modèles (Stage 1)
    # --------------------------------------------------------
    """
    On reconstruit exactement la même architecture qu'en entraînement,
    puis on charge les poids sauvegardés.

    ATTENTION :
    -----------
    Toute incohérence ici (dimensions, ordre, modules)
    => résultats invalides
    """

    graph_encoder = MPNNEncoder(
        node_vocab_sizes=None,  # ⚠️ À fournir depuis data_utils (x_map)
        edge_vocab_sizes=None,  # ⚠️ À fournir depuis data_utils (e_map)
        hidden_dim=cfg.hidden_graph,
    ).to(device)

    text_encoder = FrozenT5TextEncoder(cfg.t5_name, freeze=True)

    clip = GraphTextCLIP(
        graph_encoder=graph_encoder,
        text_encoder=text_encoder,
        graph_dim=cfg.hidden_graph,
        text_dim=768,
    ).to(device)

    # Chargement poids Stage 1
    ckpt1 = torch.load(cfg.stage1_path, map_location=device)
    clip.graph_encoder.load_state_dict(ckpt1["graph_encoder"])
    clip.graph_proj.load_state_dict(ckpt1["graph_proj"])
    clip.text_proj.load_state_dict(ckpt1["text_proj"])

    clip.eval()

    # --------------------------------------------------------
    # Chargement du générateur (Stage 2)
    # --------------------------------------------------------
    generator = GraphSoftPromptT5(
        model_name=cfg.t5_name,
        graph_emb_dim=cfg.hidden_graph,
        prompt_len=cfg.prompt_len,
    ).to(device)

    ckpt2 = torch.load(cfg.stage2_path, map_location=device)
    generator.load_state_dict(ckpt2["generator"])
    generator.eval()

    # --------------------------------------------------------
    # Construction de l'index de retrieval
    # --------------------------------------------------------
    """
    CRUCIAL :
    ---------
    L'index doit être construit UNIQUEMENT à partir du train set,
    jamais du test set.

    Ici, on suppose que l'index a été construit et sauvegardé,
    ou reconstruit depuis le train avant l'inférence.
    """
    print("[Inference] Building retrieval index from training data...")

    # ⚠️ À adapter selon votre organisation :
    # - soit recharger un index pré-calculé
    # - soit reconstruire depuis train_dl

    # Placeholder :
    # retriever = RetrievalIndex(...)
    # retriever.build(train_dl)

    # --------------------------------------------------------
    # Dataset test
    # --------------------------------------------------------
    test_ds = PreprocessedGraphTestDataset(cfg.test_graphs)
    test_dl = DataLoader(
        test_ds,
        batch_size=1,  # batch=1 pour simplicité & alignement IDs
        shuffle=False,
        collate_fn=collate_graph_only,
    )

    # --------------------------------------------------------
    # Boucle d'inférence
    # --------------------------------------------------------
    predictions = []

    for batch_graph, ids in tqdm(test_dl, desc="[Inference]"):
        batch_graph = batch_graph.to(device)

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------
        with torch.no_grad():
            retrieved_descs = retriever.retrieve(
                batch_graph,
                idxs=None,        # pas de self-retrieval en test
                topk=cfg.topk,
            )

        # ----------------------------------------------------
        # Construction input texte
        # ----------------------------------------------------
        inputs = [
            FIXED_PROMPT + retrieved_descs[0]
        ]

        tok_in = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=cfg.max_in_len,
            return_tensors="pt",
        ).to(device)

        # ----------------------------------------------------
        # Encodage graphe → soft prompt
        # ----------------------------------------------------
        graph_emb, _ = clip.graph_encoder(batch_graph)

        # ----------------------------------------------------
        # Génération
        # ----------------------------------------------------
        gen_ids = generator.generate(
            graph_emb=graph_emb,
            input_ids=tok_in["input_ids"],
            attention_mask=tok_in["attention_mask"],
            max_length=cfg.max_out_len,
            num_beams=cfg.num_beams,
            temperature=cfg.temperature,
        )

        text = tokenizer.decode(
            gen_ids[0],
            skip_special_tokens=True,
        )

        predictions.append(
            {
                "ID": ids[0],
                "description": text,
            }
        )

    # --------------------------------------------------------
    # Écriture du fichier submission.csv
    # --------------------------------------------------------
    """
    FORMAT ATTENDU PAR KAGGLE :
    --------------------------
    ID,description
    <id1>,<text1>
    <id2>,<text2>
    ...
    """

    print(f"[Inference] Writing submission to {cfg.submission_path}")

    with open(cfg.submission_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ID", "description"],
        )
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)

    print("[Inference] Done. Submission file ready.")
