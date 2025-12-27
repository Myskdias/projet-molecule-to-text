from __future__ import annotations

import os
import pickle
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from retrieval.architecture import DeepGINEEncoder, GraphTextCLIP
from retrieval.retrieval import RetrievalIndex
from retrieval.text_encoder import MiniLMTextEncoder
from utils.data_utils import (
    PreprocessedGraphDataset,
    collate_fn,
    load_descriptions_from_graphs,
)
from metrics.eval_metrics import evaluate_all


# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS = "data/validation_graphs.pkl"
CLIP_WEIGHTS = "checkpoints/checkpoint_1_15.pt"

NODE_VOCAB = [200, 20]
EDGE_VOCAB = [50, 20]
HIDDEN_GRAPH = 300

INDEX_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
TOP_K = 5
NEIGHBOR_RANK = 0  # top-1


# =========================================================
# MAIN EVALUATION
# =========================================================
@torch.no_grad()
def main():

    # -----------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------
    for path in [TRAIN_GRAPHS, VAL_GRAPHS, CLIP_WEIGHTS]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    print(f"[VAL EVAL] Device: {DEVICE}")

    # -----------------------------------------------------
    # Load TRAIN dataset
    # -----------------------------------------------------
    print("[VAL EVAL] Loading TRAIN graphs...")
    ds_train = PreprocessedGraphDataset(TRAIN_GRAPHS)
    dl_train = DataLoader(
        ds_train,
        batch_size=INDEX_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    id2desc_train = load_descriptions_from_graphs(TRAIN_GRAPHS)

    # -----------------------------------------------------
    # Load VAL dataset
    # -----------------------------------------------------
    print("[VAL EVAL] Loading VAL graphs...")
    ds_val = PreprocessedGraphDataset(VAL_GRAPHS)
    dl_val = DataLoader(
        ds_val,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    id2desc_val = load_descriptions_from_graphs(VAL_GRAPHS)

    # -----------------------------------------------------
    # Build CLIP model (MiniLM + GNN)
    # -----------------------------------------------------
    print("[VAL EVAL] Loading CLIP model...")

    graph_encoder = DeepGINEEncoder(
        NODE_VOCAB,
        EDGE_VOCAB,
        hidden_dim=HIDDEN_GRAPH,
    ).to(DEVICE)

    text_encoder = MiniLMTextEncoder(device=DEVICE, use_lora=False)

    clip_model = GraphTextCLIP(
        graph_encoder=graph_encoder,
        graph_dim=HIDDEN_GRAPH,
        text_encoder=text_encoder,
    ).to(DEVICE)

    checkpoint = torch.load(CLIP_WEIGHTS, map_location=DEVICE)
    clip_model.graph_encoder.load_state_dict(checkpoint["graph_encoder"])
    clip_model.graph_proj.load_state_dict(checkpoint["graph_proj"])
    #clip_model.text_encoder.load_state_dict(checkpoint["text_encoder"])
    clip_model.logit_scale.data = checkpoint["logit_scale"]

    clip_model.eval()

    # -----------------------------------------------------
    # Build retrieval index (TRAIN graphs)
    # -----------------------------------------------------
    print("[VAL EVAL] Building retrieval index...")
    retriever = RetrievalIndex(device=DEVICE)
    retriever.build_index(clip_model, dl_train)

    # -----------------------------------------------------
    # Retrieval on VAL set
    # -----------------------------------------------------
    preds: List[str] = []
    refs: List[str] = []

    print("[VAL EVAL] Running retrieval...")
    for batch_graph, _ in tqdm(dl_val, desc="Retrieval"):
        batch_graph = batch_graph.to(DEVICE)

        nn_indices, _ = retriever.query(
            clip_model,
            batch_graph,
            k=TOP_K,
        )

        chosen = nn_indices[:, NEIGHBOR_RANK]  # top-1

        val_graphs = batch_graph.to_data_list()

        for b in range(len(val_graphs)):
            # Prediction (TRAIN)
            train_idx = int(chosen[b].item())
            train_graph = ds_train.graphs[train_idx]
            pred_text = id2desc_train[train_graph.id]

            # Reference (VAL)
            val_graph = val_graphs[b]
            ref_text = id2desc_val[val_graph.id]

            preds.append(pred_text)
            refs.append(ref_text)

    print(f"[VAL EVAL] Collected {len(preds)} predictions.")

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    bleu4, bert_f1 = evaluate_all(preds, refs, device=DEVICE)

    print("\n[VAL EVAL] Final metrics:")
    print(f"  BLEU-4: {bleu4:.4f}")
    print(f"  BERTScore F1: {bert_f1:.4f}")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()
