from __future__ import annotations

import os
import pickle
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from retrieval.cross_encoder.cross_encoder import GraphTextCrossEncoder
from retrieval.reranker import rerank_topk_hybrid, rerank_topk_mbr, rerank_topk_mbr_weighted
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
CLIP_WEIGHTS = "other/weights_stage1_clip.pt"
CROSS_ENCODER_WEIGHTS = "other/weights_cross_encoder.pt"

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

    text_encoder = MiniLMTextEncoder(device=DEVICE)

    clip_model = GraphTextCLIP(
        graph_encoder=graph_encoder,
        graph_dim=HIDDEN_GRAPH,
        text_encoder=text_encoder,
    ).to(DEVICE)

    checkpoint = torch.load(CLIP_WEIGHTS, map_location=DEVICE)
    clip_model.graph_encoder.load_state_dict(checkpoint["graph_encoder"])
    clip_model.graph_proj.load_state_dict(checkpoint["graph_proj"])
    clip_model.logit_scale.data = checkpoint["logit_scale"]

    clip_model.eval()

    # =========================================================
    # Load Cross-Encoder
    # =========================================================
    ckpt = torch.load(CROSS_ENCODER_WEIGHTS, map_location=DEVICE)

    cross_encoder = GraphTextCrossEncoder(
        graph_dim=ckpt["graph_dim"],
        text_dim=ckpt["text_dim"],
    ).to(DEVICE)

    cross_encoder.load_state_dict(ckpt["state_dict"])
    cross_encoder.eval()

    print("[VAL EVAL] Cross-encoder loaded.")

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

        nn_indices, scores = retriever.query(
            clip_model,
            batch_graph,
            k=TOP_K,
        )

        val_graphs = batch_graph.to_data_list()
        with torch.no_grad():
            graph_embs = clip_model.encode_graph(batch_graph)  # [B, D]

        for b in range(len(val_graphs)):
            # Top-k indices and scores for this graph
            idx_b = nn_indices[b].tolist()    # [k]
            scores_b = scores[b].tolist()     # [k]

            # Candidate captions from TRAIN
            captions_b = []
            for train_idx in idx_b:
                train_graph = ds_train.graphs[int(train_idx)]
                captions_b.append(id2desc_train[train_graph.id])
            
            
            #  RERANK HERE
            pred_text = rerank_topk_hybrid(
                captions=captions_b,
                graph_scores=scores_b,
                text_encoder=text_encoder,
                alpha=0.7,
            )
            '''
            # CROSS ENCODER HERE
            with torch.no_grad():
                # Graph embedding (1 seul graphe)
                graph_emb = graph_embs[b]

                ce_scores = []
                text_embs = text_encoder(captions_b)  # [k, Dt]
                for i in range(len(captions_b)):
                    score = cross_encoder(graph_emb, text_embs[i])
                    ce_scores.append(score.item())

            best_idx = int(torch.tensor(ce_scores).argmax())
            pred_text = captions_b[best_idx]
            '''
            # Reference (VAL)
            val_graph = val_graphs[b]
            ref_text = id2desc_val[val_graph.id]

            preds.append(pred_text)
            refs.append(ref_text)

    print(f"[VAL EVAL] Collected {len(preds)} predictions.")
    df = pd.DataFrame({"prediction": preds, "ground_truth": refs})
    df.to_csv("pred_v_reel.csv")
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
