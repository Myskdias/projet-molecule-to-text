import os
import csv
from typing import List, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from editor.text_editor import graph_consistency_fix
from retrieval.architecture import DeepGINEEncoder, GraphTextCLIP
from retrieval.reranker import rerank_topk_hybrid_pruned
from retrieval.retrieval import RetrievalIndex
from retrieval.text_encoder import MiniLMTextEncoder
from utils.data_utils import (
    PreprocessedGraphDataset,
    collate_fn,
    collate_fn_test,
    load_descriptions_from_graphs,
)

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_GRAPHS = "data/train_graphs.pkl"
TEST_GRAPHS = "data/test_graphs.pkl"
CLIP_WEIGHTS = "other/weights_stage1_clip.pt"

SUBMISSION_PATH = "submission.csv"

NODE_VOCAB = [200, 20]
EDGE_VOCAB = [50, 20]
HIDDEN_GRAPH = 300

INDEX_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
TOP_K = 1  # match evaluation_validation_val.py


def _unpack_batch(batch: Union[torch.Tensor, Tuple]) -> torch.Tensor:
    """Handle both (batch_graph, y) and batch_graph-only loaders."""
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


# =========================================================
# MAIN
# =========================================================
@torch.no_grad()
def main():

    # -------------------------
    # Sanity checks
    # -------------------------
    for path in [TRAIN_GRAPHS, TEST_GRAPHS, CLIP_WEIGHTS]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    print(f"[INFER] Device: {DEVICE}")

    # -------------------------
    # Load TRAIN dataset
    # -------------------------
    print("[INFER] Loading TRAIN graphs...")
    ds_train = PreprocessedGraphDataset(TRAIN_GRAPHS)
    dl_train = DataLoader(
        ds_train,
        batch_size=INDEX_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    id2desc_train = load_descriptions_from_graphs(TRAIN_GRAPHS)

    # -------------------------
    # Load CLIP model
    # -------------------------
    print("[INFER] Loading CLIP model...")
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

    ckpt = torch.load(CLIP_WEIGHTS, map_location=DEVICE)
    clip_model.graph_encoder.load_state_dict(ckpt["graph_encoder"])
    clip_model.graph_proj.load_state_dict(ckpt["graph_proj"])
    clip_model.logit_scale.data = ckpt["logit_scale"]

    clip_model.eval()

    # -------------------------
    # Build retrieval index (TRAIN only)
    # -------------------------
    print("[INFER] Building retrieval index on TRAIN...")
    retriever = RetrievalIndex(device=DEVICE)
    retriever.build_index(clip_model, dl_train)

    # -------------------------
    # Load TEST dataset
    # -------------------------
    print("[INFER] Loading TEST graphs...")
    ds_test = PreprocessedGraphDataset(TEST_GRAPHS, mode=False)

    # Prefer collate_fn_test when available (test graphs usually have no labels).
    # Use batching to match the validation evaluation loop.
    dl_test = DataLoader(
        ds_test,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_test,
    )

    # -------------------------
    # Inference (same as evaluation_validation_val.py but without VAL-only parts)
    #   - USE_GEN = False
    #   - no editor / cross-encoder
    #   - rerank_topk_hybrid_pruned
    # -------------------------
    print("[INFER] Running inference...")
    rows: List[List[str]] = []

    for batch in tqdm(dl_test, desc="Infer"):
        batch_graph = _unpack_batch(batch).to(DEVICE)

        nn_indices, scores = retriever.query(
            clip_model,
            batch_graph,
            k=TOP_K,
        )

        test_graphs = batch_graph.to_data_list()

        for b in range(len(test_graphs)):
            idx_b = nn_indices[b].tolist()   # [k]
            scores_b = scores[b].tolist()    # [k]

            # --------------------------------------------------
            # Candidate captions from TRAIN (baseline)
            # (mirrors evaluation_validation_val.py)
            # --------------------------------------------------
            captions_b: List[str] = []
            scores_b_expanded: List[float] = []

            for train_idx, s in zip(idx_b, scores_b):
                train_graph = ds_train.graphs[int(train_idx)]
                gid = train_graph.id
                captions_b.append(id2desc_train[gid])
                scores_b_expanded.append(s)

            # --------------------------------------------------
            # RERANK (pruned hybrid)
            # --------------------------------------------------
            pred_text = rerank_topk_hybrid_pruned(
                captions=captions_b,
                graph_scores=scores_b_expanded,
                text_encoder=text_encoder,
                alpha=0.7,
            )

            # NIVEAU 0: graph consistency fix
            pred_text = graph_consistency_fix(pred_text, test_graphs[b])

            rows.append([test_graphs[b].id, pred_text])

    # -------------------------
    # Write submission
    # -------------------------
    print(f"[INFER] Writing {SUBMISSION_PATH}")
    with open(SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "description"])
        writer.writerows(rows)

    print("[INFER] Done. submission.csv ready.")


if __name__ == "__main__":
    main()
