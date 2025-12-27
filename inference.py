import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from retrieval.architecture import DeepGINEEncoder, GraphTextCLIP
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
CLIP_WEIGHTS = "weights_stage1_clip.pt"

SUBMISSION_PATH = "submission.csv"

NODE_VOCAB = [200, 20]
EDGE_VOCAB = [50, 20]
HIDDEN_GRAPH = 300

INDEX_BATCH_SIZE = 32
TOP_K = 1   # retrieval-only → top-1


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
    print("[INFER] Loading model...")
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
    # Build retrieval index
    # -------------------------
    print("[INFER] Building retrieval index...")
    retriever = RetrievalIndex(device=DEVICE)
    retriever.build_index(clip_model, dl_train)

    # -------------------------
    # Load TEST dataset
    # -------------------------
    print("[INFER] Loading TEST graphs...")
    ds_test = PreprocessedGraphDataset(TEST_GRAPHS, mode=False)
    dl_test = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_test,
    )

    # -------------------------
    # Inference (retrieval-only)
    # -------------------------
    print("[INFER] Running inference...")
    rows = []

    for batch_graph in tqdm(dl_test, desc="Infer"):
        batch_graph = batch_graph.to(DEVICE)

        nn_indices, _ = retriever.query(
            clip_model,
            batch_graph,
            k=TOP_K,
        )

        train_idx = int(nn_indices[0, 0].item())
        train_graph = ds_train.graphs[train_idx]
        caption = id2desc_train[train_graph.id]

        test_graph = batch_graph.to_data_list()[0]
        rows.append([test_graph.id, caption])

    # -------------------------
    # Write submission
    # -------------------------
    print(f"[INFER] Writing {SUBMISSION_PATH}")
    with open(SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "caption"])
        writer.writerows(rows)

    print("[INFER] Done. submission.csv ready.")


if __name__ == "__main__":
    main()
