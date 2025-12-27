import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from retrieval.architecture import DeepGINEEncoder, GraphTextCLIP, TextEncoder, MolecularCaptionModel
from retrieval.retrieval import RetrievalIndex
from utils.data_utils import (
    load_id2emb,
    load_descriptions_from_graphs,
    PreprocessedGraphDataset,
    collate_fn,
)

# =========================
# CONFIG (match training_pipeline.py)
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_GRAPHS = "data/train_graphs.pkl"
TEST_GRAPHS = "data/test_graphs.pkl"
TRAIN_EMB_CSV = "data/train_embeddings.csv"

# Poids Stage1 CLIP de ta v0.6 (contient graph_encoder + text_encoder)
CLIP_WEIGHTS = "weights_stage1_clip.pt"
FINAL_MODEL_WEIGHTS = "weights_stage2_final.pt"
SUBMISSION_PATH = "submission.csv"

# Hyperparams (identiques à training_pipeline.py)
NODE_VOCAB = [200, 20]
EDGE_VOCAB = [50, 20]
HIDDEN_GRAPH = 300
HIDDEN_TEXT = 256
PAD_IDX = 0

# Retrieval
INDEX_BATCH_SIZE = 32
TOP_K = 5
NEIGHBOR_RANK = 0  # 0 => meilleur voisin (en test il n'y a pas de leak)

MAX_GEN_LEN = 200
BOS_IDX = 1
EOS_IDX = 2


def main():
    # =========================
    # Sanity checks
    # =========================
    if not os.path.exists(TRAIN_GRAPHS):
        raise FileNotFoundError(f"Missing: {TRAIN_GRAPHS}")
    if not os.path.exists(TEST_GRAPHS):
        raise FileNotFoundError(f"Missing: {TEST_GRAPHS}")
    if not os.path.exists(TRAIN_EMB_CSV):
        raise FileNotFoundError(f"Missing: {TRAIN_EMB_CSV}")
    if not os.path.exists(FINAL_MODEL_WEIGHTS):
        raise FileNotFoundError(f"Missing: {FINAL_MODEL_WEIGHTS}")

    print(f"Device: {DEVICE}")

    # =========================
    # Load train embeddings (token ids stored as floats in CSV)
    # =========================
    print("Loading train embeddings/tokens...")
    train_emb = load_id2emb(TRAIN_EMB_CSV)  # dict id -> Tensor(seq_len) (float)
    ds_train = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)

    # =========================
    # Compute VOCAB_SIZE exactly like training_pipeline.py
    # =========================
    print("Scanning vocab size (same logic as training)...")
    max_id = 0
    temp_dl = DataLoader(ds_train, batch_size=64, shuffle=False, collate_fn=collate_fn)
    for _, txt in tqdm(temp_dl, desc="Scan Vocab"):
        max_id = max(max_id, txt.long().max().item())
    vocab_size = max_id + 100
    print(f"Detected VOCAB_SIZE: {vocab_size}")

    # =========================
    # Instantiate model EXACTLY with your class signatures
    # =========================
    print("Loading model...")
    graph_encoder = DeepGINEEncoder(NODE_VOCAB, EDGE_VOCAB, HIDDEN_GRAPH)
    text_encoder = TextEncoder(vocab_size, HIDDEN_TEXT)  # signature: (vocab_size, d_model, ...)

    clip_model = GraphTextCLIP(
        graph_encoder,
        text_encoder,
        HIDDEN_GRAPH,
        HIDDEN_TEXT
    ).to(DEVICE)

    ckpt = torch.load(CLIP_WEIGHTS, map_location=DEVICE)

    clip_model.graph_encoder.load_state_dict(ckpt["graph_encoder"])
    clip_model.text_encoder.load_state_dict(ckpt["text_encoder"])
    clip_model.graph_proj.load_state_dict(ckpt["graph_proj"])
    clip_model.text_proj.load_state_dict(ckpt["text_proj"])
    clip_model.logit_scale.data = ckpt["logit_scale"]
    #graph_encoder.eval()
    #text_encoder.eval()
    clip_model.eval()
    # =========================
    # Build retrieval index from TRAIN (same as stage2)
    # captions_tokens = list of tensors [seq_len] (CPU)
    # =========================
    print("Building retrieval index from train graphs...")
    index_dl = DataLoader(ds_train, batch_size=INDEX_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    captions_tokens = []
    for _, caps in tqdm(index_dl, desc="Extract train captions"):
        caps = torch.clamp(caps.long(), min=0, max=vocab_size - 1)
        for i in range(caps.size(0)):
            captions_tokens.append(caps[i].clone().detach().cpu())

    retriever = RetrievalIndex(device=DEVICE)
    # IMPORTANT: build_index attend "for batch in dataloader: graph_batch = batch[0]"
    # => on lui passe un dataloader qui yield (batch_graph, caps)
    retriever.build_index(clip_model, index_dl, captions_tokens)

    # =========================
    # Load train descriptions for retrieval-only submission
    # =========================
    # This is the only guaranteed way to output valid text with the provided codebase.
    id2desc = load_descriptions_from_graphs(TRAIN_GRAPHS)

    # Also keep train IDs list to map neighbor index -> train id
    train_ids = ds_train.ids  # list aligned with ds_train order (and index order)

    # =========================
    # Load test graphs
    # =========================
    print("Loading test graphs...")
    ds_test = PreprocessedGraphDataset(TEST_GRAPHS, emb_dict=None)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # =========================
    # Inference
    # =========================
    print("Running inference on test...")
    rows = []

    for batch_graph in tqdm(dl_test, desc="Infer"):
        # batch_graph is a PyG Batch (batch_size=1)
        nn_indices, _ = retriever.query(clip_model, batch_graph, k=TOP_K)  # [1,k]
        chosen = nn_indices[:, NEIGHBOR_RANK]  # [1]

        # Map neighbor index -> train id -> description
        neighbor_idx = int(chosen.item())
        neighbor_id = train_ids[neighbor_idx]
        retrieved_caption = id2desc[neighbor_id]
        
        caption_out = retrieved_caption #retrieval only
        # test graphs store unique id attribute
        test_id = batch_graph.id[0]
        rows.append([test_id, caption_out])

    # =========================
    # Write submission
    # =========================
    print(f"Writing: {SUBMISSION_PATH}")
    with open(SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "caption"])
        writer.writerows(rows)

    print("Done. submission.csv ready.")


if __name__ == "__main__":
    main()
