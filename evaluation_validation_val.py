from __future__ import annotations

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from architecture import DeepGINEEncoder, GraphTextCLIP, TextEncoder  # v0.6
from retrieval import RetrievalIndex
from data_utils import (
    load_id2emb,
    load_descriptions_from_graphs,
    PreprocessedGraphDataset,
    collate_fn,
)
from eval_metrics import evaluate_all


# =========================
# CONFIG (doit matcher ta v0.6)
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS = "data/validation_graphs.pkl"
TRAIN_EMB_CSV = "data/train_embeddings.csv"

# Poids Stage1 CLIP de ta v0.6 (contient graph_encoder + text_encoder)
CLIP_WEIGHTS = "weights_stage1_clip.pt"

# Hyperparams (identiques à inference.py / training_pipeline.py v0.6)
NODE_VOCAB = [200, 20]
EDGE_VOCAB = [50, 20]
HIDDEN_GRAPH = 300
HIDDEN_TEXT = 256
PAD_IDX = 0

# Retrieval
INDEX_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
TOP_K = 5
NEIGHBOR_RANK = 0  # 0 => top-1 voisin


def _detect_vocab_size_from_train_tokens(ds_train: PreprocessedGraphDataset) -> int:
    """
    Reproduit exactement la logique de inference.py v0.6 :
    scan du max token id sur train_embeddings.csv puis +100 marge.
    """
    max_id = 0
    dl = DataLoader(ds_train, batch_size=64, shuffle=False, collate_fn=collate_fn)
    for _, txt in tqdm(dl, desc="Scan Vocab"):
        max_id = max(max_id, txt.long().max().item())
    return max_id + 100


@torch.no_grad()
def main():
    # -------------------------
    # Sanity checks
    # -------------------------
    for path in [TRAIN_GRAPHS, VAL_GRAPHS, TRAIN_EMB_CSV, CLIP_WEIGHTS]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

    print(f"[v0.6 VAL EVAL] Device: {DEVICE}")

    # -------------------------
    # Load train token sequences (from CSV)
    # -------------------------
    print("[v0.6 VAL EVAL] Loading train token sequences...")
    train_emb = load_id2emb(TRAIN_EMB_CSV)  # dict id -> Tensor(seq_len) float
    ds_train = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)

    # Vocab size EXACT comme ton inference.py
    print("[v0.6 VAL EVAL] Detecting vocab size...")
    vocab_size = _detect_vocab_size_from_train_tokens(ds_train)
    print(f"[v0.6 VAL EVAL] Detected VOCAB_SIZE: {vocab_size}")

    # -------------------------
    # Instantiate encoders + load CLIP weights
    # -------------------------
    print("[v0.6 VAL EVAL] Loading Stage1 CLIP encoders...")
    graph_encoder = DeepGINEEncoder(NODE_VOCAB, EDGE_VOCAB, hidden_dim=HIDDEN_GRAPH).to(DEVICE)
    text_encoder = TextEncoder(vocab_size, HIDDEN_TEXT).to(DEVICE)

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

    # -------------------------
    # Build retrieval index from TRAIN (same as inference.py)
    # -------------------------
    print("[v0.6 VAL EVAL] Building retrieval index from TRAIN...")
    index_dl = DataLoader(ds_train, batch_size=INDEX_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    captions_tokens = []
    for _, caps in tqdm(index_dl, desc="Extract train captions"):
        caps = torch.clamp(caps.long(), min=0, max=vocab_size - 1)
        for i in range(caps.size(0)):
            captions_tokens.append(caps[i].clone().detach().cpu())

    retriever = RetrievalIndex(device=DEVICE)
    retriever.build_index(clip_model, index_dl, captions_tokens)

    # Pour sortir du texte (pas des tokens), on map idx voisin -> train_id -> train_description
    id2desc_train = load_descriptions_from_graphs(TRAIN_GRAPHS)
    train_ids = ds_train.ids  # ordre cohérent avec l'index car shuffle=False

    # -------------------------
    # Load validation graphs + GT descriptions
    # -------------------------
    print("[v0.6 VAL EVAL] Loading validation graphs...")
    ds_val = PreprocessedGraphDataset(VAL_GRAPHS, emb_dict=None)
    dl_val = DataLoader(ds_val, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    id2desc_val = load_descriptions_from_graphs(VAL_GRAPHS)

    # -------------------------
    # Retrieval-only predictions on validation
    # -------------------------
    preds, refs = [], []

    print("[v0.6 VAL EVAL] Running retrieval on validation...")
    for batch_graph in tqdm(dl_val, desc="VAL Retrieval"):
        # batch_graph is PyG Batch
        nn_indices, _ = retriever.query(clip_model, batch_graph, k=TOP_K)  # [B, k]
        chosen = nn_indices[:, NEIGHBOR_RANK]  # [B]

        # Map each neighbor index -> train id -> train description
        for b in range(chosen.size(0)):
            neighbor_idx = int(chosen[b].item())
            neighbor_id = train_ids[neighbor_idx]
            pred_text = id2desc_train[neighbor_id]

            # Reference = validation GT text, keyed by graph id
            val_id = batch_graph.id[b]
            ref_text = id2desc_val[val_id]

            preds.append(pred_text)
            refs.append(ref_text)

    print(f"[v0.6 VAL EVAL] Collected {len(preds)} predictions.")

    # -------------------------
    # Metrics (officielles)
    # -------------------------
    bleu4, bert_f1 = evaluate_all(preds, refs, device=DEVICE)

    print("\n[v0.6 VAL EVAL] Final metrics:")
    print(f"  BLEU-4: {bleu4:.4f}")
    print(f"  BERTScore F1: {bert_f1:.4f}")


if __name__ == "__main__":
    main()
