from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import T5TokenizerFast

from architecture import (
    MPNNEncoder,
    FrozenT5TextEncoder,
    GraphTextCLIP,
)
from data_utils import (
    PreprocessedGraphTextDataset,
    collate_graph_text,
)
from retrieval import RetrievalIndex
from eval_metrics import evaluate_all


# ============================================================
# CONFIG
# ============================================================

class EvalConfig:
    # Paths
    train_graphs = "data/train_graphs.pkl"
    val_graphs = "data/validation_graphs.pkl"
    stage1_path = "backup/checkpoints_stage2_bis/weights_stage1_clip.pt"

    # Model
    t5_name = "t5-base"
    hidden_graph = 300

    # Vocab sizes (DOIVENT correspondre au preprocessing)
    node_vocab_sizes = [86, 3, 7, 10, 5, 5, 7, 2, 2]
    edge_vocab_sizes = [13, 4, 2]

    # Dataloader
    batch_size = 32

    # Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# RETRIEVAL-ONLY EVAL ON VALIDATION
# ============================================================

@torch.no_grad()
def run_retrieval_on_validation(cfg: EvalConfig):
    device = cfg.device
    print(f"[Retrieval-only VAL] Using device: {device}")

    tokenizer = T5TokenizerFast.from_pretrained(cfg.t5_name)

    # --------------------------------------------------------
    # Load CLIP model (Stage 1 only)
    # --------------------------------------------------------
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

    # Load Stage 1 weights
    ckpt = torch.load(cfg.stage1_path, map_location=device)
    clip.graph_encoder.load_state_dict(ckpt["graph_encoder"])
    clip.graph_proj.load_state_dict(ckpt["graph_proj"])
    clip.text_proj.load_state_dict(ckpt["text_proj"])

    clip.eval()
    print("[Retrieval-only VAL] Loaded Stage 1 CLIP weights")

    # --------------------------------------------------------
    # Build retrieval index from TRAIN set
    # --------------------------------------------------------
    train_ds = PreprocessedGraphTextDataset(cfg.train_graphs)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_graph_text,
    )

    retriever = RetrievalIndex(clip, tokenizer, device)
    retriever.build(train_dl)

    # --------------------------------------------------------
    # Validation set
    # --------------------------------------------------------
    val_ds = PreprocessedGraphTextDataset(cfg.val_graphs)
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_graph_text,
    )

    all_preds = []
    all_refs = []

    print("[Retrieval-only VAL] Running retrieval on validation set...")

    for batch_graph, descs, idxs in tqdm(val_dl):
        batch_graph = batch_graph.to(device)

        # IMPORTANT: idxs fournis → empêche le self-retrieval
        retrieved_descs = retriever.retrieve(
            batch_graph,
            idxs=idxs,
            topk=1,
        )

        all_preds.extend(retrieved_descs)
        all_refs.extend(descs)

    print(f"[Retrieval-only VAL] Done. {len(all_preds)} samples evaluated.")

    return all_preds, all_refs


# ============================================================
# MAIN (debug / sanity check)
# ============================================================

if __name__ == "__main__":
    cfg = EvalConfig()
    preds, refs = run_retrieval_on_validation(cfg)
    bleu4, bert_f1 = evaluate_all(preds, refs, device=cfg.device)
    # Afficher quelques exemples pour sanity check
    for i in range(3):
        print("\n---")
        print("GT :", refs[i])
        print("RET:", preds[i])
