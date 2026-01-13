from __future__ import annotations

import os
from typing import List, Dict
from collections import Counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer

from editor.text_editor import graph_consistency_fix
from editor.edit_model import EditModel
from editor.utils import analyze_editability

from retrieval.reranker import rerank_topk_hybrid
from retrieval.architecture import DeepGINEEncoder, GraphTextCLIP
from retrieval.text_encoder import MiniLMTextEncoder

from retrieval.text_retrieval import TextRetrievalIndex

from utils.data_utils import (
    PreprocessedGraphDataset,
    collate_fn,
    load_descriptions_from_graphs,
)

from metrics.eval_metrics import evaluate_all


# =========================================================
# CONFIG (aligné avec evaluation_validation_val.py)
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS = "data/validation_graphs.pkl"
CLIP_WEIGHTS = "other/weights_stage1_clip.pt"

NODE_VOCAB = [200, 20]
EDGE_VOCAB = [50, 20]
HIDDEN_GRAPH = 300

INDEX_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32

# Retrieval sur le pool texte
TOP_K_TEXT = 10

# Reranker prend ensuite ces TOP_K_TEXT et sort un texte final
RERANK_ALPHA = 0.7

EDITOR_CKPT = "weights/editor_level1_head.pt"
EDITOR_THRESHOLD = 0.75

# Pool B: GT(train) + generated(val)
GEN_VAL_CSV = "validation_captions.csv"   # colonnes: ID, caption_1..caption_5
NUM_GEN_CAPTIONS = 5


def load_generated_csv(csv_path: str, k: int = 5) -> Dict[str, List[str]]:
    """
    CSV attendu:
      ID, caption_1, ..., caption_k
    """
    df = pd.read_csv(csv_path)
    id2gen: Dict[str, List[str]] = {}

    expected_cols = ["ID"] + [f"caption_{i}" for i in range(1, k + 1)]
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"[GEN CSV] Missing column '{c}' in {csv_path}")

    for _, row in df.iterrows():
        gid = str(row["ID"])
        caps = []
        for i in range(1, k + 1):
            s = row[f"caption_{i}"]
            if isinstance(s, str) and len(s) > 0:
                caps.append(s)
        if len(caps) > 0:
            id2gen[gid] = caps

    return id2gen


@torch.no_grad()
def main():
    # -----------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------
    for path in [TRAIN_GRAPHS, VAL_GRAPHS, CLIP_WEIGHTS]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    if not os.path.exists(GEN_VAL_CSV):
        raise FileNotFoundError(f"Missing GEN_VAL_CSV: {GEN_VAL_CSV}")

    print(f"[VAL TEXT] Device: {DEVICE}")

    # -----------------------------------------------------
    # Load TRAIN / VAL datasets + descriptions
    # -----------------------------------------------------
    print("[VAL TEXT] Loading TRAIN graphs.")
    ds_train = PreprocessedGraphDataset(TRAIN_GRAPHS)
    dl_train = DataLoader(
        ds_train,
        batch_size=INDEX_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    id2desc_train = load_descriptions_from_graphs(TRAIN_GRAPHS)

    print("[VAL TEXT] Loading VAL graphs.")
    ds_val = PreprocessedGraphDataset(VAL_GRAPHS)
    dl_val = DataLoader(
        ds_val,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    id2desc_val = load_descriptions_from_graphs(VAL_GRAPHS)

    # Generated captions for VAL
    print("[VAL TEXT] Loading generated captions (VAL).")
    id2gen_val = load_generated_csv(GEN_VAL_CSV, k=NUM_GEN_CAPTIONS)
    print(f"[VAL TEXT] Generated captions loaded for {len(id2gen_val)} VAL IDs.")

    # -----------------------------------------------------
    # Build CLIP model (MiniLM + GNN)
    # -----------------------------------------------------
    print("[VAL TEXT] Loading CLIP model.")
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

    # -----------------------------------------------------
    # Load Editor (identique à evaluation_validation_val.py)
    # -----------------------------------------------------
    print("[VAL TEXT] Loading editor.")
    editor_model = EditModel("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)
    ckpt = torch.load(EDITOR_CKPT, map_location=DEVICE)
    editor_model.classifier.load_state_dict(ckpt["classifier"])
    editor_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # -----------------------------------------------------
    # Build TEXT pool = GT(train) + GEN(val)
    # -----------------------------------------------------
    print("[VAL TEXT] Building caption pool.")
    pool_texts: List[str] = []
    pool_meta: List[dict] = []

    # GT(train)
    for g in ds_train.graphs:
        gid = g.id
        pool_texts.append(id2desc_train[gid])
        pool_meta.append({"mol_id": gid, "source": "gt_train"})
    
    # GEN(val)
    for gid, caps in id2gen_val.items():
        for c in caps:
            pool_texts.append(c)
            pool_meta.append({"mol_id": gid, "source": "gen_val"})

    print(f"[VAL TEXT] Pool size: {len(pool_texts)} captions.")
    
    # -----------------------------------------------------
    # Build TEXT retrieval index (using MiniLMTextEncoder)
    # -----------------------------------------------------
    print("[VAL TEXT] Building text index.")
    text_retriever = TextRetrievalIndex(device=DEVICE)
    text_retriever.build_index_from_texts(
        text_encoder=text_encoder,
        texts=pool_texts,
        metas=pool_meta,
        batch_size=256,
    )

    # -----------------------------------------------------
    # Retrieval on VAL set: graph -> TOP_K_TEXT captions
    # then reranker + editor (same as your pipeline)
    # -----------------------------------------------------
    preds: List[str] = []
    refs: List[str] = []

    top1_source_counter = Counter()
    topk_source_counter = Counter()
    print("[VAL TEXT] Running retrieval (graph -> text pool).")
    for batch_graph, _ in tqdm(dl_val, desc="Retrieval"):
        batch_graph = batch_graph.to(DEVICE)

        nn_indices, scores = text_retriever.query_with_graphs(
            clip_model,
            batch_graph,
            k=TOP_K_TEXT,
        )

        val_graphs = batch_graph.to_data_list()

        for b in range(len(val_graphs)):
            idx_b = nn_indices[b].tolist()
            scores_b = scores[b].tolist()

            captions_b = text_retriever.get_texts(idx_b)
            top_meta = text_retriever.get_meta(idx_b)
            # TOP-K stats
            for m in top_meta:
                topk_source_counter[m["source"]] += 1
            # ---- RERANK (identique à evaluation_validation_val.py)
            pred_text = rerank_topk_hybrid(
                captions=captions_b,
                graph_scores=scores_b,
                text_encoder=text_encoder,
                alpha=RERANK_ALPHA,
            )

            chosen_idx = captions_b.index(pred_text)
            chosen_source = top_meta[chosen_idx]["source"]
            top1_source_counter[chosen_source] += 1
            # ---- EDITOR analyse (identique)
            _ = analyze_editability(
                text=pred_text,
                tokenizer=tokenizer,
                editor_model=editor_model,
                device=DEVICE,
                threshold=EDITOR_THRESHOLD,
            )

            # ---- Editor fix niveau 0 (identique)
            pred_text = graph_consistency_fix(pred_text, val_graphs[b])

            # ---- refs
            ref_text = id2desc_val[val_graphs[b].id]
            preds.append(pred_text)
            refs.append(ref_text)

    print(f"[VAL TEXT] Collected {len(preds)} predictions.")

    # -----------------------------------------------------
    # Metrics (identique)
    # -----------------------------------------------------
    bleu4, bert_f1 = evaluate_all(preds, refs, device=DEVICE)

    print("\n[VAL TEXT] Final metrics:")
    print(f"  BLEU-4: {bleu4:.4f}")
    print(f"  BERTScore F1: {bert_f1:.4f}")
    print("\n[ANALYSIS] Top-1 source distribution:")
    for k, v in top1_source_counter.items():
        print(f"  {k}: {v}")

    print("\n[ANALYSIS] Top-K source distribution:")
    for k, v in topk_source_counter.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
