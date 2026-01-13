from __future__ import annotations

import os
import pickle
from typing import List

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer

from editor.text_editor import graph_consistency_fix, graph_consistency_fix_v2, graph_consistency_fix_v3
from editor.edit_model import EditModel
from editor.utils import analyze_editability
from retrieval.cross_encoder.cross_encoder import GraphTextCrossEncoder
from retrieval.reranker import rerank_topk_hybrid, rerank_topk_mbr, rerank_topk_mbr_weighted, rerank_topk_hybrid_pruned
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
TOP_K = 10
NEIGHBOR_RANK = 0  # top-1

EDITOR_CKPT = "weights/editor_level1_head.pt"

GEN_VAL_CSV = "validation_captions.csv"
NUM_GEN_CAPTIONS = 5
MAX_KEEP_GEN = 2
GEN_MARGIN = 0.10   # garde si sim(gen) >= sim(best_train) - margin
GEN_MIN_SIM = 0.0   # garde-fou optionnel, laisse 0.0 au début

USE_GEN = True

def load_generated_val(csv_path):
    df = pd.read_csv(csv_path)
    id2gen = {}
    for _, row in df.iterrows():
        gid = str(row["ID"])
        caps = []
        for i in range(1, NUM_GEN_CAPTIONS + 1):
            c = row[f"caption_{i}"]
            if isinstance(c, str) and len(c) > 0:
                caps.append(c)
        if caps:
            id2gen[gid] = caps
    return id2gen

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
    # Load gen captions
    # -----------------------------------------------------

    id2gen_val = load_generated_val(GEN_VAL_CSV)
    print(f"[INFO] Loaded generated captions for {len(id2gen_val)} VAL graphs")

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

    # =======================================================
    # Load Editor
    # =======================================================
    editor_model = EditModel("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)

    ckpt = torch.load(EDITOR_CKPT, map_location=DEVICE)
    editor_model.classifier.load_state_dict(ckpt["classifier"])
    editor_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------------------------------
    # MERGE TRAIN + VAL FOR RETRIEVAL
    # -----------------------------------------------------
    print("[VAL EVAL] Merging TRAIN + VAL for retrieval index...")

    ds_all = ds_train.graphs + ds_val.graphs
    id2desc_all = {}
    id2desc_all.update(id2desc_train)
    id2desc_all.update(id2desc_val)

    dl_all = DataLoader(
        PreprocessedGraphDataset.from_graphs(ds_all, mode=True),
        batch_size=INDEX_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # -----------------------------------------------------
    # Build retrieval index (TRAIN graphs)
    # -----------------------------------------------------
    print("[VAL EVAL] Building retrieval index...")
    retriever = RetrievalIndex(device=DEVICE)
    #retriever.build_index(clip_model, dl_train)
    retriever.build_index(clip_model, dl_train)
    #retriever.build_index(clip_model, dl_all)

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
            '''
            # Candidate captions from TRAIN
            captions_b = []
            for train_idx in idx_b:
                #train_graph = ds_all[int(train_idx)] when using ds_all and dl_all
                #captions_b.append(id2desc_all[train_graph.id])
                train_graph = ds_train.graphs[int(train_idx)]
                captions_b.append(id2desc_train[train_graph.id])
            '''
            # --------------------------------------------------
            # Candidate captions from TRAIN (baseline)
            # --------------------------------------------------
            captions_b = []
            scores_b_expanded = []

            for train_idx, s in zip(idx_b, scores_b):
                train_graph = ds_train.graphs[int(train_idx)]
                gid = train_graph.id

                # GT caption (baseline)
                captions_b.append(id2desc_train[gid])
                scores_b_expanded.append(s)

            # --------------------------------------------------
            # Filter generated captions by graph-text similarity
            # --------------------------------------------------
            filtered_gen_caps = []

            val_graph = val_graphs[b]
            val_gid = val_graph.id

            if val_gid in id2gen_val and USE_GEN:
                # graph embedding in shared latent space
                # (encode_graph attends un Batch, donc on crée un mini-batch de 1 graphe)
                one_batch = Batch.from_data_list([val_graph]).to(DEVICE)
                g_emb = clip_model.encode_graph(one_batch)  # [1, D], normalized

                # text embeddings for the 5 generated captions
                gen_caps = id2gen_val[val_gid]
                t_emb = text_encoder(gen_caps)  # [5, D], normalized

                # cosine similarity: [5]
                sims = (t_emb @ g_emb.t()).squeeze(1)  # dot product

                # dynamic threshold relative to best retrieved TRAIN neighbor
                ref_sim = float(scores_b[NEIGHBOR_RANK])
                thr = max(GEN_MIN_SIM, ref_sim - GEN_MARGIN)

                # keep those above threshold
                keep_idx = (sims >= thr).nonzero(as_tuple=False).view(-1).tolist()

                # if too many, keep top MAX_KEEP_GEN
                if len(keep_idx) > 0:
                    # rank kept indices by sim desc
                    keep_idx = sorted(keep_idx, key=lambda i: float(sims[i]), reverse=True)
                    keep_idx = keep_idx[:MAX_KEEP_GEN]
                    filtered_gen_caps = [gen_caps[i] for i in keep_idx]


            # --------------------------------------------------
            # + Add filtered generated captions (VAL)
            # --------------------------------------------------

            for cap in filtered_gen_caps:
                captions_b.append(cap)
                scores_b_expanded.append(scores_b[NEIGHBOR_RANK])
            
            #  RERANK HERE
            pred_text = rerank_topk_hybrid_pruned(
                captions=captions_b,
                graph_scores=scores_b_expanded,
                text_encoder=text_encoder,
                alpha=0.7,
            )
            # --- NIVEAU 1 : ANALYSE ---
            editable_tokens = analyze_editability(
                text=pred_text,
                tokenizer=tokenizer,
                editor_model=editor_model,
                device=DEVICE,
                threshold=0.75,
            )

            # LOG (debug seulement)
            if len(editable_tokens) > 0 and b == 0 and False:
                print("EDITABLE:", editable_tokens)

            # NIVEAU 0
            pred_text = graph_consistency_fix(pred_text, val_graphs[b])
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
