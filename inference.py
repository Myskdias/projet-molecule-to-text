"""
inference.py
============

But:
----
Générer submission.csv pour Kaggle:
- colonne "ID"
- colonne "description"

Pipeline:
---------
1) Charger Stage1 (CLIP) + Stage2 (T5 rewrite)
2) Construire l'index retrieval sur toutes les captions train
3) Pour chaque graphe test:
   - retrieve top-1 caption
   - input = FIXED_PROMPT + retrieved
   - graph_emb -> soft prompt
   - T5.generate -> texte final
4) écrire CSV

Important:
----------
- Kaggle est sensible au nom exact des colonnes.
  Dans votre historique, vous aviez l'erreur "ID column not found".
  Ici on écrit "ID" + "description".
"""

from __future__ import annotations

import os
import csv
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5TokenizerFast

from data_utils import PreprocessedGraphTextDataset, PreprocessedGraphTestDataset, collate_graph_text, collate_graph_only
from architecture import MPNNEncoder, FrozenT5TextEncoder, GraphTextCLIP, GraphSoftPromptT5
from retrieval import RetrievalIndex


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_GRAPHS = "data/train_graphs.pkl"
TEST_GRAPHS = "data/test_graphs.pkl"

STAGE1_WEIGHTS = "weights_stage1_clip.pt"
STAGE2_WEIGHTS = "weights_stage2_t5.pt"

SUBMISSION_PATH = "submission.csv"

NODE_VOCAB_SIZES = [200, 20]
EDGE_VOCAB_SIZES = [50, 20]
HIDDEN_GRAPH = 300
SHARED_DIM = 256

T5_NAME = "t5-base"
PROMPT_LEN = 8

FIXED_PROMPT = (
    "Rephrase the following molecular description so that it accurately reflects "
    "the structure and roles of the given molecule.\n"
    "Description:\n"
)

MAX_INPUT_LEN = 256
MAX_NEW_TOKENS = 220


@torch.no_grad()
def main():
    for p in [TRAIN_GRAPHS, TEST_GRAPHS, STAGE1_WEIGHTS, STAGE2_WEIGHTS]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    tokenizer = T5TokenizerFast.from_pretrained(T5_NAME)

    # -------- Load CLIP (Stage1)
    graph_enc = MPNNEncoder(NODE_VOCAB_SIZES, EDGE_VOCAB_SIZES, hidden_dim=HIDDEN_GRAPH, num_layers=5, dropout=0.1)
    text_enc = FrozenT5TextEncoder(T5_NAME, freeze=True)
    clip = GraphTextCLIP(graph_enc, text_enc, graph_dim=HIDDEN_GRAPH, text_dim=768, shared_dim=SHARED_DIM).to(DEVICE)

    s1 = torch.load(STAGE1_WEIGHTS, map_location=DEVICE)
    clip.graph_encoder.load_state_dict(s1["graph_encoder"])
    clip.graph_proj.load_state_dict(s1["graph_proj"])
    clip.text_proj.load_state_dict(s1["text_proj"])
    clip.eval()

    # -------- Load generator (Stage2)
    gen = GraphSoftPromptT5(
        model_name=T5_NAME,
        graph_emb_dim=HIDDEN_GRAPH,
        prompt_len=PROMPT_LEN,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=("q", "v"),
    ).to(DEVICE)

    s2 = torch.load(STAGE2_WEIGHTS, map_location=DEVICE)
    gen.load_state_dict(s2["t5_lora_softprompt"])
    gen.eval()

    # -------- Build retrieval index on train captions
    ds_train = PreprocessedGraphTextDataset(TRAIN_GRAPHS)
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=False, collate_fn=collate_graph_text)

    train_texts = []
    for _bg, descs, _idx in dl_train:
        train_texts.extend(descs)

    retriever = RetrievalIndex(device=DEVICE)
    retriever.build_text_index(clip, tokenizer, train_texts, batch_size=64, max_len=MAX_INPUT_LEN)

    # -------- Test loader
    ds_test = PreprocessedGraphTestDataset(TEST_GRAPHS)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, collate_fn=collate_graph_only)

    rows = []
    for batch_graph, ids in tqdm(dl_test, desc="Infer test"):
        batch_graph = batch_graph.to(DEVICE)

        # retrieve top-1
        nn_idx, _ = retriever.query(clip, batch_graph, k=5)
        retrieved = retriever.get_texts(nn_idx[:, 0])[0]

        # graph embedding for soft prompt
        graph_emb, _ = clip.graph_encoder(batch_graph)

        # build input to T5
        inp = FIXED_PROMPT + retrieved
        tok_in = tokenizer(
            inp, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN
        ).to(DEVICE)

        out_ids = gen.generate(
            graph_emb=graph_emb,
            input_ids=tok_in["input_ids"],
            attention_mask=tok_in["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            early_stopping=True,
        )
        out_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        rows.append([ids[0], out_text])

    # -------- Write Kaggle submission
    with open(SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "description"])
        w.writerows(rows)

    print("Wrote:", SUBMISSION_PATH)


if __name__ == "__main__":
    main()
