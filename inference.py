import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import T5TokenizerFast

from architecture import (
    MPNNEncoder,
    FrozenT5TextEncoder,
    GraphTextCLIP,
    GraphSoftPromptT5,
)
from retrieval import RetrievalIndex
from data_utils import PreprocessedGraphTextDataset, collate_graph_text, collate_graph_only


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_GRAPHS = "data/train_graphs.pkl"
TEST_GRAPHS = "data/test_graphs.pkl"

WEIGHTS_STAGE1 = "weights_stage1_clip.pt"
WEIGHTS_STAGE2 = "weights_stage2_t5.pt"

SUBMISSION_PATH = "submission.csv"

NODE_VOCAB = [200, 20]
EDGE_VOCAB = [50, 20]
HIDDEN_GRAPH = 300
SHARED_DIM = 256

T5_NAME = "t5-base"
PROMPT_LEN = 8

FIXED_PROMPT = (
    "Rephrase the following molecular description so that it accurately reflects "
    "the structure and roles of the given molecule:\n"
)

MAX_INPUT_LEN = 256
MAX_NEW_TOKENS = 220


@torch.no_grad()
def main():
    for p in [TRAIN_GRAPHS, TEST_GRAPHS, WEIGHTS_STAGE1, WEIGHTS_STAGE2]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # Load tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(T5_NAME)

    # Build CLIP model skeleton + load weights
    graph_enc = MPNNEncoder(NODE_VOCAB, EDGE_VOCAB, hidden_dim=HIDDEN_GRAPH, num_layers=5, dropout=0.1)
    text_enc = FrozenT5TextEncoder(T5_NAME, freeze=True)
    clip = GraphTextCLIP(graph_enc, text_enc, graph_dim=HIDDEN_GRAPH, text_dim=768, shared_dim=SHARED_DIM).to(DEVICE)

    s1 = torch.load(WEIGHTS_STAGE1, map_location=DEVICE)
    clip.graph_encoder.load_state_dict(s1["graph_encoder"])
    clip.graph_proj.load_state_dict(s1["graph_proj"])
    clip.text_proj.load_state_dict(s1["text_proj"])
    clip.eval()

    # Load generator (T5 LoRA + softprompt) and weights
    gen = GraphSoftPromptT5(
        model_name=T5_NAME,
        graph_emb_dim=HIDDEN_GRAPH,
        prompt_len=PROMPT_LEN,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ).to(DEVICE)

    s2 = torch.load(WEIGHTS_STAGE2, map_location=DEVICE)
    gen.load_state_dict(s2["t5_lora_and_softprompt"])
    gen.eval()

    # Load train dataset (to build retrieval index over train descriptions)
    ds_train = PreprocessedGraphTextDataset(TRAIN_GRAPHS)
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=False, collate_fn=collate_graph_text)

    train_texts = []
    for _, descs, _ in dl_train:
        train_texts.extend(descs)

    retriever = RetrievalIndex(device=DEVICE)
    retriever.build_text_index(clip, tokenizer, train_texts, batch_size=64, max_len=MAX_INPUT_LEN)

    # Test loader (graphs only)
    with open(TEST_GRAPHS, "rb") as f:
        import pickle
        test_graphs = pickle.load(f)

    dl_test = DataLoader(test_graphs, batch_size=1, shuffle=False, collate_fn=collate_graph_only)

    rows = []
    for batch_graph in tqdm(dl_test, desc="Infer"):
        # retrieve top-1
        nn_idx, _ = retriever.query(clip, batch_graph.to(DEVICE), k=5)
        chosen = nn_idx[:, 0]
        retrieved = retriever.get_texts(chosen)[0]

        # graph emb for softprompt
        graph_emb, _ = clip.graph_encoder(batch_graph.to(DEVICE))

        # build input
        inp = FIXED_PROMPT + retrieved
        tok_in = tokenizer(inp, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN).to(DEVICE)

        out_ids = gen.generate(
            graph_emb=graph_emb,
            input_ids=tok_in["input_ids"],
            attention_mask=tok_in["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            early_stopping=True,
        )
        out_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        test_id = batch_graph.id[0]
        rows.append([test_id, out_text])

    # Kaggle expects columns: ID, description
    with open(SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "description"])
        w.writerows(rows)

    print("Wrote", SUBMISSION_PATH)


if __name__ == "__main__":
    main()
