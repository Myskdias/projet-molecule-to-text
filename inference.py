import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from architecture import DeepGINEEncoder, TextEncoder, MolecularCaptionModel
from retrieval import RetrievalIndex
from data_utils import (
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

# Generation (optionnel)
DO_GENERATE_WITH_DECODER = False  # True si tu as un mapping id->token pour décoder derrière
MAX_GEN_LEN = 200
BOS_IDX = 1
EOS_IDX = 2


# =========================
# UTIL: génération greedy d'IDs (optionnel)
# =========================
@torch.no_grad()
def greedy_generate_ids(model: MolecularCaptionModel, batch_graph, retrieved_ids, retrieved_mask,
                        max_len: int = 200):
    """
    Renvoie une séquence d'IDs (pas de décodage texte ici).
    model.forward attend: (batch_graph, retrieved_ids, retrieved_mask, target_ids, target_mask, target_padding_mask)
    """
    model.eval()
    batch_graph = batch_graph.to(DEVICE)
    retrieved_ids = retrieved_ids.to(DEVICE)
    retrieved_mask = retrieved_mask.to(DEVICE)

    generated = torch.tensor([[BOS_IDX]], device=DEVICE, dtype=torch.long)

    for _ in range(max_len):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1)).to(DEVICE)
        tgt_pad_mask = (generated == PAD_IDX)

        logits = model(
            batch_graph,
            retrieved_ids,
            retrieved_mask,
            generated,
            tgt_mask,
            tgt_pad_mask,
        )  # [B, T, V]

        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)  # [B,1]
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == EOS_IDX:
            break

    return generated.squeeze(0)  # [T]


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
    model = MolecularCaptionModel(
        graph_encoder,
        text_encoder,          # text_encoder_rag
        vocab_size,
        d_model=HIDDEN_TEXT,
        graph_dim=HIDDEN_GRAPH
    ).to(DEVICE)

    state = torch.load(FINAL_MODEL_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.")

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
    retriever.build_index(model.graph_encoder, index_dl, captions_tokens)

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
        nn_indices, _ = retriever.query(model.graph_encoder, batch_graph, k=TOP_K)  # [1,k]
        chosen = nn_indices[:, NEIGHBOR_RANK]  # [1]

        # Map neighbor index -> train id -> description
        neighbor_idx = int(chosen.item())
        neighbor_id = train_ids[neighbor_idx]
        retrieved_caption = id2desc[neighbor_id]

        if DO_GENERATE_WITH_DECODER:
            # Build retrieved tokens for RAG memory
            retrieved_ids = retriever.get_retrieved_tokens(chosen).to(DEVICE)  # [1, L]
            retrieved_ids = torch.clamp(retrieved_ids.long(), min=0, max=vocab_size - 1)
            retrieved_mask = (retrieved_ids == PAD_IDX)

            # Generate ids (NO text decoding available in this repo)
            gen_ids = greedy_generate_ids(model, batch_graph, retrieved_ids, retrieved_mask, max_len=MAX_GEN_LEN)
            # Fallback representation: join ids (NOT Kaggle-friendly)
            caption_out = " ".join(map(str, gen_ids.tolist()))
        else:
            # Retrieval-only text output (Kaggle-friendly)
            caption_out = retrieved_caption

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
