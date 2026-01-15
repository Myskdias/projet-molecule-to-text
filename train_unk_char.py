import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from editor.masker import load_vocab, mask_out_of_vocab
from editor.tokenizer import tokenize
from editor.transformer.unk_transformer import UNKCharDecoder
from editor.transformer.utils import UNKCharDataset, build_char_vocab, extract_unk_samples

# ========= Imports depuis ton code =========
# - tokenize
# - mask_out_of_vocab
# - extract_unk_samples
# - UNKCharDataset
# - UNKCharDecoder
# - build_char_vocab
# - load_vocab

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_PKL = "vocab_minfreq15_nodigits.pkl"
TRAIN_DATA_PKL = "train_graphs.pkl"     # doit contenir caption + graph_emb
VAL_DATA_PKL = "val_graphs.pkl"

CTX_SIZE = 8
GRAPH_DIM = 512          # à adapter à ton graph encoder
GRAPH_TOKENS = 4
MAX_CHAR_LEN = 64

BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4

SAVE_DIR = "checkpoints_unk_char"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)

print("[INFO] Loading vocab...")
vocab = load_vocab(VOCAB_PKL)
token2id = {t: i for i, t in enumerate(sorted(vocab))}
vocab_size_ctx = len(token2id)

print("[INFO] Loading train / val data...")
train_data = load_data(TRAIN_DATA_PKL)
val_data = load_data(VAL_DATA_PKL)

# ------------------------------------------------------------------
# BUILD UNK SAMPLES
# ------------------------------------------------------------------
def build_samples(dataset):
    all_samples = []

    for mol in tqdm(dataset, desc="Extracting UNK samples"):
        caption = mol["caption"]
        graph_emb = mol["graph_emb"]

        masked = mask_out_of_vocab(caption, vocab, tokenize)

        samples = extract_unk_samples(
            original_text=caption,
            masked_text=masked,
            graph_emb=graph_emb,
            tokenize_fn=tokenize,
            token2id=token2id,
            ctx_size=CTX_SIZE,
        )
        all_samples.extend(samples)

    return all_samples

print("[INFO] Building UNK samples...")
train_samples = build_samples(train_data)
val_samples = build_samples(val_data)

print(f"[INFO] Train UNK samples: {len(train_samples)}")
print(f"[INFO] Val   UNK samples: {len(val_samples)}")

# ------------------------------------------------------------------
# BUILD CHAR VOCAB
# ------------------------------------------------------------------
print("[INFO] Building char vocab...")
all_spans = [s["target_span"] for s in train_samples]
char2id, id2char = build_char_vocab(all_spans)

vocab_size_chars = len(char2id)
print(f"[INFO] Char vocab size: {vocab_size_chars}")

# ------------------------------------------------------------------
# DATASETS / LOADERS
# ------------------------------------------------------------------
train_ds = UNKCharDataset(
    train_samples,
    char2id,
    max_len=MAX_CHAR_LEN,
)
val_ds = UNKCharDataset(
    val_samples,
    char2id,
    max_len=MAX_CHAR_LEN,
)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False
)

# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
model = UNKCharDecoder(
    vocab_size_chars=vocab_size_chars,
    vocab_size_ctx=vocab_size_ctx,
    graph_dim=GRAPH_DIM,
    graph_tokens=GRAPH_TOKENS,
    max_len=MAX_CHAR_LEN,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=char2id["<PAD>"])

# ------------------------------------------------------------------
# TRAIN / EVAL LOOPS
# ------------------------------------------------------------------
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0

    for batch in loader:
        ctx_left = batch["ctx_left"].to(DEVICE)
        ctx_right = batch["ctx_right"].to(DEVICE)
        graph_emb = batch["graph_emb"].to(DEVICE)
        target = batch["target"].to(DEVICE)

        if train:
            optimizer.zero_grad()

        logits = model(
            ctx_left=ctx_left,
            ctx_right=ctx_right,
            graph_emb=graph_emb,
            target_ids=target,
        )

        loss = loss_fn(
            logits[:, :-1].reshape(-1, vocab_size_chars),
            target[:, 1:].reshape(-1),
        )

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------
best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)

    print(f"Train loss: {train_loss:.4f}")
    print(f"Val   loss: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        ckpt_path = os.path.join(SAVE_DIR, "best.pt")
        torch.save({
            "model": model.state_dict(),
            "char2id": char2id,
            "id2char": id2char,
        }, ckpt_path)
        print("⭐ New best model saved")

print("\n[INFO] Training finished.")
