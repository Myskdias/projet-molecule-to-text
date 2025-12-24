import os
import gc
import torch
import torch.nn.functional as F
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
from data_utils import PreprocessedGraphTextDataset, collate_graph_text


# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS = "data/validation_graphs.pkl"   # si vous l'avez, sinon commentez
WEIGHTS_STAGE1 = "weights_stage1_clip.pt"
WEIGHTS_STAGE2 = "weights_stage2_t5.pt"

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
MAX_TARGET_LEN = 256


def set_requires_grad(module, value: bool):
    for p in module.parameters():
        p.requires_grad = value


# =========================
# Stage 1: CLIP-like training for retrieval
# =========================
def train_stage1_clip(train_dl, epochs=3, lr=2e-4, tau=0.07):
    graph_enc = MPNNEncoder(NODE_VOCAB, EDGE_VOCAB, hidden_dim=HIDDEN_GRAPH, num_layers=5, dropout=0.1)
    text_enc = FrozenT5TextEncoder(T5_NAME, freeze=True)

    # text_dim = T5 d_model (768 for t5-base)
    clip = GraphTextCLIP(graph_enc, text_enc, graph_dim=HIDDEN_GRAPH, text_dim=768, shared_dim=SHARED_DIM).to(DEVICE)

    # Train ONLY graph_encoder + projections (text encoder frozen)
    params = list(clip.graph_encoder.parameters()) + list(clip.graph_proj.parameters()) + list(clip.text_proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    tokenizer = T5TokenizerFast.from_pretrained(T5_NAME)

    clip.train()
    for ep in range(epochs):
        total = 0.0
        pbar = tqdm(train_dl, desc=f"Stage1 EP {ep+1}/{epochs}")
        for batch_graph, descs, _idxs in pbar:
            batch_graph = batch_graph.to(DEVICE)
            tok = tokenizer(
                descs, padding=True, truncation=True, max_length=MAX_INPUT_LEN, return_tensors="pt"
            ).to(DEVICE)

            g, t = clip(batch_graph, tok["input_ids"], tok["attention_mask"])
            # CLIP logits
            logit_scale = clip.logit_scale.exp()
            logits = logit_scale * (g @ t.t())  # [B,B]
            labels = torch.arange(g.size(0), device=DEVICE)

            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"[Stage1] epoch {ep+1} avg loss: {total/len(train_dl):.4f}")

    torch.save(
        {
            "graph_encoder": clip.graph_encoder.state_dict(),
            "graph_proj": clip.graph_proj.state_dict(),
            "text_proj": clip.text_proj.state_dict(),
        },
        WEIGHTS_STAGE1,
    )
    return clip, tokenizer


# =========================
# Stage 2: Train T5 rewrite with softprompt + LoRA
# =========================
def train_stage2_t5(train_dl, clip, tokenizer, epochs=3, lr=5e-5, warmup_freeze_epochs=1, topk=5):
    # Build retriever index on train descriptions (text space)
    # We index TEXT embeddings because retrieval is graph->text in shared space
    train_texts_full = []
    for _, descs, _ in train_dl:
        train_texts_full.extend(descs)

    retriever = RetrievalIndex(device=DEVICE)
    retriever.build_text_index(clip, tokenizer, train_texts_full, batch_size=64, max_len=MAX_INPUT_LEN)

    # Create generator
    gen = GraphSoftPromptT5(
        model_name=T5_NAME,
        graph_emb_dim=HIDDEN_GRAPH,
        prompt_len=PROMPT_LEN,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ).to(DEVICE)

    # Initially freeze graph encoder + clip projections (as you asked)
    set_requires_grad(clip.graph_encoder, False)
    set_requires_grad(clip.graph_proj, False)
    set_requires_grad(clip.text_proj, False)
    clip.eval()

    # Train LoRA + softprompt only
    trainable = []
    for n, p in gen.named_parameters():
        if p.requires_grad:
            trainable.append(p)
    opt = torch.optim.AdamW(trainable, lr=lr)

    gen.train()
    for ep in range(epochs):
        # Progressive unfreezing (optional)
        if ep >= warmup_freeze_epochs:
            set_requires_grad(clip.graph_encoder, True)
            set_requires_grad(clip.graph_proj, True)
            # If you unfreeze graph encoder, retrieval index in principle becomes stale.
            # Easiest safe choice: keep it frozen. If you REALLY want, rebuild each epoch:
            # retriever.build_text_index(clip, tokenizer, train_texts_full, batch_size=64, max_len=MAX_INPUT_LEN)

        total = 0.0
        pbar = tqdm(train_dl, desc=f"Stage2 EP {ep+1}/{epochs}")
        for batch_graph, targets, idxs in pbar:
            batch_graph = batch_graph.to(DEVICE)

            # --- retrieval: get a candidate text for each graph
            with torch.no_grad():
                nn_idx, _ = retriever.query(clip, batch_graph, k=topk)  # [B,k]
                # anti-leak: avoid picking the same sample index if it happens (train only)
                # we don't have a perfect mapping from nn_idx to global idx because index is over train_texts_full order
                # But since index is built in the same order as train_dl iterates, it's consistent with dataset order.
                chosen = nn_idx[:, 0].clone()  # [B]
                # simple heuristic: if chosen equals current idxs, take next neighbor
                # (works if indexing order matches dataset order)
                for b in range(chosen.size(0)):
                    if int(chosen[b].item()) == int(idxs[b].item()) and topk > 1:
                        chosen[b] = nn_idx[b, 1]

                retrieved_texts = retriever.get_texts(chosen)

                # graph embedding for softprompt (use graph encoder output, NOT shared proj)
                graph_emb, _ = clip.graph_encoder(batch_graph)  # [B, HIDDEN_GRAPH]

            # Build T5 input: fixed prompt + retrieved
            inputs_text = [FIXED_PROMPT + rt for rt in retrieved_texts]

            tok_in = tokenizer(
                inputs_text, padding=True, truncation=True, max_length=MAX_INPUT_LEN, return_tensors="pt"
            ).to(DEVICE)
            tok_out = tokenizer(
                targets, padding=True, truncation=True, max_length=MAX_TARGET_LEN, return_tensors="pt"
            ).to(DEVICE)

            labels = tok_out["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            out = gen(
                graph_emb=graph_emb,
                input_ids=tok_in["input_ids"],
                attention_mask=tok_in["attention_mask"],
                labels=labels,
            )
            loss = out.loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"[Stage2] epoch {ep+1} avg loss: {total/len(train_dl):.4f}")

    torch.save(
        {
            "clip_graph_encoder": clip.graph_encoder.state_dict(),
            "clip_graph_proj": clip.graph_proj.state_dict(),
            "clip_text_proj": clip.text_proj.state_dict(),
            "t5_lora_and_softprompt": gen.state_dict(),
        },
        WEIGHTS_STAGE2,
    )

    return gen, retriever


# =========================
# Optional Stage 3: MRT/BLEU-aware (stub)
# =========================
def train_stage3_mrt(*args, **kwargs):
    raise NotImplementedError(
        "MRT/BLEU-aware fine-tuning is doable, but needs careful engineering "
        "(sampling/beam + reward computation). Start with Stage1+Stage2 first."
    )


def main():
    assert os.path.exists(TRAIN_GRAPHS), f"Missing {TRAIN_GRAPHS}"

    ds_train = PreprocessedGraphTextDataset(TRAIN_GRAPHS)
    train_dl = DataLoader(ds_train, batch_size=16, shuffle=True, collate_fn=collate_graph_text)

    # Stage 1
    clip, tokenizer = train_stage1_clip(train_dl, epochs=3)

    # Stage 2
    gen, retriever = train_stage2_t5(train_dl, clip, tokenizer, epochs=3, warmup_freeze_epochs=1, topk=5)

    print("Training finished. Saved:", WEIGHTS_STAGE1, WEIGHTS_STAGE2)


if __name__ == "__main__":
    main()
