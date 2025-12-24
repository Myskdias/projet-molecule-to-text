"""
training_pipeline.py
====================

Ce script orchestre l'entraînement multi-étapes, comme discuté :

Etape 1 (prétraining retrieval):
-------------------------------
- Apprendre un espace latent commun graph/text via loss contrastive CLIP-like.
- But: retrieval performant (baseline Kaggle améliorée)
- On gèle l'encodeur texte (T5 encoder) pour stabilité + VRAM.

Etape 2 (SFT génération / rewrite):
----------------------------------
- On construit un retriever (index text embeddings).
- Pour chaque graphe:
    retrieve caption candidate (top-k)
    input = FIXED_PROMPT + retrieved_caption
    conditionnement graphe = soft prompt (MLP(graph_emb) -> k tokens)
    générateur = T5-base + LoRA
- Loss: cross-entropy (teacher forcing) = stable.

Etape 3 (optionnel):
-------------------
- Fine-tuning metric-aware (MRT / BLEU-aware / BERTScore-aware)
- Très coûteux et instable => à faire en dernier, éventuellement sur subset.

Important (engineering):
------------------------
Si on dégèle l'encodeur graphe après avoir construit l'index retrieval,
l'espace commun change => index incohérent.
=> soit on garde graph encoder gelé à l'étape 2,
=> soit on reconstruit l'index périodiquement (plus coûteux).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import T5TokenizerFast

from data_utils import PreprocessedGraphTextDataset, collate_graph_text
from architecture import MPNNEncoder, FrozenT5TextEncoder, GraphTextCLIP, GraphSoftPromptT5
from retrieval import RetrievalIndex


# =========================
# Reproductibilité (utile en MVA)
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Config
# =========================
@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    train_graphs: str = "data/train_graphs.pkl"
    val_graphs: str = "data/validation_graphs.pkl"  # optionnel
    use_val: bool = False

    # save
    stage1_path: str = "weights_stage1_clip.pt"
    stage2_path: str = "weights_stage2_t5.pt"

    # model dims
    node_vocab_sizes: list = None
    edge_vocab_sizes: list = None
    hidden_graph: int = 300
    shared_dim: int = 256

    # T5
    t5_name: str = "t5-base"
    prompt_len: int = 8

    # tokenization
    max_in_len: int = 256
    max_out_len: int = 256

    # retrieval
    topk: int = 5

    # training stage1
    stage1_epochs: int = 3
    stage1_lr: float = 2e-4

    # training stage2
    stage2_epochs: int = 3
    stage2_lr: float = 5e-5
    freeze_warmup_epochs: int = 1

    # loader
    batch_size: int = 16
    num_workers: int = 0


CFG = Config(
    node_vocab_sizes=[200, 20],   # <-- à remplacer par vos vrais vocabs si besoin
    edge_vocab_sizes=[50, 20],    # idem
)

FIXED_PROMPT = (
    "Rephrase the following molecular description so that it accurately reflects "
    "the structure and roles of the given molecule.\n"
    "Description:\n"
)


# =========================
# Helpers gel/dégel
# =========================
def set_requires_grad(module: torch.nn.Module, value: bool):
    for p in module.parameters():
        p.requires_grad = value


# ============================================================
# Stage 1: Graph–Text Alignment (CLIP-like contrastive learning)
# ============================================================
def train_stage1_clip(train_dl, cfg: Config):
    """
    On apprend :
    - MPNN graph encoder
    - projections (graph_proj, text_proj)
    pour aligner graphe et texte.

    Encodeur texte:
    --------------
    T5 encoder gelé (FrozenT5TextEncoder). Pourquoi ?
    - VRAM/compute
    - dataset ~32k : on évite d'overfit un gros encodeur
    - on garde un "anchor" sémantique stable
    """
    print("\n" + "=" * 70)
    print("STAGE 1: CLIP-like graph-text alignment (retrieval pretraining)")
    print("=" * 70)

    tokenizer = T5TokenizerFast.from_pretrained(cfg.t5_name)

    graph_enc = MPNNEncoder(
        node_vocab_sizes=cfg.node_vocab_sizes,
        edge_vocab_sizes=cfg.edge_vocab_sizes,
        hidden_dim=cfg.hidden_graph,
        num_layers=5,
        dropout=0.1,
    )

    text_enc = FrozenT5TextEncoder(cfg.t5_name, freeze=True)

    clip = GraphTextCLIP(
        graph_encoder=graph_enc,
        text_encoder=text_enc,
        graph_dim=cfg.hidden_graph,
        text_dim=768,  # t5-base d_model
        shared_dim=cfg.shared_dim,
    ).to(cfg.device)

    # On optimise seulement graph encoder + proj (text encoder freeze)
    params = list(clip.graph_encoder.parameters()) + list(clip.graph_proj.parameters()) + list(clip.text_proj.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.stage1_lr)

    clip.train()
    for ep in range(cfg.stage1_epochs):
        total = 0.0
        pbar = tqdm(train_dl, desc=f"[Stage1] epoch {ep+1}/{cfg.stage1_epochs}")

        for batch_graph, descs, _idxs in pbar:
            batch_graph = batch_graph.to(cfg.device)

            tok = tokenizer(
                descs,
                padding=True,
                truncation=True,
                max_length=cfg.max_in_len,
                return_tensors="pt",
            ).to(cfg.device)

            g, t = clip(batch_graph, tok["input_ids"], tok["attention_mask"])   # [B,D], [B,D]

            # CLIP logits = scale * dot
            logit_scale = clip.logit_scale.exp()
            logits = logit_scale * (g @ t.t())  # [B,B]
            labels = torch.arange(g.size(0), device=cfg.device)

            # Loss contrastive symétrique (graph->text et text->graph)
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
        cfg.stage1_path,
    )
    print(f"[Stage1] Saved weights -> {cfg.stage1_path}")

    return clip, tokenizer


# ============================================================
# Stage 2: Retrieval-Augmented Generation (SoftPrompt + T5 + LoRA)
# ============================================================
def train_stage2_t5(train_dl, clip, tokenizer, cfg: Config):
    """
    Objectif:
    ---------
    Apprendre un modèle génératif qui corrige/réécrit une caption récupérée.

    Pourquoi "rewrite" plutôt que génération from scratch ?
    -------------------------------------------------------
    - le dataset a un style très template + long
    - BLEU est sensible aux paraphrases
    - retrieval fournit un "prior" lexical et structurel robuste
    - le LLM apprend à faire des corrections locales + alignement au graphe
    """
    print("\n" + "=" * 70)
    print("STAGE 2: RAG rewrite (SoftPrompt + T5-base + LoRA)")
    print("=" * 70)

    # 1) Construire l'index retrieval sur toutes les captions train
    # NB: on reconstruit depuis le DataLoader (simple et sûr).
    train_texts = []
    for _bg, descs, _idx in train_dl:
        train_texts.extend(descs)

    retriever = RetrievalIndex(device=cfg.device)
    retriever.build_text_index(clip, tokenizer, train_texts, batch_size=64, max_len=cfg.max_in_len)

    # 2) Générateur T5 + LoRA + soft prompt
    gen = GraphSoftPromptT5(
        model_name=cfg.t5_name,
        graph_emb_dim=cfg.hidden_graph,
        prompt_len=cfg.prompt_len,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=("q", "v"),
    ).to(cfg.device)

    # 3) Gel initial (comme tu voulais)
    # On gèle retrieval (clip) pour stabiliser la SFT.
    set_requires_grad(clip.graph_encoder, False)
    set_requires_grad(clip.graph_proj, False)
    set_requires_grad(clip.text_proj, False)
    clip.eval()

    # LoRA + softprompt sont trainables (PEFT)
    trainable_params = [p for p in gen.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=cfg.stage2_lr)

    gen.train()
    for ep in range(cfg.stage2_epochs):
        # Optionnel: dégel progressif
        # Attention: si on dégel le graph encoder, l'index retrieval devient "stale".
        # => solution rigoureuse = rebuild index régulièrement.
        if ep >= cfg.freeze_warmup_epochs:
            # Ici je te le laisse OFF par défaut (plus stable).
            # Décommenter si vous voulez expérimenter:
            # set_requires_grad(clip.graph_encoder, True)
            # set_requires_grad(clip.graph_proj, True)
            # retriever.build_text_index(clip, tokenizer, train_texts, batch_size=64, max_len=cfg.max_in_len)
            pass

        total = 0.0
        pbar = tqdm(train_dl, desc=f"[Stage2] epoch {ep+1}/{cfg.stage2_epochs}")

        for batch_graph, targets, idxs in pbar:
            batch_graph = batch_graph.to(cfg.device)

            # --- Retrieval step (no grad)
            with torch.no_grad():
                nn_idx, _scores = retriever.query(clip, batch_graph, k=cfg.topk)  # [B,k]
                chosen = nn_idx[:, 0].clone()

                # Heuristique anti self-retrieval :
                # marche si l'ordre d'index == ordre dataset.
                if cfg.topk > 1:
                    for b in range(chosen.size(0)):
                        if int(chosen[b].item()) == int(idxs[b].item()):
                            chosen[b] = nn_idx[b, 1]

                retrieved_texts = retriever.get_texts(chosen)

                # Graph embedding pour softprompt (AVANT projection CLIP)
                graph_emb, _ = clip.graph_encoder(batch_graph)  # [B, hidden_graph]

            # --- Build T5 inputs
            inputs = [FIXED_PROMPT + rt for rt in retrieved_texts]

            tok_in = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=cfg.max_in_len,
                return_tensors="pt",
            ).to(cfg.device)

            tok_out = tokenizer(
                targets,
                padding=True,
                truncation=True,
                max_length=cfg.max_out_len,
                return_tensors="pt",
            ).to(cfg.device)

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
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"[Stage2] epoch {ep+1} avg loss: {total/len(train_dl):.4f}")

    torch.save(
        {
            "clip_graph_encoder": clip.graph_encoder.state_dict(),
            "clip_graph_proj": clip.graph_proj.state_dict(),
            "clip_text_proj": clip.text_proj.state_dict(),
            "t5_lora_softprompt": gen.state_dict(),
        },
        cfg.stage2_path,
    )
    print(f"[Stage2] Saved weights -> {cfg.stage2_path}")

    return gen, retriever


# ============================================================
# Stage 3 (Optionnel): Metric-aware (MRT / BLEU/BERTScore)
# ============================================================
def train_stage3_metric_aware(*_args, **_kwargs):
    """
    Placeholder volontaire.

    Pourquoi je le laisse en stub ?
    ------------------------------
    - MRT / BLEU-aware implique sampling/beam + reward sur texte généré
    - BERTScore -> coûteux (RoBERTa-base) à calculer dans la boucle
    - Risque fort de rendre l'entraînement instable si fait trop tôt

    Recommandation MVA:
    -------------------
    - faire Stage1+Stage2 d'abord
    - valider que la génération est "propre"
    - ensuite faire un "polish" metric-aware sur un subset
    """
    raise NotImplementedError("Stage3 à implémenter proprement après validation Stage1+Stage2.")


def main():
    set_seed(42)

    assert os.path.exists(CFG.train_graphs), f"Missing {CFG.train_graphs}"

    ds_train = PreprocessedGraphTextDataset(CFG.train_graphs)
    dl_train = DataLoader(
        ds_train,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        collate_fn=collate_graph_text,
    )

    # Stage1
    clip, tokenizer = train_stage1_clip(dl_train, CFG)

    # Stage2
    _gen, _retriever = train_stage2_t5(dl_train, clip, tokenizer, CFG)

    print("\nTraining completed.")


if __name__ == "__main__":
    main()
