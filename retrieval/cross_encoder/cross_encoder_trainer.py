import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from retrieval.cross_encoder.cross_encoder import GraphTextCrossEncoder


def train_cross_encoder_batched(
    train_loader,
    graph_encoder,
    text_encoder,
    dest: str,
    device: torch.device,

    # -------------------------------
    # 🔧 Hyperparams recommandés
    # -------------------------------
    graph_dim: int = 384,
    text_dim: int = 384,
    num_epochs: int = 8,
    lr: float = 1e-4,
    margin: float = 0.2,
    scheduler_tmax: int = 8,   # = num_epochs par défaut
    min_lr: float = 1e-6,
):
    """
    Batched training of a graph-text cross-encoder for reranking.

    train_loader yields:
        graph_batch        : PyG Batch
        gt_texts           : list[str] (len = B)
        neg_texts_batch    : list[list[str]] (len = B, each size = N)

    The graph encoder and text encoder are frozen.
    Only the cross-encoder is trained.
    """

    # ============================================================
    # 1. Init model, optimizer, scheduler, loss
    # ============================================================
    model = GraphTextCrossEncoder(
        graph_dim=graph_dim,
        text_dim=text_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=scheduler_tmax,
        eta_min=min_lr,
    )

    criterion = nn.MarginRankingLoss(margin=margin)

    # Freeze encoders (safety)
    graph_encoder.eval()
    text_encoder.eval()
    for p in graph_encoder.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False

    print(
        f"[Cross-Encoder] Trainable params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # ============================================================
    # 2. Training loop
    # ============================================================
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"[Cross-Encoder] Epoch {epoch+1}/{num_epochs}",
        )

        for graph_batch, gt_texts, neg_texts_batch in pbar:
            B = len(gt_texts)
            graph_batch = graph_batch.to(device)

            # ----------------------------------------------------
            # Encode graph + GT texts (no grad)
            # ----------------------------------------------------
            with torch.no_grad():
                graph_emb = graph_encoder.encode_graph(graph_batch)  # [B, Dg]
                text_emb_pos = text_encoder(gt_texts)                # [B, Dt]

            # ----------------------------------------------------
            # Encode negatives (flatten → reshape)
            # ----------------------------------------------------
            neg_texts_flat = []
            for negs in neg_texts_batch:
                neg_texts_flat.extend(negs)

            with torch.no_grad():
                text_emb_neg = text_encoder(neg_texts_flat)  # [B*N, Dt]

            N = len(neg_texts_batch[0])
            text_emb_neg = text_emb_neg.view(B, N, -1)  # [B, N, Dt]

            # ----------------------------------------------------
            # Scores
            # ----------------------------------------------------
            score_pos = model(graph_emb, text_emb_pos)  # [B]

            graph_emb_exp = graph_emb.unsqueeze(1).expand(-1, N, -1)
            score_neg = model(
                graph_emb_exp.reshape(B * N, -1),
                text_emb_neg.reshape(B * N, -1),
            ).view(B, N)

            # ----------------------------------------------------
            # Margin ranking loss (vectorized)
            # ----------------------------------------------------
            score_pos_exp = score_pos.unsqueeze(1).expand(-1, N)
            target = torch.ones_like(score_neg)

            loss = criterion(
                score_pos_exp.reshape(-1),
                score_neg.reshape(-1),
                target.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

            pbar.set_postfix(
                loss=f"{total_loss / n_steps:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        scheduler.step()
        print(
            f"[Epoch {epoch+1}] "
            f"avg_loss={total_loss / max(1, n_steps):.6f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    # ============================================================
    # 3. Save model
    # ============================================================
    torch.save(
        {
            "state_dict": model.state_dict(),
            "graph_dim": graph_dim,
            "text_dim": text_dim,
        },
        dest,
    )

    print(f"[Cross-Encoder] Training finished. Model saved to: {dest}")