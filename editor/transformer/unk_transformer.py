import torch
import torch.nn as nn
import math

class UNKCharDecoder(nn.Module):
    def __init__(
        self,
        vocab_size_chars,
        vocab_size_ctx,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        graph_dim=512,
        graph_tokens=4,
        max_len=64,
    ):
        super().__init__()

        self.d_model = d_model
        self.graph_tokens = graph_tokens

        # Embeddings
        self.char_emb = nn.Embedding(vocab_size_chars, d_model)
        self.ctx_emb = nn.Embedding(vocab_size_ctx, d_model)

        self.graph_proj = nn.Linear(graph_dim, graph_tokens * d_model)

        self.pos_emb = nn.Embedding(max_len + 64, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        self.out_proj = nn.Linear(d_model, vocab_size_chars)

    def forward(self, ctx_left, ctx_right, graph_emb, target_ids):
        B, T = target_ids.shape

        # ---- graph prefix ----
        graph_tokens = self.graph_proj(graph_emb)
        graph_tokens = graph_tokens.view(B, self.graph_tokens, self.d_model)

        # ---- context ----
        ctx = torch.cat([
            self.ctx_emb(ctx_left),
            self.ctx_emb(ctx_right)
        ], dim=1)

        memory = torch.cat([graph_tokens, ctx], dim=1)

        # ---- target ----
        pos = torch.arange(T, device=target_ids.device).unsqueeze(0)
        tgt = self.char_emb(target_ids) + self.pos_emb(pos)

        # causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=target_ids.device), diagonal=1
        ).bool()

        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
        )

        logits = self.out_proj(out)
        return logits


def decode_chars(model, ctx_left, ctx_right, graph_emb, char2id, id2char, max_len=64):
    model.eval()
    B = graph_emb.size(0)

    cur = torch.full((B, 1), char2id["<BOS>"], device=graph_emb.device)

    for _ in range(max_len):
        logits = model(ctx_left, ctx_right, graph_emb, cur)
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        cur = torch.cat([cur, next_id], dim=1)

        if (next_id == char2id["<EOS>"]).all():
            break

    # strip BOS/EOS
    results = []
    for seq in cur.tolist():
        chars = []
        for i in seq:
            c = id2char[i]
            if c in ("<BOS>", "<EOS>", "<PAD>"):
                continue
            chars.append(c)
        results.append("".join(chars))
    return results
