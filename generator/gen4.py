# Char-level Transformer Decoder (from scratch)
# - Input: encoder_hidden_states [B, T_enc, d_model]
# - Output: logits over character vocab [B, T_dec, vocab_size]
# - Includes: masked self-attention + cross-attention + FFN
# - Provides: training forward (teacher forcing) + greedy generate()

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Vocab (character-level)
# -----------------------------
@dataclass
class CharVocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad: int
    bos: int
    eos: int

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, s: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos)
        for ch in s:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            else:
                # If you truly want a closed vocab without <unk>,
                # you can raise here instead. For robustness we map to space.
                ids.append(self.stoi[" "])
        if add_eos:
            ids.append(self.eos)
        return ids

    def decode(self, ids: List[int], stop_at_eos: bool = True) -> str:
        out = []
        for i in ids:
            if stop_at_eos and i == self.eos:
                break
            if i in (self.pad, self.bos):
                continue
            out.append(self.itos[i])
        return "".join(out)


def build_default_char_vocab() -> CharVocab:
    # Minimal but practical closed character set for scientific text + chemistry
    specials = ["<pad>", "<bos>", "<eos>"]
    charset = []
    charset += [" "]  # keep space explicit
    charset += list("abcdefghijklmnopqrstuvwxyz")
    charset += list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    charset += list("0123456789")
    charset += list(".,;:!?")
    charset += list("()[]{}")
    charset += list("+-=*/")
    charset += list("_'\"")
    charset += list("\\|@#%^&~`")
    charset += ["\n", "\t", "-"]  # include hyphen (again is ok; we'll uniq)

    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for x in specials + charset:
        if x not in seen:
            uniq.append(x)
            seen.add(x)

    stoi = {ch: i for i, ch in enumerate(uniq)}
    itos = uniq
    pad = stoi["<pad>"]
    bos = stoi["<bos>"]
    eos = stoi["<eos>"]
    return CharVocab(stoi=stoi, itos=itos, pad=pad, bos=bos, eos=eos)


# -----------------------------
# Masks
# -----------------------------
def causal_attn_mask(T: int, device: torch.device) -> torch.Tensor:
    # nn.MultiheadAttention expects attn_mask shape [T, T] with True = masked (PyTorch >= 2)
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


def key_padding_mask(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    # shape [B, T] with True = pad positions (to be masked)
    return (input_ids == pad_id)


# -----------------------------
# Decoder blocks
# -----------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # [B, T_dec, d_model]
        encoder_hidden_states: torch.Tensor,  # [B, T_enc, d_model]
        self_attn_mask: torch.Tensor,  # [T_dec, T_dec]
        dec_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T_dec] True for PAD
        enc_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T_enc] True for PAD
    ) -> torch.Tensor:
        # 1) Masked self-attention
        sa_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=self_attn_mask,
            key_padding_mask=dec_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(sa_out))

        # 2) Cross-attention (decoder queries attend to encoder keys/values)
        ca_out, _ = self.cross_attn(
            query=x,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            key_padding_mask=enc_key_padding_mask,
            need_weights=False,
        )
        x = self.norm2(x + self.dropout(ca_out))

        # 3) FFN
        ff_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class CharTransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab: CharVocab,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 1024,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab.size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab.size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._reset_parameters()

    def _reset_parameters(self):
        # Reasonable init for from-scratch training
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        # lm_head tied or init similarly
        if self.lm_head.weight is not self.tok_emb.weight:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,        # [B, T_dec]
        encoder_hidden_states: torch.Tensor,    # [B, T_enc, d_model]
        encoder_padding_mask: Optional[torch.Tensor] = None,  # [B, T_enc] True for PAD
    ) -> torch.Tensor:
        """
        Returns logits over vocab for each decoder position: [B, T_dec, vocab_size]
        """
        B, T = decoder_input_ids.shape
        if T > self.max_len:
            raise ValueError(f"T_dec={T} > max_len={self.max_len}. Increase max_len or truncate.")

        pos = torch.arange(T, device=decoder_input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(decoder_input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        attn_mask = causal_attn_mask(T, decoder_input_ids.device)
        dec_pad_mask = key_padding_mask(decoder_input_ids, self.vocab.pad)

        for layer in self.layers:
            x = layer(
                x=x,
                encoder_hidden_states=encoder_hidden_states,
                self_attn_mask=attn_mask,
                dec_key_padding_mask=dec_pad_mask,
                enc_key_padding_mask=encoder_padding_mask,
            )

        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate_greedy(
        self,
        encoder_hidden_states: torch.Tensor,               # [B, T_enc, d_model]
        encoder_padding_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
    ) -> torch.Tensor:
        """
        Greedy decoding. Returns generated ids including <bos> and ending with <eos> if produced.
        Shape: [B, T_out]
        """
        B = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device

        # Start with <bos>
        ys = torch.full((B, 1), self.vocab.bos, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(ys, encoder_hidden_states, encoder_padding_mask=encoder_padding_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [B,1]
            ys = torch.cat([ys, next_token], dim=1)

            # stop if all EOS
            if torch.all(next_token.squeeze(1) == self.vocab.eos):
                break

            if ys.size(1) >= self.max_len:
                break

        return ys


# -----------------------------
# Training helpers
# -----------------------------
def teacher_forcing_loss(
    decoder: CharTransformerDecoder,
    encoder_hidden_states: torch.Tensor,     # [B, T_enc, d_model]
    target_ids: torch.Tensor,               # [B, T_tgt] (should include <bos> ... <eos>)
    encoder_padding_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    target_ids: includes BOS and EOS.
    We feed target_ids[:, :-1] and predict target_ids[:, 1:].
    """
    inp = target_ids[:, :-1]
    gold = target_ids[:, 1:]
    logits = decoder(inp, encoder_hidden_states, encoder_padding_mask=encoder_padding_mask)  # [B, T-1, V]
    V = logits.size(-1)

    # Flatten
    logits = logits.reshape(-1, V)
    gold = gold.reshape(-1)

    if label_smoothing > 0.0:
        # Manual label smoothing CE to keep ignore_index
        pad = decoder.vocab.pad
        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs, gold, ignore_index=pad, reduction="none")
        smooth = -log_probs.mean(dim=-1)
        mask = (gold != pad).float()
        nll = (nll * mask).sum() / mask.sum().clamp_min(1.0)
        smooth = (smooth * mask).sum() / mask.sum().clamp_min(1.0)
        return (1.0 - label_smoothing) * nll + label_smoothing * smooth
    else:
        return F.cross_entropy(logits, gold, ignore_index=decoder.vocab.pad)


# -----------------------------
# Minimal usage example (shape check)
# -----------------------------
if __name__ == "__main__":
    vocab = build_default_char_vocab()
    dec = CharTransformerDecoder(vocab, d_model=256, n_heads=8, n_layers=3, d_ff=1024, max_len=512)

    B = 2
    T_enc = 64
    d_model = dec.d_model
    encoder_out = torch.randn(B, T_enc, d_model)

    # Example targets (string -> ids)
    s1 = "The molecule is C6H12O6."
    s2 = "It has a role as a metabolite."
    ids1 = torch.tensor([vocab.encode(s1)], dtype=torch.long)
    ids2 = torch.tensor([vocab.encode(s2)], dtype=torch.long)
    # Pad to same length
    T = max(ids1.size(1), ids2.size(1))
    tgt = torch.full((B, T), vocab.pad, dtype=torch.long)
    tgt[0, :ids1.size(1)] = ids1[0]
    tgt[1, :ids2.size(1)] = ids2[0]

    loss = teacher_forcing_loss(dec, encoder_out, tgt, label_smoothing=0.1)
    print("loss:", float(loss))

    gen = dec.generate_greedy(encoder_out, max_new_tokens=80)
    print("gen[0]:", vocab.decode(gen[0].tolist()))
    print("gen[1]:", vocab.decode(gen[1].tolist()))
