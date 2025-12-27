import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled


class MiniLMTextEncoder(nn.Module):
    """
    MiniLM encoder (trainable) + optional LoRA adapters.
    Input: list[str]
    Output: normalized embeddings [B, H]
    """
    def __init__(
        self,
        device="cuda",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base = AutoModel.from_pretrained(model_name)

        if use_lora:
            if not _HAS_PEFT:
                raise ImportError("peft is not installed. Run: pip install peft")

            # MiniLM is BERT-like: attention projections are usually named 'query'/'value'
            cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["query", "value"],
            )
            base = get_peft_model(base, cfg)

        self.model = base.to(self.device)

        # If NOT using LoRA, freeze everything by default (as before)
        if not use_lora:
            for p in self.model.parameters():
                p.requires_grad = False

        self.use_lora = use_lora

    def forward(self, texts: list[str]) -> torch.Tensor:
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**tok)
        sent = mean_pooling(out.last_hidden_state, tok["attention_mask"])  # [B, H]
        sent = F.normalize(sent, dim=1)
        return sent

    def print_trainable_parameters(self):
        n_trainable = 0
        n_total = 0
        for p in self.parameters():
            n_total += p.numel()
            if p.requires_grad:
                n_trainable += p.numel()
        print(f"[MiniLM] trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.4f}%)")
