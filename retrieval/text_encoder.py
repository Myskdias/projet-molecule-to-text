from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniLMTextEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=device
        )
        for p in self.model.parameters():
            p.requires_grad = False  # IMPORTANT

    def forward(self, texts: list[str]):
        emb = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return emb.clone()  # [B, 384]