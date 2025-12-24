"""
retrieval.py
============

Rôle:
-----
Gérer l'index de retrieval dans l'espace latent commun (GraphTextCLIP).

Lien avec le cours / RAG:
-------------------------
On est dans un schéma "Retrieve-then-Generate":
- Retrieve: nearest neighbors dans un espace vectoriel (cosine/dot-product)
- Generate: LLM conditionné par l'info récupérée + soft prompt graphe

Pourquoi un index "texte" plutôt qu'un index "graphe" ?
-------------------------------------------------------
Dans CLIP-like, graph_emb et text_emb sont dans le même espace.
Le plus naturel: query = graph_emb, database = text_emb (captions train).
Cela correspond directement à: "trouver la caption dont l'embedding est le plus proche du graphe".
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


class RetrievalIndex:
    """
    Stocke:
    - embeddings textuels normalisés dans l'espace commun (shared_dim)
    - textes bruts correspondants

    On interroge avec:
    - embeddings graphe normalisés (via clip_model.encode_graph)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.text_emb: torch.Tensor | None = None   # [N, D] sur device
        self.texts: List[str] | None = None

    @torch.no_grad()
    def build_text_index(
        self,
        clip_model,
        tokenizer,
        train_texts: List[str],
        batch_size: int = 64,
        max_len: int = 256,
    ):
        """
        Construit l'index des embeddings de textes.

        Pourquoi pré-calculer ?
        ------------------------
        - Retrieval fréquent en training/inference.
        - On évite le coût d'encoder tout le train à chaque requête.

        Important:
        ----------
        Si on dégèle et met à jour l'encodeur texte (ou text_proj),
        il faut reconstruire l'index. Dans notre design, on gèle l'encodeur texte.
        """
        clip_model.eval()

        self.texts = list(train_texts)
        all_vecs = []

        for i in tqdm(range(0, len(train_texts), batch_size), desc="Build text index"):
            batch_txt = train_texts[i : i + batch_size]
            tok = tokenizer(
                batch_txt,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(self.device)

            v = clip_model.encode_text(tok["input_ids"], tok["attention_mask"])  # [B,D], déjà normalized
            all_vecs.append(v.detach().cpu())

        self.text_emb = torch.cat(all_vecs, dim=0).to(self.device)              # [N,D]
        self.text_emb = F.normalize(self.text_emb, dim=1)
        return self

    @torch.no_grad()
    def query(self, clip_model, batch_graph, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query retrieval: graph -> top-k textes.

        Returns:
            idx    : [B,k] indices dans self.texts/self.text_emb
            scores : [B,k] similarités (dot product entre vecteurs normalisés = cosine)
        """
        assert self.text_emb is not None, "Call build_text_index() first"
        clip_model.eval()

        batch_graph = batch_graph.to(self.device)
        q = clip_model.encode_graph(batch_graph)    # [B,D], normalized
        sims = q @ self.text_emb.t()                # [B,N]

        scores, idx = torch.topk(sims, k=k, dim=1)
        return idx, scores

    def get_texts(self, indices_1d: torch.Tensor) -> List[str]:
        """
        Map indices -> texts.
        indices_1d: [B] (cpu ou gpu)
        """
        assert self.texts is not None
        idx = indices_1d.detach().cpu().tolist()
        return [self.texts[i] for i in idx]
