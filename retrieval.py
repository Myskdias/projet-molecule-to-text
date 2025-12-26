"""
retrieval.py
============

RÔLE DANS LE PROJET
------------------
Ce fichier implémente le module de retrieval utilisé dans le pipeline
Retrieve-then-Generate.

Il permet, pour un graphe donné :
    - d'encoder le graphe dans l'espace latent commun
    - de retrouver la ou les descriptions textuelles
      les plus proches (similarité cosinus)

Ce module est utilisé :
    - en Stage 2 (entraînement génération)
    - en inference Kaggle

IMPORTANT :
-----------
Le retrieval est traité comme une MÉMOIRE EXTERNE.
Il est :
    - non différentiable
    - gelé pendant la génération
    - conceptuellement séparé du LLM

-----------------------------------------------------------------------
LIENS AVEC LE COURS / CONCEPTS
-----------------------------------------------------------------------

- Contrastive Learning (CLIP-style)
- Similarité cosinus
- Retrieval-Augmented Generation (RAG)
- Nearest Neighbors Search

-----------------------------------------------------------------------
PHILOSOPHIE DE DESIGN
-----------------------------------------------------------------------

1) Simplicité > micro-optimisation
2) Stabilité > sophistication
3) Aucune fuite d'information autorisée

ANTI-CHOIX :
------------

❌ Nous n'utilisons PAS FAISS.
   → dataset trop petit (~32k)
   → complexité inutile
   → torch.topk est suffisant

❌ Nous n'utilisons PAS de retrieval différentiable.
   → conceptuellement incorrect
   → instable
   → le retrieval est une mémoire externe

❌ Nous ne concaténons PAS plusieurs captions.
   → augmente la longueur d'entrée
   → bruit sémantique
   → BLEU souvent dégradé

❌ Nous n'utilisons PAS de score hybride lexical + sémantique.
   → l'espace latent CLIP capture déjà la sémantique

   
Ce module :
- est simple
- est robuste
- est fondamental pour les scores Kaggle
- implémente correctement le paradigme RAG

Toute erreur ici :
→ fuites d'information
→ scores artificiellement élevés en train
→ effondrement en test

"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm


class RetrievalIndex:
    """
    Index de retrieval simple en mémoire.

    Il stocke :
    - les embeddings texte (normalisés)
    - les descriptions correspondantes
    - les indices dataset (pour éviter le self-retrieval)

    La recherche se fait par :
        similarité cosinus
    """

    def __init__(self, clip_model, tokenizer, device: str = "cuda"):
        """
        Args:
            clip_model :
                Instance de GraphTextCLIP entraînée (Stage 1)

            tokenizer :
                Tokenizer T5 (cohérence avec l'encodeur texte)

            device :
                cpu / cuda
        """
        self.clip = clip_model
        self.tokenizer = tokenizer
        self.device = device

        # Buffers internes
        self.text_embs = None     # Tensor [N, D]
        self.texts = None         # List[str]
        self.idxs = None          # Tensor [N]

    # ========================================================
    # Construction de l'index
    # ========================================================

    def build(self, dataloader):
        """
        Construit l'index de retrieval à partir du dataset d'entraînement.

        PROCESSUS
        ---------
        - On encode TOUTES les descriptions du dataset
        - Une seule fois
        - En mode no_grad

        POURQUOI PAS À LA VOLÉE ?
        -------------------------
        - inefficace
        - embeddings texte fixes après Stage 1
        - gain énorme en temps pendant Stage 2
        """
        self.clip.eval()

        all_embs = []
        all_texts = []
        all_idxs = []

        print("[RetrievalIndex] Building text embedding index...")

        with torch.no_grad():
            for batch_graph, descs, idxs in tqdm(dataloader):
                tok = self.tokenizer(
                    descs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                # Encodage texte → espace latent
                t = self.clip.encode_text(
                    tok["input_ids"],
                    tok["attention_mask"],
                )  # [B, D], déjà normalisé

                all_embs.append(t.cpu())
                all_texts.extend(descs)
                all_idxs.append(idxs)

        self.text_embs = torch.cat(all_embs, dim=0)  # [N, D]
        self.texts = all_texts
        self.idxs = torch.cat(all_idxs, dim=0)       # [N]

        print(f"[RetrievalIndex] Indexed {len(self.texts)} captions.")

    # ========================================================
    # Retrieval
    # ========================================================

    def retrieve(self, batch_graph, idxs=None, topk: int = 1):
        """
        Récupère les captions les plus proches pour un batch de graphes.

        Args:
            batch_graph :
                PyG Batch de graphes

            idxs :
                Indices dataset des graphes du batch.
                Utilisés pour éviter le self-retrieval.
                Peut être None en inference.

            topk :
                Nombre de voisins à considérer

        Returns:
            List[str] : une caption par graphe
        """
        assert self.text_embs is not None, "Index not built. Call build() first."

        with torch.no_grad():
            # Encodage graphe → espace latent
            g = self.clip.encode_graph(batch_graph)  # [B, D], normalisé

            # Similarité cosinus = produit scalaire
            text_embs = self.text_embs.to(g.device)
            sims = g @ text_embs.t()             # [B, N]

            # ------------------------------------------------
            # Gestion du self-retrieval leak (CRUCIAL)
            # ------------------------------------------------
            if idxs is not None:
                """
                Sans cette étape :
                - le graphe i récupère EXACTEMENT sa propre description
                - la loss devient artificiellement facile
                - le modèle n'apprend RIEN

                On force donc la similarité à -∞
                pour les paires (i, i).
                """
                for i, idx in enumerate(idxs):
                    sims[i, self.idxs == idx] = -1e9

            # Top-k plus proches voisins
            topk_vals, topk_idx = torch.topk(sims, k=topk, dim=1)

            # ------------------------------------------------
            # Sélection finale
            # ------------------------------------------------
            """
            Stratégie :
            - on prend simplement le meilleur voisin (top-1)

            Extensions possibles :
            - sampling parmi top-k
            - concaténation de plusieurs captions
            """
            retrieved = [
                self.texts[j]
                for j in topk_idx[:, 0].tolist()
            ]

        return retrieved
