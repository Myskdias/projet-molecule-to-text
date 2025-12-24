"""
data_utils.py
=============

RÔLE DANS LE PROJET
------------------
Ce fichier définit l'interface entre :
- les données brutes (graphes moléculaires PyTorch Geometric)
- le pipeline d'entraînement PyTorch

Il est *critique* car toute erreur ici (ID, alignement texte/graphe,
collate incorrect) conduit à :
- erreurs Kaggle ("ID column not found")
- fuites d'information (train -> retrieval)
- bugs silencieux très coûteux à diagnostiquer

CONTEXTE DATASET (Challenge ALTEGRAD)
------------------------------------
Chaque molécule est représentée par un objet torch_geometric.data.Data avec :
- data.x           : [N, 9] features atomiques catégorielles
- data.edge_index  : [2, E] connectivité
- data.edge_attr   : [E, 3] features de liaisons catégorielles
- data.id          : identifiant unique (clé Kaggle)
- data.description : texte GT (absent pour test)

CHOIX DE DESIGN
---------------
- On travaille avec les *textes bruts* (string), PAS avec des embeddings CSV
  → plus propre, plus flexible, et compatible avec T5/BERTScore.
- On sépare explicitement train/val et test
  → évite les erreurs de logique en inference.
"""

from __future__ import annotations

import pickle
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


# ============================================================
# Dataset TRAIN / VAL
# ============================================================

class PreprocessedGraphTextDataset(Dataset):
    """
    Dataset utilisé pour l'entraînement et la validation.

    Chaque item renvoyé est :
        (graph, description, idx)

    POURQUOI inclure idx ?
    ----------------------
    - Indispensable pour le retrieval training :
      on doit éviter de récupérer la description exacte de l'exemple courant.
    - Permet aussi un debugging propre (correspondance index <-> graphe).

    ANTI-CHOIX (important):
    -----------------------
    - On ne renvoie PAS directement des embeddings texte ici.
      → l'encodage texte dépend du tokenizer et du modèle utilisé
      → il doit rester dans le pipeline modèle, pas dans le dataset.
    """

    def __init__(self, graph_path: str):
        with open(graph_path, "rb") as f:
            self.graphs = pickle.load(f)

        self.ids = [g.id for g in self.graphs]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        g = self.graphs[idx]

        # En train/val, description est toujours présente
        desc = getattr(g, "description", "")
        return g, desc, idx


# ============================================================
# Dataset TEST
# ============================================================

class PreprocessedGraphTestDataset(Dataset):
    """
    Dataset pour l'inférence Kaggle.

    Chaque item renvoyé est :
        (graph, id)

    POURQUOI un dataset séparé ?
    ----------------------------
    - Sur le test set, la description GT n'existe pas.
    - Cela évite toute ambiguïté ou bug de logique.
    """

    def __init__(self, graph_path: str):
        with open(graph_path, "rb") as f:
            self.graphs = pickle.load(f)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        g = self.graphs[idx]
        return g, g.id


# ============================================================
# Collate functions
# ============================================================

def collate_graph_text(batch: List[Tuple]):
    """
    Collate function pour train / val.

    Entrée :
        batch = [(graph, desc, idx), ...]

    Sortie :
        - batch_graph : PyG Batch (concaténation des graphes)
        - descs       : List[str] (textes bruts)
        - idxs        : Tensor [B] (indices dataset)

    IMPORTANT :
    -----------
    - Les descriptions restent des strings.
    - La tokenisation est faite PLUS TARD, avec le bon tokenizer.
    """
    graphs, descs, idxs = zip(*batch)
    batch_graph = Batch.from_data_list(list(graphs))
    idxs = torch.tensor(list(idxs), dtype=torch.long)
    return batch_graph, list(descs), idxs


def collate_graph_only(batch):
    """
    Collate function pour test.

    Entrée :
        batch = [(graph, id), ...]

    Sortie :
        - batch_graph : PyG Batch
        - ids         : List[str]
    """
    graphs, ids = zip(*batch)
    batch_graph = Batch.from_data_list(list(graphs))
    return batch_graph, list(ids)
