"""
data_utils.py
=============

Objectif (challenge Kaggle ALTEGRAD - Molecular Graph Captioning)
---------------------------------------------------------------
On veut apprendre un modèle qui, à partir d'un graphe moléculaire (PyG Data),
produit une description en langage naturel.

Pourquoi ce fichier est important ?
-----------------------------------
- Le dataset fourni est une liste de `torch_geometric.data.Data`.
- Chaque objet contient :
    data.x           : features catégorielles des atomes (9 colonnes d'entiers)
    data.edge_index  : connectivité
    data.edge_attr   : features catégorielles des liaisons (3 colonnes d'entiers)
    data.id          : identifiant unique (obligatoire pour la soumission Kaggle)
    data.description : caption GT (train/val) ; absent sur test

Design choice:
--------------
Nous revenons à une représentation *texte réelle* des descriptions.
On NE traite PAS des embeddings CSV comme des "token ids" : c'était incohérent.

On fournit donc :
- un Dataset train/val qui renvoie (graph, description, idx)
- un Dataset test qui renvoie (graph, id)
- des collate_fn compatibles avec DataLoader PyTorch
"""

from __future__ import annotations

import pickle
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class PreprocessedGraphTextDataset(Dataset):
    """
    Dataset pour train/val : renvoie (graph, description, idx).

    Pourquoi garder idx ?
    --------------------
    - Utile pour éviter le "self-retrieval leak" :
      si on fait retrieval sur le train, il faut éviter de récupérer
      exactement la description de l'exemple courant.
    - Utile aussi pour logging/diagnostics.

    Note:
    -----
    Les graphes viennent des .pkl fournis. Chaque élément est un PyG Data.
    """

    def __init__(self, graph_path: str):
        self.graph_path = graph_path
        with open(graph_path, "rb") as f:
            self.graphs = pickle.load(f)

        self.ids = [g.id for g in self.graphs]  # aligné avec l'ordre des graphes

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        g = self.graphs[idx]
        desc = getattr(g, "description", "")
        return g, desc, idx


class PreprocessedGraphTestDataset(Dataset):
    """
    Dataset pour test : renvoie (graph, id).

    Pourquoi un dataset séparé ?
    ----------------------------
    - Sur test, `description` est absent -> on veut éviter toute confusion.
    - On prépare directement la soumission Kaggle avec (ID, description_pred).
    """

    def __init__(self, graph_path: str):
        with open(graph_path, "rb") as f:
            self.graphs = pickle.load(f)
        self.ids = [g.id for g in self.graphs]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        g = self.graphs[idx]
        return g, g.id


def collate_graph_text(batch: List[Tuple]):
    """
    Collate function train/val.

    batch: List[(graph, desc, idx)]
    returns:
      batch_graph : PyG Batch
      descs       : List[str]
      idxs        : torch.LongTensor [B]
    """
    graphs, descs, idxs = zip(*batch)
    batch_graph = Batch.from_data_list(list(graphs))
    idxs = torch.tensor(list(idxs), dtype=torch.long)
    return batch_graph, list(descs), idxs


def collate_graph_only(batch):
    """
    Collate function test.

    batch: List[(graph, id)]
    returns:
      batch_graph : PyG Batch (avec .id accessible)
      ids         : List[str]
    """
    graphs, ids = zip(*batch)
    batch_graph = Batch.from_data_list(list(graphs))
    return batch_graph, list(ids)


def load_descriptions_from_graphs(graph_path: str) -> Dict[str, str]:
    """
    Utilitaire (debug / baselines) : map id -> description.

    Peut servir pour:
    - sanity check du dataset
    - baseline retrieval-only (sans LLM)
    """
    with open(graph_path, "rb") as f:
        graphs = pickle.load(f)
    return {g.id: getattr(g, "description", "") for g in graphs}
