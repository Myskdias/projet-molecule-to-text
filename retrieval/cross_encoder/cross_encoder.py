import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Batch
from torch.utils.data import Dataset

import random

class CrossEncoderTrainDataset(Dataset):
    def __init__(
        self,
        graphs,
        id2desc,
        neighbors,          # List[List[int]] : indices des top-k voisins pour chaque idx
        n_negatives=5,
    ):
        self.graphs = graphs
        self.id2desc = id2desc
        self.neighbors = neighbors
        self.n_neg = n_negatives

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        gt_text = self.id2desc[graph.id]

        nn_indices = self.neighbors[idx]  # liste d’indices (top-k)

        neg_texts = []
        for j in nn_indices:
            cand_graph = self.graphs[j]
            cand_text = self.id2desc[cand_graph.id]
            if cand_text != gt_text:
                neg_texts.append(cand_text)
            if len(neg_texts) == self.n_neg:
                break

        # fallback rare
        if len(neg_texts) == 0:
            neg_texts = [gt_text] * self.n_neg
        elif len(neg_texts) < self.n_neg:
            neg_texts += random.choices(neg_texts, k=self.n_neg - len(neg_texts))

        return graph, gt_text, neg_texts


def cross_encoder_collate_fn(batch):
    graphs, gt_texts, neg_texts_batch = zip(*batch)

    graph_batch = Batch.from_data_list(graphs)
    gt_texts = list(gt_texts)
    neg_texts_batch = list(neg_texts_batch)

    return graph_batch, gt_texts, neg_texts_batch

class GraphTextCrossEncoder(nn.Module):
    def __init__(self, graph_dim, text_dim, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(graph_dim + text_dim + graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph_emb, text_emb):
        diff = torch.abs(graph_emb - text_emb)
        x = torch.cat([graph_emb, text_emb, diff], dim=-1)
        return self.mlp(x).squeeze(-1)

