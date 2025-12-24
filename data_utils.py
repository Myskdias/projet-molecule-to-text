import pickle
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


# =========================================================
# Load descriptions from preprocessed graphs
# =========================================================
def load_descriptions_from_graphs(graph_path: str) -> Dict[str, str]:
    with open(graph_path, "rb") as f:
        graphs = pickle.load(f)
    return {g.id: g.description for g in graphs}


# =========================================================
# Dataset returning (graph, description, global_index)
# =========================================================
class PreprocessedGraphTextDataset(Dataset):
    """
    Returns (graph, description:str, idx:int) for train/val splits.
    """
    def __init__(self, graph_path: str):
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, "rb") as f:
            self.graphs = pickle.load(f)
        self.ids = [g.id for g in self.graphs]
        print(f"Loaded {len(self.graphs)} graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: int):
        g = self.graphs[idx]
        desc = getattr(g, "description", None)
        if desc is None:
            # test set
            desc = ""
        return g, desc, idx


def collate_graph_text(batch: List[Tuple]):
    """
    batch: list of (graph, desc, idx)
    """
    graphs, descs, idxs = zip(*batch)
    batch_graph = Batch.from_data_list(list(graphs))
    idxs = torch.tensor(list(idxs), dtype=torch.long)
    return batch_graph, list(descs), idxs


def collate_graph_only(batch):
    return Batch.from_data_list(batch)
