"""
Data loading and processing utilities for molecule-text retrieval.
Includes dataset classes and data loading functions.
"""
from typing import Dict
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


# =========================================================
# Feature maps for atom and bond attributes
# =========================================================
from typing import Dict, List, Any

x_map: Dict[str, List[Any]] = {
    'atomic_num': list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER',
        'CHI_TETRAHEDRAL','CHI_ALLENE','CHI_SQUAREPLANAR','CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2','OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE',
        'ONEANDAHALF','TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF',
        'AROMATIC','IONIC','HYDROGEN','THREECENTER','DATIVEONE','DATIVE','DATIVEL',
        'DATIVER','OTHER','ZERO',
    ],
    'stereo': [
        'STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


# =========================================================
# Load precomputed text embeddings
# =========================================================
def load_id2emb(csv_path: str) -> Dict[str, torch.Tensor]:
    """
    Load precomputed text embeddings from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: ID, embedding
                  where embedding is comma-separated floats
        
    Returns:
        Dictionary mapping ID (str) to embedding tensor
    """
    df = pd.read_csv(csv_path)
    id2emb = {}
    for _, row in df.iterrows():
        id_ = str(row["ID"])
        emb_str = row["embedding"]
        emb_vals = [float(x) for x in str(emb_str).split(',')]
        id2emb[id_] = torch.tensor(emb_vals, dtype=torch.float32)
    return id2emb


# =========================================================
# Load descriptions from preprocessed graphs
# =========================================================
def load_descriptions_from_graphs(graph_path: str) -> Dict[str, str]:
    """
    Load ID to description mapping from preprocessed graph file.
    
    Args:
        graph_path: Path to .pkl file containing list of pre-saved graphs
        
    Returns:
        Dictionary mapping ID (str) to description (str)
    """
    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)
    
    id2desc = {}
    for graph in graphs:
        id2desc[graph.id] = graph.description
    
    return id2desc


# =========================================================
# Dataset that loads preprocessed graphs and text
# =========================================================
class PreprocessedGraphDataset(Dataset):
    """
    Dataset that loads preprocessed molecule graphs and their textual descriptions.
    Compatible with MiniLM text encoder.
    """
    def __init__(self, graph_path: str, mode: bool=True):
        self.mode = mode
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        print(f"Loaded {len(self.graphs)} graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.mode:
            text = graph.description   # <-- TEXTE BRUT
            return graph, text
        else:
            return graph
        
    @classmethod
    def from_graphs(cls, graphs, mode: bool = False):
        """
        Build dataset directly from a list of preloaded graphs.
        Useful for retrieval over TRAIN + VAL without reloading disk.
        """
        obj = cls.__new__(cls)   # bypass __init__
        obj.graphs = graphs
        obj.mode = mode
        return obj




def collate_fn(batch):
    graphs, texts = zip(*batch)
    batch_graph = Batch.from_data_list(list(graphs))
    return batch_graph, list(texts)

def collate_fn_test(batch):
    # batch = [graph, graph, ...]
    batch_graph = Batch.from_data_list(list(batch))
    return batch_graph