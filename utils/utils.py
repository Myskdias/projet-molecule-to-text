import torch
from torch.utils.data import DataLoader
import os

from retrieval.architecture import DeepGINEEncoder, GraphTextCLIP
from retrieval.cross_encoder.cross_encoder import GraphTextCrossEncoder
from retrieval.retrieval import RetrievalIndex
from retrieval.text_encoder import MiniLMTextEncoder
from utils.data_utils import PreprocessedGraphDataset, collate_fn, load_descriptions_from_graphs

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS = "data/validation_graphs.pkl"
CLIP_WEIGHTS = "other/weights_stage1_clip.pt"
CROSS_ENCODER_WEIGHTS = "other/weights_cross_encoder.pt"

NODE_VOCAB = [200, 20]
EDGE_VOCAB = [50, 20]
HIDDEN_GRAPH = 300

INDEX_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32

def build():
    # -----------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------
    for path in [TRAIN_GRAPHS, VAL_GRAPHS, CLIP_WEIGHTS]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    print(f"[VAL EVAL] Device: {DEVICE}")

    # -----------------------------------------------------
    # Load TRAIN dataset
    # -----------------------------------------------------
    print("[VAL EVAL] Loading TRAIN graphs...")
    ds_train = PreprocessedGraphDataset(TRAIN_GRAPHS)
    dl_train = DataLoader(
        ds_train,
        batch_size=INDEX_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    id2desc_train = load_descriptions_from_graphs(TRAIN_GRAPHS)

    # -----------------------------------------------------
    # Load VAL dataset
    # -----------------------------------------------------
    print("[VAL EVAL] Loading VAL graphs...")
    ds_val = PreprocessedGraphDataset(VAL_GRAPHS)
    dl_val = DataLoader(
        ds_val,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    id2desc_val = load_descriptions_from_graphs(VAL_GRAPHS)

    # -----------------------------------------------------
    # Build CLIP model (MiniLM + GNN)
    # -----------------------------------------------------
    print("[VAL EVAL] Loading CLIP model...")

    graph_encoder = DeepGINEEncoder(
        NODE_VOCAB,
        EDGE_VOCAB,
        hidden_dim=HIDDEN_GRAPH,
    ).to(DEVICE)

    text_encoder = MiniLMTextEncoder(device=DEVICE)

    clip_model = GraphTextCLIP(
        graph_encoder=graph_encoder,
        graph_dim=HIDDEN_GRAPH,
        text_encoder=text_encoder,
    ).to(DEVICE)

    checkpoint = torch.load(CLIP_WEIGHTS, map_location=DEVICE)
    clip_model.graph_encoder.load_state_dict(checkpoint["graph_encoder"])
    clip_model.graph_proj.load_state_dict(checkpoint["graph_proj"])
    clip_model.logit_scale.data = checkpoint["logit_scale"]

    clip_model.eval()

    '''
    # =========================================================
    # Load Cross-Encoder
    # =========================================================
    ckpt = torch.load(CROSS_ENCODER_WEIGHTS, map_location=DEVICE)

    cross_encoder = GraphTextCrossEncoder(
        graph_dim=ckpt["graph_dim"],
        text_dim=ckpt["text_dim"],
    ).to(DEVICE)

    cross_encoder.load_state_dict(ckpt["state_dict"])
    cross_encoder.eval()

    print("[VAL EVAL] Cross-encoder loaded.")
    '''
    # -----------------------------------------------------
    # Build retrieval index (TRAIN graphs)
    # -----------------------------------------------------
    print("[VAL EVAL] Building retrieval index...")
    retriever = RetrievalIndex(device=DEVICE)
    retriever.build_index(clip_model, dl_train)

    return {"dl_train": dl_train,
            "ds_train": ds_train,
            "dl_val": dl_val,
            "ds_val": ds_val,
            "id2desc_train": id2desc_train,
            "id2desc_val": id2desc_val,
            "clip_model": clip_model,
            "retriever": retriever
            }