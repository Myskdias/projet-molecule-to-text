import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from retrieval.cross_encoder.cross_encoder_trainer import train_cross_encoder_batched
from retrieval.cross_encoder.cross_encoder import (
    CrossEncoderTrainDataset,
    cross_encoder_collate_fn,
)
from retrieval.retrieval import RetrievalIndex
from retrieval.architecture import DeepGINEEncoder, GraphTextCLIP
from retrieval.text_encoder import MiniLMTextEncoder
from utils.data_utils import PreprocessedGraphDataset, collate_fn, load_descriptions_from_graphs

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_GRAPHS = "data/train_graphs.pkl"
CLIP_WEIGHTS = "other/weights_stage1_clip.pt"
CROSS_ENCODER_WEIGHTS = "other/weights_cross_encoder.pt"

NODE_VOCAB = [200, 20]
EDGE_VOCAB = [50, 20]
HIDDEN_GRAPH = 300

INDEX_BATCH_SIZE = 32
BATCH_SIZE = 32
NUM_NEGATIVES = 5
K_RETRIEVAL = 10

# ============================================================
# MAIN
# ============================================================
def main():
    print("[Cross-Encoder] Loading data...")

    # ----------------------------
    # Load graphs & descriptions
    # ----------------------------
    # Dataset de base
    ds_train = PreprocessedGraphDataset(TRAIN_GRAPHS, mode=True)
    dl_train = DataLoader(
        ds_train,
        batch_size=INDEX_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    # Accès direct aux graphes
    train_graphs = ds_train.graphs
    id2desc_train = load_descriptions_from_graphs(TRAIN_GRAPHS)

    # ----------------------------
    # Load CLIP model (frozen)
    # ----------------------------
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

    # ----------------------------
    # Build retrieval index (TRAIN)
    # ----------------------------
    retriever = RetrievalIndex(device=DEVICE)
    retriever.build_index(clip_model, dl_train)

    # ------------------------------------------------
    # Precompute neighbors (top-k) for every train graph
    # ------------------------------------------------
    print("[Cross-Encoder] Precomputing train neighbors (top-k)...")

    neighbors = []

    with torch.no_grad():
        for batch_graph, _ in tqdm(dl_train, desc="Precompute NN"):
            batch_graph = batch_graph.to(DEVICE)
            nn_idx, _ = retriever.query(clip_model, batch_graph, k=K_RETRIEVAL)  # [B, K]
            neighbors.extend(nn_idx.cpu().tolist())

    assert len(neighbors) == len(train_graphs)
    print("[Cross-Encoder] Neighbors ready.")

    # ----------------------------
    # Dataset & DataLoader
    # ----------------------------
    train_dataset = CrossEncoderTrainDataset(
        graphs=train_graphs,
        id2desc=id2desc_train,
        neighbors=neighbors,
        n_negatives=NUM_NEGATIVES,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=cross_encoder_collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # ----------------------------
    # Train cross-encoder
    # ----------------------------
    train_cross_encoder_batched(
        train_loader=train_loader,
        graph_encoder=clip_model,
        text_encoder=text_encoder,
        dest=CROSS_ENCODER_WEIGHTS,
        device=DEVICE,
    )

    print("[Cross-Encoder] Done.")


if __name__ == "__main__":
    main()
