import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

# Import de NOS fichiers
from retrieval.architecture import DeepGINEEncoder, GraphTextCLIP
from retrieval.retrieval import RetrievalIndex
from retrieval.text_encoder import MiniLMTextEncoder

from utils.data_utils import load_id2emb, PreprocessedGraphDataset, collate_fn

# --- CONFIGURATION GLOBALE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_GRAPHS = "data/train_graphs.pkl"

# Hyperparamètres Architecture
NODE_VOCAB = [200, 20] 
EDGE_VOCAB = [50, 20]
HIDDEN_GRAPH = 300
HIDDEN_TEXT = 256

# Fichiers de sauvegarde
CLIP_WEIGHTS = "weights_stage1_clip.pt"

# ==================================================================================
# ÉTAPE 1 : ALIGNEMENT CLIP (Contrastive Pre-training)
# ==================================================================================
def train_stage_1_clip(train_dl, epochs=5):
    print("\n" + "="*50)
    print("DEMARRAGE ETAPE 1 : ALIGNEMENT CLIP")
    print("Objectif : Apprendre aux encodeurs à rapprocher Graphe et Texte")
    print("="*50)

    # 1. Instanciation des Encodeurs
    graph_enc = DeepGINEEncoder(NODE_VOCAB, EDGE_VOCAB, HIDDEN_GRAPH)
    text_enc  = MiniLMTextEncoder(device=DEVICE)
    
    # Wrapper CLIP
    model = GraphTextCLIP(
        graph_encoder=graph_enc,
        graph_dim=HIDDEN_GRAPH,
        text_encoder=text_enc,
        shared_dim=384
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [CLIP]")
        
        for batch in pbar:
            batch_graph, batch_text = batch
            batch_graph = batch_graph.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward CLIP
            I_g, I_t = model(batch_graph, batch_text)
            
            # Loss Contrastive Symétrique
            logits = (model.logit_scale.exp()) * (I_g @ I_t.t())
            labels = torch.arange(I_g.size(0), device=DEVICE)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        print("Average loss: ", total_loss / num_batches)
        scheduler.step()
        print(f"LR: {scheduler.get_last_lr()[0]:.2e}")
        if(epoch % 5 == 0 and not epoch == 0):
            torch.save({
                'graph_encoder': model.graph_encoder.state_dict(),
                'text_encoder': model.text_encoder.state_dict(),
                'graph_proj': model.graph_proj.state_dict(),
                'logit_scale': model.logit_scale.data
            }, f"checkpoints/checkpoint_1_{epoch}.pt")

    # Sauvegarde des encodeurs pré-entraînés
    torch.save({
        'graph_encoder': model.graph_encoder.state_dict(),
        'text_encoder': model.text_encoder.state_dict(),
        'graph_proj': model.graph_proj.state_dict(),
        'logit_scale': model.logit_scale.data
    }, CLIP_WEIGHTS)
    print(f"Étape 1 terminée. Poids sauvegardés dans {CLIP_WEIGHTS}")
    
    return model #model.graph_encoder, model.text_encoder


# ==================================================================================
# MAIN
# ==================================================================================
def main():
    if not os.path.exists(TRAIN_GRAPHS):
        print("Erreur: Données introuvables.")
        return

    print(f"Device: {DEVICE}")
    print("Chargement des données...")

    # Dataset : (graph, description: str)
    ds = PreprocessedGraphDataset(TRAIN_GRAPHS)

    # DataLoader
    train_dl = DataLoader(
        ds,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Entraînement CLIP (retrieval only)
    train_stage_1_clip(train_dl, epochs=25)

if __name__ == "__main__":
    main()