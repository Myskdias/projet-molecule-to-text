import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

# Import de NOS fichiers
from retrieval.architecture import DeepGINEEncoder, TextEncoder, GraphTextCLIP, MolecularCaptionModel
from retrieval.retrieval import RetrievalIndex
# On suppose que data_utils existe (comme vu précédemment)
from utils.data_utils import load_id2emb, PreprocessedGraphDataset, collate_fn

# --- CONFIGURATION GLOBALE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_GRAPHS = "data/train_graphs.pkl"
TRAIN_EMB_CSV = "data/train_embeddings.csv"

# Hyperparamètres Architecture
NODE_VOCAB = [200, 20] 
EDGE_VOCAB = [50, 20]
HIDDEN_GRAPH = 300
HIDDEN_TEXT = 256
PAD_IDX = 0

# Fichiers de sauvegarde
CLIP_WEIGHTS = "weights_stage1_clip.pt"
FINAL_MODEL_WEIGHTS = "weights_stage2_final.pt"

# ==================================================================================
# ÉTAPE 1 : ALIGNEMENT CLIP (Contrastive Pre-training)
# ==================================================================================
def train_stage_1_clip(train_dl, vocab_size, epochs=5):
    print("\n" + "="*50)
    print("DEMARRAGE ETAPE 1 : ALIGNEMENT CLIP")
    print("Objectif : Apprendre aux encodeurs à rapprocher Graphe et Texte")
    print("="*50)

    # 1. Instanciation des Encodeurs
    graph_enc = DeepGINEEncoder(NODE_VOCAB, EDGE_VOCAB, HIDDEN_GRAPH)
    text_enc = TextEncoder(vocab_size, 256) # Projecteur CLIP vers 256
    
    # Wrapper CLIP
    model = GraphTextCLIP(graph_enc, text_enc, HIDDEN_GRAPH, 256).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [CLIP]")
        
        for batch in pbar:
            batch_graph, texts = batch
            batch_graph = batch_graph.to(DEVICE)
            texts = texts.to(DEVICE).long()
            texts = torch.clamp(texts, min=0, max=vocab_size-1)
            text_mask = (texts == PAD_IDX)
            
            optimizer.zero_grad()
            
            # Forward CLIP
            I_g, I_t = model(batch_graph, texts, text_mask)
            
            # Loss Contrastive Symétrique
            logits = (model.logit_scale.exp()) * (I_g @ I_t.t())
            labels = torch.arange(I_g.size(0)).to(DEVICE)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    # Sauvegarde des encodeurs pré-entraînés
    torch.save({
        'graph_encoder': model.graph_encoder.state_dict(),
        'text_encoder': model.text_encoder.state_dict(),
        'graph_proj': model.graph_proj.state_dict(),
        'text_proj': model.text_proj.state_dict(),
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
    # 1. Chargement Données & Vocab
    print("Chargement des données...")
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    
    # Calcul Vocabulaire Réel
    max_id = 0
    temp_dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    for _, txt in tqdm(temp_dl, desc="Scan Vocab"):
        max_id = max(max_id, txt.long().max().item())
    VOCAB_SIZE = max_id + 100
    print(f"Vocabulaire détecté : {VOCAB_SIZE}")

    # Loader pour entraînement
    train_dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 2. Exécution du Pipeline
    # Étape A : CLIP
    train_stage_1_clip(train_dl, VOCAB_SIZE, epochs=10)

if __name__ == "__main__":
    main()