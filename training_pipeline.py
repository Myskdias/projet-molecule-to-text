import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

# Import de NOS fichiers
from architecture import DeepGINEEncoder, TextEncoder, GraphTextCLIP, MolecularCaptionModel
from retrieval import RetrievalIndex
# On suppose que data_utils existe (comme vu précédemment)
from data_utils import load_id2emb, PreprocessedGraphDataset, collate_fn

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
        'text_encoder': model.text_encoder.state_dict()
    }, CLIP_WEIGHTS)
    print(f"Étape 1 terminée. Poids sauvegardés dans {CLIP_WEIGHTS}")
    
    return model.graph_encoder, model.text_encoder

# ==================================================================================
# ÉTAPE 2 : GÉNÉRATION RAG (End-to-End avec Decoder)
# ==================================================================================
def train_stage_2_rag(train_dl, ds_full, vocab_size, epochs=10):
    print("\n" + "="*50)
    print("DEMARRAGE ETAPE 2 : GÉNÉRATION RAG")
    print("Objectif : Apprendre à écrire en utilisant l'antisèche (Retrieved Text)")
    print("="*50)
    
    # 1. Chargement des Poids Pré-entraînés
    if not os.path.exists(CLIP_WEIGHTS):
        raise FileNotFoundError("Lancez l'étape 1 d'abord !")
    
    checkpoint = torch.load(CLIP_WEIGHTS)
    
    # 2. Instanciation (On recrée des instances propres)
    graph_enc = DeepGINEEncoder(NODE_VOCAB, EDGE_VOCAB, HIDDEN_GRAPH)
    text_enc = TextEncoder(vocab_size, 256)
    
    graph_enc.load_state_dict(checkpoint['graph_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    print("Poids CLIP chargés dans les encodeurs.")

    # 3. Construction de l'Index RAG (Avec l'encodeur Intelligent !)
    # On utilise l'encodeur de graphe pré-entraîné pour vectoriser la base
    retriever = RetrievalIndex(device=DEVICE)
    
    # Préparation des données pour l'index
    # On a besoin d'un loader séquentiel pour extraire les textes
    index_dl = DataLoader(ds_full, batch_size=32, shuffle=False, collate_fn=collate_fn)
    captions_tokens = []
    
    print("Extraction des textes pour l'index...")
    for _, caps in tqdm(index_dl):
        caps = torch.clamp(caps.long(), min=0, max=vocab_size-1)
        for i in range(caps.size(0)): captions_tokens.append(caps[i].clone().detach().cpu())
            
    # Construction effective
    retriever.build_index(graph_enc, index_dl, captions_tokens)

    # 4. Modèle Complet
    # Note : text_enc sert ici à encoder le texte récupéré (antisèche)
    model = MolecularCaptionModel(graph_enc, text_enc, vocab_size, d_model=256, graph_dim=HIDDEN_GRAPH).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5) # LR plus faible pour Fine-Tuning
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 5. Boucle d'entraînement
    model.train()
    grad_acc_steps = 4 # Accumulation pour simuler un batch de 32 (si batch=8)
    
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [RAG]")
        
        for batch_idx, batch in enumerate(pbar):
            batch_graph, targets = batch
            batch_graph = batch_graph.to(DEVICE)
            targets = targets.to(DEVICE).long()
            targets = torch.clamp(targets, min=0, max=vocab_size-1)
            
            # --- LOGIQUE RAG ---
            with torch.no_grad():
                # Recherche Voisins (Utilise l'intelligence CLIP)
                neighbor_indices, _ = retriever.query(graph_enc, batch_graph, k=5)
                selected_indices = neighbor_indices[:, 2] # 3ème voisin pour éviter le leak
                
                retrieved_ids = retriever.get_retrieved_tokens(selected_indices).to(DEVICE)
                retrieved_ids = torch.clamp(retrieved_ids, min=0, max=vocab_size-1)
                
                # Anti-Leakage (Firewall)
                # Si le texte est trop similaire (>80%), on masque
                for i in range(targets.size(0)):
                    t_set = set(targets[i].tolist()) - {PAD_IDX, 1, 2}
                    r_set = set(retrieved_ids[i].tolist()) - {PAD_IDX, 1, 2}
                    if len(t_set) > 0 and (len(t_set.intersection(r_set)) / len(t_set) > 0.8):
                        retrieved_ids[i, :] = PAD_IDX
                
                retrieved_mask = (retrieved_ids == PAD_IDX)

            # Inputs Décodeur
            tgt_in = targets[:, :-1]
            tgt_out = targets[:, 1:]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(DEVICE)
            tgt_pad_mask = (tgt_in == PAD_IDX)
            
            # Forward
            logits = model(batch_graph, retrieved_ids, retrieved_mask, tgt_in, tgt_mask, tgt_pad_mask)
            
            # Loss
            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            loss = loss / grad_acc_steps
            loss.backward()
            
            if (batch_idx + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_acc_steps
            pbar.set_postfix({'loss': f"{loss.item() * grad_acc_steps:.4f}"})
            
    torch.save(model.state_dict(), FINAL_MODEL_WEIGHTS)
    print("✅ Entraînement terminé.")

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
    train_dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # 2. Exécution du Pipeline
    # Étape A : CLIP
    train_stage_1_clip(train_dl, VOCAB_SIZE, epochs=10)
    
    # Nettoyage mémoire avant étape 2
    gc.collect()
    torch.cuda.empty_cache()
    
    # Étape B : RAG
    train_stage_2_rag(train_dl, ds, VOCAB_SIZE, epochs=10)

if __name__ == "__main__":
    main()