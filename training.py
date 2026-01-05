import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tokenizers import Tokenizer

# Imports LOCAUX (Attention au nom du fichier architecture)
from architecture_bpe import DeepGINEEncoder, Graph2TextTransformer
from data_loader import MolecularDataset, custom_collate_fn

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_GRAPHS = "data/train_graphs.pkl"
TOKENIZER_FILE = "bpe_tokenizer.json" 

# --- HYPERPARAMÈTRES BOOSTÉS ---
BATCH_SIZE = 16 # Si OOM, baisse à 16 et utilise gradient_accumulation
EPOCHS = 30     # On augmente car le modèle est plus gros et a besoin de temps
LR_MAX = 5e-4   # Pic du scheduler

# Architecture "Large"
NODE_VOCAB = [200, 20] 
EDGE_VOCAB = [50, 20]
GRAPH_DIM = 300
D_MODEL = 512   # Doublé
N_HEAD = 8      # Doublé
N_LAYERS = 6    # Doublé
DROPOUT = 0.2

def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device, pad_id):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        batch_graph, targets = batch
        batch_graph = batch_graph.to(device)
        targets = targets.to(device).long()
        
        tgt_input = targets[:, :-1]
        tgt_output = targets[:, 1:]
        
        seq_len = tgt_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        tgt_padding_mask = (tgt_input == pad_id)
        
        optimizer.zero_grad()
        
        logits = model(
            batch_graph=batch_graph,
            target_ids=tgt_input,
            target_mask=tgt_mask,
            target_padding_mask=tgt_padding_mask
        )
        
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step() # Mise à jour du LR à chaque batch !
        
        total_loss += loss.item()
        
        # Affichage du LR actuel
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})
        
    return total_loss / len(dataloader)

def main():
    print("--- DÉMARRAGE TRAINING V2 (SCALED UP) ---")
    
    if not os.path.exists(TOKENIZER_FILE): return
    
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    pad_id = tokenizer.token_to_id("[PAD]") or 0
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulaire: {vocab_size} | Model Dim: {D_MODEL} | Layers: {N_LAYERS}")

    dataset = MolecularDataset(TRAIN_GRAPHS, TOKENIZER_FILE, NODE_VOCAB, EDGE_VOCAB)
    train_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=2)

    # Modèle V2
    graph_enc = DeepGINEEncoder(NODE_VOCAB, EDGE_VOCAB, GRAPH_DIM, dropout=DROPOUT).to(DEVICE)
    model = Graph2TextTransformer(
        graph_encoder=graph_enc,
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=N_LAYERS,
        dropout=DROPOUT,
        graph_dim=GRAPH_DIM
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.01) # Weight decay aide à généraliser
    
    # SCHEDULER : OneCycleLR
    # Monte le LR puis le descend. Excellent pour la convergence.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LR_MAX, 
        steps_per_epoch=len(train_dl), 
        epochs=EPOCHS,
        pct_start=0.1 # 10% du temps pour chauffer (Warmup)
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        loss = train_one_epoch(model, train_dl, optimizer, criterion, scheduler, DEVICE, pad_id)
        print(f"Loss Moyenne: {loss:.4f}")
        torch.save(model.state_dict(), f"model_v2_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()