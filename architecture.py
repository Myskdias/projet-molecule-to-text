import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.utils import to_dense_batch

# ==========================================
# 1. COMPOSANTS DE BASE (EMBEDDINGS)
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim, list_of_vocab_sizes):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(v, hidden_dim) for v in list_of_vocab_sizes])
        for emb in self.embeddings: nn.init.xavier_uniform_(emb.weight.data)
    def forward(self, x):
        out = 0
        for i, emb in enumerate(self.embeddings): out += emb(x[:, i])
        return out

class BondEncoder(nn.Module):
    def __init__(self, hidden_dim, list_of_vocab_sizes):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(v, hidden_dim) for v in list_of_vocab_sizes])
        for emb in self.embeddings: nn.init.xavier_uniform_(emb.weight.data)
    def forward(self, edge_attr):
        out = 0
        for i, emb in enumerate(self.embeddings): out += emb(edge_attr[:, i])
        return out

# ==========================================
# 2. ENCODEURS (GRAPH & TEXT)
# ==========================================

class DeepGINEEncoder(nn.Module):
    """Encodeur de Graphe (L'Oeil du modèle)"""
    def __init__(self, num_node_vocab, num_edge_vocab, hidden_dim=300, num_layers=5, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.atom_encoder = AtomEncoder(hidden_dim, num_node_vocab)
        self.bond_encoder = BondEncoder(hidden_dim, num_edge_vocab)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.atom_encoder(x)
        edge_emb = self.bond_encoder(edge_attr)

        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index, edge_attr=edge_emb)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity # Skip connection

        graph_emb = global_add_pool(x, batch)
        return graph_emb, x

class TextEncoder(nn.Module):
    """Encodeur de Texte (Utilisé pour CLIP et pour lire l'antisèche RAG)"""
    def __init__(self, vocab_size, d_model, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, src, src_key_padding_mask=None):
        x = self.embedding(src)
        x = self.pos_encoder(x)
        return self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

# ==========================================
# 3. MODULE ÉTAPE 1 : CLIP (ALIGNEMENT)
# ==========================================

class GraphTextCLIP(nn.Module):
    """Modèle pour l'étape 1 : Apprend à aligner Graphe et Texte"""
    def __init__(self, graph_encoder, text_encoder, graph_dim, text_dim, shared_dim=256):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        self.graph_proj = nn.Linear(graph_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052) # ln(100) ~ 4.6

    def forward(self, batch_graph, batch_text, text_mask):
        # Encodage Graphe
        graph_feat, _ = self.graph_encoder(batch_graph)
        graph_emb = self.graph_proj(graph_feat)
        
        # Encodage Texte
        text_seq = self.text_encoder(batch_text, src_key_padding_mask=text_mask)
        # Pooling: On prend la moyenne des tokens non-padded (simplifié ici par mean global)
        text_feat = text_seq.mean(dim=1) 
        text_emb = self.text_proj(text_feat)
        
        # Normalisation L2 (Crucial pour CLIP)
        return F.normalize(graph_emb, dim=1), F.normalize(text_emb, dim=1)

# ==========================================
# 4. MODULE ÉTAPE 2 : GÉNÉRATEUR RAG
# ==========================================

class MolecularCaptionModel(nn.Module):
    """Modèle pour l'étape 2 : Génération assistée (RAG)"""
    def __init__(self, graph_encoder, text_encoder_rag, vocab_size, d_model=256, graph_dim=300):
        super().__init__()
        self.graph_encoder = graph_encoder
        # Note : On peut réutiliser le text_encoder de CLIP comme "lecteur d'antisèche"
        self.retrieved_text_encoder = text_encoder_rag 
        
        self.graph_projector = nn.Linear(graph_dim, d_model)
        
        # Décodeur (L'écrivain)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, batch_graph, retrieved_ids, retrieved_mask, target_ids, target_mask, target_padding_mask):
        # A. Traitement Graphe
        _, graph_node_emb = self.graph_encoder(batch_graph)
        graph_feats, graph_mask = to_dense_batch(graph_node_emb, batch_graph.batch)
        graph_feats = self.graph_projector(graph_feats)
        
        # B. Traitement Antisèche (RAG)
        retrieved_feats = self.retrieved_text_encoder(retrieved_ids, src_key_padding_mask=retrieved_mask)
        
        # C. Fusion (Concaténation)
        memory = torch.cat([graph_feats, retrieved_feats], dim=1)
        graph_padding_mask = ~graph_mask
        memory_key_padding_mask = torch.cat([graph_padding_mask, retrieved_mask], dim=1)
        
        # D. Décodage
        tgt_emb = self.pos_encoder(self.tgt_embedding(target_ids))
        output = self.transformer_decoder(
            tgt_emb, memory, 
            tgt_mask=target_mask, 
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.fc_out(output)