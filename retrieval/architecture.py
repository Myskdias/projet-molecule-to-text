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
    def __init__(self, num_node_vocab, num_edge_vocab, hidden_dim=300, num_layers=5, dropout=0.1):#old drop out 0.5
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.atom_encoder = AtomEncoder(hidden_dim, num_node_vocab)
        self.bond_encoder = BondEncoder(hidden_dim, num_edge_vocab)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.readout_proj = nn.Linear(
            hidden_dim * num_layers,
            hidden_dim
        )
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
        xs = [] #Extremlmy important, necessary to do pooling
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index, edge_attr=edge_emb)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity # Skip connection
            xs.append(x)
        #graph_emb = global_add_pool(x, batch) #without pooling
        pooled = [global_add_pool(h, batch) for h in xs]
        graph_emb = torch.cat(pooled, dim=1)
        graph_emb = self.readout_proj(graph_emb)
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
    
    @torch.no_grad()
    def encode_graph(self, batch_graph):
        graph_feat, _ = self.graph_encoder(batch_graph)
        graph_emb = self.graph_proj(graph_feat)
        return F.normalize(graph_emb, dim=1)
    