import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.utils import to_dense_batch

# --- POSITIONAL ENCODING ---
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

# --- EMBEDDINGS ---
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

# --- ENCODEUR GRAPHE (Plus profond et robuste) ---
class DeepGINEEncoder(nn.Module):
    def __init__(self, num_node_vocab, num_edge_vocab, hidden_dim=300, num_layers=5, dropout=0.3):
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
                nn.Dropout(dropout), # Ajout Dropout dans le MLP
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
            x = x + identity
        
        return x

# --- DÉCODEUR (Scaled Up) ---
class Graph2TextTransformer(nn.Module):
    def __init__(self, graph_encoder, vocab_size, d_model=512, nhead=8, num_layers=6, dropout=0.2, graph_dim=300):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.d_model = d_model
        
        # Projection Graphe (300) -> Transformer (512)
        self.graph_projector = nn.Sequential(
            nn.Linear(graph_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Embeddings Texte
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer Decoder (Plus gros)
        # dim_feedforward est généralement 4x d_model -> 2048
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True # Astuce : norm_first aide souvent la convergence
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Sortie
        self.output_norm = nn.LayerNorm(d_model) # Stabilisation finale
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, batch_graph, target_ids, target_mask, target_padding_mask):
        # 1. Encodage Graphe
        node_feats = self.graph_encoder(batch_graph)
        dense_feats, node_mask = to_dense_batch(node_feats, batch_graph.batch)
        memory = self.graph_projector(dense_feats)
        memory_padding_mask = ~node_mask
        
        # 2. Encodage Cible
        tgt_emb = self.pos_encoder(self.tgt_embedding(target_ids) * math.sqrt(self.d_model))
        
        # 3. Décodage
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )
        
        output = self.output_norm(output)
        return self.fc_out(output)