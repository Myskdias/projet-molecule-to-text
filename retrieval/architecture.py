import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn import GINConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj

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

def batched_dense_adj_to_edge_index(adj: torch.Tensor, thresh: float = 1e-8):
    """
    adj: [B, N, N] dense adjacency (float)
    returns edge_index: [2, E] for a batched graph with offset nodes
    """
    B, N, _ = adj.shape
    device = adj.device

    rows = []
    cols = []
    for b in range(B):
        a = adj[b]
        idx = (a > thresh).nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        r = idx[:, 0] + b * N
        c = idx[:, 1] + b * N
        rows.append(r)
        cols.append(c)

    if len(rows) == 0:
        # pas d'arêtes -> edge_index vide
        return torch.empty((2, 0), dtype=torch.long, device=device)

    row = torch.cat(rows, dim=0)
    col = torch.cat(cols, dim=0)
    return torch.stack([row, col], dim=0)

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
        # ============================
        # DiffPool (1 niveau)
        # ============================
        self.num_clusters = 50  # à tuner (ex: 30, 50, 80)
        self.assign_lin = nn.Linear(hidden_dim, self.num_clusters)
        self.embed_lin = nn.Linear(hidden_dim, hidden_dim)

        # Post-pooling GINE
        self.post_convs = nn.ModuleList()
        self.post_bns = nn.ModuleList()

        post_layers = 2

        for _ in range(post_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.post_convs.append(GINConv(mlp, train_eps=True))
            self.post_bns.append(nn.BatchNorm1d(hidden_dim))

        # Fusion multi-scale (pré + post pooling)
        self.final_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        

    def forward(self, data):
        # ============================
        # 1) Multi-layer GINE (inchangé)
        # ============================
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.atom_encoder(x)
        edge_emb = self.bond_encoder(edge_attr)

        xs = []
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index, edge_attr=edge_emb)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity
            xs.append(x)

        # ----------------------------
        # Readout pré-pooling (niveau atomique)
        # ----------------------------
        h_pre = global_add_pool(x, batch)  # [B, H]

        # ----------------------------
        # DiffPool (dense)
        # ----------------------------
        x_dense, mask = to_dense_batch(x, batch)          # [B, N, H], [B, N]
        adj_dense = to_dense_adj(edge_index, batch=batch) # [B, N, N]

        s = self.assign_lin(x_dense)  # [B, N, K]
        z = self.embed_lin(x_dense)   # [B, N, H]

        x_pool, adj_pool, lp_loss, ent_loss = dense_diff_pool(z, adj_dense, s, mask)

        # ----------------------------
        # Post-pooling GIN (sur super-noeuds)
        # ----------------------------
        B, Np, H = x_pool.shape

        edge_index_pool = batched_dense_adj_to_edge_index(adj_pool)
        x_post = x_pool.reshape(B * Np, H)

        batch_post = torch.arange(B, device=x.device).repeat_interleave(Np)

        for conv, bn in zip(self.post_convs, self.post_bns):
            identity = x_post
            x_post = conv(x_post, edge_index_pool)
            x_post = bn(x_post)
            x_post = F.relu(x_post)
            x_post = F.dropout(x_post, p=self.dropout, training=self.training)
            x_post = x_post + identity

        h_post = global_add_pool(x_post, batch_post)  # [B, H]

        # ----------------------------
        # Multi-scale fusion
        # ----------------------------
        graph_emb = torch.cat([h_pre, h_post], dim=1)     # [B, 2H]
        graph_emb = self.final_proj(graph_emb)            # [B, H]

        return graph_emb, lp_loss, ent_loss

# ==========================================
# 3. MODULE ÉTAPE 1 : CLIP (ALIGNEMENT)
# ==========================================

class GraphTextCLIP(nn.Module):
    """
    CLIP-like model for Graph ↔ Text retrieval
    Text side = MiniLM only
    """
    def __init__(self, graph_encoder, graph_dim, text_encoder, shared_dim=384):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder  # MiniLM
        self.graph_proj = nn.Linear(graph_dim, shared_dim)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))  # exp(0)=1

    def forward(self, batch_graph, batch_texts):
        # Graph encoding
        graph_feat, lp_loss, ent_loss = self.graph_encoder(batch_graph)
        self.lp_loss = lp_loss
        self.ent_loss = ent_loss
        graph_emb = F.normalize(self.graph_proj(graph_feat), dim=1)

        # Text encoding (MiniLM)
        text_emb = self.text_encoder(batch_texts)  # already normalized

        return graph_emb, text_emb

    @torch.no_grad()
    def encode_graph(self, batch_graph):
        out = self.graph_encoder(batch_graph)

        # Compatibilité: graph_encoder peut retourner 2 ou 3 valeurs
        if isinstance(out, tuple):
            graph_feat = out[0]
        else:
            graph_feat = out

        graph_feat = self.graph_proj(graph_feat)
        graph_feat = F.normalize(graph_feat, dim=-1)
        return graph_feat