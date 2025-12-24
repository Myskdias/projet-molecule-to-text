import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

class MPNNConv(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add')  # agrégation par somme des messages
        # MLP pour calculer le message à partir du voisin et de l'arête
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # MLP pour l'update du nœud (combine h_i actuel et message agrégé)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        # x: [N, node_dim], edge_index: [2, E], edge_attr: [E, edge_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        # Pour chaque arête j->i, calcule le message à partir du voisin j et de l'attribut de l'arête (j,i)
        msg_input = torch.cat([x_j, edge_attr], dim=1)  # concatène h_j et e_{j,i}
        return self.msg_mlp(msg_input)
    
    def update(self, aggr_out, x):
        # Combine le message agrégé avec la feature initiale du nœud i
        upd_input = torch.cat([x, aggr_out], dim=1)
        return self.update_mlp(upd_input)

class GraphEncoderMPNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Projection initiale des features nœud vers hidden_dim
        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)
        # Empile num_layers couches MPNNConv
        self.convs = nn.ModuleList([
            MPNNConv(hidden_dim, edge_feat_dim, hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(self, data):
        # data.x: features des nœuds [N_total, node_feat_dim]
        # data.edge_index: indices des arêtes [2, E_total]
        # data.edge_attr: features des arêtes [E_total, edge_feat_dim]
        # data.batch: identifiant du graphe pour chaque nœud [N_total]
        x = F.relu(self.node_embed(data.x))
        for conv in self.convs:
            # Message passing itératif
            x = conv(x, data.edge_index, data.edge_attr)
        # Agrégation globale (moyenne) pour obtenir l'embedding de chaque graphe du batch
        graph_emb = global_mean_pool(x, data.batch)  # shape [batch_size, hidden_dim]
        return graph_emb
