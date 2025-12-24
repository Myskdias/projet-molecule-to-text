import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import to_dense_batch

from transformers import T5ForConditionalGeneration, T5EncoderModel
from peft import LoraConfig, get_peft_model


# ==========================================
# 1) Embedders for categorical node/edge feats
# ==========================================
class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim: int, list_of_vocab_sizes):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(v, hidden_dim) for v in list_of_vocab_sizes])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: [N, F] categorical ints
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(x_cat[:, i])
        return out


class BondEncoder(nn.Module):
    def __init__(self, hidden_dim: int, list_of_vocab_sizes):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(v, hidden_dim) for v in list_of_vocab_sizes])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, edge_attr_cat: torch.Tensor) -> torch.Tensor:
        # edge_attr_cat: [E, Fe] categorical ints
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(edge_attr_cat[:, i])
        return out


# ==========================================
# 2) MPNN layer using edge embeddings
# ==========================================
class MPNNLayer(MessagePassing):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__(aggr="add")
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        # edge_attr already embedded: [E, edge_dim]
        m = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        out = self.upd_mlp(torch.cat([x, m], dim=-1))
        out = self.norm(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # residual
        return out + x

    def message(self, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))


# ==========================================
# 3) Graph encoder MPNN
# ==========================================
class MPNNEncoder(nn.Module):
    """
    Returns:
      graph_emb: [B, hidden_dim]
      node_emb:  [N_total, hidden_dim]
    """
    def __init__(self, num_node_vocab, num_edge_vocab, hidden_dim=300, num_layers=5, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_encoder = AtomEncoder(hidden_dim, num_node_vocab)
        self.bond_encoder = BondEncoder(hidden_dim, num_edge_vocab)
        self.layers = nn.ModuleList([MPNNLayer(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)])

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.atom_encoder(x)                        # [N, H]
        e = self.bond_encoder(edge_attr)                # [E, H]
        for layer in self.layers:
            x = layer(x, edge_index, e)

        graph_emb = global_add_pool(x, batch)           # [B, H]
        return graph_emb, x


# ==========================================
# 4) Text encoder for retrieval = frozen T5 encoder
# ==========================================
class FrozenT5TextEncoder(nn.Module):
    """
    Encodes tokenized text to a single embedding using T5 encoder (frozen by default).
    """
    def __init__(self, model_name: str = "t5-base", freeze: bool = True):
        super().__init__()
        self.enc = T5EncoderModel.from_pretrained(model_name)
        if freeze:
            for p in self.enc.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: [B, L, d_model]
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()  # [B,L,1]
        pooled = (out * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
        return pooled  # [B, d_model]


# ==========================================
# 5) Stage-1: CLIP-like aligner (graph->shared, text->shared)
# ==========================================
class GraphTextCLIP(nn.Module):
    def __init__(self, graph_encoder: nn.Module, text_encoder: nn.Module,
                 graph_dim: int, text_dim: int, shared_dim: int = 256):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        self.graph_proj = nn.Linear(graph_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_graph(self, batch_graph) -> torch.Tensor:
        g, _ = self.graph_encoder(batch_graph)      # [B, graph_dim]
        g = self.graph_proj(g)                      # [B, shared]
        return F.normalize(g, dim=1)

    def encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        t = self.text_encoder(input_ids, attention_mask)  # [B, text_dim]
        t = self.text_proj(t)                             # [B, shared]
        return F.normalize(t, dim=1)

    def forward(self, batch_graph, input_ids, attention_mask):
        return self.encode_graph(batch_graph), self.encode_text(input_ids, attention_mask)


# ==========================================
# 6) Stage-2: Graph-conditioned T5 with soft prompt + LoRA
# ==========================================
class GraphSoftPromptT5(nn.Module):
    def __init__(
        self,
        model_name: str = "t5-base",
        graph_emb_dim: int = 300,
        prompt_len: int = 8,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules=("q", "v"),
    ):
        super().__init__()
        self.prompt_len = prompt_len

        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

        # LoRA on attention projections
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            target_modules=list(target_modules),
        )
        self.t5 = get_peft_model(self.t5, lora_cfg)

        d_model = self.t5.config.d_model

        # soft prompt MLP: graph_emb -> (prompt_len * d_model)
        self.softprompt = nn.Sequential(
            nn.Linear(graph_emb_dim, 2 * graph_emb_dim),
            nn.ReLU(),
            nn.Linear(2 * graph_emb_dim, prompt_len * d_model),
        )

    def _build_inputs_embeds(self, graph_emb: torch.Tensor, input_ids: torch.Tensor):
        """
        graph_emb: [B, graph_emb_dim]
        input_ids: [B, L]
        returns inputs_embeds: [B, prompt_len + L, d_model]
        """
        B = graph_emb.size(0)
        d_model = self.t5.config.d_model

        sp = self.softprompt(graph_emb).view(B, self.prompt_len, d_model)  # [B,k,d]
        tok_emb = self.t5.get_input_embeddings()(input_ids)               # [B,L,d]
        return torch.cat([sp, tok_emb], dim=1)

    def forward(self, graph_emb, input_ids, attention_mask=None, labels=None):
        inputs_embeds = self._build_inputs_embeds(graph_emb, input_ids)

        if attention_mask is not None:
            B = attention_mask.size(0)
            sp_mask = torch.ones(B, self.prompt_len, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([sp_mask, attention_mask], dim=1)

        return self.t5(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, graph_emb, input_ids, attention_mask=None, **gen_kwargs):
        inputs_embeds = self._build_inputs_embeds(graph_emb, input_ids)
        if attention_mask is not None:
            B = attention_mask.size(0)
            sp_mask = torch.ones(B, self.prompt_len, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([sp_mask, attention_mask], dim=1)
        return self.t5.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kwargs)
