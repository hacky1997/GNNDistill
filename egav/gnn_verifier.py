from typing import Optional

import torch
from torch import nn


class MLPVerifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HeteroGATVerifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        try:
            from torch_geometric.nn import HeteroConv, GATConv, Linear
        except Exception as exc:
            raise ImportError("torch-geometric is required for HeteroGATVerifier") from exc

        self.lin_dict = nn.ModuleDict({
            "QENT": Linear(in_dim, hidden_dim),
            "CENT": Linear(in_dim, hidden_dim),
            "SPAN": Linear(in_dim, hidden_dim),
            "SENT": Linear(in_dim, hidden_dim),
        })
        conv1 = HeteroConv(
            {
                ("QENT", "qent_span", "SPAN"): GATConv((-1, -1), hidden_dim, add_self_loops=False),
                ("QENT", "qent_sent", "SENT"): GATConv((-1, -1), hidden_dim, add_self_loops=False),
                ("CENT", "cent_sent", "SENT"): GATConv((-1, -1), hidden_dim, add_self_loops=False),
                ("SPAN", "span_cent", "CENT"): GATConv((-1, -1), hidden_dim, add_self_loops=False),
                ("CENT", "cent_cooccur", "CENT"): GATConv((-1, -1), hidden_dim, add_self_loops=False),
            },
            aggr="sum",
        )
        conv2 = HeteroConv(
            {
                ("QENT", "qent_span", "SPAN"): GATConv((-1, -1), hidden_dim, add_self_loops=False),
                ("QENT", "qent_sent", "SENT"): GATConv((-1, -1), hidden_dim, add_self_loops=False),
                ("CENT", "cent_sent", "SENT"): GATConv((-1, -1), hidden_dim, add_self_loops=False),
                ("SPAN", "span_cent", "CENT"): GATConv((-1, -1), hidden_dim, add_self_loops=False),
                ("CENT", "cent_cooccur", "CENT"): GATConv((-1, -1), hidden_dim, add_self_loops=False),
            },
            aggr="sum",
        )
        self.convs = nn.ModuleList([conv1, conv2])
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: self.lin_dict[k](v) for k, v in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        span_out = self.out(x_dict["SPAN"]).squeeze(-1)
        return span_out
