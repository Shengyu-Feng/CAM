import torch
from torch import nn
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn import GraphNorm
from models.nn_utils import (
    ScalarEmbeddingSine,
    timestep_embedding
)
import torch.nn.functional as F

def rw_normalize(edge_index, nodes):
    """D^{-1/2} A normalization — aggregates neighbor messages scaled by 1/sqrt(deg(receiver))"""
    num_nodes = nodes.shape[0]
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)

    _, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[col] * edge_weight
    return torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size((num_nodes, num_nodes))).to_sparse_csr()

class ReLUMLP(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, out_channels):
        super(ReLUMLP, self).__init__()

        assert num_layers >= 2   
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.linears.append(nn.Linear(in_channels, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        for i in range(num_layers-2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.linears.append(nn.Linear(hidden_dim, out_channels))
        self.norms.append(nn.LayerNorm(out_channels))
        self.activation = nn.ReLU()

    def reset_parameters(self):
        for linear in self.linears:
            nn.init.kaiming_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

        for norm in self.norms:
            norm.reset_parameters()
        
    def forward(self, x):
        
        for linear, norm in zip(self.linears, self.norms):
            x = norm(self.activation(linear(x)))

        return x
            
class LinearMessagePassing(nn.Module):
    def __init__(self, hidden_dim, graph_norm=True):
        super(LinearMessagePassing, self).__init__()
        self.node_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm = nn.LayerNorm(hidden_dim)
        self.node_mlp = ReLUMLP(2*hidden_dim, hidden_dim, 2, hidden_dim)
        self.graph_norm = GraphNorm(hidden_dim) if graph_norm else None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.node_proj.weight)
        nn.init.kaiming_uniform_(self.message_proj.weight)

        self.norm.reset_parameters()
        self.node_mlp.reset_parameters()

    def forward(self, X, adj, batch=None):
        message = self.message_proj(X)
        nei = torch.spmm(adj, message.flatten(1))
        nei = nei.reshape(message.shape)
        if self.graph_norm is not None:
            nei = self.graph_norm(nei, batch)
        nodes_out = self.node_mlp(torch.cat([X, nei], -1))
        X_new = self.norm(self.node_proj(X) + nodes_out)
        return X_new


class EncodeProcessDecode(nn.Module):
    def __init__(self, hidden_dim, n_layers, out_channels, time_scale=1000, graph_norm=True, **kwargs):
        super(EncodeProcessDecode, self).__init__()
        self.time_embed_dim = hidden_dim // 2
        self.node_encoder = ReLUMLP(1, hidden_dim, 2, hidden_dim)
        self.convs = nn.ModuleList()
        self.time_scale = time_scale

        for _ in range(n_layers):
            self.convs.append(LinearMessagePassing(hidden_dim, graph_norm=graph_norm))

        self.node_decoder = ReLUMLP(hidden_dim, hidden_dim, 2, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.node_decoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.head.reset_parameters()

    def forward(self, graph, solution, time=None):
        adj = rw_normalize(graph.edge_index, solution)
        batch = graph.batch if hasattr(graph, 'batch') else None

        X = self.node_encoder(solution)

        for conv in self.convs:
            X = conv(X, adj, batch)
        X = self.node_decoder(X)
        return self.head(X)

class EncodeProcessDecodeTime1(nn.Module):
    def __init__(self, hidden_dim, n_layers, out_channels, time_scale=1000, graph_norm=True, **kwargs):
        super(EncodeProcessDecodeTime1, self).__init__()
        self.time_embed_dim = hidden_dim // 2
        self.node_encoder = ReLUMLP(2 + self.time_embed_dim, hidden_dim, 2, hidden_dim)
        self.convs = nn.ModuleList()
        self.time_scale = time_scale

        for _ in range(n_layers):
            self.convs.append(LinearMessagePassing(hidden_dim, graph_norm=graph_norm))

        self.node_decoder = ReLUMLP(hidden_dim, hidden_dim, 2, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.node_decoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.head.reset_parameters()

    def forward(self, graph, solution, time=None):
        adj = rw_normalize(graph.edge_index, solution)
        batch = graph.batch if hasattr(graph, 'batch') else None

        time_emb = timestep_embedding(time*self.time_scale, self.time_embed_dim)

        X_one_hot = F.one_hot(solution.squeeze(-1).long(), num_classes=2)
        X = torch.cat([X_one_hot, time_emb], dim=-1)
        X = self.node_encoder(X)

        for conv in self.convs:
            X = conv(X, adj, batch)
        X = self.node_decoder(X)
        return self.head(X)
    
class EncodeProcessDecodeTime2(nn.Module):
    def __init__(self, hidden_dim, n_layers, out_channels, time_scale=1000, graph_norm=True, **kwargs):
        super(EncodeProcessDecodeTime2, self).__init__()
        self.time_scale = time_scale

        self.hidden_dim = hidden_dim
        time_embed_dim = hidden_dim // 2

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.node_encoder = ReLUMLP(1, hidden_dim, 2, hidden_dim)

        self.convs = nn.ModuleList()
        self.time_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(LinearMessagePassing(hidden_dim, graph_norm=graph_norm))
            self.time_layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(time_embed_dim, hidden_dim),
                )
            )

        self.node_decoder = ReLUMLP(hidden_dim, hidden_dim, 2, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.node_decoder.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

        for layer in self.time_layers:
            for m in layer:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

        self.head.reset_parameters()

    def forward(self, graph, solution, time=None):
        adj = rw_normalize(graph.edge_index, solution)
        batch = graph.batch if hasattr(graph, 'batch') and graph.batch is not None else None

        X = self.node_encoder(solution)

        if time is not None:
            if time.dim() == 0:
                time = time[None]

            time_emb = self.time_embed(
                timestep_embedding(time*self.time_scale, self.hidden_dim)
            )

            if batch is not None:
                time_emb_node = time_emb[batch]
            else:
                time_emb_node = time_emb.expand(X.size(0), -1)
        else:
            time_emb_node = None

        for conv, time_layer in zip(self.convs, self.time_layers):
            if time_emb_node is not None:
                X = X + time_layer(time_emb_node)

            X = conv(X, adj, batch)

        X = self.node_decoder(X)
        return self.head(X)