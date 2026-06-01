import torch
from torch import nn
import torch.nn.functional as F

import functools
from models.nn_utils import (
    zero_module,
    timestep_embedding,
    PositionEmbeddingSine,
    ScalarEmbeddingSine,
)
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max
import torch.utils.checkpoint as activation_checkpoint

class AnGNNLayer(nn.Module):
    """Configurable GNN Layer
    Implements the Gated Graph ConvNet layer:
      h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
      sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
      e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
      where Aggr. is an aggregation function: sum/mean/max.
    References:
      - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
      - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
    """
    def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True, node_only=False, edge_only=False):
        """
        Args:
            hidden_dim: Hidden dimension size (int)
            aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
            norm: Feature normalization scheme ("layer"/"batch"/None)
            learn_norm: Whether the normalizer has learnable affine parameters (True/False)
            track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
            gated: Whether to use edge gating (True/False)
        """
        super(AnGNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        self.node_only = node_only
        self.edge_only = edge_only
          
        assert self.gated, "Use gating with GCN, pass the `--gated` flag"
        
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        
        if not self.node_only:
            self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
            }.get(self.norm, None)
          
        if not self.edge_only:
            self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.norm_h = {
                "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
                "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
            }.get(self.norm, None)


    def forward(self, h, e, graph, mode="residual", edge_index=None):
        """
        Args:
            h: Input node features (V x B x H)
            e: Input edge features (E x B x H)
            graph: torch_sparse.SparseTensor
            mode: str
            edge_index: Edge indices (2 x E)
        Returns:
            Updated node and edge features
        """
        h_in = h
        e_in = e
        
        # Linear transformations for edge update and gating
        Ah = self.A(h)  # V x B x H, source
        Bh = self.B(h)  # V x B x H, target
        Ce = self.C(e)
        
        De = Ah[edge_index[1]] + Bh[edge_index[0]] + Ce  # E x B x H
        
        if not self.node_only:
            e = self.norm_e(De.flatten(0,-2)).view(De.shape) if self.norm_e else De
            e = F.relu(e)
            if mode == "residual":
                e = e + e_in
          
        if not self.edge_only:
            # Linear transformations for node update
            Uh = self.U(h)  # V x B x H
            
            Vh = self.V(h[edge_index[1]])  # E x B x H
            gates = torch.sigmoid(De)  # B x V x V x H / E x H
            
            # Update node features
            h = Uh + self.aggregate(Vh, graph, gates, edge_index=edge_index)  # B x V x H
            
            h = self.norm_h(h.flatten(0,-2)).view(h.shape) if self.norm_h else h
            
            # Apply non-linearity
            h = F.relu(h)
            
            # Make residual connection
            if mode == "residual":
                h = h_in + h
        
        return h, e
    
    def aggregate(self, Vh, graph, gates, mode=None, edge_index=None, sparse=False):
        """
        Args:
            Vh: Neighborhood features (E x B x H)
            graph: torch_sparse.SparseTensor (E edges for V x V adjacency matrix)
            gates: Edge gates (E x B x H)
            mode: str
            edge_index: Edge indices (2 x E)
            sparse: Whether to use sparse tensors (True/False)
        Returns:
            Aggregated neighborhood features (V x B x H)
        """
        # Perform feature-wise gating mechanism
        
        Vh = gates * Vh  # E x B x H
        
        # Aggregate neighborhood features
        sparseVh = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=Vh,
            sparse_sizes=(graph.size(0), graph.size(1))
        )
        
        
        if (mode or self.aggregation) == "mean":
            return sparse_mean(sparseVh, dim=1)
        
        elif (mode or self.aggregation) == "max":
            return sparse_max(sparseVh, dim=1)
        else:
            return sparse_sum(sparseVh, dim=1)

def run_sparse_layer(layer, time_layer, out_layer, adj_matrix, edge_index, add_time_on_edge=True):
    def custom_forward(*inputs):
        x_in = inputs[0]
        e_in = inputs[1]
        time_emb = inputs[2]
        x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index)
        if add_time_on_edge:
            e = e + time_layer(time_emb)
            e = e_in + out_layer(e.flatten(0,-2)).reshape(e.shape)
            x = x + x_in
        else:
            x = x + time_layer(time_emb)
            x = x_in + out_layer(x.flatten(0,-2)).reshape(x.shape)
            e = e + e_in
        return x, e
    return custom_forward

class AnGNN(nn.Module):
    """Anisotropic GNN
    """
    
    def __init__(self, n_layers, hidden_dim, out_channels=1, aggregation="sum", norm="layer",
               learn_norm=True, track_norm=False, gated=True, use_activation_checkpoint=False, problem='tsp', time_scale=1000, *args, **kwargs):
        super(AnGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        time_embed_dim = hidden_dim // 2
        self.node_embed = nn.Linear(hidden_dim, hidden_dim)
        self.problem = problem
        self.time_scale = time_scale

        if self.problem == 'tsp':
            self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True) # coordinate
            self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False) # solution
            self.edge_embed = nn.Linear(hidden_dim, hidden_dim)
        elif self.problem == 'cvrp':
            self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True) # coordinate
            
            self.edge_pos_embed = nn.Sequential(
                ScalarEmbeddingSine(hidden_dim, normalize=False),
                nn.Linear(hidden_dim, hidden_dim)
            ) # solution
            self.demand_emb = ScalarEmbeddingSine(hidden_dim) # demand
            self.is_depot_embed = nn.Embedding(2, hidden_dim) # depot
            self.distance_embed = nn.Sequential(
                ScalarEmbeddingSine(hidden_dim, normalize=False),
                nn.Linear(hidden_dim, hidden_dim)
            ) # distance
            self.edge_embed = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.pos_embed =  ScalarEmbeddingSine(hidden_dim, normalize=False)
        
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.out = nn.Linear(hidden_dim, out_channels)
        
        self.layers = nn.ModuleList([
            AnGNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
            for _ in range(n_layers-1)
          ]
        )
        if self.problem in ['tsp', 'cvrp']:
            self.layers.append(AnGNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated, edge_only=True))
        else:
            self.layers.append(AnGNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated, node_only=True))
          
        self.time_embed_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(time_embed_dim,hidden_dim),
            ) for _ in range(n_layers)
        ])
        
        
        self.per_layer_out = nn.ModuleList(
          [
            nn.Sequential(
              nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
              nn.SiLU(),
              zero_module(
                  nn.Linear(hidden_dim, hidden_dim)
              ),
            ) for _ in range(n_layers)
          ]
        )
        self.use_activation_checkpoint = use_activation_checkpoint

    def sparse_forward(self, graph, solution, time):
        """
        Args:
            graph: torch geometric graphs
            timesteps: Input edge timestep features (E x B)
        Returns:
            Updated solution
        """
        # Embed edge features
        if self.problem == 'tsp':
            pos = graph.x
            if pos.ndim < solution.ndim:
                pos = pos.unsqueeze(1)

            x = self.node_embed(self.pos_embed(pos))
            e = self.edge_embed(self.edge_pos_embed(solution))
        else:
            pos = graph.x
            demand = graph.demand
            distance = graph.distance
            is_depot = graph.is_depot
            if pos.ndim < solution.ndim:
                pos = pos.unsqueeze(1)
                demand = demand.unsqueeze(1)
                is_depot = is_depot.unsqueeze(1)
                distance = distance.unsqueeze(1)

            pos_embed = self.pos_embed(pos)
            demand_embed = self.demand_emb(demand)
            is_depot_embed = self.is_depot_embed(is_depot)
            x = self.node_embed(pos_embed + demand_embed + is_depot_embed) # (B, V, H)

            distance_embed = self.distance_embed(distance)
            label_embed = self.edge_pos_embed(solution)
            e = self.edge_embed(distance_embed + label_embed)

        time_emb = self.time_embed(timestep_embedding(time*self.time_scale, self.hidden_dim))
        
        edge_index = graph.edge_index
        edge_index = edge_index.long()
        
        x, e = self.sparse_encoding(x, e, edge_index, time_emb)
        logits = self.out(e)

        return logits
      
    def sparse_forward_node_feature_only(self, graph, solution, time):
        x = solution
        edge_index = graph.edge_index
        solution_shape = solution.shape
 
        x = self.node_embed(self.pos_embed(x))
        e = torch.zeros((graph.edge_index.shape[1], *x.shape[1:]), dtype=torch.float, device=solution.device)
        
        time_emb = self.time_embed(timestep_embedding(time*self.time_scale, self.hidden_dim))
        
        edge_index = edge_index.long()
        
        x, e = self.sparse_encoding(x, e, edge_index, time_emb)
        logits = self.out(x)

        return logits


    def sparse_encoding(self, x, e, edge_index, time_emb):
        adj_matrix = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=torch.ones_like(edge_index[0].float()),
            sparse_sizes=(x.shape[0], x.shape[0]),
        )
        adj_matrix = adj_matrix.to(x.device)
        
        for layer, out_layer, time_layer in zip(self.layers,  self.per_layer_out, self.time_embed_layers):
            x_in, e_in = x, e
            if self.use_activation_checkpoint and self.training:
                run_sparse_layer_fn = functools.partial(
                    run_sparse_layer,
                    add_time_on_edge=not self.node_feature_only
                )
                
                out = activation_checkpoint.checkpoint(
                    run_sparse_layer_fn(layer, time_layer, out_layer, adj_matrix, edge_index),
                    x_in, e_in, time_emb
                )
                x = out[0]
                e = out[1]
            else:
                x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index)
                
                if self.problem in ['tsp', 'cvrp']:
                    e = e + time_layer(time_emb)
                    e = e_in + out_layer(e.flatten(0,-2)).reshape(e.shape)
                    x = x + x_in
                else:
                    x = x + time_layer(time_emb)
                    x = x_in + out_layer(x.flatten(0,-2)).reshape(x.shape)
                    e = e + e_in
                
        return x, e

    def forward(self, graph, solution, time):
        if self.problem in ['tsp', 'cvrp']:
            return self.sparse_forward(graph, solution, time)
        else:
            return self.sparse_forward_node_feature_only(graph, solution, time)
        