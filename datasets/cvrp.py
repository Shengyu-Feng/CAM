"""TSP (Traveling Salesman Problem) Graph Dataset"""

import numpy as np
import torch 
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import dense_to_sparse, remove_self_loops

class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_file, dtype, num_samples=-1, args=None):
    self.data_file = data_file
    self.dtype = dtype

    self.locs = []
    self.demands = []
    self.tours = []
    for i, line in enumerate(open( self.data_file, "r").readlines()[:num_samples]):
        line = line.strip().split(" ")
        depot_index = int(line.index('depots'))
        customer_index = int(line.index('points'))
        capacity_index = int(line.index('capacity'))
        demand_index = int(line.index('demands'))
        tour_index = int(line.index('output'))

        depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
        customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, demand_index, 2)]

        loc = np.array(depot + customer)
        self.locs.append(loc)


        demand = [0] + [float(line[idx]) for idx in range(demand_index + 1, capacity_index)]
        capacity = float(line[capacity_index+1])

        demand = np.array(demand)/capacity
        self.demands.append(demand)

        tour = np.array([int(line[idx]) for idx in range(tour_index + 1, len(line))])
        
        self.tours.append(tour)


  def __len__(self):
    return len(self.locs)

  def __getitem__(self, idx):
    points = torch.tensor(self.locs[idx], dtype=self.dtype)
    demand = torch.tensor(self.demands[idx], dtype=self.dtype)
    n = points.shape[0]

    is_depot = torch.zeros(n, dtype=torch.long, device=points.device)
    is_depot[0] = 1

    tour = torch.as_tensor(self.tours[idx], dtype=torch.long, device=points.device)

    dist_mat = torch.cdist(points, points)   # (n, n)

    row = torch.arange(n, device=points.device).repeat_interleave(n)
    col = torch.arange(n, device=points.device).repeat(n)
    edge_index = torch.stack([row, col], dim=0)   # (2, n*n)

    tour_mask = torch.zeros((n, n), dtype=self.dtype, device=points.device)
    tour_mask[tour[:-1], tour[1:]] = 1

    # if undirected
    tour_mask[tour[1:], tour[:-1]] = 1

    dist_edge = dist_mat.reshape(-1)         # (n*n,)
    tour_edge = tour_mask.reshape(-1)        # (n*n,)

    graph_data = GraphData(
        x=points,
        edge_index=edge_index,
        edge_attr=tour_edge.unsqueeze(-1),
        demand=demand.unsqueeze(-1),
        is_depot=is_depot,
        distance=dist_edge.unsqueeze(-1),
        idx = idx,
    )
    return graph_data