import os
import numpy as np
import torch
import pickle
import glob
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, is_undirected, to_undirected
import time

class Dataset(Dataset):
    def __init__(self, data_dir, dtype=torch.float32, num_samples=-1, args=None):
        super(Dataset, self).__init__()
        self.dtype = dtype
             
        self.files = sorted(glob.glob(data_dir + '/*gpickle'))
        self.load_data = self.load_gpickle
    
        if os.path.exists(os.path.join(data_dir, 'anno')):
            self.data_label_path = os.path.join(data_dir, 'anno')
        else:
            self.data_label_path = None

        if num_samples>0:
            self.files = self.files[:num_samples]
    
    def __len__(self):
        return len(self.files)
    
    def load_dimacs(self, idx):
        G = nx.Graph()
        
        with open(self.files[idx], 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Parse the line
            parts = line.split()
            
            # Problem line (p edge NODES EDGES)
            if parts[0] == 'p' and parts[1] == 'edge':
                num_nodes = int(parts[2])
                # Pre-add all nodes
                G.add_nodes_from(range(1, num_nodes + 1))
            
            # Edge line (e NODE1 NODE2)
            elif parts[0] == 'e':
                node1 = int(parts[1])
                node2 = int(parts[2])
                G.add_edge(node1, node2)

        data = from_networkx(G)
        if not is_undirected(data.edge_index):
            data.edge_index = to_undirected(data.edge_index)
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_weight = torch.ones_like(data.edge_index[0], dtype=self.dtype)/2
        data.x = torch.ones(data.num_nodes,1, dtype=self.dtype)
        data.b = -torch.ones_like(data.x)
        data.idx = idx
        return data

    def load_gpickle(self, idx):
        if self.data_label_path is not None:
            file_name = os.path.basename(self.files[idx])
            anno_path = os.path.join(self.data_label_path, file_name.replace('.gpickle', '_unweighted.result'))
            if os.path.exists(anno_path):
                solution = open(anno_path).readlines()
                solution = [int(x.strip()) for x in solution]
            else:
                solution = None
        else:
            solution = None
            
        G = nx.read_gpickle(self.files[idx])
        data = from_networkx(G)
        if not is_undirected(data.edge_index):
            data.edge_index = to_undirected(data.edge_index)
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_weight = torch.ones_like(data.edge_index[0], dtype=self.dtype)/2
        if solution is None:
            data.x = torch.ones(data.num_nodes,1, dtype=self.dtype)
        else:
            data.x = torch.tensor(solution).reshape(data.num_nodes,1).to(self.dtype)

        data.b = -torch.ones_like(data.x)
        data.idx = idx
        return data
        
    def __getitem__(self, idx):
        return self.load_data(idx)
    