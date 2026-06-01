import os
import numpy as np
import torch
import pickle
import glob
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, is_undirected, to_undirected, degree
from pathlib import Path

class Dataset(Dataset):
    def __init__(self, data_dir, dtype=torch.float32, num_samples=-1, args=None):
        super(Dataset, self).__init__()
        self.dtype = dtype
             
        self.files = sorted(glob.glob(data_dir + '/*gpickle'))
    
        if os.path.exists(os.path.join(data_dir, 'anno')):
            self.data_label_path = os.path.join(data_dir, 'anno')
        else:
            self.data_label_path = None
        
        if num_samples>0:
            self.files = self.files[:num_samples]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        G = nx.read_gpickle(self.files[idx])
        data = from_networkx(G)
        if not is_undirected(data.edge_index):
            data.edge_index = to_undirected(data.edge_index)
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_weight = torch.ones_like(data.edge_index[0], dtype=self.dtype)
        degrees = degree(data.edge_index[0], data.num_nodes).unsqueeze(1)
        data.b = -degrees
        data.idx = idx

        if self.data_label_path is not None:
            stem = Path(self.files[idx]).stem
            x = np.load(os.path.join(self.data_label_path, stem + ".npy"))
            data.x = torch.from_numpy(x).to(self.dtype).unsqueeze(1)

        return data
