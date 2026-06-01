import torch
from torch_geometric.utils import scatter

def g_sum(x, batch):
    if batch.max() == 0:
        out = x.sum(dim=0, keepdim=True)
    else:
        out = scatter(x, batch, dim=0, reduce='sum')
    return out
