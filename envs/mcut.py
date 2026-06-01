import torch
from utils.graph_ops import g_sum

class Env(object):
    def __init__(self, beta):
        self.beta = beta
        self.solution_type = 'node'

    def energy_func(self, graph, solution, compute_grad=False):
        # out: num_nodes x ...
        A, b = graph.A, graph.b
        L = A@solution
        if compute_grad:
            grad = 2*L+b
        else:
            grad = None

        energy_per_node = solution*(L+b)
        loss = g_sum(energy_per_node, graph.batch)
        return loss, grad


    def evaluate(self, graph, solution):
        x = torch.round(solution)
        batch_idx = torch.arange(x.shape[1], device=x.device).long()
        energy, grad = self.energy_func(graph, x, True)
        best_energy = energy
        delta = grad*(2*x-1)/2
         
        while not (delta<=0).all():
            value, index = torch.max(delta, dim=0)
            x[index, batch_idx] = torch.where(value>0, 1-x[index, batch_idx], x[index, batch_idx])
            best_energy, grad = self.energy_func(graph, x, True)
            delta = grad*(2*x-1)/2
        
        return best_energy, x