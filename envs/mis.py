import torch
from utils.graph_ops import g_sum

class Env(object):
    def __init__(self, args):
        self.beta = args.beta
        self.solution_type = 'node'
        self.mis_decoding = args.mis_decoding

    # mis loss
    def energy_func(self, graph, solution, compute_grad=False):
        # out: num_nodes x ...
        A, b = graph.A, graph.b
        v = A@solution

        energy_per_node = solution*(self.beta*v+b)
        
        energy = g_sum(energy_per_node, graph.batch)
        
        if compute_grad:
            grad = 2*self.beta*v+b
        else:
            grad = None

        return energy, grad

    def evaluate(self, graph, solution):
        if self.mis_decoding == 'continuous':
            x = solution.transpose(0,1)
            mask = torch.ones_like(x).bool()
            solution = torch.zeros_like(x)
    
            while mask.any():
                prob = torch.where(mask, x, -torch.inf)
                next_node = torch.argmax(prob,1)
                indices = torch.arange(x.shape[0]).to(x.device)
                flag = mask[indices, next_node]
                indices = indices[flag]
                next_node = next_node[flag]
                solution[indices,next_node] = 1
                mask[indices,next_node] = False
                nnx_masks = graph.edge_index[0].unsqueeze(0) == next_node.unsqueeze(1)  # Create masks for matching nodes
                nnx_indices = torch.nonzero(nnx_masks).T
                mask[indices[nnx_indices[0]], graph.edge_index[1][nnx_indices[1]]] = False
    
            return -solution.sum(1), solution.T
        else:
            solution = torch.round(solution).squeeze(-1)
            
            batch_idx = torch.arange(solution.shape[1], device=solution.device).long()

            if not hasattr(graph, "A"):
                A = torch.sparse_coo_tensor(
                    graph.edge_index, 
                    graph.edge_weight, 
                    torch.Size((graph.num_nodes, graph.num_nodes))
                ).to_sparse_csr()
                graph.A = A
            
            energy, grad = self.energy_func(graph, solution, True)
            best_energy = energy.clone()
            delta = grad*(2*solution-1)/2
            
            while not (delta<=0).all():
                value, index = torch.max(delta, dim=0)
                solution[index, batch_idx] = torch.where(value>0, 1-solution[index, batch_idx], solution[index, batch_idx])
                best_energy, grad = self.energy_func(graph, solution, True)
                delta = grad*(2*solution-1)/2
            
            return best_energy, solution.T
        
        