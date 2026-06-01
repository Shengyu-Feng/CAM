from .meta_solver import ULSolver
import torch
from utils.data_utils import BatchBuffer
import numpy as np
from utils.graph_ops import g_sum

class RLNN(ULSolver):
    # not supporting edge_based TSP!
    def __init__(self, args):
        super(RLNN, self).__init__(args)
        self.lambd = args.lambd
        self.num_d = args.num_d
        
    def full_batches_per_step(self):
        return np.ceil(self.num_tp*self.num_kp/self.mini_batch_size)
          
    def train_batch(self, graph, batch):   
        solution, time = batch
        solution = solution.T
        time = time.T
        
        if self.solution_type == 'node':
            graph_idx = graph.batch
            logits = self.model(graph, solution.unsqueeze(-1), time).squeeze(-1)
            prob = torch.sigmoid(logits)
            solution = solution*(1-prob) + (1-solution)*prob
            energy, _ = self.problem.energy_func(graph, solution)
            neg_entropy = prob*torch.log(prob+1e-6) + (1-prob)*torch.log(1-prob+1e-6)
            loss = energy.mean() + self.tau*g_sum(neg_entropy, graph_idx).mean() + self.lambd*((g_sum(prob, graph_idx)-self.num_d)**2).mean()
        else:
            raise NotImplementedError
        
        return loss
    
    @torch.no_grad()
    def generate_data(self, graph):
        if self.solution_type == 'edge':
            num_vars = graph.edge_index.shape[1]
        else:
            num_vars = graph.num_nodes

        solution = torch.randint(0, 2, size=(num_vars, self.num_kp),
                                 device=self.accelerator.device, dtype=torch.float)
        all_data = []

        buffer = BatchBuffer()

        
        for t in range(self.num_tp):
            time = torch.ones_like(solution)*t/self.num_tp
            all_data.append(solution.clone())
            logits = self.model(graph, solution.unsqueeze(-1), time).squeeze(-1)
            flip_prob = torch.sigmoid(logits)
            
            rr = torch.rand_like(solution)
            solution = torch.where(rr<flip_prob, 1-solution, solution)
            
        all_data = torch.stack(all_data, dim=-1)
        all_time = (torch.arange(self.num_tp, device=solution.device, dtype=torch.float) / self.num_tp
                    ).view(1, 1, self.num_tp).expand(num_vars, self.num_kp, self.num_tp)
        
        all_data = all_data.flatten(1).unbind(1)
        all_time = all_time.flatten(1).unbind(1)
        
        buffer.add(list(zip(*(all_data, all_time))))
        
        return buffer.get_dataset()