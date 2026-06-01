from .meta_solver import ULSolver
import torch
import numpy as np
from utils.data_utils import BatchDataset
from utils.noise_distributions import Bernoulli, Annealing
import torch.nn.functional as F
from utils.graph_ops import g_sum

class SDDS(ULSolver):
    def __init__(self, args):
        super(SDDS, self).__init__(args)
        
        if args.do_train:
            if args.noise_energy == 'bernoulli':
                self.noise_energy_func = Bernoulli(self.problem, self.num_tp)
            elif args.noise_energy == 'annealing':
                self.noise_energy_func = Annealing(self.problem, self.num_tp)
            else:
                raise NotImplementedError

    def full_batches_per_step(self):
        return np.ceil(self.num_tp/self.mini_batch_size)
         
    def train_batch(self, graph, batch):
        weight, solution, time, action = batch
        
        if self.solution_type == 'edge':
            graph_idx = graph.batch[graph.edge_index[0]]
        else:
            graph_idx = graph.batch
        
 
        weight = weight.permute(1,2,0)

        solution = solution.permute(1,2,0).flatten(1)
        time = time.permute(1,2,0).flatten(1)
        action = action.permute(1,2,0).flatten(1)

        logits = self.model(graph, solution.unsqueeze(-1), time).squeeze(-1)
        prob = F.sigmoid(logits)

        log_prob = g_sum(action*torch.log(prob+1e-6) + (1-action)*torch.log(1-prob+1e-6), graph_idx)
        log_prob = log_prob.reshape(-1, self.num_kp, weight.shape[-1])
        loss = -(log_prob*weight*self.mini_batch_size).sum(1).mean()
        
        return loss

    @torch.no_grad()
    def generate_data(self, graph):
        self.model.eval()
        log_ps = []
        log_qs = []
        all_data = []
        all_actions = []

        if self.solution_type == 'edge':
            num_edges = graph.edge_index.shape[1]
            solution = torch.randint(0,2,size=(num_edges, self.num_kp), device=self.accelerator.device, dtype=torch.float)
            graph_idx = graph.batch[graph.edge_index[0]]
        else:
            num_nodes = graph.num_nodes
            solution = torch.randint(0,2,size=(num_nodes, self.num_kp), device=self.accelerator.device, dtype=torch.float)
            graph_idx = graph.batch
            
        log_qs.append(g_sum(torch.ones_like(solution), graph_idx)*np.log(0.5))
        
        for t in range(self.num_tp):    
            time = torch.ones_like(solution)*t/self.num_tp
            all_data.append(solution.clone())
            logits = self.model(graph, solution.unsqueeze(-1), time).squeeze(-1)

            flip_prob = torch.sigmoid(logits)  
            solution_prob = solution*(1-flip_prob) + flip_prob*(1-solution)

            # action, used for log_prob
            rr = torch.rand_like(solution)
            action = (rr<flip_prob).float() 
            all_actions.append(action)
            log_qs.append(g_sum(torch.log(flip_prob+1e-6)*action+ torch.log(1-flip_prob+1e-6)*(1-action), graph_idx))
            
            # (Noise) energy: p(X_t|X_{t-1}), the log prob should be shifted
            next_solution = torch.where(rr<flip_prob, 1-solution, solution)
            log_p = -self.noise_energy_func(graph, solution, next_solution, time, self.tau)
            log_ps.append(log_p)
            solution = next_solution
            
        all_data = torch.stack(all_data, dim=-1)
        all_actions = torch.stack(all_actions, dim=-1)


        if self.solution_type == 'edge':
            energy, _ = self.problem.energy_func(graph, solution_prob)
        else:
            energy, _ = self.problem.energy_func(graph, solution)

        log_ps.append(-energy/self.tau)
        
        log_qs = torch.stack(log_qs, dim=-1).sum(-1)
        log_ps = torch.stack(log_ps, dim=-1).sum(-1)

        weights = torch.softmax(log_ps-log_qs, dim=1)

        weights = weights.unsqueeze(2).repeat(1,1,all_data.shape[-1])
        
        all_time = (torch.arange(self.num_tp, device=solution.device, dtype=torch.float) / self.num_tp
                    ).view(1, 1, self.num_tp).expand(solution.shape[0], self.num_kp, self.num_tp)
        weights = weights.unbind(-1)
        all_data = all_data.unbind(-1)
        all_time = all_time.unbind(-1)
        all_actions = all_actions.unbind(-1)
        
        return BatchDataset(list(zip(weights, all_data, all_time, all_actions)))