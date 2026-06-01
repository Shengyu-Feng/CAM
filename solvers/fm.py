from .meta_solver import SLSolver
from utils.diffusion_schedulers import *
import torch
from torch import nn

class FlowMatching(SLSolver):
    def __init__(self, args):
        super(FlowMatching, self).__init__(args)
        self.diffusion = CategoricalFM()
        self.inference_schedule = FMInferenceSchedule(inference_schedule=args.inference_schedule, inference_T=args.num_t)
        
        self.loss_func = nn.BCEWithLogitsLoss()
      
    def train_batch(self, graph):
        batch_size = len(graph.idx)
        time = torch.rand(batch_size, device=self.accelerator.device)

        if self.solution_type == 'edge':
            time = time[graph.batch[graph.edge_index[0]]]
            solution_0 = graph.edge_attr.squeeze(-1)
        else:
            time = time[graph.batch]
            solution_0 = graph.x.squeeze(-1)

        solution_t = self.diffusion.sample(solution_0, time)
        target = solution_0.float()
        
        logits = self.model(graph, solution_t.unsqueeze(-1), time).squeeze(-1)
        
        # Compute loss
        loss = self.loss_func(logits, target)
        
        return loss 
    
    def evaluate_single(self, graph):
        best_energies = []
        for _ in range(self.num_repetition):
            if self.solution_type == 'edge':
                num_edges = graph.edge_index.shape[1]
                solution = torch.randint(0,2,size=(num_edges,self.num_k), device=self.accelerator.device, dtype=torch.float)
            else:
                A = torch.sparse_coo_tensor(
                    graph.edge_index, 
                    graph.edge_weight, 
                    torch.Size((graph.num_nodes, graph.num_nodes))
                ).to_sparse_csr()
                graph.A = A
                num_nodes = graph.num_nodes
                solution = torch.randint(0,2,size=(num_nodes,self.num_k), device=self.accelerator.device, dtype=torch.float)
    
    
            for t in range(self.num_t):
                t1, t2 = self.inference_schedule(t)
                time = torch.ones((solution.shape[0],self.num_k), device=self.accelerator.device, dtype=torch.float)*t1            
                logits = self.model(graph, solution.unsqueeze(-1), time).squeeze(-1)
    
                if t == self.num_t-1:
                    solution = torch.sigmoid(logits)
                else:
                    solution_prob = solution + (torch.sigmoid(logits)-solution)*(t2-t1)
                    solution = torch.bernoulli(solution_prob.clamp(0,1))
    
            best_energy, _ = self.problem.evaluate(graph, solution)
            best_energies.append(best_energy)
            
        return torch.cat(best_energies).min()
    
