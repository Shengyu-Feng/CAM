from .meta_solver import ULSolver
import torch
from torch import nn
import numpy as np
from utils.data_utils import BatchBuffer
from utils.graph_ops import g_sum

class CAM(ULSolver):
    def __init__(self, args):
        super(CAM, self).__init__(args)
        self.lambd = args.lambd
        self.num_d = args.num_d
        self.num_clips = args.num_clips
        self.loss_func = nn.BCEWithLogitsLoss()
        

    def full_batches_per_step(self):
        return np.ceil(self.num_tp*self.num_kp/self.mini_batch_size)

    def train_batch(self, graph, batch):
        solution, time, weight, target = batch
        solution = solution.T
        weight = weight.T
        time = time.T
        target = target.T
        logits = self.model(graph, solution.unsqueeze(-1), time).squeeze(-1)

        if self.solution_type == 'node':
            graph_idx = graph.batch
            prob = torch.sigmoid(logits)
            neg_entropy = prob*torch.log(prob+1e-6) + (1-prob)*torch.log(1-prob+1e-6)
            loss = -g_sum(prob*target*torch.exp(-1*(1-weight)), graph_idx).mean() + self.tau*g_sum(neg_entropy, graph_idx).mean() + self.lambd*((g_sum(prob, graph_idx)-self.num_d)**2).mean()
        else:
            graph_idx = graph.batch[graph.edge_index[0]]
            prob = torch.sigmoid(logits)
            loss = self.loss_func(logits, target) + self.lambd*((g_sum(prob, graph_idx)-self.num_d)**2).mean()

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
        all_time = []

        buffer = BatchBuffer()
        clip_frequency = np.ceil(self.num_tp/self.num_clips)
        
        for t in range(self.num_tp):
            time = torch.ones_like(solution)*t/self.num_tp
            all_time.append(time)
            all_data.append(solution.clone())
            logits = self.model(graph, solution.unsqueeze(-1), time).squeeze(-1)
            flip_prob = torch.sigmoid(logits)

            if ((t+1)%clip_frequency == 0) or (t==self.num_tp-1):
                all_data = torch.stack(all_data, dim=-1)
                all_time = torch.stack(all_time, dim=-1)
                weight = torch.ones_like(all_time)*(t+1)/self.num_tp
                if self.solution_type == 'node':
                    energy, grad = self.problem.energy_func(graph, solution, True)
                    target = grad.unsqueeze(-1)*(2*all_data-1)
                else:
                    solution_prob = solution*(1-flip_prob) + flip_prob*(1-solution)
                    energy, grad = self.problem.energy_func(graph, solution_prob, True)
                    target = ((1-2*all_data)*(2*grad.unsqueeze(-1)-1)+1)/2

                all_data = all_data.flatten(1).unbind(1)
                all_time = all_time.flatten(1).unbind(1)
                weight = weight.flatten(1).unbind(1)
                target = target.flatten(1).unbind(1)
                buffer.add(list(zip(*(all_data, all_time, weight, target))))
                all_data = []
                all_time = []
            
            rr = torch.rand_like(solution)
            solution = torch.where(rr<flip_prob, 1-solution, solution)

        return buffer.get_dataset()
