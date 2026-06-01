from .meta_solver import ULSolver
import torch
from torch import nn
from utils.noise_distributions import Bernoulli, Annealing
from utils.data_utils import BatchDataset
from utils.graph_ops import g_sum

class DiffUCO(ULSolver):
    # not supporting edge_based TSP!
    def __init__(self, args):
        super(DiffUCO, self).__init__(args)
       
        if args.do_train:
            if args.noise_energy == 'bernoulli':
                self.noise_energy_func = Bernoulli(self.problem, self.num_tp)
            elif args.noise_energy == 'annealing':
                self.noise_energy_func = Annealing(self.problem, self.num_tp)
            else:
                raise NotImplementedError

    def full_batches_per_step(self):
        return 1

    @torch.no_grad()
    def generate_data(self, graph):
        all_data = []
        all_entropies = []
        all_noise_energies = []
        all_actions = []

        if self.solution_type == 'edge':
            raise NotImplementedError
        else:
            num_nodes = graph.num_nodes
            solution = torch.randint(0,2,size=(num_nodes, self.num_kp), device=self.accelerator.device, dtype=torch.float)
            graph_idx = graph.batch

        for t in range(self.num_tp):    
            time = torch.ones_like(solution)*t/self.num_tp
            all_data.append(solution.clone())
            logits = self.model(graph, solution.unsqueeze(-1), time).squeeze(-1)

            flip_prob = torch.sigmoid(logits)  

            # entropy: S(q(X_{t+1}|X_t))
            entropy_per_node = -(flip_prob * torch.log(flip_prob+1e-6) + (1 - flip_prob) * torch.log(1 - flip_prob +1e-6))
            entropy = g_sum(entropy_per_node, graph_idx)
            all_entropies.append(entropy-entropy.mean(dim=1, keepdim=True))

            # action, used for log_prob
            rr = torch.rand_like(solution)
            all_actions.append((rr<flip_prob).float())

            solution_prob = solution*(1-flip_prob) + (1-solution)*flip_prob

            # (Noise) energy: p(X_t|X_{t-1}), the log prob should be shifted
            if t < self.num_tp-1:
                noise_energy = self.noise_energy_func(graph, solution, solution_prob, time, self.tau)*self.tau
            else:
                # this is in fact the energy
                noise_energy, _ = self.problem.energy_func(graph, solution_prob)   

            all_noise_energies.append(noise_energy - noise_energy.mean(dim=1, keepdim=True))

            solution = torch.where(rr<flip_prob, 1-solution, solution)

        all_entropies = torch.stack(all_entropies, dim=-1)
        all_noise_energies = torch.stack(all_noise_energies, dim=-1)

        all_data = torch.stack(all_data, dim=-1)
        all_actions = torch.stack(all_actions, dim=-1)
        all_time = (torch.arange(self.num_tp, device=solution.device, dtype=torch.float) / self.num_tp
                    ).view(1, 1, self.num_tp).expand(num_nodes, self.num_kp, self.num_tp).contiguous()
        return BatchDataset([(all_data, all_time, all_actions, all_entropies, all_noise_energies)])
        
    
    def train_batch(self, graph, batch):
        if self.solution_type == 'edge':
            raise NotImplementedError
        else:
            graph_idx = graph.batch
        
        
        all_data, all_time, all_actions, all_entropies, all_noise_energies = batch

        all_data = all_data[0]
        all_time = all_time[0]
        all_actions = all_actions[0]
        all_entropies = all_entropies[0]
        all_noise_energies = all_noise_energies[0]
        
        logits = self.model(graph, all_data.flatten(1).unsqueeze(-1), all_time.flatten(1)).squeeze(-1) # N X (num_kp * (num_tp))
        flip_prob = torch.sigmoid(logits)
        flip_prob = flip_prob.reshape(-1, self.num_kp, self.num_tp)
        
        solution_prob = all_data*(1-flip_prob) + (1-all_data)*flip_prob
        
        noise_energy = self.noise_energy_func(graph, all_data[...,:-1].flatten(1), solution_prob[...,:-1].flatten(1), all_time[...,:-1].flatten(1), self.tau)*self.tau
        noise_energy = noise_energy.reshape(-1, self.num_kp, self.num_tp-1)

        final_energy, _ = self.problem.energy_func(graph, solution_prob[...,-1])
        energy_loss = (noise_energy.sum(-1) + final_energy).mean()
        
        
        entropy_per_node = -(flip_prob * torch.log(flip_prob+1e-6) + (1 - flip_prob) * torch.log(1 - flip_prob +1e-6))
        entropy = g_sum(entropy_per_node, graph_idx)

        entropy_loss = -entropy.sum(-1).mean()*self.tau    

        all_rewards = all_entropies*self.tau - all_noise_energies
        cum_rewards = all_rewards[...,1:].flip(-1).cumsum(-1).flip(-1)

        node_log_prob = torch.log(flip_prob+1e-6)*all_actions + torch.log(1-flip_prob+1e-6)*(1-all_actions)
        graph_log_prob = g_sum(node_log_prob, graph_idx)

        reinforce_loss = -(cum_rewards*graph_log_prob[...,:-1]).sum(-1).mean()

        loss = energy_loss + entropy_loss + reinforce_loss

        return loss
           