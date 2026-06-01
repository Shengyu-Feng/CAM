from .meta_solver import SLSolver
from utils.diffusion_schedulers import *
import torch
from torch import nn
import torch.nn.functional as F

class DIFUSCO(SLSolver):
    def __init__(self, args):
        super(DIFUSCO, self).__init__(args)            
        self.diffusion = CategoricalDiffusion(T=args.num_tp, schedule=args.diffusion_schedule)
        self.inference_schedule = DiffusionInferenceSchedule(inference_schedule=args.inference_schedule,
                                              T=args.num_tp, inference_T=args.num_t)
        self.loss_func = nn.CrossEntropyLoss()
    
    def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
        """Sample from the categorical posterior for a given time step.
           See https://arxiv.org/pdf/2107.03006.pdf for details.
        """
        diffusion = self.diffusion
        
        if target_t is None:
            target_t = t - 1
        
        Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
        Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device) # 2 x 2

        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device) # 2 x 2
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device) # 2 x 2

        xt = F.one_hot(xt.long(), num_classes=2).float() # |V| x B x x 2
        xt = xt.reshape(x0_pred_prob.shape) # |V| x B x  2 
        
        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous()) # |V| x B x 2
        x_t_target_prob_part_2 = Q_bar_t_target[0] # 2
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True) # |V| x B x 1

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3 # |V| x B x 2

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0] # |V| x B 
        x_t_target_prob_part_2_new = Q_bar_t_target[1] # 2
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True) # |V| x B 
        
        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new # |V| x B x 2
        
        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1] # |V| x B 

        if target_t > 0:
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)
        
        return xt
    
    def tour2adj(self, tour, points, sparse, sparse_factor, edge_index):
        adj_matrix = np.zeros(points.shape[0], dtype=np.int64)
        adj_matrix[tour[:-1]] = tour[1:]
        adj_matrix = torch.from_numpy(adj_matrix)
        adj_matrix = adj_matrix.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
        adj_matrix = torch.eq(edge_index[1].cpu(), adj_matrix).to(torch.int)
        return adj_matrix
    
    def guided_categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
        torch.set_grad_enabled(True)
        xt = xt.float()  # b, n, n
        xt.requires_grad = True
        t = torch.from_numpy(t).view(1)
        
        # [b, 2, n, n]
        with torch.inference_mode(False):
            # print(points.shape, xt.shape)
            x0_pred = self.forward(
                points.float().to(device),
                xt.to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )
        
        if not self.sparse:
            x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        else:
            x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)
        
        dis_matrix = torch.sqrt(torch.sum((points[edge_index.T[:, 0]] - points[edge_index.T[:, 1]]) ** 2, dim=1))
        dis_matrix = dis_matrix.reshape((1, points.shape[0], -1))
        cost_est = (dis_matrix * x0_pred_prob[..., 1]).sum()
        cost_est.requires_grad_(True)
        cost_est.backward()
        
        if self.args.norm is True:
            xt.grad = nn.functional.normalize(xt.grad, p=2, dim=-1)
            xt = self.guided_categorical_posterior(target_t, t, x0_pred_prob, xt)
     
        return xt.detach()
    
    def train_batch(self, graph):
        batch_size = len(graph.idx)
        time = np.random.randint(0, self.num_tp+1, batch_size)
        
        if self.solution_type == 'edge':
            time = time[graph.batch[graph.edge_index[0]].cpu().numpy()]
            solution_0 = graph.edge_attr
        else:
            time = time[graph.batch.cpu().numpy()]
            solution_0 = graph.x

        solution_0_onehot = F.one_hot(solution_0.long(), num_classes=2).float()
        solution_t = self.diffusion.sample(solution_0_onehot, time)
        time = torch.from_numpy(time).float().to(self.accelerator.device)
        target = solution_0.long().squeeze(1)          
        
        logits = self.model(graph, solution_t, time)

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
                logits = self.model(graph, solution.unsqueeze(-1), time)
                x0_pred_prob = logits.softmax(dim=-1)
                solution = self.categorical_posterior(t2, t1, x0_pred_prob, solution)
    
            best_energy, _ = self.problem.evaluate(graph, solution)
            best_energies.append(best_energy)
        return torch.cat(best_energies).min() 