import torch
from .meta_solver import MetaSolver
from utils.temp_schedulers import TempScheduler

class RLSA(MetaSolver):
    # not supporting edge_based TSP!
    def __init__(self, args):
        super(RLSA, self).__init__(args)
        self.num_d = args.num_d
        self.temp_scheduler = TempScheduler(args.temp_schedule, args.tau0, args.tau1, 0, args.num_t, 0)

    def evaluate_single(self, graph):
        if self.solution_type == 'edge':
            raise NotImplementedError
        else:
            A = torch.sparse_coo_tensor(
                graph.edge_index, 
                graph.edge_weight, 
                torch.Size((graph.num_nodes, graph.num_nodes))
            ).to_sparse_csr()
            graph.A = A
        
        best_energies = []
        for _ in range(self.num_repetition):
            x = torch.randint(0,2, (graph.num_nodes, self.num_k), dtype=self.dtype, device=self.accelerator.device)
            energy, grad = self.problem.energy_func(graph, x, True) 
            energy = energy[0]
            best_sol = x.clone()
            best_energy = energy.clone()
            for epoch in range(self.num_t):          
                tau = self.temp_scheduler(epoch)
                delta = grad*(2*x-1)/2
                
                #The kth value method
                term2 = -torch.kthvalue(-delta, self.num_d, dim=0, keepdim=True).values
                flip_prob = torch.sigmoid((delta-term2)/tau)

                #The normalization method
                #flip_prob = torch.sigmoid(delta/tau)
                #flip_prob = flip_prob*self.num_d/flip_prob.sum(dim=0, keepdim=True)
                
                rr = torch.rand_like(x)
                x = torch.where(rr<flip_prob, 1-x, x)
                energy, grad = self.problem.energy_func(graph, x, True if epoch < self.num_t-1 else False)

                energy = energy[0]

                best_sol = torch.where((energy<best_energy).unsqueeze(0).repeat(graph.num_nodes,1), x, best_sol)
                best_energy = torch.where(energy<best_energy, energy, best_energy)
                   
            if not self.skip_decode:
                best_energy, _ = self.problem.evaluate(graph, best_sol)
                
            best_energies.append(best_energy)
        return torch.cat(best_energies).min() 