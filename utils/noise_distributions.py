import torch
from utils.graph_ops import g_sum

class Base(object):
    def __init__(self, problem, num_tp):
        self.problem = problem
        self.num_tp = num_tp

    def __call__(self, graph, solution, solution_prob, t, tau):
        # t is the next time not the current time
        pass
        
class Bernoulli(Base):
    def __init__(self, problem, num_tp):
        super(Bernoulli, self).__init__(problem, num_tp)
        
    def __call__(self, graph, solution, solution_prob, t, tau):
        beta_t = torch.pow(2.0, -6.0 * t) * 0.5 
        p_up = solution_prob
        p_down = 1-p_up
        
        if self.problem.solution_type == 'edge':
            graph_idx = graph.batch[graph.edge_index[0]]
        else:
            graph_idx = graph.batch

        noise_energy_per_node = solution*(p_up*torch.log(1-beta_t)+p_down*torch.log(beta_t)) +\
            (1-solution)*(p_down*torch.log(1-beta_t)+p_up*torch.log(beta_t))
        noise_energy = g_sum(noise_energy_per_node, graph_idx)
        return -noise_energy        
    
    
class Annealing(Base):
    def __init__(self, problem, num_tp):
        super(Annealing, self).__init__(problem, num_tp)
        
    def __call__(self, graph, solution, solution_prob, t, tau):
        beta_t = t[0]
        noise_energy, _ = self.problem.energy_func(graph, solution_prob)
        noise_energy = noise_energy*beta_t
        return noise_energy/tau