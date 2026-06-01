import torch
import numpy as np
import ctypes
from utils.c_classic import c_cvrp_local_search


def cvrp_ls(
    init_tour,
    coord, 
    demand,
    coord_scale: int = 1000,
    demand_scale: int = 1000,
    seed: int = 1234
):
    """Classic local search for CVRP problems using C implementation."""
    # Preparation
    nodes_num = coord.shape[0] # Get number of nodes before reshaping

    init_tour = np.ascontiguousarray(init_tour, dtype=np.int16)
    coord = np.ascontiguousarray(coord.reshape(-1), dtype=np.float32)
    demand = np.ascontiguousarray(demand.reshape(-1), dtype=np.float32)

    ls_tour = c_cvrp_local_search(
        init_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),  
        coord.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
        demand.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
        nodes_num,
        len(init_tour),
        coord_scale,
        demand_scale,
        seed
    )

    ls_tour = np.ctypeslib.as_array(ls_tour, shape=(len(init_tour)+2,))
    
    # Store the optimized tour in the task data
    if ls_tour[0] != -1:
        ls_tour = ls_tour[:np.where(ls_tour==-1)[0][0]]
        return ls_tour
    else:
        return init_tour


def cvrp_decode_score_savings(
    edge_score,
    coord,
    demand,
    capacity=1.0,
    alpha=1.0,
    beta=0.0,
):
    """
    Decode a CVRP solution from edge scores using a simple score-guided
    Clarke-Wright style merging heuristic.

    Args:
        edge_score: (V, V), larger means edge is more preferred
        coord:      (V, 2)
        demand:     (V,), demand[0] = 0
        capacity:   vehicle capacity
        alpha:      weight for learned edge score
        beta:       weight for classical savings term

    Returns:
        flat tour with depot separators, e.g. [0, ..., 0, ..., 0]
    """
    edge_score = np.asarray(edge_score, dtype=np.float64)
    coord = np.asarray(coord, dtype=np.float64)
    demand = np.asarray(demand, dtype=np.float64)

    V = len(demand)
    assert edge_score.shape == (V, V)
    assert coord.shape[0] == V
    assert demand[0] == 0

    if np.any(demand[1:] > capacity):
        raise ValueError("Some customer demand exceeds vehicle capacity.")

    # pairwise distances
    dist = np.linalg.norm(coord[:, None, :] - coord[None, :, :], axis=-1)

    # initial routes: one customer per route
    routes = {i: [i] for i in range(1, V)}
    route_load = {i: demand[i] for i in range(1, V)}
    node_to_route = {i: i for i in range(1, V)}

    # candidate merges only between customers
    candidates = []
    for i in range(1, V):
        for j in range(i + 1, V):
            savings = dist[i, 0] + dist[0, j] - dist[i, j]
            priority = alpha * edge_score[i, j] + beta * savings
            candidates.append((priority, i, j))

    # highest priority first
    candidates.sort(reverse=True, key=lambda x: x[0])

    def is_endpoint(route, node):
        return route[0] == node or route[-1] == node

    def merge_routes(route_a, route_b, i, j):
        """
        Merge route_a and route_b through endpoints i and j.
        Returns merged customer list or None.
        """
        a = route_a
        b = route_b

        # orient so that i is at the end of a
        if a[0] == i:
            a = a[::-1]
        elif a[-1] != i:
            return None

        # orient so that j is at the start of b
        if b[-1] == j:
            b = b[::-1]
        elif b[0] != j:
            return None

        return a + b

    # perform feasible merges
    for _, i, j in candidates:
        ri = node_to_route.get(i, None)
        rj = node_to_route.get(j, None)

        if ri is None or rj is None or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        # only merge route endpoints
        if not is_endpoint(route_i, i) or not is_endpoint(route_j, j):
            continue

        new_load = route_load[ri] + route_load[rj]
        if new_load > capacity + 1e-8:
            continue

        merged = merge_routes(route_i, route_j, i, j)
        if merged is None:
            merged = merge_routes(route_i, route_j, j, i)
        if merged is None:
            continue

        # keep ri, remove rj
        routes[ri] = merged
        route_load[ri] = new_load

        for node in merged:
            node_to_route[node] = ri

        del routes[rj]
        del route_load[rj]

    # build flat tour
    flat_tour = [0]
    for route in routes.values():
        flat_tour.extend(route)
        flat_tour.append(0)

    return np.array(flat_tour, dtype=np.int64)

def cvrp_greedy(heatmap, demand, capacity=1.0):
    """
    heatmap: (V, V)
    demand:  (V,), demand[0] = 0
    """
    heatmap = heatmap.copy()
    V = len(demand)

    if demand[0] != 0:
        raise ValueError("demand[0] must be 0 for depot")
    if np.any(demand[1:] > capacity):
        raise ValueError("Some customer demand exceeds vehicle capacity")

    np.fill_diagonal(heatmap, -np.inf)

    served = np.zeros(V, dtype=bool)
    served[0] = True

    current = 0
    remaining = capacity
    tour = [0]

    while not np.all(served[1:]):
        feasible = (~served) & (demand <= remaining+1e-8)
        feasible[0] = False

        if np.any(feasible[1:]):
            scores = heatmap[current].copy()
            scores[~feasible] = -np.inf

            nxt = int(np.argmax(scores))
            served[nxt] = True
            remaining -= demand[nxt]
            tour.append(nxt)
            current = nxt
        else:
            if current != 0:
                tour.append(0)
                current = 0
                remaining = capacity
            else:
                raise RuntimeError("No feasible node can be selected from depot.")

    if current != 0:
        tour.append(0)

    return np.array(tour, dtype=np.int64)

class Env:
    """
    Single-step, bandit-style env:
    - step(action) -> obs_next (can be dummy), reward, done=True, info
    """
    def __init__(self, args=None):
        self.solution_type = 'edge'
                
    def energy_func(self, graph, solution, compute_grad=False):  
        data_list = graph.to_data_list()
        
        graph_idx = graph.batch[graph.edge_index[0]]  # [E] graph id per edge

        num_graphs = graph.num_graphs  # or int(graph.batch.max()) + 1
        solution_list = [solution[graph_idx == i] for i in range(num_graphs)]

        energy_list, grad_list = [], []
        for g, s in zip(data_list, solution_list):
            energy, tours = self.evaluate(g, s)
            energy_list.append(energy)
            if compute_grad:
                terminal_state = []
                for tour in tours:
                    tour_matrix = torch.zeros((g.num_nodes, g.num_nodes), dtype=solution.dtype, device=solution.device)
                    tour_matrix[tour[:-1], tour[1:]] = 1
                    tour_matrix[tour[1:], tour[:-1]] = 1
                    tour_edge = tour_matrix.reshape(-1)
                    terminal_state.append(tour_edge)

                terminal_state = torch.stack(terminal_state, dim=-1) 
                grad_list.append(terminal_state)
            
        return torch.stack(energy_list), torch.cat(grad_list) if compute_grad else None

    def evaluate(self, graph, solution):
        """
        instance: graph instance
        solution: shape [|E|, batch_size]
        """

        coord = graph.x.cpu().numpy()
        demand = graph.demand.cpu().numpy()[:,0]
        device = solution.device
        num_nodes = graph.num_nodes
        solution =  solution.cpu().numpy()
        batch_size = solution.shape[1]
        tours = []
        costs = []

        for i in range(batch_size):
            #init_tour = sum([[0,i] for i in range(1,101)],[]) +[0]
            init_tour = cvrp_decode_score_savings(solution[:,i].reshape(num_nodes, num_nodes), coord, demand)
            ls_tour = cvrp_ls(init_tour, coord, demand)
            cost = np.linalg.norm(coord[ls_tour[:-1]]-coord[ls_tour[1:]], axis=1).sum()
            tours.append(torch.LongTensor(ls_tour).to(device))
            costs.append(cost)

        costs = torch.Tensor(np.stack(costs)).to(device)

        return costs, tours