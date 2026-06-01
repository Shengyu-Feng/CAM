import os
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import scipy.spatial
import torch
from utils.cython_merge.cython_merge import merge_cython


def batched_two_opt_torch(points, tour, max_iterations=1000, device="cpu"):
  iterator = 0
  tour = tour.copy()
  with torch.inference_mode():
    cuda_points = torch.from_numpy(points).to(device)
    cuda_tour = torch.from_numpy(tour).to(device)
    batch_size = cuda_tour.shape[0]
    min_change = -1.0
    while min_change < 0.0:
      points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
      points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

      A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
      A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
      A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
      A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

      change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
      valid_change = torch.triu(change, diagonal=2)

      min_change = torch.min(valid_change)
      flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
      min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
      min_j = torch.remainder(flatten_argmin_index, len(points))

      if min_change < -1e-6:
        for i in range(batch_size):
          cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
        iterator += 1
      else:
        break

      if iterator >= max_iterations:
        break
    tour = cuda_tour.cpu().numpy()
  return tour, iterator


def numpy_merge(points, adj_mat):
  dists = np.linalg.norm(points[:, None] - points, axis=-1)

  components = np.zeros((adj_mat.shape[0], 2)).astype(int)
  components[:] = np.arange(adj_mat.shape[0])[..., None]
  real_adj_mat = np.zeros_like(adj_mat)
  merge_iterations = 0
  for edge in (-adj_mat/dist).flatten().argsort():
    merge_iterations += 1
    a, b = edge // adj_mat.shape[0], edge % adj_mat.shape[0]
    if not (a in components and b in components):
      continue
    ca = np.nonzero((components == a).sum(1))[0][0]
    cb = np.nonzero((components == b).sum(1))[0][0]
    if ca == cb:
      continue
    cca = sorted(components[ca], key=lambda x: x == a)
    ccb = sorted(components[cb], key=lambda x: x == b)
    newc = np.array([[cca[0], ccb[0]]])
    m, M = min(ca, cb), max(ca, cb)
    real_adj_mat[a, b] = 1
    components = np.concatenate([components[:m], components[m + 1:M], components[M + 1:], newc], 0)
    if len(components) == 1:
      break
  real_adj_mat[components[0, 1], components[0, 0]] = 1
  real_adj_mat += real_adj_mat.T
  return real_adj_mat, merge_iterations


def cython_merge(points, adj_mat):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    real_adj_mat, merge_iterations = merge_cython(points.astype("double"), adj_mat.astype("double"))
    real_adj_mat = np.asarray(real_adj_mat)
  return real_adj_mat, merge_iterations


def merge_tours(adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1):
  """
  To extract a tour from the inferred adjacency matrix A, we used the following greedy edge insertion
  procedure.
  • Initialize extracted tour with an empty graph with N vertices.
  • Sort all the possible edges (i, j) in decreasing order of Aij/kvi − vjk (i.e., the inverse edge weight,
  multiplied by inferred likelihood). Call the resulting edge list (i1, j1),(i2, j2), . . . .
  • For each edge (i, j) in the list:
    – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
    – If inserting (i, j) results in a graph with cycles (of length < N), continue.
    – Otherwise, insert (i, j) into the tour.
  • Return the extracted tour.
  """
  splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)

  if not sparse_graph:
    splitted_adj_mat = [
        adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat
    ]
  else:
    splitted_adj_mat = [
        scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[0], edge_index_np[1])),
        ).toarray() + scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[1], edge_index_np[0])),
        ).toarray() for adj_mat in splitted_adj_mat
    ]

  splitted_points = [
      np_points for _ in range(parallel_sampling)
  ]

  if np_points.shape[0] > 1000 and parallel_sampling > 1:
    with Pool(parallel_sampling) as p:
      results = p.starmap(
          cython_merge,
          zip(splitted_points, splitted_adj_mat),
      )
  else:
    results = [
        cython_merge(_np_points, _adj_mat) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)
    ]

  splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

  tours = []
  for i in range(parallel_sampling):
    tour = [0]
    while len(tour) < splitted_adj_mat[i].shape[0] + 1:
      n = np.nonzero(splitted_real_adj_mat[i][tour[-1]])[0]
      if len(tour) > 1:
        n = n[n != tour[-2]]
      tour.append(n.max())
    tours.append(tour)

  merge_iterations = np.mean(splitted_merge_iterations)
  return tours, merge_iterations


class TSPEvaluator(object):
    def __init__(self, points):
        self.dist_mat = scipy.spatial.distance_matrix(points, points)

    def evaluate(self, route):
        total_cost = 0
        for i in range(len(route) - 1):
          total_cost += self.dist_mat[route[i], route[i + 1]]
        return total_cost

class Env:
    """
    Single-step, bandit-style env:
    - step(action) -> obs_next (can be dummy), reward, done=True, info
    """
    def __init__(self, args):
        self.two_opt_iterations = args.two_opt_iterations
        self.solution_type = 'edge'
        
    def energy_func(self, graph, solution, compute_grad=False):    
        data_list = graph.to_data_list()
        
        graph_idx = graph.batch[graph.edge_index[0]]  # [E] graph id per edge

        # (optional safety check)
        # assert torch.all(edge_batch == graph.batch[col])

        num_graphs = graph.num_graphs  # or int(graph.batch.max()) + 1
        solution_list = [solution[graph_idx == i] for i in range(num_graphs)]
        
        energy_list, grad_list = [], []
        for g, s in zip(data_list, solution_list):
            energy, tours = self.evaluate(g, s)
            energy_list.append(energy)
            if compute_grad:
                edge_index_0 = tours[:-1, :]
                edge_index_1 = tours[1:, :]
                edge_hash = g.edge_index[0] * g.num_nodes + g.edge_index[1]
                subset_hash = edge_index_0 * g.num_nodes + edge_index_1

                terminal_state = []
                for i in range(tours.shape[1]):
                    terminal_state.append(torch.isin(edge_hash, subset_hash[:,i].flatten(), assume_unique=True).long())

                terminal_state = torch.stack(terminal_state, dim=-1)
                grad_list.append(terminal_state)
            
        return torch.stack(energy_list), torch.cat(grad_list) if compute_grad else None
        
    def evaluate(self, graph, solution):
        """
        instance: graph instance
        solution: shape [|E|, batch_size]
        """

        np_points = graph.x.cpu().numpy()
        np_edge_index = graph.edge_index.cpu().numpy()
        
        device = solution.device
        batch_size = solution.shape[1]
        
        adj_mat = solution.float().cpu().detach().flatten(1).numpy() + 1e-6
        adj_mat = adj_mat.transpose().reshape(-1)
        
        tours, merge_iterations = merge_tours(
            adj_mat, np_points, np_edge_index,
            sparse_graph=True,
            parallel_sampling=batch_size,
        )

        # Refine using 2-opt
        if self.two_opt_iterations > 0:
            tours, ns = batched_two_opt_torch(
                np_points.astype("float64"), np.array(tours).astype('int64'),
                max_iterations=self.two_opt_iterations, device=device)
        
        tsp_solver = TSPEvaluator(np_points)
        all_solved_costs = torch.Tensor([tsp_solver.evaluate(tours[i]) for i in range(batch_size)]).to(device)
        return all_solved_costs, torch.LongTensor(tours).T.to(device)
