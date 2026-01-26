from graph_structure import Graph
import numpy as np


def reward(action: int, v: int, graph: Graph, ans_vector: np.ndarray):
    adj_vector = np.array(graph.get_adjacency_vector(v))
    k = np.sum(adj_vector)
    if k == 0:
        return 1.0
    if action == 0:
        return np.sum(adj_vector * (1 - ans_vector)) / k
    else:
        return np.sum(adj_vector * ans_vector) / k