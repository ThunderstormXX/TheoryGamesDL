import networkx as nx
import numpy as np

class Graph():
    def __init__(self):
        self.G = None

    def get_adj_matrix(self):
        if self.G is None:
            return None
        return nx.to_numpy_array(self.G)

    def get_degree(self):
        if self.G is None:
            return None
        # Return degrees as a list/array indexed by node
        return [d for n, d in sorted(self.G.degree())]

    def get_node_degree(self, node):
        if self.G is None:
            return 0
        return self.G.degree(node)

    def get_neibhours(self):
        if self.G is None:
            return None
        return {n: list(self.G.neighbors(n)) for n in self.G.nodes()}

    def get_clasterizotion_rate(self):
        if self.G is None:
            return 0
        return nx.average_clustering(self.G)

    def get_all_clasterization_rate(self):
        if self.G is None:
            return {}
        return nx.clustering(self.G)

    def get_cooperation_rate(self, strategies):
        # strategies should be an array-like of 0s and 1s corresponding to nodes
        if not len(strategies):
            return 0
        return np.mean(strategies)

    def get_all_cooperation_rate(self, strategies):
        # Simply returns the strategies distribution or array
        return strategies


class StarGraph(Graph):
    def __init__(self, n):
        super().__init__()
        # Star graph with n nodes (1 center, n-1 leaves)
        self.G = nx.star_graph(n - 1)

class WheelGraph(Graph):
    def __init__(self, n):
        super().__init__()
        # Wheel graph with n nodes (1 center, n-1 rim)
        self.G = nx.wheel_graph(n)

class SmallWorldGraph(Graph):
    def __init__(self, n, k, p):
        super().__init__()
        # Watts-Strogatz small-world graph
        # n nodes, each connected to k nearest neighbors in ring topology,
        # p probability of rewiring
        self.G = nx.watts_strogatz_graph(n, k, p)
