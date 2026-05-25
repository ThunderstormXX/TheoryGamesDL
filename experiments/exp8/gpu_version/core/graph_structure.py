import torch
import numpy as np

class BaseGraph:
    def __init__(self, num_nodes, device=None):
        self.num_nodes = num_nodes
        self.device = device or torch.device('cpu')

    def generate_adjacency_matrix(self):
        raise NotImplementedError("Subclasses must implement generate_adjacency_matrix")

    def get_batched_adjacency(self, batch_size):
        base_adj = self.generate_adjacency_matrix()
        return base_adj.unsqueeze(0).expand(batch_size, -1, -1)

class SmallWorldGraph(BaseGraph):
    def __init__(self, num_nodes, k=4, p=0.1, device=None):
        super().__init__(num_nodes, device)
        self.k = k
        self.p = p
        
    def generate_adjacency_matrix(self):
        """
        Creates a Small World (Watts-Strogatz) graph.
        """
        # Start with a regular ring
        adj = torch.zeros((self.num_nodes, self.num_nodes), device=self.device, dtype=torch.float32)
        
        # Connect to k nearest neighbors (k/2 on each side)
        for i in range(self.num_nodes):
            for j in range(1, self.k // 2 + 1):
                neighbor_plus = (i + j) % self.num_nodes
                neighbor_minus = (i - j) % self.num_nodes
                adj[i, neighbor_plus] = 1.0
                adj[neighbor_plus, i] = 1.0
                adj[i, neighbor_minus] = 1.0
                adj[neighbor_minus, i] = 1.0
                
        # Rewiring
        if self.p > 0:
            for i in range(self.num_nodes):
                for j in range(1, self.k // 2 + 1):
                    target = (i + j) % self.num_nodes
                    if torch.rand(1).item() < self.p:
                        # Remove edge (i, target)
                        adj[i, target] = 0
                        adj[target, i] = 0
                        
                        # Add new edge (i, new_target)
                        while True:
                            new_target = torch.randint(0, self.num_nodes, (1,)).item()
                            if new_target != i and adj[i, new_target] == 0:
                                adj[i, new_target] = 1.0
                                adj[new_target, i] = 1.0
                                break
                                
        return adj

class StarGraph(BaseGraph):
    def __init__(self, num_nodes, device=None):
        """
        One central node connected to all other nodes.
        """
        super().__init__(num_nodes, device)

    def generate_adjacency_matrix(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes), device=self.device, dtype=torch.float32)
        # Node 0 is the center
        adj[0, 1:] = 1.0
        adj[1:, 0] = 1.0
        return adj

class WheelGraph(BaseGraph):
    def __init__(self, num_nodes, device=None):
        """
        A star graph where the peripheral nodes also form a ring.
        """
        super().__init__(num_nodes, device)

    def generate_adjacency_matrix(self):
        # 1. Start with a Star Graph (center connected to all)
        adj = torch.zeros((self.num_nodes, self.num_nodes), device=self.device, dtype=torch.float32)
        adj[0, 1:] = 1.0
        adj[1:, 0] = 1.0
        
        # 2. Add the Ring (connecting peripheral nodes 1 to N-1)
        num_peripheral = self.num_nodes - 1
        for i in range(1, self.num_nodes):
            # Next node in the ring (wrapping around 1 to N-1)
            next_node = i + 1
            if next_node == self.num_nodes:
                next_node = 1
            
            adj[i, next_node] = 1.0
            adj[next_node, i] = 1.0
            
        return adj


class EdgeGraph(BaseGraph):
    def __init__(self, device=None):
        """Two nodes connected by a single undirected edge."""
        super().__init__(num_nodes=2, device=device)

    def generate_adjacency_matrix(self):
        adj = torch.zeros((2, 2), device=self.device, dtype=torch.float32)
        adj[0, 1] = 1.0
        adj[1, 0] = 1.0
        return adj


class TriangleGraph(BaseGraph):
    def __init__(self, device=None):
        """Three nodes in a triangle (complete graph K3)."""
        super().__init__(num_nodes=3, device=device)

    def generate_adjacency_matrix(self):
        adj = torch.ones((3, 3), device=self.device, dtype=torch.float32)
        adj.fill_diagonal_(0.0)
        return adj


class RingGraph(BaseGraph):
    """2-regular cycle graph on n nodes."""
    def __init__(self, num_nodes, device=None):
        super().__init__(num_nodes=num_nodes, device=device)

    def generate_adjacency_matrix(self):
        n = self.num_nodes
        adj = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        for i in range(n):
            adj[i, (i + 1) % n] = 1.0
            adj[(i + 1) % n, i] = 1.0
        return adj


class CubicCirculantGraph(BaseGraph):
    """
    3-regular circulant graph C(n, {1, n/2}).
    Each node connects to its two ring neighbours AND its antipodal node.
    Requires n to be even and n >= 4.
    """
    def __init__(self, num_nodes, device=None):
        assert num_nodes % 2 == 0 and num_nodes >= 4, \
            f"CubicCirculant requires even n >= 4, got {num_nodes}"
        super().__init__(num_nodes=num_nodes, device=device)

    def generate_adjacency_matrix(self):
        n = self.num_nodes
        adj = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        half = n // 2
        for i in range(n):
            adj[i, (i + 1) % n] = 1.0
            adj[(i + 1) % n, i] = 1.0
            adj[i, (i + half) % n] = 1.0
            adj[(i + half) % n, i] = 1.0
        return adj


class QuarticCirculantGraph(BaseGraph):
    """
    4-regular circulant graph C(n, {1, 2}).
    Each node connects to 2 neighbors on each side.
    Requires n >= 5.
    """
    def __init__(self, num_nodes, device=None):
        assert num_nodes >= 5, f"QuarticCirculant requires n >= 5, got {num_nodes}"
        super().__init__(num_nodes=num_nodes, device=device)

    def generate_adjacency_matrix(self):
        n = self.num_nodes
        adj = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        for i in range(n):
            adj[i, (i + 1) % n] = 1.0
            adj[(i + 1) % n, i] = 1.0
            adj[i, (i + 2) % n] = 1.0
            adj[(i + 2) % n, i] = 1.0
        return adj


class QuinticCirculantGraph(BaseGraph):
    """
    5-regular circulant graph C(n, {1, 2, n/2}).
    4-regular + antipodal edges.
    Requires n to be even and n >= 6.
    """
    def __init__(self, num_nodes, device=None):
        assert num_nodes % 2 == 0 and num_nodes >= 6, \
            f"QuinticCirculant requires even n >= 6, got {num_nodes}"
        super().__init__(num_nodes=num_nodes, device=device)

    def generate_adjacency_matrix(self):
        n = self.num_nodes
        adj = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        half = n // 2
        for i in range(n):
            adj[i, (i + 1) % n] = 1.0
            adj[(i + 1) % n, i] = 1.0
            adj[i, (i + 2) % n] = 1.0
            adj[(i + 2) % n, i] = 1.0
            adj[i, (i + half) % n] = 1.0
            adj[(i + half) % n, i] = 1.0
        return adj


class Mixed23Graph(BaseGraph):
    """
    Almost-2-regular graph: ring + one antipodal chord.
    Exactly 2 nodes have degree 3, the rest have degree 2.
    """
    def __init__(self, num_nodes, device=None):
        assert num_nodes >= 4, f"Mixed23Graph requires n >= 4, got {num_nodes}"
        super().__init__(num_nodes=num_nodes, device=device)

    def generate_adjacency_matrix(self):
        n = self.num_nodes
        adj = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        # Ring
        for i in range(n):
            adj[i, (i + 1) % n] = 1.0
            adj[(i + 1) % n, i] = 1.0
        # One antipodal chord: 0 <-> n//2
        half = n // 2
        adj[0, half] = 1.0
        adj[half, 0] = 1.0
        return adj


class Mixed34Graph(BaseGraph):
    """
    Almost-3-regular graph: 3-regular + one additional chord.
    Exactly 2 nodes have degree 4, the rest have degree 3.
    """
    def __init__(self, num_nodes, device=None):
        assert num_nodes % 2 == 0 and num_nodes >= 6, \
            f"Mixed34Graph requires even n >= 6, got {num_nodes}"
        super().__init__(num_nodes=num_nodes, device=device)

    def generate_adjacency_matrix(self):
        n = self.num_nodes
        # Start with 3-regular
        adj = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        half = n // 2
        for i in range(n):
            adj[i, (i + 1) % n] = 1.0
            adj[(i + 1) % n, i] = 1.0
            adj[i, (i + half) % n] = 1.0
            adj[(i + half) % n, i] = 1.0
        
        # Add one chord, e.g. 0 <-> n//4 (ensure it's not already connected by ring or antipodal)
        quarter = max(2, n // 4)
        if quarter == half: # Fallback if n is too small to avoid antipodal
            quarter = half - 1 
        adj[0, quarter] = 1.0
        adj[quarter, 0] = 1.0
        return adj
