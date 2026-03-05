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

