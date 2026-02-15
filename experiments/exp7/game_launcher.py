import numpy as np

class Game():
    def __init__(self, graph, learners, reward_model):
        self.graph = graph
        self.learners = learners
        self.reward_model = reward_model
        # Strategies 0 or 1.
        self.strategies = np.random.randint(0, 2, size=len(learners))
        self.history = []

    def round(self):
        pass

    def result(self):
        return self.history
    
    def _get_state(self, agent_id, adj, strategies):
        # State: number of cooperating neighbors
        # adj is the relevant adjacency matrix for the current game round
        neighbors = np.nonzero(adj[agent_id])[0]
        if len(neighbors) == 0:
            return 0
        return int(np.sum(strategies[neighbors]))

class PairGame(Game):
    def round(self):
        adj = self.graph.get_adj_matrix()
        
        # 1. Choose actions
        transitions = {}
        for i, learner in enumerate(self.learners):
            state = self._get_state(i, adj, self.strategies)
            action = learner.choose_action(state)
            transitions[i] = (state, action)
            
        # Update strategies
        for i, (s, a) in transitions.items():
            self.strategies[i] = a
            
        # 2. Rewards
        degrees = self.graph.get_degree()
        rewards = self.reward_model.get_all_rewards(self.strategies, adj, degrees)
        
        # 3. Learn
        for i, learner in enumerate(self.learners):
            s, a = transitions[i]
            r = rewards[i]
            next_state = self._get_state(i, adj, self.strategies)
            
            # For SARSA next action needed
            next_action = learner.choose_action(next_state)
            
            if learner.__class__.__name__ == 'SARSALearner':
                 learner.step(s, a, r, next_state, next_action)
            else:
                 learner.step(s, a, r, next_state)
            
        self.history.append({
            'strategies': self.strategies.copy(),
            'rewards': rewards
        })

class MonteKarloPairGame(Game):
    def __init__(self, graph, learners, reward_model, k_anchors=1):
        super().__init__(graph, learners, reward_model)
        self.k = k_anchors

    def round(self):
        adj = self.graph.get_adj_matrix()
        n_agents = len(self.learners)
        
        # Select anchors
        anchors = np.random.choice(n_agents, self.k, replace=False)
        active_nodes = set(anchors)
        for a in anchors:
            active_nodes.update(np.nonzero(adj[a])[0])
        active_nodes = list(active_nodes)
        
        # Induced Subgraph
        # Actions
        transitions = {} 
        for i in active_nodes:
            # We use global adj to determine state? 
            # Or induced sub-adj? 
            # "Game happens on induced subgraph" -> Local state.
            
            # We need sub_adj first to get state? 
            # Yes. Constructing mask.
            mask = np.zeros_like(adj)
            ixgrid = np.ix_(active_nodes, active_nodes)
            mask[ixgrid] = 1
            sub_adj = adj * mask
            
            state = self._get_state(i, sub_adj, self.strategies)
            action = self.learners[i].choose_action(state)
            transitions[i] = (state, action)
        
        # Update active strategies
        for i, (s, a) in transitions.items():
            self.strategies[i] = a
            
        # Rewards (Recalculate sub_adj/mask is same)
        mask = np.zeros_like(adj)
        ixgrid = np.ix_(active_nodes, active_nodes)
        mask[ixgrid] = 1
        sub_adj = adj * mask
        sub_degrees = np.sum(sub_adj, axis=1)
        
        full_rewards = self.reward_model.get_all_rewards(self.strategies, sub_adj, sub_degrees)
        
        # Learn
        for i in active_nodes:
            s, a = transitions[i]
            r = full_rewards[i]
            next_state = self._get_state(i, sub_adj, self.strategies)
            next_action = self.learners[i].choose_action(next_state)
            
            if self.learners[i].__class__.__name__ == 'SARSALearner':
                 self.learners[i].step(s, a, r, next_state, next_action)
            else:
                 self.learners[i].step(s, a, r, next_state)
                 
        self.history.append({
            'active_nodes': active_nodes,
            'strategies': self.strategies.copy()
        })

class MonteKarloNotPairGame(Game):
    def __init__(self, graph, learners, reward_model, k_anchors=1):
        super().__init__(graph, learners, reward_model)
        self.k = k_anchors

    def round(self):
        adj = self.graph.get_adj_matrix()
        n_agents = len(self.learners)
        
        # Select Anchors
        anchors = np.random.choice(n_agents, self.k, replace=False)
        active_nodes = set(anchors)
        for a in anchors:
            active_nodes.update(np.nonzero(adj[a])[0])
        active_nodes = list(active_nodes)
        
        # Clique Topology
        clique_adj = np.zeros_like(adj)
        ixgrid = np.ix_(active_nodes, active_nodes)
        clique_adj[ixgrid] = 1
        np.fill_diagonal(clique_adj, 0)
        
        transitions = {}
        for i in active_nodes:
            state = self._get_state(i, clique_adj, self.strategies)
            action = self.learners[i].choose_action(state)
            transitions[i] = (state, action)
            
        for i, (s, a) in transitions.items():
            self.strategies[i] = a
            
        clique_degrees = np.sum(clique_adj, axis=1)
        full_rewards = self.reward_model.get_all_rewards(self.strategies, clique_adj, clique_degrees)
        
        for i in active_nodes:
            s, a = transitions[i]
            r = full_rewards[i]
            next_state = self._get_state(i, clique_adj, self.strategies)
            next_action = self.learners[i].choose_action(next_state)
            
            if self.learners[i].__class__.__name__ == 'SARSALearner':
                 self.learners[i].step(s, a, r, next_state, next_action)
            else:
                 self.learners[i].step(s, a, r, next_state)

        self.history.append({
            'active_nodes': active_nodes,
            'strategies': self.strategies.copy()
        })

