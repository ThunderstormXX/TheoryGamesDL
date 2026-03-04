import numpy as np

class Learner():
    def __init__(self, action_space_size=2, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, strategy='epsilon_greedy', temperature=1.0):
        self.q_table = {} 
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.strategy = strategy 
        self.temp = temperature
        self.action_space_size = action_space_size
        self.last_state = None
        self.last_action = None

    def step(self):
        pass

    def get_q(self, state):
        state_key = str(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        return self.q_table[state_key]

    def get_probs(self, state):
        q_values = self.get_q(state)
        
        if self.strategy == 'epsilon_greedy':
            probs = np.ones(self.action_space_size) * (self.epsilon / self.action_space_size)
            best_action = np.argmax(q_values)
            probs[best_action] += (1.0 - self.epsilon)
            return probs
            
        elif self.strategy == 'boltzmann':
            # numerical stability
            q_stable = q_values - np.max(q_values)
            exp_q = np.exp(q_stable / self.temp)
            probs = exp_q / np.sum(exp_q)
            return probs
            
        return np.ones(self.action_space_size) / self.action_space_size

    def choose_action(self, state):
        probs = self.get_probs(state)
        return np.random.choice(len(probs), p=probs)


class QLearner(Learner):
    def step(self, state, action, reward, next_state, done=False):
        # Q-Learning update
        # Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
        
        current_q = self.get_q(state)
        next_q = self.get_q(next_state)
        
        max_next_q = np.max(next_q) if not done else 0
        
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q[action]
        
        current_q[action] += self.lr * td_error

class SARSALearner(Learner):
    def step(self, state, action, reward, next_state, next_action, done=False):
        # SARSA update
        # Q(s,a) = Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
        
        current_q = self.get_q(state)
        next_q = self.get_q(next_state)
        
        next_q_val = next_q[next_action] if not done else 0
        
        td_target = reward + self.gamma * next_q_val
        td_error = td_target - current_q[action]
        
        current_q[action] += self.lr * td_error

