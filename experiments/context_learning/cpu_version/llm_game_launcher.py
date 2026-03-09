"""
Extended Game classes that pass rich context to LLM agents.

These subclass the existing Game/PairGame but detect when agents
are LLM-based and pass additional kwargs (global cooperation rate,
per-neighbor actions) to choose_action() and step().
"""

import numpy as np
from game_launcher import Game, PairGame


def _is_llm_agent(agent):
    """Check if agent is an LLM-based agent (has prompt_mode attribute)."""
    return hasattr(agent, "prompt_mode")


class LLMPairGame(PairGame):
    """
    PairGame that provides LLM agents with extra information
    depending on their prompt_mode.
    """

    def __init__(self, graph, learners, reward_model):
        super().__init__(graph, learners, reward_model)
        self.round_num = 0
        # Precompute neighbor lists
        self._neighbors = graph.get_neibhours()

    def round(self):
        adj = self.graph.get_adj_matrix()
        self.round_num += 1

        # Global cooperation rate (from previous round strategies)
        global_coop = float(np.mean(self.strategies))

        # ── 1. Choose actions ──
        transitions = {}
        for i, learner in enumerate(self.learners):
            state = self._get_state(i, adj, self.strategies)

            if _is_llm_agent(learner):
                # Build extra context
                kwargs = {}
                if learner.prompt_mode == "history_and_global":
                    kwargs["global_coop_rate"] = global_coop
                elif learner.prompt_mode == "neighbors_detail":
                    nbs = self._neighbors.get(i, [])
                    kwargs["neighbor_actions"] = {
                        nb: int(self.strategies[nb]) for nb in nbs
                    }
                action = learner.choose_action(state, **kwargs)
            else:
                action = learner.choose_action(state)

            transitions[i] = (state, action)

        # ── Update strategies ──
        for i, (s, a) in transitions.items():
            self.strategies[i] = a

        # ── 2. Rewards ──
        degrees = self.graph.get_degree()
        rewards = self.reward_model.get_all_rewards(self.strategies, adj, degrees)

        # Post-round global coop (after action update)
        post_global_coop = float(np.mean(self.strategies))

        # ── 3. Learn ──
        for i, learner in enumerate(self.learners):
            s, a = transitions[i]
            r = rewards[i]
            next_state = self._get_state(i, adj, self.strategies)

            if _is_llm_agent(learner):
                extra = {
                    "round_num": self.round_num,
                    "global_coop": post_global_coop,
                }
                if learner.prompt_mode == "neighbors_detail":
                    nbs = self._neighbors.get(i, [])
                    extra["neighbor_actions"] = {
                        nb: int(self.strategies[nb]) for nb in nbs
                    }
                learner.step(s, a, r, next_state, **extra)
            elif learner.__class__.__name__ == "SARSALearner":
                next_action = learner.choose_action(next_state)
                learner.step(s, a, r, next_state, next_action)
            else:
                learner.step(s, a, r, next_state)

        self.history.append({
            "strategies": self.strategies.copy(),
            "rewards": rewards,
            "round": self.round_num,
            "global_coop": post_global_coop,
        })