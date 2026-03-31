"""
LLM agent for network Prisoner's Dilemma via OpenAI API.
Drop-in replacement for LLMAgentOpenRouter — identical interface,
same prompt logic, different endpoint and auth header.

Supports multiple prompt modes to study how information
presentation affects LLM cooperation behavior:

  MODE 1 ("history_only"):
      Agent sees its own action history + rewards + neighbor coop count.
      Analogous to Interactive Identity (II) in Q-learning paper.

  MODE 2 ("history_and_global"):
      Same as mode 1, but each round also reports what fraction
      of ALL players in the network cooperated.

  MODE 3 ("neighbors_detail"):
      Agent sees per-neighbor outcomes (who cooperated, who defected)
      from previous rounds. Analogous to Interactive Diversity (ID).

  MODE 4 ("blind"):
      Agent knows nothing about the game structure, other players,
      or the dilemma. Sees only its own past actions (labeled 0/1)
      and the reward received. Pure bandit-style in-context learning.
"""

import numpy as np
import re
import time
import os
import requests
import json


class LLMAgentOpenAI:
    """
    Drop-in replacement for LLMAgentOpenRouter using the OpenAI API directly.

    Interface contract (used by Game classes):
      - choose_action(state) -> int
      - step(state, action, reward, next_state, ...) -> None
      - __class__.__name__ != 'SARSALearner'
    """

    VALID_MODES = ("history_only", "history_and_global", "neighbors_detail", "blind")

    VALID_REASONING = (None, "low", "medium", "high")

    def __init__(
        self,
        agent_id=0,
        degree=None,
        model="gpt-4o-mini",
        temperature=0.0,
        max_history=30,
        api_key=None,
        prompt_mode="history_only",
        neighbor_ids=None,
        verbose=False,
        api_delay=0.0,
        reasoning_effort=None,
        chain_of_thought=False,
    ):
        """
        Args:
            agent_id:         identifier (for logging)
            degree:           number of neighbors (enriches prompt)
            model:            OpenAI model string (e.g. "gpt-4o-mini", "gpt-4o", "o3-mini")
            temperature:      LLM sampling temperature (0 = deterministic)
            max_history:      how many past rounds to include in prompt
            api_key:          OpenAI API key; falls back to OPENAI_API_KEY env
            prompt_mode:      one of VALID_MODES — controls what info goes into prompt
            neighbor_ids:     list of neighbor node ids (needed for "neighbors_detail" mode)
            verbose:          if True, print every prompt and response to stdout
            api_delay:        seconds to sleep before each API call (for rate limiting)
            reasoning_effort: None (disabled) or "low"/"medium"/"high" for o-series models
            chain_of_thought: if True, prompt model to reason in <think> tags before answering
        """
        if prompt_mode not in self.VALID_MODES:
            raise ValueError(
                f"prompt_mode must be one of {self.VALID_MODES}, got '{prompt_mode}'"
            )
        if chain_of_thought and reasoning_effort is not None:
            raise ValueError(
                "Cannot use both chain_of_thought and reasoning_effort simultaneously"
            )

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Set OPENAI_API_KEY env var or pass api_key= to constructor"
            )

        self.model = model
        self.llm_temp = temperature
        self.agent_id = agent_id
        self.degree = degree
        self.max_history = max_history
        self.prompt_mode = prompt_mode
        self.neighbor_ids = neighbor_ids or []

        self.reasoning_effort = reasoning_effort
        self.chain_of_thought = chain_of_thought

        self.history = []          # core of in-context learning
        self.action_space_size = 2
        self.verbose = verbose
        self.api_delay = api_delay

        self._api_url = "https://api.openai.com/v1/chat/completions"
        self._system = self._build_system_prompt()

        # Stats
        self.total_api_calls = 0
        self.total_api_errors = 0

        # File logging
        self._log_file = None

    # ── prompts ─────────────────────────────────────────────

    def _build_system_prompt(self):
        if self.chain_of_thought:
            tail = (
                " First, reason step-by-step about your decision inside <think>...</think> tags."
                " Consider your history, your neighbors' behavior, and the payoff structure."
                " Then, after the closing </think> tag, output ONLY the single digit 0 or 1."
            )
        else:
            tail = " Respond with ONLY the single digit 0 or 1. No explanation."

        if self.prompt_mode == "blind":
            return (
                "You are making repeated decisions. Each round you pick "
                "action 0 or action 1. After each round you receive a numerical "
                "reward. Your goal is to maximize your total reward over many rounds. "
                "You will see your past actions and the rewards you received."
                + tail
            )

        deg = f" You have {self.degree} neighbors." if self.degree else ""

        base = (
            "You are a player in a repeated Prisoner's Dilemma game on a network."
            f"{deg}"
            " Each round you choose: 1 = Cooperate, 0 = Defect."
            " For each neighbor who cooperates, you receive benefit b."
            " If you cooperate, you pay cost c for each of your neighbors."
            " If you defect, you pay nothing but also give nothing to your neighbors."
            " Your reward = b * (number of cooperating neighbors) - c * (your number of neighbors if you cooperate, 0 if you defect)."
            " Your goal is to maximize your own cumulative reward over many rounds."
        )

        if self.prompt_mode == "history_only":
            base += (
                " Each round you will see how many of your neighbors cooperated "
                "and your own past actions and rewards."
            )
        elif self.prompt_mode == "history_and_global":
            base += (
                " Each round you will see how many of your neighbors cooperated, "
                "what fraction of ALL players in the network cooperated, "
                "and your own past actions and rewards."
            )
        elif self.prompt_mode == "neighbors_detail":
            base += (
                " Each round you will see the individual actions of each "
                "of your neighbors (who cooperated, who defected) "
                "and your own past actions and rewards."
            )

        base += tail
        return base

    def _build_user_prompt(self, state, global_coop_rate=None, neighbor_actions=None,
                           is_first_round=False):
        lines = []

        # ── blind mode: minimal info ──
        if self.prompt_mode == "blind":
            if self.history:
                lines.append("Your past rounds:")
                for h in self.history[-self.max_history:]:
                    lines.append(
                        f"  Round {h.get('round', '?')}: "
                        f"action={h['action']}, reward={h['reward']:.2f}"
                    )
                lines.append("")
            lines.append("Choose your next action (0 or 1):")
            return "\n".join(lines)

        # ── other modes: history block — the core of in-context learning ──
        if self.history:
            lines.append("=== Your history (most recent last) ===")
            for h in self.history[-self.max_history:]:
                act_str = "COOPERATE" if h["action"] == 1 else "DEFECT"
                line = f"  Round {h.get('round', '?')}: you={act_str}, reward={h['reward']:.2f}"

                if self.prompt_mode == "history_only":
                    line += f", neighbors_cooperating={h['state']}/{self.degree or '?'}"
                elif self.prompt_mode == "history_and_global":
                    line += f", neighbors_cooperating={h['state']}/{self.degree or '?'}"
                    if "global_coop" in h:
                        line += f", network_cooperation={h['global_coop']:.1%}"
                elif self.prompt_mode == "neighbors_detail":
                    if "neighbor_actions" in h:
                        na = h["neighbor_actions"]
                        detail = ", ".join(
                            f"n{nid}={'C' if a == 1 else 'D'}"
                            for nid, a in na.items()
                        )
                        line += f", [{detail}]"

                lines.append(line)
            lines.append("")

        # ── current observation ──
        lines.append("=== Current round ===")

        if is_first_round:
            lines.append("This is the first round. No actions have been taken yet.")
        elif self.prompt_mode == "history_only":
            if self.degree is not None:
                lines.append(
                    f"{state} out of {self.degree} neighbors are cooperating."
                )
            else:
                lines.append(f"{state} neighbors are cooperating.")

        elif self.prompt_mode == "history_and_global":
            if self.degree is not None:
                lines.append(
                    f"{state} out of {self.degree} neighbors are cooperating."
                )
            else:
                lines.append(f"{state} neighbors are cooperating.")
            if global_coop_rate is not None:
                lines.append(
                    f"Overall network cooperation rate: {global_coop_rate:.1%}"
                )

        elif self.prompt_mode == "neighbors_detail":
            if neighbor_actions is not None:
                details = []
                for nid, act in neighbor_actions.items():
                    details.append(f"  Neighbor {nid}: {'COOPERATE' if act == 1 else 'DEFECT'}")
                lines.append("Your neighbors' current actions:")
                lines.extend(details)
            else:
                lines.append(f"{state} neighbors cooperating.")

        lines.append("")
        lines.append("Your choice (0=Defect, 1=Cooperate):")
        return "\n".join(lines)

    # ── API call ─────────────────────────────────────────────

    def _call_api(self, system_msg, user_msg):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # o-series models require "developer" role instead of "system"
        sys_role = "developer" if self.reasoning_effort is not None else "system"
        messages = [
            {"role": sys_role, "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        if self.reasoning_effort is not None:
            # o-series models: no temperature, max_completion_tokens, reasoning_effort
            payload = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": 500,
                "reasoning_effort": self.reasoning_effort,
            }
        else:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.llm_temp,
                "max_tokens": 500 if self.chain_of_thought else 5,
            }
        resp = requests.post(self._api_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Safely extract content — can be None or missing
        try:
            msg = data["choices"][0]["message"]
            content = msg["content"]
            reasoning_content = msg.get("reasoning_content") or msg.get("reasoning")
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"Unexpected API response: {data}")
        if content is None:
            raise ValueError(f"API returned null content: {data}")
        return content.strip(), reasoning_content

    # ── interface ───────────────────────────────────────────

    def choose_action(self, state, global_coop_rate=None, neighbor_actions=None,
                      is_first_round=False):
        """
        Pick action 0 or 1.

        Extended kwargs are used by the LLM game launcher to pass
        extra context depending on prompt_mode.
        """
        prompt = self._build_user_prompt(state, global_coop_rate, neighbor_actions,
                                         is_first_round=is_first_round)

        for attempt in range(3):
            try:
                if self.api_delay > 0:
                    time.sleep(self.api_delay)
                self.total_api_calls += 1
                text, reasoning_content = self._call_api(self._system, prompt)

                if self.chain_of_thought:
                    cot_reasoning, action = self._parse_cot(text)
                    if cot_reasoning:
                        reasoning_content = cot_reasoning
                else:
                    action = self._parse(text)

                sep = "─" * 60
                if self.reasoning_effort:
                    reasoning_tag = f"  │  reasoning={self.reasoning_effort}"
                elif self.chain_of_thought:
                    reasoning_tag = "  │  chain_of_thought=on"
                else:
                    reasoning_tag = ""
                reasoning_block = (
                    f"[REASONING]\n{reasoning_content}\n\n"
                    if reasoning_content else ""
                )
                log_lines = (
                    f"\n{sep}\n"
                    f"  Agent {self.agent_id}  │  mode={self.prompt_mode}{reasoning_tag}  │  round {len(self.history)+1}\n"
                    f"{sep}\n"
                    f"[SYSTEM]\n{self._system}\n\n"
                    f"[USER]\n{prompt}\n\n"
                    f"{reasoning_block}"
                    f"[LLM RAW] \"{text}\"  →  action={action}\n"
                    f"{sep}\n"
                )
                if self.verbose:
                    print(log_lines)
                if self._log_file is not None:
                    self._log_file.write(log_lines)
                    self._log_file.flush()

                return action
            except Exception as e:
                self.total_api_errors += 1
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"[Agent {self.agent_id}] API fail after 3 tries: {e}")
                    return np.random.choice([0, 1])

    def step(self, state, action, reward, next_state, *args,
             round_num=None, global_coop=None, neighbor_actions=None, **kwargs):
        """
        Store experience for in-context learning.
        """
        entry = {
            "state": state,
            "action": action,
            "reward": reward,
            "round": round_num,
        }
        if global_coop is not None:
            entry["global_coop"] = global_coop
        if neighbor_actions is not None:
            entry["neighbor_actions"] = neighbor_actions

        self.history.append(entry)

    # ── compatibility stubs ──────────────────────────────────

    def get_q(self, state):
        return np.zeros(self.action_space_size)

    def get_probs(self, state):
        return np.array([0.5, 0.5])

    # ── helpers ─────────────────────────────────────────────

    @staticmethod
    def _parse(text):
        m = re.search(r"[01]", text)
        return int(m.group()) if m else np.random.choice([0, 1])

    @staticmethod
    def _parse_cot(text):
        """Parse response with <think>...</think> tags.
        Returns (reasoning, action) tuple."""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
            after_think = text[think_match.end():]
            m = re.search(r"[01]", after_think)
            if m:
                return reasoning, int(m.group())
            # fallback: search entire text
            m = re.search(r"[01]", text)
            if m:
                return reasoning, int(m.group())
            return reasoning, int(np.random.choice([0, 1]))
        # no think tags — fallback
        m = re.search(r"[01]", text)
        if m:
            return None, int(m.group())
        return None, int(np.random.choice([0, 1]))

    def set_log_file(self, path):
        """Open a text file for logging all prompts and responses."""
        if self._log_file is not None:
            self._log_file.close()
        self._log_file = open(path, "a", encoding="utf-8")

    def reset(self):
        """Clear history for a fresh experiment."""
        self.history = []

    def get_stats(self):
        return {
            "agent_id": self.agent_id,
            "api_calls": self.total_api_calls,
            "api_errors": self.total_api_errors,
            "history_len": len(self.history),
        }
