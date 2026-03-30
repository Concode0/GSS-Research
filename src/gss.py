"""Complete Geometric Superposition Search.

Combines:
- Beam of K hypotheses, each a rotor R_k = exp(-B_k/2) in Cl(3,0)
- LogManifoldManager: merges similar, splits when diversity drops
- NullAwareManager: Cl(1,1) causal frontier detection
- EntropyGatedAttention: suppresses disordered beams
- Selection: Q[s,a] + tanh(c * ||B|| / sqrt(N))
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.embeddings import alg30, alg11, BV_IDX, relational_bv_norm
from layers import EntropyGatedAttention


# ---------------------------------------------------------------------------
# Dict-based Q/N table (scales to any state space)
# ---------------------------------------------------------------------------

class QTable:
    """Sparse Q-table backed by defaultdict. Works for huge state spaces."""

    def __init__(self, n_actions: int):
        self.na = n_actions
        self._q: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(self.na))
        self._n: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(self.na, dtype=int))

    def get_q(self, sid: int) -> np.ndarray:
        return self._q[sid]

    def get_n(self, sid: int, a: int) -> int:
        return int(self._n[sid][a])

    def update(self, sid: int, a: int, target: float):
        self._n[sid][a] += 1
        lr = 1.0 / self._n[sid][a]
        self._q[sid][a] += lr * (target - self._q[sid][a])

    def max_q(self, sid: int) -> float:
        return float(self._q[sid].max()) if sid in self._q else 0.0

    def copy(self) -> "QTable":
        c = QTable(self.na)
        for sid, q in self._q.items():
            c._q[sid] = q.copy()
        for sid, n in self._n.items():
            c._n[sid] = n.copy()
        return c

    def merge_with(self, other: "QTable"):
        """Average Q-values, take max N."""
        all_sids = set(self._q.keys()) | set(other._q.keys())
        for sid in all_sids:
            self._q[sid] = (self._q[sid] + other._q[sid]) / 2.0
            self._n[sid] = np.maximum(self._n[sid], other._n[sid])


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------

class Hypothesis:
    def __init__(self, bivector: torch.Tensor, n_actions: int, hyp_id: int = 0):
        self.bivector = bivector
        self.table = QTable(n_actions)
        self.id = hyp_id
        self.total_reward = 0.0


# ---------------------------------------------------------------------------
# LogManifoldManager
# ---------------------------------------------------------------------------

class LogManifoldManager:
    """Manages beam hypotheses on the bivector (log-rotor) manifold."""

    def __init__(self, tau_collapse: float = 0.3, tau_min_diversity: float = 0.2,
                 max_beam: int = 8, min_beam: int = 2):
        self.tau_c = tau_collapse
        self.tau_min_div = tau_min_diversity
        self.max_beam = max_beam
        self.min_beam = min_beam

    @staticmethod
    def bv_distance(Bi: torch.Tensor, Bj: torch.Tensor) -> float:
        return (Bi - Bj)[BV_IDX].norm().item()

    def mean_diversity(self, bivectors: List[torch.Tensor]) -> float:
        if len(bivectors) < 2:
            return 0.0
        dists = []
        for i in range(len(bivectors)):
            for j in range(i + 1, len(bivectors)):
                dists.append(self.bv_distance(bivectors[i], bivectors[j]))
        return float(np.mean(dists))

    def step(self, hypotheses: List[Hypothesis],
             ) -> Tuple[List[Hypothesis], List[tuple], List[tuple]]:
        merge_events = []
        skip = set()
        for i in range(len(hypotheses)):
            if i in skip:
                continue
            for j in range(i + 1, len(hypotheses)):
                if j in skip:
                    continue
                d = self.bv_distance(hypotheses[i].bivector, hypotheses[j].bivector)
                if d < self.tau_c:
                    hypotheses[i].bivector = (hypotheses[i].bivector + hypotheses[j].bivector) / 2.0
                    hypotheses[i].table.merge_with(hypotheses[j].table)
                    skip.add(j)
                    merge_events.append((i, j, d))

        merged = [h for idx, h in enumerate(hypotheses) if idx not in skip]

        split_events = []
        if len(merged) >= 2:
            div = self.mean_diversity([h.bivector for h in merged])
            if div < self.tau_min_div and len(merged) < self.max_beam:
                scores = [h.total_reward for h in merged]
                best_idx = int(np.argmax(scores))
                new_bv = merged[best_idx].bivector.clone()
                new_bv[BV_IDX] += torch.randn(len(BV_IDX)) * 0.15
                new_hyp = Hypothesis(new_bv, merged[best_idx].table.na, hyp_id=-1)
                new_hyp.table = merged[best_idx].table.copy()
                merged.append(new_hyp)
                split_events.append((best_idx, div))

        return merged, merge_events, split_events


# ---------------------------------------------------------------------------
# NullAwareManager
# ---------------------------------------------------------------------------

class NullAwareManager:
    """Detects null boundary crossings in Cl(1,1) for frontier expansion."""

    def __init__(self, null_eps: float = 0.15):
        self.null_eps = null_eps

    def project_to_search(self, bv: torch.Tensor) -> torch.Tensor:
        bv_comps = bv[BV_IDX]
        a = bv_comps.norm()
        b = bv_comps.std() + 0.01
        h = torch.zeros(alg11.dim)
        h[1] = a
        h[2] = b
        return h

    def classify(self, bv: torch.Tensor) -> Tuple[float, str]:
        h = self.project_to_search(bv)
        nsq = alg11.norm_sq(h).item()
        if abs(nsq) < self.null_eps:
            return nsq, 'null'
        elif nsq > 0:
            return nsq, 'timelike'
        else:
            return nsq, 'spacelike'

    def frontier_check(self, bivectors: List[torch.Tensor],
                       ) -> List[Tuple[bool, float, str]]:
        results = []
        for bv in bivectors:
            nsq, label = self.classify(bv)
            results.append((label == 'null', nsq, label))
        return results


# ---------------------------------------------------------------------------
# CompleteGSS
# ---------------------------------------------------------------------------

class CompleteGSS:
    """Complete Geometric Superposition Search.

    Beam of rotor hypotheses + LogManifoldManager + NullAwareManager + EGA.
    Supports single-agent and adversarial (minimax) modes.
    """

    def __init__(self, num_states: int, num_actions: int,
                 state_mvs: torch.Tensor,
                 beam_size: int = 4,
                 c_explore: float = 1.0,
                 gamma: float = 0.99,
                 adversarial: bool = False):
        self.ns = num_states
        self.na = num_actions
        self.state_mvs = state_mvs
        self.c = c_explore
        self.gamma = gamma
        self.adversarial = adversarial

        # Initialize beam
        self.beam: List[Hypothesis] = []
        self._next_id = 0
        for _ in range(beam_size):
            bv = torch.zeros(alg30.dim)
            bv[BV_IDX] = torch.randn(len(BV_IDX)) * 0.3
            self.beam.append(Hypothesis(bv, num_actions, self._next_id))
            self._next_id += 1

        # Managers
        self.log_mgr = LogManifoldManager(
            tau_collapse=0.3, tau_min_diversity=0.2,
            max_beam=beam_size * 2, min_beam=2,
        )
        self.null_mgr = NullAwareManager(null_eps=0.15)

        # Entropy gating
        self.ega = EntropyGatedAttention(alg30, channels=1, num_heads=1,
                                          eta=1.0, H_base=0.5)
        self.ega.eval()

        # Step counter for merge warmup
        self._step = 0

        # State visit tracking (for heatmaps)
        self.state_visits: Dict[int, int] = defaultdict(int)

        # Logging
        self.log: Dict[str, list] = {
            'beam_sizes': [], 'merge_events': [], 'split_events': [],
            'frontier_events': [], 'causal_labels': [],
            'best_scores': [], 'diversity': [],
        }

    def _get_state_mv(self, state_id: int) -> torch.Tensor:
        if state_id < self.state_mvs.shape[0]:
            return self.state_mvs[state_id]
        return self.state_mvs[state_id % self.state_mvs.shape[0]]

    def _score_hypothesis(self, hyp: Hypothesis, state_id: int,
                          legal: Optional[List[int]] = None) -> np.ndarray:
        actions = legal if legal is not None else list(range(self.na))
        scores = np.full(self.na, -1e9)
        state_mv = self._get_state_mv(state_id)

        # Compute rotor and rotated state once per hypothesis
        R = alg30.exp(-hyp.bivector / 2)
        state_2d = state_mv.unsqueeze(0).unsqueeze(0)
        R_2d = R.unsqueeze(0)
        with torch.no_grad():
            rotated = alg30.sandwich_product(R_2d, state_2d).squeeze()

        bv = relational_bv_norm(rotated, state_mv)

        for a in actions:
            n = max(1, hyp.table.get_n(state_id, a))
            explore = math.tanh(self.c * bv / math.sqrt(n))
            scores[a] = hyp.table.get_q(state_id)[a] + explore

        return scores

    def select_action(self, state_id: int,
                      legal: Optional[List[int]] = None) -> Tuple[int, int]:
        best_score = -float('inf')
        best_action = 0
        best_hyp_idx = 0

        for i, hyp in enumerate(self.beam):
            scores = self._score_hypothesis(hyp, state_id, legal)
            max_score = scores.max()
            if max_score > best_score:
                best_score = max_score
                best_action = int(np.argmax(scores))
                best_hyp_idx = i

        self.log['best_scores'].append(best_score)
        return best_action, best_hyp_idx

    def update(self, state_id: int, action: int, reward: float,
               next_state_id: int, done: bool):
        self._step += 1
        self.state_visits[state_id] += 1

        for hyp in self.beam:
            target = reward + (0.0 if done else self.gamma * hyp.table.max_q(next_state_id))
            hyp.table.update(state_id, action, target)
            hyp.total_reward += reward

            # Decay bivector
            n = hyp.table.get_n(state_id, action)
            decay = max(0.99, 1.0 - 0.01 / math.sqrt(max(1, n)))
            hyp.bivector[BV_IDX] *= decay

        # NullAwareManager: frontier check
        frontier_results = self.null_mgr.frontier_check(
            [h.bivector for h in self.beam])
        labels = [r[2] for r in frontier_results]
        self.log['causal_labels'].append(labels)

        # Only run managers after warmup (let hypotheses differentiate first)
        if self._step > 50:
            for i, (is_frontier, nsq, label) in enumerate(frontier_results):
                if is_frontier and len(self.beam) < self.log_mgr.max_beam:
                    new_bv = self.beam[i].bivector.clone()
                    new_bv[BV_IDX] += torch.randn(len(BV_IDX)) * 0.1
                    new_hyp = Hypothesis(new_bv, self.na, self._next_id)
                    new_hyp.table = self.beam[i].table.copy()
                    self._next_id += 1
                    self.beam.append(new_hyp)
                    self.log['frontier_events'].append(len(self.log['beam_sizes']))

            # LogManifoldManager: merge/split
            self.beam, merges, splits = self.log_mgr.step(self.beam)

            for h in self.beam:
                if h.id == -1:
                    h.id = self._next_id
                    self._next_id += 1

            for _ in merges:
                self.log['merge_events'].append(len(self.log['beam_sizes']))
            for _ in splits:
                self.log['split_events'].append(len(self.log['beam_sizes']))

        self.log['beam_sizes'].append(len(self.beam))
        self.log['diversity'].append(
            self.log_mgr.mean_diversity([h.bivector for h in self.beam]))

    def solve_single_agent(self, env, n_episodes: int, max_steps: int) -> dict:
        episode_rewards = []
        episode_steps = []

        for ep in range(n_episodes):
            state = env.reset()
            sid = env.state_id(state)
            total_reward = 0.0
            for step in range(max_steps):
                legal = env.legal_actions(state)
                action, _ = self.select_action(sid, legal)
                next_state, reward, done = env.step(action)
                next_sid = env.state_id(next_state)
                self.update(sid, action, reward, next_sid, done)
                state = next_state
                sid = next_sid
                total_reward += reward
                if done:
                    break
            episode_rewards.append(total_reward)
            episode_steps.append(step + 1)

        return {
            'rewards': episode_rewards,
            'steps': episode_steps,
            'log': self.log,
        }

    def solve_adversarial(self, env, opponent_fn, n_games: int,
                          max_steps: int = 200) -> dict:
        wins = 0
        losses = 0
        draws = 0
        game_rewards = []

        for game in range(n_games):
            state = env.reset()
            gss_player = game % 2
            total_reward = 0.0

            for step in range(max_steps):
                sid = env.state_id(state)
                legal = env.legal_actions(state)
                if not legal:
                    draws += 1
                    break

                current_player = getattr(env, 'current_player', 0)

                if current_player == gss_player:
                    action, _ = self.select_action(sid, legal)
                else:
                    action = opponent_fn(env, legal)

                next_state, reward, done = env.step(action)
                next_sid = env.state_id(next_state)

                # Only update Q on GSS's own moves to avoid contradictory values
                gss_reward = reward if gss_player == 0 else -reward
                if current_player == gss_player:
                    self.update(sid, action, gss_reward, next_sid, done)

                state = next_state
                total_reward += gss_reward

                if done:
                    if gss_reward > 0:
                        wins += 1
                    elif gss_reward < 0:
                        losses += 1
                    else:
                        draws += 1
                    break
            else:
                draws += 1

            game_rewards.append(total_reward)

        return {
            'rewards': game_rewards,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / max(1, n_games),
            'log': self.log,
        }


# ---------------------------------------------------------------------------
# GSS-Lite: single hypothesis, no managers (for ablation)
# ---------------------------------------------------------------------------

class GSSLite:
    """Single-hypothesis GSS without managers. For ablation comparison."""

    def __init__(self, num_states: int, num_actions: int,
                 state_mvs: torch.Tensor,
                 c_explore: float = 1.0, gamma: float = 0.99):
        self.na = num_actions
        self.state_mvs = state_mvs
        self.c = c_explore
        self.gamma = gamma
        self.table = QTable(num_actions)

        self.bivector = torch.zeros(alg30.dim)
        self.bivector[BV_IDX] = torch.randn(len(BV_IDX)) * 0.3

    def select_action(self, state_id: int,
                      legal: Optional[List[int]] = None) -> int:
        actions = legal if legal is not None else list(range(self.na))
        scores = np.full(self.na, -1e9)
        state_mv = self.state_mvs[state_id % self.state_mvs.shape[0]]

        for a in actions:
            n = max(1, self.table.get_n(state_id, a))
            bv = relational_bv_norm(state_mv, state_mv)
            explore = math.tanh(self.c * bv / math.sqrt(n))
            scores[a] = self.table.get_q(state_id)[a] + explore

        return int(np.argmax(scores))

    def update(self, state_id: int, action: int, reward: float,
               next_state_id: int, done: bool):
        target = reward + (0.0 if done else self.gamma * self.table.max_q(next_state_id))
        self.table.update(state_id, action, target)

    def solve(self, env, n_episodes: int, max_steps: int) -> dict:
        episode_rewards = []
        episode_steps = []

        for ep in range(n_episodes):
            state = env.reset()
            sid = env.state_id(state)
            total_reward = 0.0
            for step in range(max_steps):
                action = self.select_action(sid)
                next_state, reward, done = env.step(action)
                next_sid = env.state_id(next_state)
                self.update(sid, action, reward, next_sid, done)
                state = next_state
                sid = next_sid
                total_reward += reward
                if done:
                    break
            episode_rewards.append(total_reward)
            episode_steps.append(step + 1)

        return {'rewards': episode_rewards, 'steps': episode_steps}
