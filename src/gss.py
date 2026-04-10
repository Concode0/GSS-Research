"""Complete Geometric Superposition Search.

Combines:
- Beam of K hypotheses, each a rotor R_k = exp(-B_k/2) in Cl(3,0)
- LogManifoldManager: merges similar, splits when diversity drops
- NullAwareManager: Cl(1,1) causal frontier detection
- EntropyGatedAttention: computes beam entropy H and gating lam for coherence
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

# Pin(3,0) grade decomposition
EVEN_IDX = []  # grades 0, 2 (rotor/rotation part)
ODD_IDX = []   # grades 1, 3 (reflection/opponent part)
for g in [0, 2]:
    EVEN_IDX.extend(alg30.grade_masks[g].nonzero(as_tuple=False).squeeze(-1).tolist())
for g in [1, 3]:
    ODD_IDX.extend(alg30.grade_masks[g].nonzero(as_tuple=False).squeeze(-1).tolist())
EVEN_IDX = torch.tensor(EVEN_IDX)
ODD_IDX = torch.tensor(ODD_IDX)
G1_IDX = alg30.grade_masks[1].nonzero(as_tuple=False).squeeze(-1)  # grade-1 vectors
G3_IDX = alg30.grade_masks[3].nonzero(as_tuple=False).squeeze(-1)  # grade-3 pseudoscalar


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

    def q_similarity(self, other: "QTable", states=None) -> float:
        """Cosine similarity of Q-vectors over shared states."""
        if states is None:
            states = set(self._q.keys()) & set(other._q.keys())
        if not states:
            return 0.0
        a = np.concatenate([self._q[s] for s in states])
        b = np.concatenate([other._q[s] for s in states])
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------

class Hypothesis:
    def __init__(self, bivector: torch.Tensor, n_actions: int, hyp_id: int = 0):
        self.bivector = bivector
        self.table = QTable(n_actions)
        self.id = hyp_id
        self.total_reward = 0.0


class PinHypothesis:
    """Hypothesis as a full Pin(3,0) versor.

    versor[EVEN_IDX] = rotation part (own strategy, Spin group)
    versor[ODD_IDX]  = reflection part (opponent model, extends to Pin group)
    """
    def __init__(self, versor: torch.Tensor, n_actions: int, hyp_id: int = 0):
        self.versor = versor
        self.table = QTable(n_actions)
        self.id = hyp_id
        self.total_reward = 0.0

    @property
    def bivector(self):
        """Alias for compatibility with existing code paths."""
        return self.versor

    @property
    def rotor_part(self):
        """Even subalgebra: grades 0+2 (rotation)."""
        r = torch.zeros_like(self.versor)
        r[EVEN_IDX] = self.versor[EVEN_IDX]
        return r

    @property
    def reflection_part(self):
        """Odd part: grades 1+3 (reflection/opponent model)."""
        r = torch.zeros_like(self.versor)
        r[ODD_IDX] = self.versor[ODD_IDX]
        return r


# ---------------------------------------------------------------------------
# LogManifoldManager
# ---------------------------------------------------------------------------

class LogManifoldManager:
    """Manages beam hypotheses on the bivector (log-rotor) manifold.

    Supports adaptive threshold adjustment based on diversity trends.
    """

    def __init__(self, tau_collapse: float = 0.3, tau_min_diversity: float = 0.2,
                 max_beam: int = 8, min_beam: int = 2, adaptive: bool = True):
        self.tau_c_init = tau_collapse
        self.tau_min_div_init = tau_min_diversity
        self.tau_c = tau_collapse
        self.tau_min_div = tau_min_diversity
        self.max_beam = max_beam
        self.min_beam = min_beam
        self.adaptive = adaptive
        self._diversity_history: List[float] = []

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

    def _adapt_thresholds(self):
        """Adjust merge/split thresholds based on diversity trends."""
        if not self.adaptive or len(self._diversity_history) < 20:
            return
        recent = self._diversity_history[-20:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        if trend < 0:
            # Diversity declining -> relax merge (less aggressive), tighten split (split sooner)
            self.tau_c *= 0.99
            self.tau_min_div = min(self.tau_min_div_init * 1.5, self.tau_min_div * 1.01)
        else:
            # Diversity growing -> tighten merge
            self.tau_c = min(self.tau_c_init * 1.5, self.tau_c * 1.01)

    def step(self, hypotheses: List[Hypothesis],
             ) -> Tuple[List[Hypothesis], List[tuple], List[tuple]]:
        # Adaptive threshold adjustment
        div = self.mean_diversity([h.bivector for h in hypotheses])
        self._diversity_history.append(div)
        self._adapt_thresholds()

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
        if len(merged) < self.min_beam or (
                div < self.tau_min_div and len(merged) < self.max_beam):
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
# FunctionalManager (for adversarial mode)
# ---------------------------------------------------------------------------

class FunctionalManager:
    """Manages hypotheses based on Q-value similarity, not geometric distance.

    - Merge: when two hypotheses produce similar Q-values (cosine sim > threshold)
    - Split: when a hypothesis underperforms the beam mean
    - Divergent splits perturb the ODD part (opponent model) more than EVEN part
    """

    def __init__(self, max_beam: int = 8, min_beam: int = 2,
                 merge_sim_threshold: float = 0.95,
                 split_reward_threshold: float = 0.5):
        self.max_beam = max_beam
        self.min_beam = min_beam
        self.merge_sim = merge_sim_threshold
        self.split_reward_ratio = split_reward_threshold

    def step(self, hypotheses, visited_states=None):
        merge_events = []
        skip = set()

        for i in range(len(hypotheses)):
            if i in skip:
                continue
            for j in range(i + 1, len(hypotheses)):
                if j in skip:
                    continue
                sim = hypotheses[i].table.q_similarity(
                    hypotheses[j].table, visited_states)
                if sim > self.merge_sim:
                    if hypotheses[i].total_reward >= hypotheses[j].total_reward:
                        skip.add(j)
                        merge_events.append((i, j, sim))
                    else:
                        skip.add(i)
                        merge_events.append((j, i, sim))
                        break

        merged = [h for idx, h in enumerate(hypotheses) if idx not in skip]

        split_events = []
        if len(merged) < self.min_beam:
            best = max(merged, key=lambda h: h.total_reward)
            new_hyp = self._create_divergent(best)
            merged.append(new_hyp)
            split_events.append(('force', best.id))
        elif len(merged) >= 2 and len(merged) < self.max_beam:
            rewards = [h.total_reward for h in merged]
            mean_r = np.mean(rewards)
            worst_idx = int(np.argmin(rewards))
            best_idx = int(np.argmax(rewards))
            if mean_r > 0 and rewards[worst_idx] < mean_r * self.split_reward_ratio:
                new_hyp = self._create_divergent(merged[best_idx])
                merged[worst_idx] = new_hyp
                split_events.append(('replace', best_idx))

        return merged, merge_events, split_events

    def _create_divergent(self, source):
        """Create a divergent hypothesis. Perturbs odd (opponent model) part more."""
        new_v = source.versor.clone() if hasattr(source, 'versor') else source.bivector.clone()
        new_v[ODD_IDX] += torch.randn(len(ODD_IDX)) * 0.3
        new_v[BV_IDX] += torch.randn(len(BV_IDX)) * 0.1
        hyp = PinHypothesis(new_v, source.table.na, hyp_id=-1)
        hyp.table = source.table.copy()
        return hyp


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
        self.beam = []
        self._next_id = 0
        for _ in range(beam_size):
            if self.adversarial:
                v = torch.zeros(alg30.dim)
                v[BV_IDX] = torch.randn(len(BV_IDX)) * 0.3
                v[G1_IDX] = torch.randn(len(G1_IDX)) * 0.2
                self.beam.append(PinHypothesis(v, num_actions, self._next_id))
            else:
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
        if self.adversarial:
            self.func_mgr = FunctionalManager(
                max_beam=beam_size * 2, min_beam=2,
                merge_sim_threshold=0.95, split_reward_threshold=0.5)

        # Entropy gating
        self.ega = EntropyGatedAttention(alg30, channels=1, num_heads=1,
                                          eta=1.0, H_base=0.5)
        self.ega.eval()

        # Step counter for merge warmup
        self._step = 0

        # State visit tracking (for heatmaps)
        self.state_visits: Dict[int, int] = defaultdict(int)
        self._visited_states: set = set()

        # EGA state (cached for use in update/momentum)
        self._ega_H = 0.0
        self._ega_lam = 0.0

        # Logging
        self.log: Dict[str, list] = {
            'beam_sizes': [], 'merge_events': [], 'split_events': [],
            'frontier_events': [], 'causal_labels': [],
            'best_scores': [], 'diversity': [],
            'ega_H': [], 'ega_lam': [],
            'robustness': [], 'even_norms': [], 'odd_norms': [],
        }

    def _get_state_mv(self, state_id: int) -> torch.Tensor:
        if state_id < self.state_mvs.shape[0]:
            return self.state_mvs[state_id]
        return self.state_mvs[state_id % self.state_mvs.shape[0]]

    def _compute_robustness(self, hyp, state_id: int) -> float:
        """Robustness = stability of state under reflection by opponent model.

        robustness = 1.0 - ||state - Reflect(state)|| / ||state||
        High = minimax-safe, Low = vulnerable to opponent's reflection.
        """
        state_mv = self._get_state_mv(state_id)
        state_norm = state_mv.norm().item()
        if state_norm < 1e-8:
            return 1.0

        n_vec = hyp.versor.clone() if hasattr(hyp, 'versor') else hyp.bivector.clone()
        n_vec[EVEN_IDX] = 0  # keep only odd part
        n_sq = alg30.norm_sq(n_vec.unsqueeze(0)).squeeze().item()

        if abs(n_sq) < 1e-8:
            return 1.0  # no opponent model yet

        n_inv = n_vec / n_sq
        n_2d = n_vec.unsqueeze(0)
        state_2d = state_mv.unsqueeze(0)
        with torch.no_grad():
            nx = alg30.geometric_product(n_2d, state_2d)
            reflected = alg30.geometric_product(nx, n_inv.unsqueeze(0)).squeeze(0)

        diff_norm = (state_mv - reflected).norm().item()
        robustness = 1.0 - min(1.0, diff_norm / state_norm)
        return max(0.0, robustness)

    def _score_hypothesis(self, hyp, state_id: int,
                          legal: Optional[List[int]] = None) -> np.ndarray:
        actions = legal if legal is not None else list(range(self.na))
        scores = np.full(self.na, -1e9)
        state_mv = self._get_state_mv(state_id)

        # For PinHypothesis: use even part only as rotor
        if self.adversarial and hasattr(hyp, 'rotor_part'):
            bv = hyp.rotor_part
        else:
            bv = hyp.bivector

        R = alg30.exp(-bv / 2)
        state_2d = state_mv.unsqueeze(0).unsqueeze(0)
        R_2d = R.unsqueeze(0)
        with torch.no_grad():
            rotated = alg30.sandwich_product(R_2d, state_2d).squeeze()

        bv_norm = relational_bv_norm(rotated, state_mv)

        # Robustness-modulated exploration in adversarial mode
        if self.adversarial and hasattr(hyp, 'versor'):
            robustness = self._compute_robustness(hyp, state_id)
            explore_factor = 1.5 * (2.0 - robustness)
        else:
            explore_factor = 1.0

        for a in actions:
            n = max(1, hyp.table.get_n(state_id, a))
            explore = math.tanh(self.c * explore_factor * bv_norm / math.sqrt(n))
            scores[a] = hyp.table.get_q(state_id)[a] + explore

        return scores

    def select_action(self, state_id: int,
                      legal: Optional[List[int]] = None) -> Tuple[int, int]:
        # EGA: compute entropy gating on beam bivectors
        beam_mvs = torch.stack([h.bivector for h in self.beam])  # [K, 8]
        beam_4d = beam_mvs.unsqueeze(0).unsqueeze(2)  # [1, K, 1, 8]
        with torch.no_grad():
            _, H, lam = self.ega(beam_4d, return_gating=True)
        self._ega_H = H.item()
        self._ega_lam = lam.item()
        self.log['ega_H'].append(self._ega_H)
        self.log['ega_lam'].append(self._ega_lam)

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

        # 2.1: Performance-weighted hypothesis updates
        rewards = np.array([h.total_reward for h in self.beam])
        if len(self.beam) > 1 and rewards.std() > 1e-6:
            weights = np.exp((rewards - rewards.max()) / max(1.0, rewards.std()))
            weights /= weights.sum()
        else:
            weights = np.ones(len(self.beam)) / len(self.beam)

        for hyp, w in zip(self.beam, weights):
            # Probabilistic update: better hypotheses more likely to be updated
            if random.random() < w * len(self.beam):
                target = reward + (0.0 if done else self.gamma * hyp.table.max_q(next_state_id))
                hyp.table.update(state_id, action, target)

            # Temporal decay on total_reward so recent performance dominates
            hyp.total_reward = hyp.total_reward * 0.999 + reward

            # 2.2: EGA-driven momentum decay
            n = hyp.table.get_n(state_id, action)
            base_decay = max(0.99, 1.0 - 0.01 / math.sqrt(max(1, n)))
            # EGA lam: high when disordered -> faster decay; low when coherent -> preserve momentum
            beta = 0.3  # coupling constant
            ega_mod = 1.0 - beta * self._ega_lam
            # Modulate by hypothesis weight: high-performing + coherent = max momentum
            momentum_factor = 0.5 + 0.5 * w * len(self.beam)
            decay = min(1.0, base_decay * ega_mod * momentum_factor)
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

    def update_adversarial(self, state_id: int, action: int, reward: float,
                           next_state_id: int, done: bool, is_our_turn: bool):
        """Pin-aware adversarial Q-update with minimax value propagation.

        Even part (rotor) decays on own turn. Odd part (reflection) learns
        from opponent observations on opponent turns.
        """
        self._step += 1
        self.state_visits[state_id] += 1
        self._visited_states.add(state_id)

        # Performance weights
        rewards_arr = np.array([h.total_reward for h in self.beam])
        if len(self.beam) > 1 and rewards_arr.std() > 1e-6:
            weights = np.exp((rewards_arr - rewards_arr.max()) / max(1.0, rewards_arr.std()))
            weights /= weights.sum()
        else:
            weights = np.ones(len(self.beam)) / len(self.beam)

        for hyp, w in zip(self.beam, weights):
            if random.random() < w * len(self.beam):
                if is_our_turn:
                    next_val = hyp.table.max_q(next_state_id) if not done else 0.0
                else:
                    next_val = float(hyp.table.get_q(next_state_id).min()) if not done else 0.0
                target = reward + self.gamma * next_val
                hyp.table.update(state_id, action, target)

            hyp.total_reward = hyp.total_reward * 0.999 + reward

            # EGA-driven momentum decay
            n = hyp.table.get_n(state_id, action)
            base_decay = max(0.99, 1.0 - 0.01 / math.sqrt(max(1, n)))
            beta = 0.3
            ega_mod = 1.0 - beta * self._ega_lam
            momentum_factor = 0.5 + 0.5 * w * len(self.beam)
            decay = min(1.0, base_decay * ega_mod * momentum_factor)

            # Pin-aware versor decay
            if hasattr(hyp, 'versor'):
                if is_our_turn:
                    hyp.versor[BV_IDX] *= decay
                    # Prevent total collapse of rotor part
                    even_norm = hyp.versor[BV_IDX].norm().item()
                    if even_norm < 0.05:
                        hyp.versor[BV_IDX] += torch.randn(len(BV_IDX)) * 0.02
                else:
                    # Opponent turn: blend odd part toward observed state direction
                    state_mv = self._get_state_mv(state_id)
                    g1_component = alg30.grade_projection(state_mv, 1)
                    alpha = 0.05
                    hyp.versor[G1_IDX] = (
                        (1 - alpha) * hyp.versor[G1_IDX]
                        + alpha * g1_component[G1_IDX]
                    )
                    hyp.versor[ODD_IDX] *= max(0.995, decay)
            else:
                hyp.bivector[BV_IDX] *= decay

        # Log even/odd norms and robustness
        if self.beam and hasattr(self.beam[0], 'versor'):
            even_norms = [h.versor[EVEN_IDX].norm().item() for h in self.beam]
            odd_norms = [h.versor[ODD_IDX].norm().item() for h in self.beam]
            self.log['even_norms'].append(np.mean(even_norms))
            self.log['odd_norms'].append(np.mean(odd_norms))
            best_hyp = max(self.beam, key=lambda h: h.total_reward)
            self.log['robustness'].append(self._compute_robustness(best_hyp, state_id))

        # FunctionalManager or LogManifoldManager (every 20 steps after warmup)
        if self._step > 50 and self._step % 20 == 0:
            if hasattr(self, 'func_mgr'):
                self.beam, merges, splits = self.func_mgr.step(
                    self.beam, self._visited_states)
            else:
                self.beam, merges, splits = self.log_mgr.step(self.beam)
            for h in self.beam:
                if h.id == -1:
                    h.id = self._next_id
                    self._next_id += 1
            for _ in merges:
                self.log['merge_events'].append(len(self.log['beam_sizes']))
            for _ in splits:
                self.log['split_events'].append(len(self.log['beam_sizes']))

        # NullAwareManager
        bvs = [h.versor if hasattr(h, 'versor') else h.bivector for h in self.beam]
        frontier_results = self.null_mgr.frontier_check(bvs)
        labels = [r[2] for r in frontier_results]
        self.log['causal_labels'].append(labels)

        self.log['beam_sizes'].append(len(self.beam))
        self.log['diversity'].append(
            self.log_mgr.mean_diversity(bvs))

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
                    current_player = getattr(env, 'current_player', 0)
                    if current_player == gss_player:
                        losses += 1
                    else:
                        wins += 1
                    break

                current_player = getattr(env, 'current_player', 0)

                if current_player == gss_player:
                    action, _ = self.select_action(sid, legal)
                else:
                    action = opponent_fn(env, legal)

                next_state, reward, done = env.step(action)
                next_sid = env.state_id(next_state)

                # Minimax update: update on both turns with perspective-aware Q-backup
                gss_reward = reward if gss_player == 0 else -reward
                self.update_adversarial(
                    sid, action, gss_reward, next_sid, done,
                    is_our_turn=(current_player == gss_player)
                )

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

        # Compute rotor and rotated state (mirrors CompleteGSS._score_hypothesis)
        R = alg30.exp(-self.bivector / 2)
        state_2d = state_mv.unsqueeze(0).unsqueeze(0)
        R_2d = R.unsqueeze(0)
        with torch.no_grad():
            rotated = alg30.sandwich_product(R_2d, state_2d).squeeze()
        bv = relational_bv_norm(rotated, state_mv)

        for a in actions:
            n = max(1, self.table.get_n(state_id, a))
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
