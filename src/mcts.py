"""Monte Carlo Tree Search with UCT selection, expansion, rollout, backpropagation.

Supports single-agent MDPs and 2-player adversarial games.
Does NOT use geometric embeddings -- uses only environment interaction.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tree Node
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ('state', 'state_id', 'parent', 'parent_action',
                 'children', 'visits', 'value_sum', 'untried_actions',
                 'player', '_expand_reward', '_expand_done')

    def __init__(self, state: Any, state_id: int,
                 parent: Optional[MCTSNode] = None,
                 parent_action: Optional[int] = None,
                 untried_actions: Optional[List[int]] = None,
                 player: int = 0):
        self.state = state
        self.state_id = state_id
        self.parent = parent
        self.parent_action = parent_action
        self.children: Dict[int, MCTSNode] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.untried_actions = list(untried_actions) if untried_actions else []
        self.player = player
        self._expand_reward = 0.0
        self._expand_done = False

    @property
    def q_value(self) -> float:
        return self.value_sum / max(1, self.visits)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c: float = math.sqrt(2)) -> MCTSNode:
        """UCT selection."""
        best_score = -float('inf')
        best_node = None
        ln_parent = math.log(max(1, self.visits))
        for child in self.children.values():
            if child.visits == 0:
                return child
            uct = child.q_value + c * math.sqrt(ln_parent / child.visits)
            if uct > best_score:
                best_score = uct
                best_node = child
        return best_node

    def best_action(self) -> int:
        """Most-visited child (robust action selection)."""
        if not self.children:
            return 0
        return max(self.children.items(), key=lambda x: x[1].visits)[0]


# ---------------------------------------------------------------------------
# MCTS (single-agent)
# ---------------------------------------------------------------------------

class MCTS:
    """Standard MCTS for single-agent episodic tasks."""

    def __init__(self, n_simulations: int = 50, c_uct: float = math.sqrt(2),
                 rollout_depth: int = 20, gamma: float = 0.99):
        self.n_sims = n_simulations
        self.c = c_uct
        self.rollout_depth = rollout_depth
        self.gamma = gamma

        self.log: Dict[str, list] = {
            'tree_sizes': [],
            'rollout_returns': [],
            'node_visits': defaultdict(int),
        }

    # -- Core MCTS phases --

    def _select(self, node: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        """Phase 1: traverse tree via UCT to a leaf."""
        path = [node]
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.c)
            if node is None:
                break
            path.append(node)
        return node, path

    def _expand(self, node: MCTSNode, env) -> MCTSNode:
        """Phase 2: add one new child."""
        if not node.untried_actions:
            return node
        action = node.untried_actions.pop()
        env_copy = env.clone()
        env_copy.set_state(node.state)
        next_state, reward, done = env_copy.step(action)
        child = MCTSNode(
            state=next_state,
            state_id=env_copy.state_id(next_state),
            parent=node,
            parent_action=action,
            untried_actions=env_copy.legal_actions(next_state) if not done else [],
        )
        child._expand_reward = reward
        child._expand_done = done
        node.children[action] = child
        return child

    def _rollout(self, node: MCTSNode, env) -> float:
        """Phase 3: random rollout from node."""
        if hasattr(node, '_expand_done') and node._expand_done:
            return getattr(node, '_expand_reward', 0.0)

        env_copy = env.clone()
        env_copy.set_state(node.state)
        total_return = getattr(node, '_expand_reward', 0.0)
        discount = self.gamma

        for _ in range(self.rollout_depth):
            legal = env_copy.legal_actions(env_copy.current_state())
            if not legal:
                break
            action = random.choice(legal)
            state, reward, done = env_copy.step(action)
            total_return += discount * reward
            discount *= self.gamma
            if done:
                break

        return total_return

    def _backpropagate(self, path: List[MCTSNode], value: float):
        """Phase 4: propagate value up the path."""
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value *= self.gamma

    def _count_nodes(self, node: MCTSNode) -> int:
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    # -- Public API --

    def search(self, root_state: Any, root_state_id: int, env) -> int:
        """Run simulations and return the best action."""
        root = MCTSNode(
            state=root_state,
            state_id=root_state_id,
            untried_actions=env.legal_actions(root_state),
        )

        for _ in range(self.n_sims):
            leaf, path = self._select(root)
            if leaf.untried_actions:
                leaf = self._expand(leaf, env)
                path.append(leaf)
            value = self._rollout(leaf, env)
            self.log['rollout_returns'].append(value)
            self._backpropagate(path, value)

        self.log['tree_sizes'].append(self._count_nodes(root))
        self.log['node_visits'][root_state_id] += root.visits
        return root.best_action()

    def solve(self, env, n_episodes: int, max_steps: int) -> dict:
        """Run MCTS on a single-agent env. Returns results dict."""
        episode_rewards = []
        episode_steps = []

        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            for step in range(max_steps):
                sid = env.state_id(state)
                action = self.search(state, sid, env)
                state, reward, done = env.step(action)
                total_reward += reward
                if done:
                    break
            episode_rewards.append(total_reward)
            episode_steps.append(step + 1)

        return {
            'rewards': episode_rewards,
            'steps': episode_steps,
            'log': dict(self.log),
        }


# ---------------------------------------------------------------------------
# MCTS for 2-player games
# ---------------------------------------------------------------------------

class MCTSTwoPlayer:
    """MCTS for 2-player zero-sum games with alternating turns."""

    def __init__(self, n_simulations: int = 100, c_uct: float = math.sqrt(2),
                 rollout_depth: int = 50):
        self.n_sims = n_simulations
        self.c = c_uct
        self.rollout_depth = rollout_depth

        self.log: Dict[str, list] = {
            'tree_sizes': [],
            'rollout_returns': [],
            'node_visits': defaultdict(int),
        }

    def _select(self, node: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        path = [node]
        while node.is_fully_expanded() and node.children:
            # Negate UCT for opponent's nodes (minimax)
            if node.player == 0:
                node = node.best_child(self.c)
            else:
                # For opponent: pick child with lowest Q from our perspective
                best_score = -float('inf')
                best_node = None
                ln_parent = math.log(max(1, node.visits))
                for child in node.children.values():
                    if child.visits == 0:
                        best_node = child
                        break
                    # Opponent maximizes negative of our value
                    uct = -child.q_value + self.c * math.sqrt(ln_parent / child.visits)
                    if uct > best_score:
                        best_score = uct
                        best_node = child
                node = best_node
            if node is None:
                break
            path.append(node)
        return node, path

    def _expand(self, node: MCTSNode, env) -> MCTSNode:
        if not node.untried_actions:
            return node
        action = node.untried_actions.pop()
        env_copy = env.clone()
        env_copy.set_state(node.state)
        if hasattr(env_copy, '_current_player'):
            env_copy._current_player = node.player
        elif hasattr(env_copy, 'current_player') and not isinstance(
                type(env_copy).current_player, property):
            env_copy.current_player = node.player
        next_state, reward, done = env_copy.step(action)
        next_player = 1 - node.player
        child = MCTSNode(
            state=next_state,
            state_id=env_copy.state_id(next_state),
            parent=node,
            parent_action=action,
            untried_actions=env_copy.legal_actions(next_state) if not done else [],
            player=next_player,
        )
        child._expand_reward = reward
        child._expand_done = done
        node.children[action] = child
        return child

    def _rollout(self, node: MCTSNode, env) -> float:
        if hasattr(node, '_expand_done') and node._expand_done:
            return getattr(node, '_expand_reward', 0.0)

        env_copy = env.clone()
        env_copy.set_state(node.state)
        if hasattr(env_copy, '_current_player'):
            env_copy._current_player = node.player
        elif hasattr(env_copy, 'current_player') and not isinstance(
                type(env_copy).current_player, property):
            env_copy.current_player = node.player

        for _ in range(self.rollout_depth):
            legal = env_copy.legal_actions(env_copy.current_state())
            if not legal:
                return 0.0
            action = random.choice(legal)
            state, reward, done = env_copy.step(action)
            if done:
                return reward  # from player 0's perspective
        return 0.0

    def _backpropagate(self, path: List[MCTSNode], value: float):
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            # Value is always from player 0's perspective

    def _count_nodes(self, node: MCTSNode) -> int:
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def search(self, root_state: Any, root_state_id: int,
               env, player: int = 0) -> int:
        root = MCTSNode(
            state=root_state,
            state_id=root_state_id,
            untried_actions=env.legal_actions(root_state),
            player=player,
        )

        for _ in range(self.n_sims):
            leaf, path = self._select(root)
            if leaf.untried_actions:
                leaf = self._expand(leaf, env)
                path.append(leaf)
            value = self._rollout(leaf, env)
            self.log['rollout_returns'].append(value)
            self._backpropagate(path, value)

        self.log['tree_sizes'].append(self._count_nodes(root))
        self.log['node_visits'][root_state_id] += root.visits

        # Select: most visited for our turn, least visited for opponent
        if player == 0:
            return root.best_action()
        else:
            # Opponent: pick action with lowest value for player 0
            if not root.children:
                return 0
            return min(root.children.items(),
                       key=lambda x: x[1].q_value)[0]

    def solve_vs_random(self, env, n_games: int, max_steps: int = 200) -> dict:
        """Play n_games against random opponent, alternating who goes first."""
        wins = 0
        losses = 0
        draws = 0
        game_rewards = []

        for game in range(n_games):
            state = env.reset()
            mcts_player = game % 2

            for step in range(max_steps):
                sid = env.state_id(state)
                legal = env.legal_actions(state)
                if not legal:
                    draws += 1
                    break

                current = getattr(env, 'current_player', 0)

                if current == mcts_player:
                    action = self.search(state, sid, env, player=current)
                else:
                    action = random.choice(legal)

                state, reward, done = env.step(action)

                if done:
                    mcts_reward = reward if mcts_player == 0 else -reward
                    if mcts_reward > 0:
                        wins += 1
                    elif mcts_reward < 0:
                        losses += 1
                    else:
                        draws += 1
                    game_rewards.append(mcts_reward)
                    break
            else:
                draws += 1
                game_rewards.append(0.0)

        return {
            'rewards': game_rewards,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / max(1, n_games),
            'log': dict(self.log),
        }

    def solve_vs_opponent(self, env, opponent_fn, n_games: int,
                          max_steps: int = 200, mcts_is_player: int = 0) -> dict:
        """Play against a specific opponent function."""
        wins = 0
        losses = 0
        draws = 0
        game_rewards = []

        for game in range(n_games):
            state = env.reset()
            mcts_player = mcts_is_player if mcts_is_player >= 0 else game % 2

            for step in range(max_steps):
                legal = env.legal_actions(state)
                if not legal:
                    draws += 1
                    break

                current = getattr(env, 'current_player', 0)

                if current == mcts_player:
                    sid = env.state_id(state)
                    action = self.search(state, sid, env, player=current)
                else:
                    action = opponent_fn(env, legal)

                state, reward, done = env.step(action)

                if done:
                    mcts_reward = reward if mcts_player == 0 else -reward
                    if mcts_reward > 0:
                        wins += 1
                    elif mcts_reward < 0:
                        losses += 1
                    else:
                        draws += 1
                    game_rewards.append(mcts_reward)
                    break
            else:
                draws += 1
                game_rewards.append(0.0)

        return {
            'rewards': game_rewards,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / max(1, n_games),
            'log': dict(self.log),
        }
