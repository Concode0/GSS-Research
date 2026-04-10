"""8 diverse environments with uniform interface for GSS vs MCTS comparison."""

from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class Env(ABC):
    """Base environment interface."""

    num_states: int
    num_actions: int
    two_player: bool = False

    @abstractmethod
    def reset(self) -> Any:
        ...

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool]:
        ...

    @abstractmethod
    def clone(self) -> "Env":
        ...

    @abstractmethod
    def set_state(self, state: Any) -> None:
        ...

    @abstractmethod
    def state_id(self, state: Any) -> int:
        ...

    def legal_actions(self, state: Any = None) -> List[int]:
        return list(range(self.num_actions))

    def current_state(self) -> Any:
        return self._state


# ---------------------------------------------------------------------------
# 1. Multi-Armed Bandit
# ---------------------------------------------------------------------------

class BanditEnv(Env):
    """10-armed Gaussian bandit. Each step is a full episode."""

    def __init__(self, n_arms: int = 10, seed: int = 0):
        self.n_arms = n_arms
        self.num_states = 1
        self.num_actions = n_arms
        self._rng = random.Random(seed)
        self._mu = [self._rng.gauss(0, 1) for _ in range(n_arms)]
        self._state = 0

    def reset(self):
        self._state = 0
        return self._state

    def step(self, action):
        reward = random.gauss(self._mu[action], 1.0)
        return self._state, reward, True

    def clone(self):
        c = copy.copy(self)
        c._mu = list(self._mu)
        return c

    def set_state(self, state):
        self._state = state

    def state_id(self, state):
        return 0

    @property
    def optimal_mean(self):
        return max(self._mu)

    @property
    def means(self):
        return list(self._mu)


# ---------------------------------------------------------------------------
# 2. Chain MDP
# ---------------------------------------------------------------------------

class ChainMDP(Env):
    """Linear chain: left/right actions, reward +1 at rightmost state."""

    def __init__(self, n: int = 8):
        self.n = n
        self.num_states = n
        self.num_actions = 2  # 0=left, 1=right
        self._state = 0

    def reset(self):
        self._state = 0
        return self._state

    def step(self, action):
        if action == 1:
            self._state = min(self._state + 1, self.n - 1)
        else:
            self._state = max(self._state - 1, 0)
        reward = 1.0 if self._state == self.n - 1 else 0.0
        done = self._state == self.n - 1
        return self._state, reward, done

    def clone(self):
        c = ChainMDP(self.n)
        c._state = self._state
        return c

    def set_state(self, state):
        self._state = state

    def state_id(self, state):
        return state

    def current_state(self):
        return self._state


# ---------------------------------------------------------------------------
# 3. Cliff Walking
# ---------------------------------------------------------------------------

class CliffWalking(Env):
    """Classic cliff walking on a width x height grid.

    Start: bottom-left (height-1, 0).  Goal: bottom-right (height-1, width-1).
    Cliff: bottom row between start and goal.
    Step: -1, cliff: -100 (reset to start), goal: +10.
    """

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def __init__(self, width: int = 12, height: int = 4):
        self.width = width
        self.height = height
        self.num_states = width * height
        self.num_actions = 4
        self._start = (height - 1, 0)
        self._goal = (height - 1, width - 1)
        self._cliff = {(height - 1, c) for c in range(1, width - 1)}
        self._state = self._start

    def reset(self):
        self._state = self._start
        return self._state

    def step(self, action):
        r, c = self._state
        dr, dc = self.ACTIONS[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.height and 0 <= nc < self.width:
            self._state = (nr, nc)

        if self._state in self._cliff:
            self._state = self._start
            return self._state, -100.0, False
        if self._state == self._goal:
            return self._state, 10.0, True
        return self._state, -1.0, False

    def clone(self):
        c = CliffWalking(self.width, self.height)
        c._state = self._state
        return c

    def set_state(self, state):
        self._state = state

    def state_id(self, state):
        return state[0] * self.width + state[1]

    def current_state(self):
        return self._state


# ---------------------------------------------------------------------------
# 4. Maze 5x5
# ---------------------------------------------------------------------------

class MazeEnv(Env):
    """5x5 grid maze with walls. Start (0,0), goal (size-1, size-1)."""

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def __init__(self, size: int = 5, walls=None):
        self.size = size
        self.walls = walls if walls is not None else {(1, 1), (1, 3), (3, 1), (3, 3)}
        self.num_states = size * size
        self.num_actions = 4
        self._goal = (size - 1, size - 1)
        self._state = (0, 0)

    def reset(self):
        self._state = (0, 0)
        return self._state

    def step(self, action):
        r, c = self._state
        dr, dc = self.ACTIONS[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.walls:
            self._state = (nr, nc)
        reward = 1.0 if self._state == self._goal else -0.01
        done = self._state == self._goal
        return self._state, reward, done

    def clone(self):
        c = MazeEnv(self.size, set(self.walls))
        c._state = self._state
        c._goal = self._goal
        return c

    def set_state(self, state):
        self._state = state

    def state_id(self, state):
        return state[0] * self.size + state[1]

    def current_state(self):
        return self._state


# ---------------------------------------------------------------------------
# 5. Larger Maze 8x8
# ---------------------------------------------------------------------------

class LargerMaze(MazeEnv):
    """8x8 maze with corridors and dead ends."""

    def __init__(self):
        walls = {
            (1, 1), (1, 2), (1, 5), (1, 6),
            (2, 4),
            (3, 1), (3, 3), (3, 5),
            (4, 2), (4, 6),
            (5, 1), (5, 4), (5, 5),
            (6, 3), (6, 6),
        }
        super().__init__(size=8, walls=walls)


# ---------------------------------------------------------------------------
# 6. Nim (adversarial)
# ---------------------------------------------------------------------------

class NimEnv(Env):
    """Nim game with multiple heaps. Last player to move wins.

    State: tuple of heap sizes. Actions encoded as (heap_idx, n_remove).
    Flattened to int for the action interface.
    """

    two_player = True

    def __init__(self, heaps: tuple = (3, 4, 5)):
        self._init_heaps = tuple(heaps)
        self._heaps = list(heaps)
        self._current_player = 0  # 0 or 1
        self._max_heap = max(heaps)
        self.num_heaps = len(heaps)
        # Action = heap_idx * max_heap + (n_remove - 1)
        self.num_actions = self.num_heaps * self._max_heap
        # State space: product of (heap+1) sizes
        self.num_states = 1
        for h in heaps:
            self.num_states *= (h + 1)
        self._state = tuple(self._heaps)

    def reset(self):
        self._heaps = list(self._init_heaps)
        self._current_player = 0
        self._state = tuple(self._heaps)
        return self._state

    def _decode_action(self, action: int):
        heap_idx = action // self._max_heap
        n_remove = (action % self._max_heap) + 1
        return heap_idx, n_remove

    def legal_actions(self, state=None):
        heaps = self._heaps if state is None else list(state)
        actions = []
        for h_idx in range(self.num_heaps):
            for n in range(1, heaps[h_idx] + 1):
                actions.append(h_idx * self._max_heap + (n - 1))
        return actions

    def step(self, action):
        heap_idx, n_remove = self._decode_action(action)
        n_remove = min(n_remove, self._heaps[heap_idx])
        self._heaps[heap_idx] -= n_remove
        self._state = tuple(self._heaps)

        # Check if game over (all heaps empty)
        if all(h == 0 for h in self._heaps):
            # Current player made the last move -> they win
            reward = 1.0 if self._current_player == 0 else -1.0
            return self._state, reward, True

        self._current_player = 1 - self._current_player
        return self._state, 0.0, False

    @property
    def current_player(self):
        return self._current_player

    def clone(self):
        c = NimEnv(self._init_heaps)
        c._heaps = list(self._heaps)
        c._current_player = self._current_player
        c._state = self._state
        return c

    def set_state(self, state):
        self._heaps = list(state)
        self._state = tuple(self._heaps)
        # Infer player from total stones removed (even = player 0's turn)
        total_init = sum(self._init_heaps)
        total_now = sum(self._heaps)
        removed = total_init - total_now
        self._current_player = removed % 2

    def state_id(self, state):
        sid = 0
        mul = 1
        for i, h in enumerate(state):
            sid += h * mul
            mul *= (self._init_heaps[i] + 1)
        return sid

    def current_state(self):
        return self._state

    @staticmethod
    def nim_sum(heaps):
        """Optimal Nim strategy: XOR of all heap sizes."""
        s = 0
        for h in heaps:
            s ^= h
        return s


# ---------------------------------------------------------------------------
# 7. Connect (simplified Connect-4)
# ---------------------------------------------------------------------------

class ConnectEnv(Env):
    """Simplified Connect game on rows x cols board. Connect `connect` to win.

    Action = column index (drop piece from top).
    State = tuple of tuples (the board).
    Board encoding: 0=empty, 1=player 0, -1=player 1.
    Players: 0 and 1 (consistent with NimEnv).
    """

    two_player = True

    def __init__(self, rows: int = 4, cols: int = 5, connect: int = 3):
        self.rows = rows
        self.cols = cols
        self.connect = connect
        self.num_actions = cols
        self.num_states = 3 ** (rows * cols)  # approximate upper bound
        self._board = [[0] * cols for _ in range(rows)]
        self._current_player = 0  # 0 or 1
        self._state = self._to_state()
        self.two_player = True

    def _to_state(self):
        return tuple(tuple(row) for row in self._board)

    def _from_state(self, state):
        self._board = [list(row) for row in state]

    def _player_to_board(self, player):
        """Map player index (0/1) to board value (1/-1)."""
        return 1 if player == 0 else -1

    def reset(self):
        self._board = [[0] * self.cols for _ in range(self.rows)]
        self._current_player = 0
        self._state = self._to_state()
        return self._state

    def legal_actions(self, state=None):
        board = self._board if state is None else [list(row) for row in state]
        return [c for c in range(self.cols) if board[0][c] == 0]

    def _drop(self, col, board_val):
        for r in range(self.rows - 1, -1, -1):
            if self._board[r][col] == 0:
                self._board[r][col] = board_val
                return r
        return -1

    def _check_win(self, r, c, board_val):
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in dirs:
            count = 1
            for sign in (1, -1):
                nr, nc = r + sign * dr, c + sign * dc
                while (0 <= nr < self.rows and 0 <= nc < self.cols
                       and self._board[nr][nc] == board_val):
                    count += 1
                    nr += sign * dr
                    nc += sign * dc
            if count >= self.connect:
                return True
        return False

    def step(self, action):
        col = action
        if self._board[0][col] != 0:
            # Illegal move: penalize and skip
            self._state = self._to_state()
            return self._state, -1.0, False

        board_val = self._player_to_board(self._current_player)
        row = self._drop(col, board_val)
        self._state = self._to_state()

        if self._check_win(row, col, board_val):
            # Reward from player 0's perspective (consistent with NimEnv)
            reward = 1.0 if self._current_player == 0 else -1.0
            return self._state, reward, True

        # Draw check
        if all(self._board[0][c] != 0 for c in range(self.cols)):
            return self._state, 0.0, True

        self._current_player = 1 - self._current_player  # toggle 0<->1
        return self._state, 0.0, False

    @property
    def current_player(self):
        return self._current_player

    @current_player.setter
    def current_player(self, val):
        self._current_player = val

    def clone(self):
        c = ConnectEnv(self.rows, self.cols, self.connect)
        c._board = [list(row) for row in self._board]
        c._current_player = self._current_player
        c._state = self._to_state()
        return c

    def set_state(self, state):
        self._from_state(state)
        self._state = state
        # Infer player: count total pieces on board
        pieces = sum(1 for row in state for cell in row if cell != 0)
        self._current_player = 0 if pieces % 2 == 0 else 1

    def state_id(self, state):
        sid = 0
        mul = 1
        for row in state:
            for cell in row:
                sid += (cell + 1) * mul  # map -1->0, 0->1, 1->2
                mul *= 3
        return sid

    def current_state(self):
        return self._state


# ---------------------------------------------------------------------------
# 8. Frozen Lake (stochastic)
# ---------------------------------------------------------------------------

class FrozenLake(Env):
    """4x4 grid with holes and slippery movement.

    S=start, F=frozen(safe), H=hole(done, -1), G=goal(done, +1).
    With probability slip_prob, action is replaced by a random perpendicular action.
    """

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    DEFAULT_MAP = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ]

    def __init__(self, size: int = 4, slip_prob: float = 0.2, lake_map=None):
        self.size = size
        self.slip_prob = slip_prob
        self.lake_map = lake_map or self.DEFAULT_MAP
        self.num_states = size * size
        self.num_actions = 4
        self._start = (0, 0)
        self._goal = None
        self._holes = set()
        for r in range(size):
            for c in range(size):
                ch = self.lake_map[r][c]
                if ch == 'S':
                    self._start = (r, c)
                elif ch == 'G':
                    self._goal = (r, c)
                elif ch == 'H':
                    self._holes.add((r, c))
        self._state = self._start

    def reset(self):
        self._state = self._start
        return self._state

    def step(self, action):
        # Slip: with probability slip_prob, pick a random perpendicular direction
        if random.random() < self.slip_prob:
            if action in (0, 1):  # up/down -> slip to left/right
                action = random.choice([2, 3])
            else:  # left/right -> slip to up/down
                action = random.choice([0, 1])

        r, c = self._state
        dr, dc = self.ACTIONS[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            self._state = (nr, nc)

        if self._state in self._holes:
            return self._state, -1.0, True
        if self._state == self._goal:
            return self._state, 1.0, True
        return self._state, -0.01, False

    def clone(self):
        c = FrozenLake(self.size, self.slip_prob, self.lake_map)
        c._state = self._state
        c._start = self._start
        c._goal = self._goal
        c._holes = set(self._holes)
        return c

    def set_state(self, state):
        self._state = state

    def state_id(self, state):
        return state[0] * self.size + state[1]

    def current_state(self):
        return self._state


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ENV_REGISTRY = {
    'bandit':      lambda: BanditEnv(n_arms=10),
    'chain':       lambda: ChainMDP(n=8),
    'cliff':       lambda: CliffWalking(width=12, height=4),
    'maze5':       lambda: MazeEnv(size=5),
    'maze8':       lambda: LargerMaze(),
    'nim':         lambda: NimEnv(heaps=(3, 4, 5)),
    'connect':     lambda: ConnectEnv(rows=4, cols=5, connect=3),
    'frozenlake':  lambda: FrozenLake(size=4, slip_prob=0.2),
}

SINGLE_AGENT_ENVS = ['bandit', 'chain', 'cliff', 'maze5', 'maze8', 'frozenlake']
ADVERSARIAL_ENVS = ['nim', 'connect']
