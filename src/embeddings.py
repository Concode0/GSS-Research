"""Cl(3,0) state embedding strategies and geometric helpers."""

from __future__ import annotations

import math
import torch
import numpy as np

import src  # ensures versor is on sys.path
from core.algebra import CliffordAlgebra

# Shared algebra instances
alg30 = CliffordAlgebra(3, 0, device='cpu')  # state algebra, dim=8
alg11 = CliffordAlgebra(1, 1, device='cpu')  # search algebra, dim=4

BV_IDX = alg30.grade_masks[2].nonzero(as_tuple=False).squeeze(-1)


def relational_bv_norm(mv_a: torch.Tensor, mv_b: torch.Tensor) -> float:
    """||grade-2(GP(a, b))|| -- angular uncertainty between two multivectors."""
    gp = alg30.geometric_product(mv_a.unsqueeze(0), mv_b.unsqueeze(0)).squeeze(0)
    return alg30.grade_projection(gp, 2).norm().item()


# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

def embed_bandit_arms(n_arms: int) -> torch.Tensor:
    """Arms on unit circle in grade-1 subspace. Returns [n_arms, 8]."""
    angles = torch.linspace(0, 2 * math.pi, n_arms + 1)[:-1]
    vecs = torch.stack([torch.cos(angles), torch.sin(angles),
                        torch.zeros(n_arms)], dim=1)
    return alg30.embed_vector(vecs)


def embed_chain_states(n_states: int) -> torch.Tensor:
    """Chain states on semicircle. Returns [n_states, 8]."""
    angles = torch.linspace(0, math.pi, n_states)
    vecs = torch.stack([torch.cos(angles), torch.sin(angles),
                        torch.zeros(n_states)], dim=1)
    return alg30.embed_vector(vecs)


def embed_grid_states(rows: int, cols: int) -> torch.Tensor:
    """Grid cells as normalized 2D position vectors. Returns [rows*cols, 8]."""
    ns = rows * cols
    vecs = torch.zeros(ns, 3)
    for r in range(rows):
        for c in range(cols):
            vecs[r * cols + c, 0] = (r + 0.5) / rows
            vecs[r * cols + c, 1] = (c + 0.5) / cols
    vecs = vecs / vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return alg30.embed_vector(vecs)


def embed_nim_state(heaps: tuple, max_heap: int = 5) -> torch.Tensor:
    """Nim heap sizes as 3D vector (one dimension per heap). Returns [1, 8]."""
    vec = torch.zeros(1, 3)
    for i, h in enumerate(heaps):
        if i < 3:
            vec[0, i] = h / max_heap  # normalize
    norm = vec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    vec = vec / norm
    return alg30.embed_vector(vec)


def embed_connect_state(board: tuple, rows: int = 4, cols: int = 5) -> torch.Tensor:
    """Board features -> 3D projection. Encodes piece counts and position.
    Returns [1, 8]."""
    p1 = 0
    p2 = 0
    center_mass = 0.0
    total = 0
    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            if cell == 1:
                p1 += 1
                center_mass += c / cols
                total += 1
            elif cell == 2:
                p2 += 1
                center_mass -= c / cols
                total += 1
    if total > 0:
        center_mass /= total
    vec = torch.tensor([[p1 / (rows * cols), p2 / (rows * cols), center_mass]])
    norm = vec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    vec = vec / norm
    return alg30.embed_vector(vec)


def embed_generic(state_id: int, n_states: int) -> torch.Tensor:
    """Fallback: distribute states uniformly on sphere. Returns [1, 8]."""
    # Fibonacci sphere
    golden = (1 + math.sqrt(5)) / 2
    i = state_id
    theta = 2 * math.pi * i / golden
    phi = math.acos(1 - 2 * (i + 0.5) / max(1, n_states))
    vec = torch.tensor([[
        math.sin(phi) * math.cos(theta),
        math.sin(phi) * math.sin(theta),
        math.cos(phi),
    ]])
    return alg30.embed_vector(vec)


# ---------------------------------------------------------------------------
# Precompute embeddings for each environment type
# ---------------------------------------------------------------------------

def get_state_embeddings(env_name: str, env) -> torch.Tensor:
    """Return precomputed [num_states, 8] embedding tensor for the given env."""
    if env_name == 'bandit':
        return embed_bandit_arms(env.num_actions)
    elif env_name == 'chain':
        return embed_chain_states(env.num_states)
    elif env_name in ('cliff',):
        return embed_grid_states(env.height, env.width)
    elif env_name in ('maze5', 'maze8'):
        return embed_grid_states(env.size, env.size)
    elif env_name == 'frozenlake':
        return embed_grid_states(env.size, env.size)
    elif env_name == 'nim':
        # Precompute for all possible states
        heaps = env._init_heaps
        states = []
        for h0 in range(heaps[0] + 1):
            for h1 in range(heaps[1] + 1):
                for h2 in range(heaps[2] + 1):
                    states.append((h0, h1, h2))
        n = len(states)
        vecs = torch.zeros(n, 3)
        max_h = max(heaps)
        for idx, (h0, h1, h2) in enumerate(states):
            vecs[idx, 0] = h0 / max(max_h, 1)
            vecs[idx, 1] = h1 / max(max_h, 1)
            vecs[idx, 2] = h2 / max(max_h, 1)
        norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        vecs = vecs / norms
        return alg30.embed_vector(vecs)
    elif env_name == 'connect':
        # For connect, we embed on-the-fly (too many states)
        # Return a generic embedding based on total states
        return embed_grid_states(env.rows, env.cols)
    else:
        # Generic fallback
        n = env.num_states
        vecs = torch.zeros(n, 3)
        golden = (1 + math.sqrt(5)) / 2
        for i in range(n):
            theta = 2 * math.pi * i / golden
            phi = math.acos(1 - 2 * (i + 0.5) / max(1, n))
            vecs[i, 0] = math.sin(phi) * math.cos(theta)
            vecs[i, 1] = math.sin(phi) * math.sin(theta)
            vecs[i, 2] = math.cos(phi)
        return alg30.embed_vector(vecs)
