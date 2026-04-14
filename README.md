# Geometric Superposition Search (GSS)

> **Copyright (c) 2026 Eunkyum Kim <nemonanconcode@gmail.com>. All Rights Reserved.**
> 
> **Project Identity:** A Specialized Research Extension of the **Versor Framework**
> **Base Framework DOI:** [10.5281/zenodo.18939519]
> **License:** Apache License 2.0

Research notebook exploring deterministic geometric beam search as an alternative to stochastic tree search (UCT/MCTS).

Built on the [Versor](https://github.com/Concode0/versor) geometric algebra framework for PyTorch.

## Core Idea

Use Clifford algebra structure to replace probabilistic exploration with geometric exploration:

- **Cl(3,0) state algebra**: rotor-equivariant policy + grade-0 invariant value
- **Bivector norm** as exploration signal (Fisher Information connection, Prop 1)
- **Cl(1,1) search algebra**: `tanh` from hyperbolic rotors as bounded exploration rate (Prop 2)
- **Null vectors** as causal frontier boundary (Prop 3)

## Results (vs UCB1 baseline)

| Environment     | GSS          | UCB1      | Metric            |
| --------------- | ------------ | --------- | ----------------- |
| 10-armed bandit | **38.6**     | 114.8     | Cumulative regret |
| Chain MDP       | **~7 steps** | ~11 steps | Steps to goal     |
| 5x5 Maze        | **0.930**    | 0.882     | Success rate      |

GSS receives geometric embeddings encoding environment structure; UCB1 uses only visit counts and reward means. See Section 8 in the notebook for an honest discussion of this information asymmetry.

## Key Properties

- **Fully differentiable**: all operations preserve `torch.autograd` (unlike MCTS which breaks gradients at rollout/expansion/backup)
- **Grade-selective scaling**: core ops (bivector exp, sandwich product) are O(n^2), not O(2^2n)
- **GPU-native**: standard PyTorch tensor ops throughout, no custom CUDA kernels

## Setup

```bash
git clone --recurse-submodules https://github.com/Concode0/gss-research.git
cd gss-research
uv sync
uv run jupyter lab gss_research.ipynb
```

## Status

Proof-of-concept. Open problems: formal regret bound, convergence proof on Spin(3), adversarial/game testing, scale beyond toy environments. See the notebook discussion for details.

### Current Develop..

See src/ for more expanded research with GSS vs MCTS ( it need more analysis and bug fix may not fair in potential bugs.)

* Testing the possibility of effectively depicting Zero-Sum games through reflexive computation in adversarial game situations.
* A hypothesis management concept is needed to maintain better prune, directionality, and power than currently available.
