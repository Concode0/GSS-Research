# GSS vs MCTS: Experimental Analysis

## 1. Overview

Complete Geometric Superposition Search (GSS) with Null Frontier detection
and Aware Hypothesis Manager vs standard Monte Carlo Tree Search (MCTS)
across 8 diverse environments.

## 2. Single-Agent Results

| Environment | GSS (mean reward) | MCTS (mean reward) | Winner |
|---|---|---|---|
| bandit | 0.942 | 0.383 | **GSS** |
| chain | 0.997 | 1.000 | **MCTS** |
| cliff | -11.752 | -194.962 | **GSS** |
| maze5 | 0.910 | 0.889 | **GSS** |
| maze8 | 0.247 | 0.813 | **MCTS** |
| frozenlake | 0.007 | -0.650 | **GSS** |

### bandit

![Learning curves](../figures/bandit_learning.png)

![GSS diagnostics](../figures/bandit_gss_diag.png)

![MCTS diagnostics](../figures/bandit_mcts_diag.png)

### chain

![Learning curves](../figures/chain_learning.png)

![GSS diagnostics](../figures/chain_gss_diag.png)

![MCTS diagnostics](../figures/chain_mcts_diag.png)

### cliff

![Learning curves](../figures/cliff_learning.png)

![GSS diagnostics](../figures/cliff_gss_diag.png)

![MCTS diagnostics](../figures/cliff_mcts_diag.png)

![Heatmap](../figures/cliff_heatmap.png)

### maze5

![Learning curves](../figures/maze5_learning.png)

![GSS diagnostics](../figures/maze5_gss_diag.png)

![MCTS diagnostics](../figures/maze5_mcts_diag.png)

![Heatmap](../figures/maze5_heatmap.png)

### maze8

![Learning curves](../figures/maze8_learning.png)

![GSS diagnostics](../figures/maze8_gss_diag.png)

![MCTS diagnostics](../figures/maze8_mcts_diag.png)

![Heatmap](../figures/maze8_heatmap.png)

### frozenlake

![Learning curves](../figures/frozenlake_learning.png)

![GSS diagnostics](../figures/frozenlake_gss_diag.png)

![MCTS diagnostics](../figures/frozenlake_mcts_diag.png)

![Heatmap](../figures/frozenlake_heatmap.png)

## 3. Adversarial Results

| Game | GSS Win Rate | MCTS Win Rate | Winner |
|---|---|---|---|
| nim | 0.417 | 0.970 | **MCTS** |
| connect | 0.500 | 0.500 | **Tie** |

### nim

![Adversarial results](../figures/nim_adversarial.png)

### connect

![Adversarial results](../figures/connect_adversarial.png)

## 4. Ablation: Manager Contribution

![Ablation](../figures/ablation_maze5.png)

- **gss**: final reward = 0.930
- **gss_lite**: final reward = 0.930
- **mcts**: final reward = 0.908

## 5. Computation Cost

![Timing](../figures/timing.png)

| Environment | GSS (s) | MCTS (s) |
|---|---|---|
| bandit | 0.24 | 0.05 |
| chain | 0.48 | 0.95 |
| cliff | 1.85 | 95.00 |
| maze5 | 0.65 | 8.13 |
| maze8 | 6.57 | 59.75 |
| frozenlake | 3.10 | 2.08 |
| nim | 0.13 | 0.28 |
| connect | 0.13 | 0.11 |

## 6. Discussion

### Information Asymmetry
GSS receives geometric embeddings encoding environment structure.
MCTS uses only environment interaction (rollouts). This is intentional:
each algorithm leverages its natural inductive bias.

### Computation Trade-offs
GSS: O(beam_size x num_actions x algebra_ops) per step.
MCTS: O(n_simulations x rollout_depth) per step.
MCTS uses significantly more environment interactions per decision.

### Null Frontier Activity
Frontier events (null boundary crossings in Cl(1,1)) trigger
beam expansion, increasing hypothesis diversity at critical points.
See GSS diagnostic plots for each environment.

## 7. Summary

![Summary dashboard](../figures/summary_dashboard.png)
