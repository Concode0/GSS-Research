# GSS vs MCTS: Experimental Analysis

## 1. Overview

Complete Geometric Superposition Search (GSS) with Null Frontier detection
and Aware Hypothesis Manager vs standard Monte Carlo Tree Search (MCTS)
across 8 diverse environments.

## 2. Single-Agent Results

| Environment | GSS (mean reward) | MCTS (mean reward) | Winner |
|---|---|---|---|
| bandit | 0.880 | 0.401 | **GSS** |
| chain | 0.997 | 1.000 | **MCTS** |
| cliff | -15.873 | -195.285 | **GSS** |
| maze5 | 0.897 | 0.889 | **GSS** |
| maze8 | 0.207 | 0.814 | **MCTS** |
| frozenlake | -0.091 | -0.631 | **GSS** |

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
| nim | 0.451 | 0.970 | **MCTS** |
| connect | 0.651 | 0.978 | **MCTS** |

### nim

![Adversarial results](../figures/nim_adversarial.png)

### connect

![Adversarial results](../figures/connect_adversarial.png)

## 4. Ablation: Manager Contribution

![Ablation](../figures/ablation_maze5.png)

- **gss**: final reward = 0.929
- **gss_lite**: final reward = 0.930
- **mcts**: final reward = 0.902

## 5. Computation Cost

![Timing](../figures/timing.png)

| Environment | GSS (s) | MCTS (s) |
|---|---|---|
| bandit | 0.41 | 0.05 |
| chain | 1.11 | 0.95 |
| cliff | 9.19 | 93.34 |
| maze5 | 2.70 | 7.95 |
| maze8 | 31.65 | 58.70 |
| frozenlake | 15.41 | 2.12 |
| nim | 0.36 | 0.28 |
| connect | 0.49 | 1.03 |

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
