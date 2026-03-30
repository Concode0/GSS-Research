"""Research-quality visualization utilities for GSS vs MCTS comparison."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

plt.rcParams.update({
    'figure.dpi': 130,
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
})

STYLES = {
    'gss':      ('Complete GSS',  '#2196F3', '-',  'o'),
    'mcts':     ('MCTS (UCT)',    '#F44336', '--', 's'),
    'gss_lite': ('GSS-Lite',      '#9E9E9E', ':',  '^'),
}


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(results: Dict[str, Dict],
                         metric: str = 'rewards',
                         title: str = '',
                         ylabel: str = '',
                         smooth: int = 1,
                         save_path: Optional[str] = None):
    """Plot learning curves with confidence intervals for multiple algorithms."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name in results:
        label, color, ls, marker = STYLES.get(name, (name, '#333', '-', 'o'))
        data = np.array(results[name][metric])
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if smooth > 1:
            data = uniform_filter1d(data.astype(float), size=smooth, axis=1)
        m = data.mean(axis=0)
        s = data.std(axis=0)
        episodes = np.arange(1, len(m) + 1)
        ax.plot(episodes, m, color=color, ls=ls, lw=2.2, label=label)
        ax.fill_between(episodes, m - s, m + s, color=color, alpha=0.13)

    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel or metric.capitalize())
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# GSS diagnostics
# ---------------------------------------------------------------------------

def plot_gss_diagnostics(log: dict,
                         title_prefix: str = '',
                         save_path: Optional[str] = None):
    """GSS-specific: beam size, diversity, causal classification, frontier events."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Beam size
    ax = axes[0, 0]
    if log.get('beam_sizes'):
        ax.plot(log['beam_sizes'], '#2196F3', lw=1.5)
        for t in log.get('merge_events', []):
            if t < len(log['beam_sizes']):
                ax.axvline(t, color='red', alpha=0.25, lw=0.5)
        for t in log.get('frontier_events', []):
            if t < len(log['beam_sizes']):
                ax.axvline(t, color='gold', alpha=0.3, lw=0.5)
    ax.set_title(f'{title_prefix}Beam Size', fontweight='bold')
    ax.set_ylabel('# Hypotheses')
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.2)

    # Diversity
    ax = axes[0, 1]
    if log.get('diversity'):
        ax.plot(log['diversity'], '#9C27B0', lw=1.5)
    ax.set_title(f'{title_prefix}Beam Diversity', fontweight='bold')
    ax.set_ylabel('Mean Pairwise Distance')
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.2)

    # Causal classification
    ax = axes[1, 0]
    label_map = {'timelike': 1, 'null': 0, 'spacelike': -1}
    colors_map = {'timelike': '#4CAF50', 'null': '#FFC107', 'spacelike': '#F44336'}
    if log.get('causal_labels') and len(log['causal_labels']) > 0:
        max_hyps = max(len(labels) for labels in log['causal_labels'])
        for i in range(min(max_hyps, 8)):
            vals = []
            for labels in log['causal_labels']:
                if i < len(labels):
                    vals.append(label_map.get(labels[i], 0))
            if vals:
                ax.plot(vals, alpha=0.5, lw=0.8, label=f'H{i}' if i < 4 else None)
        ax.axhline(0, color='gold', lw=2, ls='--', alpha=0.5, label='Null boundary')
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['Spacelike\n(unexplored)', 'Null\n(frontier)',
                            'Timelike\n(explored)'])
        ax.legend(fontsize=8, ncol=2)
    ax.set_title(f'{title_prefix}Causal Classification', fontweight='bold')
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.2)

    # Best scores
    ax = axes[1, 1]
    if log.get('best_scores'):
        scores = log['best_scores']
        ax.plot(scores, '#1565C0', lw=0.6, alpha=0.5)
        if len(scores) > 20:
            smoothed = uniform_filter1d(np.array(scores, dtype=float), size=20)
            ax.plot(smoothed, '#1565C0', lw=2)
    ax.set_title(f'{title_prefix}Best Hypothesis Score', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.2)

    plt.suptitle(f'{title_prefix}GSS Internals', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# MCTS diagnostics
# ---------------------------------------------------------------------------

def plot_mcts_diagnostics(log: dict,
                          title_prefix: str = '',
                          save_path: Optional[str] = None):
    """MCTS-specific: tree size, rollout returns, node visits."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # Tree size
    ax = axes[0]
    if log.get('tree_sizes'):
        ax.plot(log['tree_sizes'], '#F44336', lw=1.5)
    ax.set_title(f'{title_prefix}Tree Size', fontweight='bold')
    ax.set_ylabel('# Nodes')
    ax.set_xlabel('Decision step')
    ax.grid(True, alpha=0.2)

    # Rollout returns
    ax = axes[1]
    if log.get('rollout_returns'):
        returns = np.array(log['rollout_returns'], dtype=float)
        ax.plot(returns, '#F44336', lw=0.3, alpha=0.2)
        if len(returns) > 50:
            smoothed = uniform_filter1d(returns, size=min(50, len(returns)))
            ax.plot(smoothed, '#B71C1C', lw=2)
    ax.set_title(f'{title_prefix}Rollout Returns', fontweight='bold')
    ax.set_xlabel('Simulation')
    ax.grid(True, alpha=0.2)

    # Node visits
    ax = axes[2]
    visits = log.get('node_visits', {})
    if visits:
        if isinstance(visits, defaultdict):
            visits = dict(visits)
        states = sorted(visits.keys())
        vals = [visits[s] for s in states]
        ax.bar(range(len(states)), vals, color='#F44336', alpha=0.7)
        ax.set_xlabel('State index')
    ax.set_title(f'{title_prefix}State Visit Distribution', fontweight='bold')
    ax.set_ylabel('Total visits')
    ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle(f'{title_prefix}MCTS Internals', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Exploration heatmap (grid environments)
# ---------------------------------------------------------------------------

def plot_exploration_heatmap(gss_visits: np.ndarray,
                             mcts_visits: np.ndarray,
                             env,
                             title: str = '',
                             save_path: Optional[str] = None):
    """Side-by-side visit heatmaps for grid environments."""
    size = getattr(env, 'size', None)
    if size is None:
        rows = getattr(env, 'height', 4)
        cols = getattr(env, 'width', 12)
    else:
        rows = cols = size

    walls = getattr(env, 'walls', set())
    if hasattr(env, '_cliff'):
        walls = walls | env._cliff
    holes = getattr(env, '_holes', set())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, visits, label, cmap in [
        (axes[0], gss_visits, 'Complete GSS', 'Blues'),
        (axes[1], mcts_visits, 'MCTS', 'Reds'),
    ]:
        grid = visits.reshape(rows, cols).astype(float)
        # Mask walls
        mask = np.ones_like(grid)
        for r, c in walls:
            if 0 <= r < rows and 0 <= c < cols:
                mask[r, c] = 0
                grid[r, c] = np.nan
        for r, c in holes:
            if 0 <= r < rows and 0 <= c < cols:
                mask[r, c] = 0
                grid[r, c] = np.nan

        im = ax.imshow(grid, cmap=cmap, interpolation='nearest', aspect='equal')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Visits')

        # Mark walls
        for r, c in walls:
            if 0 <= r < rows and 0 <= c < cols:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor='black', alpha=0.7))
        for r, c in holes:
            if 0 <= r < rows and 0 <= c < cols:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor='darkred', alpha=0.5))

        # Start / goal markers (use env's actual positions if available)
        start_r = getattr(env, '_start', (0, 0))[0] if hasattr(env, '_start') else 0
        start_c = getattr(env, '_start', (0, 0))[1] if hasattr(env, '_start') else 0
        ax.plot(start_c, start_r, 'g*', ms=15, label='Start')
        goal = getattr(env, '_goal', (rows - 1, cols - 1))
        goal_r = goal[0] if goal else rows - 1
        goal_c = goal[1] if goal else cols - 1
        ax.plot(goal_c, goal_r, 'r*', ms=15, label='Goal')

        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='lower left')
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))

    plt.suptitle(f'{title} -- Exploration Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Adversarial results
# ---------------------------------------------------------------------------

def plot_adversarial_results(gss_result: dict,
                             mcts_result: dict,
                             env_name: str = '',
                             save_path: Optional[str] = None):
    """Win/loss/draw comparison for adversarial games."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Win rates bar chart
    ax = axes[0]
    categories = ['Wins', 'Losses', 'Draws']
    gss_vals = [gss_result['wins'], gss_result['losses'], gss_result['draws']]
    mcts_vals = [mcts_result['wins'], mcts_result['losses'], mcts_result['draws']]
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w / 2, gss_vals, w, label='GSS', color='#2196F3', alpha=0.8)
    ax.bar(x + w / 2, mcts_vals, w, label='MCTS', color='#F44336', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Count')
    ax.set_title(f'{env_name} vs Random', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    # Win rate over games (running average)
    ax = axes[1]
    for name, result, color in [('GSS', gss_result, '#2196F3'),
                                 ('MCTS', mcts_result, '#F44336')]:
        rewards = np.array(result['rewards'])
        cumwins = np.cumsum(rewards > 0)
        games = np.arange(1, len(rewards) + 1)
        ax.plot(games, cumwins / games, color=color, lw=2, label=name)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.4, label='50%')
    ax.set_xlabel('Game')
    ax.set_ylabel('Cumulative Win Rate')
    ax.set_title('Win Rate Over Games', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Reward distribution
    ax = axes[2]
    for name, result, color in [('GSS', gss_result, '#2196F3'),
                                 ('MCTS', mcts_result, '#F44336')]:
        ax.hist(result['rewards'], bins=20, alpha=0.5, color=color,
                label=name, edgecolor='white')
    ax.set_xlabel('Game Reward')
    ax.set_ylabel('Count')
    ax.set_title('Reward Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle(f'{env_name} -- Adversarial Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Summary dashboard
# ---------------------------------------------------------------------------

def plot_summary_dashboard(all_results: Dict[str, Dict],
                           save_path: Optional[str] = None):
    """Cross-environment comparison: one column per environment."""
    env_names = list(all_results.keys())
    n_envs = len(env_names)
    fig, axes = plt.subplots(2, n_envs, figsize=(4.5 * n_envs, 9))
    if n_envs == 1:
        axes = axes.reshape(2, 1)

    for col, env_name in enumerate(env_names):
        res = all_results[env_name]

        # Row 1: Learning curves
        ax = axes[0, col]
        for alg_name in res:
            label, color, ls, marker = STYLES.get(alg_name,
                                                    (alg_name, '#333', '-', 'o'))
            data = np.array(res[alg_name]['rewards'])
            if data.ndim == 1:
                data = data.reshape(1, -1)
            m = data.mean(axis=0)
            smooth_size = max(1, len(m) // 20)
            if smooth_size > 1:
                m = uniform_filter1d(m, size=smooth_size)
            ax.plot(m, color=color, ls=ls, lw=2, label=label)
        ax.set_title(env_name, fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('Reward')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

        # Row 2: Final metric bars
        ax = axes[1, col]
        alg_names = list(res.keys())
        final_vals = []
        bar_colors = []
        for alg_name in alg_names:
            data = np.array(res[alg_name]['rewards'])
            if data.ndim > 1:
                final_vals.append(data[:, -1].mean())
            else:
                final_vals.append(data[-1] if len(data) > 0 else 0)
            bar_colors.append(STYLES.get(alg_name, (alg_name, '#333', '-', 'o'))[1])

        bars = ax.bar(range(len(alg_names)), final_vals, color=bar_colors, alpha=0.8)
        ax.set_xticks(range(len(alg_names)))
        ax.set_xticklabels([STYLES.get(n, (n,))[0] for n in alg_names],
                           rotation=30, ha='right', fontsize=8)
        if col == 0:
            ax.set_ylabel('Final Reward')
        ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('GSS vs MCTS -- Cross-Environment Summary',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def plot_ablation(results: Dict[str, Dict],
                  title: str = 'Ablation: GSS-Lite vs Complete GSS vs MCTS',
                  save_path: Optional[str] = None):
    """Compare GSS-lite, Complete GSS, and MCTS on one environment."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Learning curves
    ax = axes[0]
    for name in results:
        label, color, ls, marker = STYLES.get(name, (name, '#333', '-', 'o'))
        data = np.array(results[name]['rewards'])
        if data.ndim == 1:
            data = data.reshape(1, -1)
        m = data.mean(axis=0)
        s = data.std(axis=0)
        eps = np.arange(1, len(m) + 1)
        smooth_size = max(1, len(m) // 20)
        if smooth_size > 1:
            m = uniform_filter1d(m, size=smooth_size)
            s = uniform_filter1d(s, size=smooth_size)
        ax.plot(eps, m, color=color, ls=ls, lw=2.2, label=label)
        ax.fill_between(eps, m - s, m + s, color=color, alpha=0.12)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Learning Curves', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.25)

    # Final metric bars
    ax = axes[1]
    names = list(results.keys())
    finals = []
    colors = []
    for name in names:
        data = np.array(results[name]['rewards'])
        if data.ndim > 1:
            finals.append(data[:, -1].mean())
        else:
            finals.append(data[-1] if len(data) > 0 else 0)
        colors.append(STYLES.get(name, (name, '#333', '-', 'o'))[1])

    ax.bar(range(len(names)), finals, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([STYLES.get(n, (n,))[0] for n in names],
                       rotation=20, ha='right')
    ax.set_ylabel('Final Reward (Mean)')
    ax.set_title('Final Performance', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def plot_timing(timing: Dict[str, Dict[str, float]],
                save_path: Optional[str] = None):
    """Bar chart of wall-clock time per environment for each algorithm."""
    env_names = list(timing.keys())
    alg_names = list(next(iter(timing.values())).keys())

    fig, ax = plt.subplots(figsize=(max(8, len(env_names) * 2), 5))
    x = np.arange(len(env_names))
    w = 0.8 / len(alg_names)

    for i, alg in enumerate(alg_names):
        label, color, _, _ = STYLES.get(alg, (alg, '#333', '-', 'o'))
        vals = [timing[env].get(alg, 0) for env in env_names]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=label,
               color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(env_names, rotation=30, ha='right')
    ax.set_ylabel('Wall-Clock Time (s)')
    ax.set_title('Computation Cost per Environment', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return fig
