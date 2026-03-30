"""Main experiment runner: GSS vs MCTS across 8 environments.

Usage:
    python -m src.run                          # full suite
    python -m src.run --envs bandit,chain      # specific envs
    python -m src.run --seeds 3 --quick        # fast smoke test
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Callable, Dict, List

import numpy as np
import torch

import src  # versor path
from src.environments import (
    ENV_REGISTRY, SINGLE_AGENT_ENVS, ADVERSARIAL_ENVS,
    BanditEnv, NimEnv, ConnectEnv,
)
from src.embeddings import get_state_embeddings
from src.gss import CompleteGSS, GSSLite
from src.mcts import MCTS, MCTSTwoPlayer
from src.visualize import (
    plot_learning_curves, plot_gss_diagnostics, plot_mcts_diagnostics,
    plot_exploration_heatmap, plot_adversarial_results,
    plot_summary_dashboard, plot_ablation, plot_timing,
)


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS = {
    'bandit':     {'n_episodes': 1000, 'max_steps': 1,   'mcts_sims': 30,  'beam': 4, 'gamma': 0.99},
    'chain':      {'n_episodes': 300,  'max_steps': 50,  'mcts_sims': 30,  'beam': 4, 'gamma': 0.99},
    'cliff':      {'n_episodes': 500,  'max_steps': 100, 'mcts_sims': 50,  'beam': 4, 'gamma': 0.99},
    'maze5':      {'n_episodes': 500,  'max_steps': 100, 'mcts_sims': 50,  'beam': 4, 'gamma': 0.95},
    'maze8':      {'n_episodes': 500,  'max_steps': 200, 'mcts_sims': 100, 'beam': 4, 'gamma': 0.95},
    'nim':        {'n_games': 200,     'max_steps': 200, 'mcts_sims': 100, 'beam': 4, 'gamma': 1.0},
    'connect':    {'n_games': 200,     'max_steps': 200, 'mcts_sims': 100, 'beam': 4, 'gamma': 1.0},
    'frozenlake': {'n_episodes': 500,  'max_steps': 100, 'mcts_sims': 50,  'beam': 4, 'gamma': 0.95},
}

QUICK_CONFIGS = {
    k: {**v,
        'n_episodes': min(v.get('n_episodes', v.get('n_games', 50)), 50),
        'n_games': min(v.get('n_games', v.get('n_episodes', 50)), 30),
        'mcts_sims': min(v.get('mcts_sims', 30), 20),
        'max_steps': min(v.get('max_steps', 50), 50),
    }
    for k, v in DEFAULT_CONFIGS.items()
}


def random_opponent(env, legal_actions):
    return random.choice(legal_actions)


# ---------------------------------------------------------------------------
# Single-agent experiment
# ---------------------------------------------------------------------------

def run_single_agent(env_name: str, config: dict, n_seeds: int,
                     fig_dir: str) -> Dict[str, Dict]:
    """Run GSS and MCTS on a single-agent environment across seeds."""
    n_episodes = config['n_episodes']
    max_steps = config['max_steps']
    mcts_sims = config['mcts_sims']
    beam_size = config['beam']
    gamma = config['gamma']

    results = {
        'gss':  {'rewards': [], 'steps': [], 'logs': [], 'state_visits': []},
        'mcts': {'rewards': [], 'steps': [], 'logs': []},
    }
    timing = {'gss': 0.0, 'mcts': 0.0}

    for seed in range(n_seeds):
        _set_seed(seed)

        # --- GSS ---
        env = ENV_REGISTRY[env_name]()
        state_mvs = get_state_embeddings(env_name, env)
        gss = CompleteGSS(
            num_states=env.num_states,
            num_actions=env.num_actions,
            state_mvs=state_mvs,
            beam_size=beam_size,
            c_explore=1.0,
            gamma=gamma,
        )
        t0 = time.time()
        gss_res = gss.solve_single_agent(env, n_episodes, max_steps)
        timing['gss'] += time.time() - t0
        results['gss']['rewards'].append(gss_res['rewards'])
        results['gss']['steps'].append(gss_res['steps'])
        results['gss']['logs'].append(gss_res['log'])
        results['gss']['state_visits'].append(dict(gss.state_visits))

        # --- MCTS ---
        _set_seed(seed)
        env = ENV_REGISTRY[env_name]()
        mcts = MCTS(
            n_simulations=mcts_sims,
            rollout_depth=max_steps,
            gamma=gamma,
        )
        t0 = time.time()
        mcts_res = mcts.solve(env, n_episodes, max_steps)
        timing['mcts'] += time.time() - t0
        results['mcts']['rewards'].append(mcts_res['rewards'])
        results['mcts']['steps'].append(mcts_res['steps'])
        results['mcts']['logs'].append(mcts_res['log'])

    timing['gss'] /= max(1, n_seeds)
    timing['mcts'] /= max(1, n_seeds)

    # --- Plots ---
    plot_learning_curves(
        results, metric='rewards',
        title=f'{env_name} -- Episode Reward',
        ylabel='Cumulative Reward',
        smooth=max(1, n_episodes // 20),
        save_path=os.path.join(fig_dir, f'{env_name}_learning.png'),
    )
    if results['gss']['logs']:
        plot_gss_diagnostics(
            results['gss']['logs'][0],
            title_prefix=f'{env_name}: ',
            save_path=os.path.join(fig_dir, f'{env_name}_gss_diag.png'),
        )
    if results['mcts']['logs']:
        plot_mcts_diagnostics(
            results['mcts']['logs'][0],
            title_prefix=f'{env_name}: ',
            save_path=os.path.join(fig_dir, f'{env_name}_mcts_diag.png'),
        )

    # Heatmaps for grid environments
    if env_name in ('maze5', 'maze8', 'cliff', 'frozenlake'):
        env_ref = ENV_REGISTRY[env_name]()
        gss_visits = _aggregate_visits(results['gss']['state_visits'], env_ref.num_states)
        mcts_visits = _aggregate_mcts_visits(results['mcts']['logs'], env_ref.num_states)
        plot_exploration_heatmap(
            gss_visits, mcts_visits, env_ref,
            title=env_name,
            save_path=os.path.join(fig_dir, f'{env_name}_heatmap.png'),
        )

    return results, timing


# ---------------------------------------------------------------------------
# Adversarial experiment
# ---------------------------------------------------------------------------

def run_adversarial(env_name: str, config: dict, n_seeds: int,
                    fig_dir: str) -> Dict[str, Dict]:
    """Run GSS and MCTS on a 2-player game against random opponent."""
    n_games = config.get('n_games', 200)
    max_steps = config['max_steps']
    mcts_sims = config['mcts_sims']
    beam_size = config['beam']
    gamma = config['gamma']

    gss_agg = {'wins': 0, 'losses': 0, 'draws': 0, 'rewards': []}
    mcts_agg = {'wins': 0, 'losses': 0, 'draws': 0, 'rewards': []}
    timing = {'gss': 0.0, 'mcts': 0.0}
    gss_logs = []
    mcts_logs = []

    for seed in range(n_seeds):
        _set_seed(seed)

        # --- GSS vs random ---
        env = ENV_REGISTRY[env_name]()
        state_mvs = get_state_embeddings(env_name, env)
        gss = CompleteGSS(
            num_states=env.num_states,
            num_actions=env.num_actions,
            state_mvs=state_mvs,
            beam_size=beam_size,
            c_explore=1.0,
            gamma=gamma,
            adversarial=True,
        )
        t0 = time.time()
        gss_res = gss.solve_adversarial(env, random_opponent, n_games, max_steps)
        timing['gss'] += time.time() - t0
        gss_agg['wins'] += gss_res['wins']
        gss_agg['losses'] += gss_res['losses']
        gss_agg['draws'] += gss_res['draws']
        gss_agg['rewards'].extend(gss_res['rewards'])
        gss_logs.append(gss_res['log'])

        # --- MCTS vs random ---
        _set_seed(seed)
        env = ENV_REGISTRY[env_name]()
        mcts = MCTSTwoPlayer(n_simulations=mcts_sims, rollout_depth=max_steps)
        t0 = time.time()
        mcts_res = mcts.solve_vs_random(env, n_games, max_steps)
        timing['mcts'] += time.time() - t0
        mcts_agg['wins'] += mcts_res['wins']
        mcts_agg['losses'] += mcts_res['losses']
        mcts_agg['draws'] += mcts_res['draws']
        mcts_agg['rewards'].extend(mcts_res['rewards'])
        mcts_logs.append(mcts_res['log'])

    timing['gss'] /= max(1, n_seeds)
    timing['mcts'] /= max(1, n_seeds)

    total_games = n_games * n_seeds
    gss_agg['win_rate'] = gss_agg['wins'] / max(1, total_games)
    mcts_agg['win_rate'] = mcts_agg['wins'] / max(1, total_games)

    plot_adversarial_results(
        gss_agg, mcts_agg, env_name=env_name,
        save_path=os.path.join(fig_dir, f'{env_name}_adversarial.png'),
    )
    if gss_logs:
        plot_gss_diagnostics(
            gss_logs[0], title_prefix=f'{env_name}: ',
            save_path=os.path.join(fig_dir, f'{env_name}_gss_diag.png'),
        )
    if mcts_logs:
        plot_mcts_diagnostics(
            mcts_logs[0], title_prefix=f'{env_name}: ',
            save_path=os.path.join(fig_dir, f'{env_name}_mcts_diag.png'),
        )

    results = {
        'gss': {'rewards': [gss_agg['rewards']], **gss_agg, 'logs': gss_logs},
        'mcts': {'rewards': [mcts_agg['rewards']], **mcts_agg, 'logs': mcts_logs},
    }
    return results, timing


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def run_ablation(fig_dir: str, n_seeds: int, config: dict) -> Dict:
    """GSS-lite vs Complete GSS vs MCTS on maze5."""
    env_name = 'maze5'
    n_episodes = config['n_episodes']
    max_steps = config['max_steps']
    gamma = config['gamma']
    mcts_sims = config['mcts_sims']

    results = {
        'gss':      {'rewards': [], 'steps': []},
        'gss_lite': {'rewards': [], 'steps': []},
        'mcts':     {'rewards': [], 'steps': []},
    }

    for seed in range(n_seeds):
        _set_seed(seed)

        # Complete GSS
        env = ENV_REGISTRY[env_name]()
        state_mvs = get_state_embeddings(env_name, env)
        gss = CompleteGSS(env.num_states, env.num_actions, state_mvs,
                          beam_size=4, gamma=gamma)
        res = gss.solve_single_agent(env, n_episodes, max_steps)
        results['gss']['rewards'].append(res['rewards'])
        results['gss']['steps'].append(res['steps'])

        # GSS-Lite
        _set_seed(seed)
        env = ENV_REGISTRY[env_name]()
        lite = GSSLite(env.num_states, env.num_actions, state_mvs, gamma=gamma)
        res = lite.solve(env, n_episodes, max_steps)
        results['gss_lite']['rewards'].append(res['rewards'])
        results['gss_lite']['steps'].append(res['steps'])

        # MCTS
        _set_seed(seed)
        env = ENV_REGISTRY[env_name]()
        mcts = MCTS(n_simulations=mcts_sims, rollout_depth=max_steps, gamma=gamma)
        res = mcts.solve(env, n_episodes, max_steps)
        results['mcts']['rewards'].append(res['rewards'])
        results['mcts']['steps'].append(res['steps'])

    plot_ablation(
        results,
        title='Ablation: Manager Contribution (maze5)',
        save_path=os.path.join(fig_dir, 'ablation_maze5.png'),
    )
    return results


# ---------------------------------------------------------------------------
# Analysis doc generation
# ---------------------------------------------------------------------------

def generate_analysis(all_results: Dict, adv_results: Dict,
                      ablation_results: Dict, timing: Dict,
                      docs_dir: str, fig_dir: str):
    """Generate docs/analysis.md from experiment results."""
    lines = [
        '# GSS vs MCTS: Experimental Analysis',
        '',
        '## 1. Overview',
        '',
        'Complete Geometric Superposition Search (GSS) with Null Frontier detection',
        'and Aware Hypothesis Manager vs standard Monte Carlo Tree Search (MCTS)',
        'across 8 diverse environments.',
        '',
        '## 2. Single-Agent Results',
        '',
    ]

    # Summary table
    lines.append('| Environment | GSS (mean reward) | MCTS (mean reward) | Winner |')
    lines.append('|---|---|---|---|')
    for env_name, res in all_results.items():
        if env_name in ADVERSARIAL_ENVS:
            continue
        gss_r = np.array(res['gss']['rewards']).mean()
        mcts_r = np.array(res['mcts']['rewards']).mean()
        winner = 'GSS' if gss_r > mcts_r else ('MCTS' if mcts_r > gss_r else 'Tie')
        lines.append(f'| {env_name} | {gss_r:.3f} | {mcts_r:.3f} | **{winner}** |')

    lines.append('')

    # Per-environment figures
    for env_name in SINGLE_AGENT_ENVS:
        if env_name not in all_results:
            continue
        lines.append(f'### {env_name}')
        lines.append('')
        lines.append(f'![Learning curves](../figures/{env_name}_learning.png)')
        lines.append('')
        lines.append(f'![GSS diagnostics](../figures/{env_name}_gss_diag.png)')
        lines.append('')
        lines.append(f'![MCTS diagnostics](../figures/{env_name}_mcts_diag.png)')
        lines.append('')
        if env_name in ('maze5', 'maze8', 'cliff', 'frozenlake'):
            lines.append(f'![Heatmap](../figures/{env_name}_heatmap.png)')
            lines.append('')

    # Adversarial
    lines.append('## 3. Adversarial Results')
    lines.append('')
    lines.append('| Game | GSS Win Rate | MCTS Win Rate | Winner |')
    lines.append('|---|---|---|---|')
    for env_name, res in adv_results.items():
        gss_wr = res['gss'].get('win_rate', 0)
        mcts_wr = res['mcts'].get('win_rate', 0)
        winner = 'GSS' if gss_wr > mcts_wr else ('MCTS' if mcts_wr > gss_wr else 'Tie')
        lines.append(f'| {env_name} | {gss_wr:.3f} | {mcts_wr:.3f} | **{winner}** |')
    lines.append('')

    for env_name in ADVERSARIAL_ENVS:
        if env_name not in adv_results:
            continue
        lines.append(f'### {env_name}')
        lines.append('')
        lines.append(f'![Adversarial results](../figures/{env_name}_adversarial.png)')
        lines.append('')

    # Ablation
    lines.append('## 4. Ablation: Manager Contribution')
    lines.append('')
    lines.append('![Ablation](../figures/ablation_maze5.png)')
    lines.append('')
    if ablation_results:
        for name in ablation_results:
            data = np.array(ablation_results[name]['rewards'])
            mean_final = data[:, -1].mean() if data.ndim > 1 else data[-1]
            lines.append(f'- **{name}**: final reward = {mean_final:.3f}')
        lines.append('')

    # Timing
    lines.append('## 5. Computation Cost')
    lines.append('')
    lines.append('![Timing](../figures/timing.png)')
    lines.append('')
    lines.append('| Environment | GSS (s) | MCTS (s) |')
    lines.append('|---|---|---|')
    for env_name, t in timing.items():
        lines.append(f'| {env_name} | {t.get("gss", 0):.2f} | {t.get("mcts", 0):.2f} |')
    lines.append('')

    # Discussion
    lines.append('## 6. Discussion')
    lines.append('')
    lines.append('### Information Asymmetry')
    lines.append('GSS receives geometric embeddings encoding environment structure.')
    lines.append('MCTS uses only environment interaction (rollouts). This is intentional:')
    lines.append('each algorithm leverages its natural inductive bias.')
    lines.append('')
    lines.append('### Computation Trade-offs')
    lines.append('GSS: O(beam_size x num_actions x algebra_ops) per step.')
    lines.append('MCTS: O(n_simulations x rollout_depth) per step.')
    lines.append('MCTS uses significantly more environment interactions per decision.')
    lines.append('')
    lines.append('### Null Frontier Activity')
    lines.append('Frontier events (null boundary crossings in Cl(1,1)) trigger')
    lines.append('beam expansion, increasing hypothesis diversity at critical points.')
    lines.append('See GSS diagnostic plots for each environment.')
    lines.append('')

    # Summary
    lines.append('## 7. Summary')
    lines.append('')
    lines.append('![Summary dashboard](../figures/summary_dashboard.png)')
    lines.append('')

    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, 'analysis.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Analysis written to {path}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _aggregate_visits(gss_agents_or_logs, n_states: int) -> np.ndarray:
    """Aggregate GSS state visit counts across seeds."""
    visits = np.zeros(n_states)
    for item in gss_agents_or_logs:
        # Item is a state_visits dict from the GSS agent
        if isinstance(item, dict):
            for sid, count in item.items():
                if 0 <= sid < n_states:
                    visits[sid] += count
    if visits.sum() == 0:
        visits = np.ones(n_states)
    return visits


def _aggregate_mcts_visits(logs: List[dict], n_states: int) -> np.ndarray:
    """Aggregate MCTS visit counts across seeds."""
    visits = np.zeros(n_states)
    for log in logs:
        nv = log.get('node_visits', {})
        for sid, count in nv.items():
            if 0 <= sid < n_states:
                visits[sid] += count
    if visits.sum() == 0:
        visits = np.ones(n_states)
    return visits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='GSS vs MCTS experiments')
    parser.add_argument('--envs', type=str, default='all',
                        help='Comma-separated env names, or "all"')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--quick', action='store_true',
                        help='Reduced episodes for fast testing')
    parser.add_argument('--output', type=str, default='figures')
    parser.add_argument('--docs', type=str, default='docs')
    args = parser.parse_args()

    fig_dir = args.output
    docs_dir = args.docs
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    configs = QUICK_CONFIGS if args.quick else DEFAULT_CONFIGS

    if args.envs == 'all':
        env_list = list(ENV_REGISTRY.keys())
    else:
        env_list = [e.strip() for e in args.envs.split(',')]

    print(f'Running experiments: {env_list}')
    print(f'Seeds: {args.seeds}, Quick: {args.quick}')
    print(f'Output: {fig_dir}/, Docs: {docs_dir}/')
    print('=' * 60)

    all_results = {}
    adv_results = {}
    all_timing = {}

    # Single-agent environments
    for env_name in env_list:
        if env_name in ADVERSARIAL_ENVS:
            continue
        if env_name not in ENV_REGISTRY:
            print(f'  [SKIP] Unknown env: {env_name}')
            continue

        print(f'\n--- {env_name} ---')
        config = configs.get(env_name, configs.get('maze5'))
        results, timing = run_single_agent(env_name, config, args.seeds, fig_dir)
        all_results[env_name] = results
        all_timing[env_name] = timing

        # Print summary
        gss_mean = np.array(results['gss']['rewards']).mean()
        mcts_mean = np.array(results['mcts']['rewards']).mean()
        print(f'  GSS:  {gss_mean:.3f}  |  MCTS: {mcts_mean:.3f}  '
              f'|  Time: GSS={timing["gss"]:.1f}s  MCTS={timing["mcts"]:.1f}s')

    # Adversarial environments
    for env_name in env_list:
        if env_name not in ADVERSARIAL_ENVS:
            continue
        if env_name not in ENV_REGISTRY:
            print(f'  [SKIP] Unknown env: {env_name}')
            continue

        print(f'\n--- {env_name} (adversarial) ---')
        config = configs.get(env_name, configs.get('nim'))
        results, timing = run_adversarial(env_name, config, args.seeds, fig_dir)
        adv_results[env_name] = results
        all_results[env_name] = results
        all_timing[env_name] = timing

        gss_wr = results['gss'].get('win_rate', 0)
        mcts_wr = results['mcts'].get('win_rate', 0)
        print(f'  GSS win rate: {gss_wr:.3f}  |  MCTS win rate: {mcts_wr:.3f}  '
              f'|  Time: GSS={timing["gss"]:.1f}s  MCTS={timing["mcts"]:.1f}s')

    # Ablation
    print('\n--- Ablation (maze5) ---')
    ablation_config = configs.get('maze5', DEFAULT_CONFIGS['maze5'])
    ablation_results = run_ablation(fig_dir, args.seeds, ablation_config)
    for name in ablation_results:
        data = np.array(ablation_results[name]['rewards'])
        mean_final = data[:, -1].mean() if data.ndim > 1 else data[-1]
        print(f'  {name}: final reward = {mean_final:.3f}')

    # Timing plot
    plot_timing(all_timing, save_path=os.path.join(fig_dir, 'timing.png'))

    # Summary dashboard
    plot_summary_dashboard(all_results,
                           save_path=os.path.join(fig_dir, 'summary_dashboard.png'))

    # Generate analysis doc
    generate_analysis(all_results, adv_results, ablation_results,
                      all_timing, docs_dir, fig_dir)

    print('\n' + '=' * 60)
    print('Done. Check figures/ and docs/analysis.md')


if __name__ == '__main__':
    main()
