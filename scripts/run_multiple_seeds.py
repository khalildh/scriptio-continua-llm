#!/usr/bin/env python3
"""
Run the experiment with multiple random seeds for statistical significance.

This script:
1. Runs the experiment N times with different seeds
2. Aggregates results across seeds
3. Computes mean, std, and confidence intervals
4. Generates publication-ready statistics

Usage:
    python scripts/run_multiple_seeds.py --config config/train_shakespeare_small.py --seeds 3
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

def run_single_experiment(config: str, seed: int, base_dir: Path):
    """Run a single experiment with a specific seed."""
    exp_name = f"seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = base_dir / exp_name

    script_dir = Path(__file__).parent.parent

    # Modify training to use specific seed
    # We'll pass it via environment variable
    import os
    env = os.environ.copy()
    env['SEED'] = str(seed)

    cmd = [
        sys.executable,
        str(script_dir / 'run_experiment.py'),
        '--config', config,
        '--experiment-name', exp_name
    ]

    print(f"\n{'='*60}")
    print(f"RUNNING SEED {seed}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=script_dir, env=env)

    # Load results
    results_path = script_dir / 'experiments' / exp_name / 'results.json'
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def aggregate_results(all_results: list):
    """Aggregate results across seeds."""
    aggregated = {
        'modern': {'losses': [], 'bpcs': []},
        'scriptio': {'losses': [], 'bpcs': []},
    }

    for result in all_results:
        if not result:
            continue

        for condition in ['modern', 'scriptio']:
            metrics = result.get(condition, {}).get('final_metrics', {})
            if metrics.get('best_val_loss'):
                aggregated[condition]['losses'].append(metrics['best_val_loss'])
            if metrics.get('bits_per_char'):
                aggregated[condition]['bpcs'].append(metrics['bits_per_char'])

    # Compute statistics
    stats = {}
    for condition in ['modern', 'scriptio']:
        stats[condition] = {}
        for metric in ['losses', 'bpcs']:
            values = aggregated[condition][metric]
            if values:
                stats[condition][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n': len(values),
                    'values': values,
                    # 95% CI assuming normal distribution
                    'ci_95': 1.96 * np.std(values) / np.sqrt(len(values)) if len(values) > 1 else None,
                }

    return stats

def compute_significance(stats: dict):
    """Compute statistical significance between conditions."""
    from scipy import stats as scipy_stats

    significance = {}

    for metric in ['losses', 'bpcs']:
        modern_vals = stats.get('modern', {}).get(metric, {}).get('values', [])
        scriptio_vals = stats.get('scriptio', {}).get(metric, {}).get('values', [])

        if len(modern_vals) >= 2 and len(scriptio_vals) >= 2:
            # Independent samples t-test
            t_stat, p_value = scipy_stats.ttest_ind(modern_vals, scriptio_vals)
            significance[metric] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_005': p_value < 0.05,
                'significant_001': p_value < 0.01,
            }

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(modern_vals)**2 + np.std(scriptio_vals)**2) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(scriptio_vals) - np.mean(modern_vals)) / pooled_std
                significance[metric]['cohens_d'] = cohens_d
                significance[metric]['effect_size'] = (
                    'large' if abs(cohens_d) > 0.8 else
                    'medium' if abs(cohens_d) > 0.5 else
                    'small'
                )

    return significance

def print_summary(stats: dict, significance: dict):
    """Print a summary of the multi-seed experiment."""
    print("\n" + "=" * 70)
    print("MULTI-SEED EXPERIMENT SUMMARY")
    print("=" * 70)

    print(f"\n{'Condition':<20} {'Loss (mean ± std)':<25} {'BPC (mean ± std)':<25}")
    print("-" * 70)

    for condition in ['modern', 'scriptio']:
        loss_stats = stats.get(condition, {}).get('losses', {})
        bpc_stats = stats.get(condition, {}).get('bpcs', {})

        loss_str = f"{loss_stats.get('mean', 0):.4f} ± {loss_stats.get('std', 0):.4f}" if loss_stats else "N/A"
        bpc_str = f"{bpc_stats.get('mean', 0):.4f} ± {bpc_stats.get('std', 0):.4f}" if bpc_stats else "N/A"

        print(f"{condition:<20} {loss_str:<25} {bpc_str:<25}")

    print("\n" + "-" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("-" * 70)

    for metric, sig in significance.items():
        print(f"\n{metric.upper()}:")
        print(f"  t-statistic: {sig.get('t_statistic', 'N/A'):.4f}")
        print(f"  p-value: {sig.get('p_value', 'N/A'):.4f}")
        print(f"  Significant (p<0.05): {sig.get('significant_005', 'N/A')}")
        print(f"  Cohen's d: {sig.get('cohens_d', 'N/A'):.4f}")
        print(f"  Effect size: {sig.get('effect_size', 'N/A')}")

def main():
    parser = argparse.ArgumentParser(description='Run multi-seed experiment')
    parser.add_argument('--config', type=str, default='config/train_shakespeare_small.py')
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds to run')
    parser.add_argument('--start-seed', type=int, default=42, help='Starting seed value')
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    base_dir = script_dir / 'experiments' / f'multiseed_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MULTI-SEED SCRIPTIO CONTINUA EXPERIMENT")
    print(f"Config: {args.config}")
    print(f"Seeds: {args.seeds} (starting from {args.start_seed})")
    print(f"Output: {base_dir}")
    print("=" * 70)

    # Run experiments
    all_results = []
    for i in range(args.seeds):
        seed = args.start_seed + i
        result = run_single_experiment(args.config, seed, base_dir)
        all_results.append(result)

    # Aggregate and analyze
    stats = aggregate_results(all_results)
    significance = compute_significance(stats)

    # Save aggregated results
    summary = {
        'config': args.config,
        'num_seeds': args.seeds,
        'start_seed': args.start_seed,
        'stats': stats,
        'significance': significance,
        'timestamp': datetime.now().isoformat(),
    }

    with open(base_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    print_summary(stats, significance)

    print(f"\n\nFull results saved to: {base_dir / 'summary.json'}")

if __name__ == '__main__':
    main()
