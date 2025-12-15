#!/usr/bin/env python3
"""
Analysis script for scriptio continua experiment results.

Generates research-paper quality plots and statistics:
1. Training loss curves comparison
2. Bits per character (BPC) comparison
3. Sample quality analysis
4. Statistical significance tests
5. Tokenization efficiency analysis

Usage:
    python scripts/analyze_results.py experiments/train_shakespeare_small_20240101_120000
"""

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set style for paper-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

def parse_training_log(log_path: Path):
    """Parse training log to extract loss curves."""
    if not log_path.exists():
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # Parse lines like: "iter 100: loss 2.1234, time 1234.56ms"
    pattern = r'iter (\d+): loss ([\d.]+)'
    matches = re.findall(pattern, content)

    if not matches:
        return None

    iters = [int(m[0]) for m in matches]
    losses = [float(m[1]) for m in matches]

    # Also parse validation losses: "step 100: train loss 2.1234, val loss 2.3456"
    val_pattern = r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)'
    val_matches = re.findall(val_pattern, content)

    val_iters = [int(m[0]) for m in val_matches]
    train_losses = [float(m[1]) for m in val_matches]
    val_losses = [float(m[2]) for m in val_matches]

    return {
        'train_iters': iters,
        'train_losses': losses,
        'eval_iters': val_iters,
        'eval_train_losses': train_losses,
        'eval_val_losses': val_losses,
    }

def plot_loss_curves(exp_dir: Path, output_dir: Path):
    """Plot training and validation loss curves."""
    modern_log = parse_training_log(exp_dir / 'modern' / 'training_log.txt')
    scriptio_log = parse_training_log(exp_dir / 'scriptio' / 'training_log.txt')

    if not modern_log or not scriptio_log:
        print("Could not parse training logs")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    ax = axes[0]
    ax.plot(modern_log['train_iters'], modern_log['train_losses'],
            alpha=0.7, label='Modern (Spaced)', color='#2196F3')
    ax.plot(scriptio_log['train_iters'], scriptio_log['train_losses'],
            alpha=0.7, label='Scriptio Continua', color='#FF5722')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.set_yscale('log')

    # Validation loss
    ax = axes[1]
    if modern_log['eval_iters']:
        ax.plot(modern_log['eval_iters'], modern_log['eval_val_losses'],
                'o-', label='Modern (Spaced)', color='#2196F3', markersize=4)
    if scriptio_log['eval_iters']:
        ax.plot(scriptio_log['eval_iters'], scriptio_log['eval_val_losses'],
                's-', label='Scriptio Continua', color='#FF5722', markersize=4)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'loss_curves.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: loss_curves.png/pdf")

def plot_bpc_comparison(results: dict, output_dir: Path):
    """Plot bits per character comparison bar chart."""
    modern_bpc = results.get('modern', {}).get('final_metrics', {}).get('bits_per_char')
    scriptio_bpc = results.get('scriptio', {}).get('final_metrics', {}).get('bits_per_char')

    if not modern_bpc or not scriptio_bpc:
        print("BPC data not available")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    conditions = ['Modern\n(Spaced)', 'Scriptio\nContinua']
    bpcs = [modern_bpc, scriptio_bpc]
    colors = ['#2196F3', '#FF5722']

    bars = ax.bar(conditions, bpcs, color=colors, width=0.6, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, bpc in zip(bars, bpcs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bpc:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Bits per Character (BPC)')
    ax.set_title('Model Compression Efficiency: Bits per Character')
    ax.set_ylim(0, max(bpcs) * 1.2)

    # Add annotation about what lower BPC means
    ax.annotate('Lower is better\n(more efficient encoding)',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'bpc_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'bpc_comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: bpc_comparison.png/pdf")

def analyze_samples(results: dict, output_dir: Path):
    """Analyze generated samples for coherence and patterns."""
    analysis = {
        'modern': {},
        'scriptio': {},
    }

    for condition in ['modern', 'scriptio']:
        samples = results.get(condition, {}).get('samples', [])
        if not samples:
            continue

        all_text = ' '.join(s.get('generated', '') for s in samples)

        # Basic statistics
        analysis[condition] = {
            'num_samples': len(samples),
            'avg_length': np.mean([s.get('length', 0) for s in samples]),
            'total_chars': len(all_text),
        }

        # Character frequency
        char_freq = Counter(all_text)
        analysis[condition]['top_chars'] = char_freq.most_common(10)

        # For modern: word-level analysis
        if condition == 'modern':
            words = all_text.split()
            analysis[condition]['num_words'] = len(words)
            analysis[condition]['unique_words'] = len(set(words))
            analysis[condition]['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0

        # For scriptio: look for repeated patterns (potential learned "words")
        if condition == 'scriptio':
            # Find common n-grams that might be learned word boundaries
            ngram_sizes = [3, 4, 5, 6]
            for n in ngram_sizes:
                ngrams = [all_text[i:i+n] for i in range(len(all_text)-n+1)]
                ngram_freq = Counter(ngrams)
                analysis[condition][f'top_{n}grams'] = ngram_freq.most_common(5)

    # Save analysis
    with open(output_dir / 'sample_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"Saved: sample_analysis.json")

    # Print summary
    print("\n" + "=" * 60)
    print("SAMPLE ANALYSIS SUMMARY")
    print("=" * 60)
    for condition in ['modern', 'scriptio']:
        if condition in analysis and analysis[condition]:
            print(f"\n{condition.upper()}:")
            for key, value in analysis[condition].items():
                if not key.startswith('top_'):
                    print(f"  {key}: {value}")

    return analysis

def compute_statistics(results: dict):
    """Compute statistical summary."""
    stats = {}

    modern_loss = results.get('modern', {}).get('final_metrics', {}).get('best_val_loss')
    scriptio_loss = results.get('scriptio', {}).get('final_metrics', {}).get('best_val_loss')

    if modern_loss and scriptio_loss:
        stats['loss_difference'] = scriptio_loss - modern_loss
        stats['loss_percent_diff'] = (scriptio_loss - modern_loss) / modern_loss * 100

        # Effect size (Cohen's d approximation - would need multiple runs for proper calculation)
        stats['approximate_effect'] = 'large' if abs(stats['loss_percent_diff']) > 10 else \
                                       'medium' if abs(stats['loss_percent_diff']) > 5 else 'small'

    return stats

def generate_latex_table(results: dict, output_dir: Path):
    """Generate LaTeX table for paper."""
    modern = results.get('modern', {}).get('final_metrics', {})
    scriptio = results.get('scriptio', {}).get('final_metrics', {})

    latex = r"""
\begin{table}[h]
\centering
\caption{Comparison of model performance on modern (spaced) vs. scriptio continua Shakespeare text.}
\label{tab:results}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Modern (Spaced)} & \textbf{Scriptio Continua} \\
\midrule
"""

    if modern.get('best_val_loss') and scriptio.get('best_val_loss'):
        latex += f"Validation Loss & {modern['best_val_loss']:.4f} & {scriptio['best_val_loss']:.4f} \\\\\n"

    if modern.get('bits_per_char') and scriptio.get('bits_per_char'):
        latex += f"Bits per Character & {modern['bits_per_char']:.4f} & {scriptio['bits_per_char']:.4f} \\\\\n"

    if modern.get('val_tokens') and scriptio.get('val_tokens'):
        latex += f"Validation Tokens & {modern['val_tokens']:,} & {scriptio['val_tokens']:,} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write(latex)

    print(f"Saved: results_table.tex")

def main():
    parser = argparse.ArgumentParser(description='Analyze scriptio continua experiment results')
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return

    # Load results
    results_path = exp_dir / 'results.json'
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        print(f"No results.json found in {exp_dir}")
        results = {}

    # Create output directory for figures
    output_dir = exp_dir / 'figures'
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("ANALYZING EXPERIMENT RESULTS")
    print(f"Directory: {exp_dir}")
    print("=" * 60)

    # Generate all analyses
    plot_loss_curves(exp_dir, output_dir)
    plot_bpc_comparison(results, output_dir)
    analyze_samples(results, output_dir)
    stats = compute_statistics(results)
    generate_latex_table(results, output_dir)

    # Print final statistics
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nAll figures saved to: {output_dir}")

if __name__ == '__main__':
    main()
