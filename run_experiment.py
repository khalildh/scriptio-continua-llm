#!/usr/bin/env python3
"""
Main experiment runner for the Scriptio Continua LLM study.

This script:
1. Prepares both datasets (modern and scriptio continua)
2. Trains identical models on each dataset
3. Collects training metrics (loss curves, timing)
4. Generates samples from each model
5. Computes comparison metrics
6. Saves results for analysis

Usage:
    python run_experiment.py --config config/train_shakespeare_small.py
    python run_experiment.py --config config/train_shakespeare_medium.py
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

def run_command(cmd, cwd=None):
    """Run a command and stream output."""
    print(f"\n>>> {cmd}\n")
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
    process.wait()
    return process.returncode, ''.join(output_lines)

def prepare_data():
    """Prepare both datasets."""
    print("=" * 60)
    print("PREPARING DATASETS")
    print("=" * 60)

    script_dir = Path(__file__).parent
    return_code, _ = run_command(
        f"{sys.executable} data/prepare_shakespeare.py",
        cwd=script_dir
    )
    if return_code != 0:
        raise RuntimeError("Data preparation failed")

def train_model(dataset_name: str, config_path: str, output_dir: Path, experiment_name: str):
    """Train a model on the specified dataset."""
    print("\n" + "=" * 60)
    print(f"TRAINING: {experiment_name}")
    print("=" * 60)

    script_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build training command
    cmd = (
        f"{sys.executable} train.py "
        f"{config_path} "
        f"--data_dir=data/{dataset_name} "
        f"--out_dir={output_dir}"
    )

    start_time = time.time()
    return_code, output = run_command(cmd, cwd=script_dir)
    training_time = time.time() - start_time

    if return_code != 0:
        print(f"WARNING: Training may have had issues (return code {return_code})")

    # Save training log
    with open(output_dir / 'training_log.txt', 'w') as f:
        f.write(output)

    return {
        'training_time_seconds': training_time,
        'return_code': return_code,
    }

def generate_samples(model_dir: Path, dataset_name: str, num_samples: int = 5, max_tokens: int = 500):
    """Generate samples from a trained model."""
    print(f"\nGenerating {num_samples} samples from {model_dir}...")

    script_dir = Path(__file__).parent
    samples = []

    # Load model and generate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load meta for encoding/decoding
    meta_path = script_dir / 'data' / dataset_name / 'meta.pkl'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    itos = meta['itos']
    stoi = meta['stoi']

    def decode(tokens):
        return ''.join([itos[t] for t in tokens])

    # Load model
    ckpt_path = model_dir / 'ckpt.pt'
    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}")
        return []

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Import model class
    sys.path.insert(0, str(script_dir))
    from model import GPT, GPTConfig

    # Create model from checkpoint config
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    # Generate samples with different starting characters
    start_chars = ['\n', 'T', 'W', 'A', 'I']  # Common Shakespeare starts

    for i, start_char in enumerate(start_chars[:num_samples]):
        if start_char in stoi:
            start_ids = [stoi[start_char]]
        else:
            start_ids = [0]  # Fallback

        x = torch.tensor([start_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            y = model.generate(x, max_new_tokens=max_tokens, temperature=0.8, top_k=200)

        generated = decode(y[0].tolist())
        samples.append({
            'start': start_char,
            'generated': generated,
            'length': len(generated)
        })
        print(f"\n--- Sample {i+1} (start='{start_char}') ---")
        print(generated[:200] + "..." if len(generated) > 200 else generated)

    return samples

def compute_final_metrics(model_dir: Path, dataset_name: str):
    """Compute final evaluation metrics on validation set."""
    script_dir = Path(__file__).parent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    ckpt_path = model_dir / 'ckpt.pt'
    if not ckpt_path.exists():
        return {}

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Get metrics from checkpoint
    metrics = {
        'best_val_loss': checkpoint.get('best_val_loss', None),
        'iter_num': checkpoint.get('iter_num', None),
    }

    # Load validation data for additional metrics
    val_data_path = script_dir / 'data' / dataset_name / 'val.bin'
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')

    metrics['val_tokens'] = len(val_data)

    # Compute bits per character (BPC) from loss
    # BPC = loss / ln(2)
    if metrics['best_val_loss'] is not None:
        metrics['bits_per_char'] = metrics['best_val_loss'] / np.log(2)

    return metrics

def compare_experiments(modern_results: dict, scriptio_results: dict):
    """Compare results between modern and scriptio continua experiments."""
    comparison = {
        'timestamp': datetime.now().isoformat(),
    }

    # Loss comparison
    if modern_results.get('final_metrics') and scriptio_results.get('final_metrics'):
        m_loss = modern_results['final_metrics'].get('best_val_loss')
        s_loss = scriptio_results['final_metrics'].get('best_val_loss')

        if m_loss and s_loss:
            comparison['loss_difference'] = s_loss - m_loss
            comparison['loss_ratio'] = s_loss / m_loss
            comparison['modern_loss'] = m_loss
            comparison['scriptio_loss'] = s_loss

            # BPC comparison
            m_bpc = modern_results['final_metrics'].get('bits_per_char')
            s_bpc = scriptio_results['final_metrics'].get('bits_per_char')
            if m_bpc and s_bpc:
                comparison['modern_bpc'] = m_bpc
                comparison['scriptio_bpc'] = s_bpc
                comparison['bpc_difference'] = s_bpc - m_bpc

    # Training time comparison
    if modern_results.get('training_time_seconds') and scriptio_results.get('training_time_seconds'):
        comparison['modern_training_time'] = modern_results['training_time_seconds']
        comparison['scriptio_training_time'] = scriptio_results['training_time_seconds']

    return comparison

def main():
    parser = argparse.ArgumentParser(description='Run scriptio continua experiment')
    parser.add_argument('--config', type=str, default='config/train_shakespeare_small.py',
                        help='Training config file')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and just analyze existing results')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for this experiment run')
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Create experiment name
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        config_name = Path(args.config).stem
        exp_name = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exp_dir = script_dir / 'experiments' / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SCRIPTIO CONTINUA LLM EXPERIMENT")
    print(f"Experiment: {exp_name}")
    print(f"Config: {args.config}")
    print(f"Output: {exp_dir}")
    print("=" * 60)

    # Step 1: Prepare data
    if not args.skip_training:
        prepare_data()

    results = {
        'experiment_name': exp_name,
        'config': args.config,
        'timestamp': datetime.now().isoformat(),
    }

    if not args.skip_training:
        # Step 2: Train modern model
        modern_dir = exp_dir / 'modern'
        modern_training = train_model(
            'shakespeare_modern',
            args.config,
            modern_dir,
            'Modern (Spaced) Shakespeare'
        )
        results['modern'] = {'training': modern_training}

        # Step 3: Train scriptio continua model
        scriptio_dir = exp_dir / 'scriptio'
        scriptio_training = train_model(
            'shakespeare_scriptio',
            args.config,
            scriptio_dir,
            'Scriptio Continua Shakespeare'
        )
        results['scriptio'] = {'training': scriptio_training}
    else:
        modern_dir = exp_dir / 'modern'
        scriptio_dir = exp_dir / 'scriptio'
        results['modern'] = {}
        results['scriptio'] = {}

    # Step 4: Compute final metrics
    print("\n" + "=" * 60)
    print("COMPUTING FINAL METRICS")
    print("=" * 60)

    results['modern']['final_metrics'] = compute_final_metrics(modern_dir, 'shakespeare_modern')
    results['scriptio']['final_metrics'] = compute_final_metrics(scriptio_dir, 'shakespeare_scriptio')

    # Step 5: Generate samples
    print("\n" + "=" * 60)
    print("GENERATING SAMPLES")
    print("=" * 60)

    print("\n--- Modern Model Samples ---")
    results['modern']['samples'] = generate_samples(modern_dir, 'shakespeare_modern')

    print("\n--- Scriptio Continua Model Samples ---")
    results['scriptio']['samples'] = generate_samples(scriptio_dir, 'shakespeare_scriptio')

    # Step 6: Compare results
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    comparison = compare_experiments(results['modern'], results['scriptio'])
    results['comparison'] = comparison

    # Print summary
    if comparison.get('modern_loss') and comparison.get('scriptio_loss'):
        print(f"\n{'Metric':<30} {'Modern':<15} {'Scriptio':<15} {'Diff':<15}")
        print("-" * 75)
        print(f"{'Validation Loss':<30} {comparison['modern_loss']:<15.4f} {comparison['scriptio_loss']:<15.4f} {comparison['loss_difference']:+.4f}")
        if comparison.get('modern_bpc'):
            print(f"{'Bits per Character':<30} {comparison['modern_bpc']:<15.4f} {comparison['scriptio_bpc']:<15.4f} {comparison['bpc_difference']:+.4f}")

    # Save results
    results_path = exp_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    if comparison.get('loss_difference'):
        if comparison['loss_difference'] > 0.1:
            print("- Scriptio continua shows HIGHER loss (harder to learn)")
        elif comparison['loss_difference'] < -0.1:
            print("- Scriptio continua shows LOWER loss (easier to learn)")
        else:
            print("- Similar performance between conditions")

    print("\nNext steps for research paper:")
    print("1. Run multiple seeds for statistical significance")
    print("2. Analyze generated samples for coherence")
    print("3. Test on held-out data from different Shakespeare plays")
    print("4. Compare tokenization patterns")
    print("5. Visualize attention patterns")

if __name__ == '__main__':
    main()
