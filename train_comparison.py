#!/usr/bin/env python3
"""
Train both scriptio continua and modern text models for comparison.
Automatically resumes from checkpoints if available.
"""

import subprocess
import sys
import os

CONFIGS = [
    'config/train_shakespeare_modern.py',  # modern text first (easier to evaluate)
    'config/train_shakespeare_small.py',   # scriptio continua
]

def get_init_from(out_dir):
    """Check if checkpoint exists and return appropriate init_from value."""
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if os.path.exists(ckpt_path):
        print(f"  Found checkpoint at {ckpt_path}, resuming...")
        return 'resume'
    else:
        print(f"  No checkpoint found, starting from scratch...")
        return 'scratch'

def run_training(config_path):
    """Run training for a single config."""
    # Extract out_dir from config to check for checkpoint
    out_dir = None
    with open(config_path, 'r') as f:
        for line in f:
            if line.strip().startswith('out_dir'):
                # Parse out_dir = 'value'
                out_dir = line.split('=')[1].strip().strip("'\"")
                break

    if out_dir is None:
        out_dir = 'out'  # default

    init_from = get_init_from(out_dir)

    cmd = [
        sys.executable, 'train.py', config_path,
        f'--init_from={init_from}'
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    print("=" * 60)

    result = subprocess.run(cmd)
    return result.returncode

def main():
    print("=" * 60)
    print("TRAINING COMPARISON: Scriptio Continua vs Modern Text")
    print("=" * 60)

    for i, config in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}] Training with {config}")

        returncode = run_training(config)

        if returncode != 0:
            print(f"\nTraining failed for {config} with return code {returncode}")
            # Continue to next config instead of failing completely
            continue

        print(f"\nCompleted training for {config}")

    print("\n" + "=" * 60)
    print("All training runs complete!")
    print("Compare results in wandb project: scriptio-continua")
    print("=" * 60)

if __name__ == '__main__':
    main()
