"""
Prepare the Shakespeare dataset for the scriptio continua experiment.

Creates two versions:
1. Modern (spaced): Original text with spaces, punctuation, and case
2. Scriptio Continua: No spaces, no punctuation, all uppercase

Both use character-level encoding for fair comparison.
"""
import os
import pickle
import requests
import numpy as np
from pathlib import Path

def download_shakespeare():
    """Download the tiny shakespeare dataset."""
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    return requests.get(data_url).text

def to_scriptio_continua(text: str) -> str:
    """
    Convert modern text to scriptio continua.

    Historical scriptio continua characteristics:
    - No word boundaries (no spaces)
    - No punctuation
    - No case distinction (we use uppercase)
    - No newlines - pure continuous text
    """
    result = []
    for char in text:
        if char.isalpha():
            result.append(char.upper())
        # Everything else (spaces, punctuation, numbers, newlines) is dropped
    return ''.join(result)

def prepare_dataset(data: str, output_dir: Path, dataset_name: str):
    """Prepare a dataset for training."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    print(f"\n=== {dataset_name} ===")
    print(f"Length in characters: {len(data):,}")
    print(f"Unique characters: {''.join(chars)}")
    print(f"Vocab size: {vocab_size}")

    # Create character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    # Train/val split (90/10)
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    # Encode
    train_ids = encode(train_data)
    val_ids = encode(val_data)

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")

    # Save binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(output_dir / 'train.bin')
    val_ids.tofile(output_dir / 'val.bin')

    # Save metadata
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'chars': chars,
        'dataset_name': dataset_name,
    }
    with open(output_dir / 'meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

    # Save raw text for inspection
    with open(output_dir / 'input.txt', 'w') as f:
        f.write(data)

    # Save sample for quick inspection
    sample_len = min(500, len(data))
    with open(output_dir / 'sample.txt', 'w') as f:
        f.write(f"=== {dataset_name} Sample (first {sample_len} chars) ===\n\n")
        f.write(data[:sample_len])

    return {
        'vocab_size': vocab_size,
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
        'total_chars': len(data),
    }

def main():
    base_dir = Path(__file__).parent

    # Download Shakespeare
    print("Downloading Shakespeare corpus...")
    raw_text = download_shakespeare()

    # Prepare modern (spaced) version
    modern_stats = prepare_dataset(
        raw_text,
        base_dir / 'shakespeare_modern',
        'Shakespeare Modern (Spaced)'
    )

    # Prepare scriptio continua version
    scriptio_text = to_scriptio_continua(raw_text)
    scriptio_stats = prepare_dataset(
        scriptio_text,
        base_dir / 'shakespeare_scriptio',
        'Shakespeare Scriptio Continua'
    )

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Modern':<15} {'Scriptio':<15} {'Ratio':<10}")
    print("-" * 60)

    print(f"{'Total characters':<25} {modern_stats['total_chars']:<15,} {scriptio_stats['total_chars']:<15,} {scriptio_stats['total_chars']/modern_stats['total_chars']:.2f}")
    print(f"{'Vocabulary size':<25} {modern_stats['vocab_size']:<15} {scriptio_stats['vocab_size']:<15} {scriptio_stats['vocab_size']/modern_stats['vocab_size']:.2f}")
    print(f"{'Train tokens':<25} {modern_stats['train_tokens']:<15,} {scriptio_stats['train_tokens']:<15,} {scriptio_stats['train_tokens']/modern_stats['train_tokens']:.2f}")
    print(f"{'Val tokens':<25} {modern_stats['val_tokens']:<15,} {scriptio_stats['val_tokens']:<15,} {scriptio_stats['val_tokens']/modern_stats['val_tokens']:.2f}")

    # The key insight: scriptio continua has fewer tokens because spaces/punctuation removed
    # but the model must learn to segment implicitly
    compression = 1 - (scriptio_stats['total_chars'] / modern_stats['total_chars'])
    print(f"\nScriptio continua compression: {compression:.1%} fewer characters")
    print("(But model must learn implicit word boundaries)")

if __name__ == '__main__':
    main()
