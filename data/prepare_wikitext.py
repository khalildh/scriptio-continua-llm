"""
Prepare the WikiText-103 dataset for the scriptio continua experiment.

Creates multiple versions to isolate experimental variables:
1. Modern (spaced): Original text with spaces, punctuation, and case
2. Scriptio Continua: No spaces, no punctuation, all uppercase
3. No-spaces: Spaces removed only (keeps punctuation and mixed case)
4. No-punctuation: Punctuation removed only (keeps spaces and mixed case)
5. Uppercase: Uppercase only (keeps spaces and punctuation)

WikiText-103 is ~100M characters, providing a much larger and more diverse
corpus than tiny Shakespeare, reducing overfitting in small models.
"""
import os
import pickle
import string
import numpy as np
from pathlib import Path


def download_wikitext():
    """Download WikiText-103 raw character-level dataset via HuggingFace."""
    from datasets import load_dataset

    print("Downloading WikiText-103 via HuggingFace datasets...")
    ds = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1")

    train_text = "\n".join(ds["train"]["text"])
    val_text = "\n".join(ds["validation"]["text"])

    print(f"Downloaded: {len(train_text):,} train chars, {len(val_text):,} val chars")
    return train_text, val_text


def to_scriptio_continua(text: str) -> str:
    """No spaces, no punctuation, all uppercase."""
    result = []
    for char in text:
        if char.isalpha():
            result.append(char.upper())
    return ''.join(result)


def to_nospaces(text: str) -> str:
    """Remove spaces only."""
    return text.replace(' ', '')


def to_nopunct(text: str) -> str:
    """Remove punctuation only."""
    return ''.join(ch for ch in text if ch not in string.punctuation)


def to_uppercase(text: str) -> str:
    """Convert to uppercase only."""
    return text.upper()


def prepare_dataset(train_data: str, val_data: str, output_dir: Path, dataset_name: str):
    """Prepare a dataset for training."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all unique characters from combined data
    all_data = train_data + val_data
    chars = sorted(list(set(all_data)))
    vocab_size = len(chars)

    print(f"\n=== {dataset_name} ===")
    print(f"Train characters: {len(train_data):,}")
    print(f"Val characters: {len(val_data):,}")
    print(f"Vocab size: {vocab_size}")

    # Create character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

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

    # Save sample for quick inspection
    sample_len = min(500, len(train_data))
    with open(output_dir / 'sample.txt', 'w') as f:
        f.write(f"=== {dataset_name} Sample (first {sample_len} chars) ===\n\n")
        f.write(train_data[:sample_len])

    return {
        'vocab_size': vocab_size,
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
        'total_chars': len(train_data) + len(val_data),
    }


def main():
    base_dir = Path(__file__).parent

    # Download WikiText-103
    train_text, val_text = download_wikitext()

    # Define all variants
    variants = [
        (None, 'wikitext_modern', 'WikiText-103 Modern (Spaced)'),
        (to_scriptio_continua, 'wikitext_scriptio', 'WikiText-103 Scriptio Continua'),
        (to_nospaces, 'wikitext_nospaces', 'WikiText-103 No-Spaces (ablation)'),
        (to_nopunct, 'wikitext_nopunct', 'WikiText-103 No-Punctuation (ablation)'),
        (to_uppercase, 'wikitext_uppercase', 'WikiText-103 Uppercase (ablation)'),
    ]

    all_stats = {}
    for transform_fn, dir_name, dataset_name in variants:
        if transform_fn is None:
            t_train, t_val = train_text, val_text
        else:
            print(f"\nApplying transform for {dataset_name}...")
            t_train = transform_fn(train_text)
            t_val = transform_fn(val_text)

        all_stats[dir_name] = prepare_dataset(
            t_train, t_val,
            base_dir / dir_name,
            dataset_name,
        )

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    header = f"{'Metric':<20}"
    for name in all_stats:
        short = name.replace('wikitext_', '')
        header += f" {short:<12}"
    print(header)
    print("-" * 80)

    for metric, key in [('Total chars', 'total_chars'), ('Vocab size', 'vocab_size'),
                         ('Train tokens', 'train_tokens'), ('Val tokens', 'val_tokens')]:
        row = f"{metric:<20}"
        for stats in all_stats.values():
            row += f" {stats[key]:<12,}"
        print(row)

    modern = all_stats['wikitext_modern']
    scriptio = all_stats['wikitext_scriptio']
    compression = 1 - (scriptio['total_chars'] / modern['total_chars'])
    print(f"\nScriptio continua compression: {compression:.1%} fewer characters")


if __name__ == '__main__':
    main()
