"""
Run scriptio continua training on Modal with GPU acceleration.

Usage:
    # Train all 5 conditions on WikiText-103 (default):
    modal run modal_train.py

    # Train on Shakespeare instead:
    modal run modal_train.py --corpus shakespeare

    # Train only original 2 conditions (no ablation):
    modal run modal_train.py --no-ablation

    # Train a single condition:
    modal run modal_train.py --condition scriptio
"""

import modal

app = modal.App("scriptio-continua-training")

# Create a persistent volume to store training outputs across runs
volume = modal.Volume.from_name("scriptio-continua-vol", create_if_missing=True)

# Build the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "requests",
        "datasets",
        "tqdm",
        "wandb",
    )
    .add_local_file("train.py", "/root/project/train.py")
    .add_local_file("model.py", "/root/project/model.py")
    .add_local_file("configurator.py", "/root/project/configurator.py")
    .add_local_dir("config", "/root/project/config")
    .add_local_dir("data", "/root/project/data")
)

# Corpus configurations
CORPORA = {
    "wikitext": {
        "prep_script": "data/prepare_wikitext.py",
        "prefix": "wikitext",
        "wandb_prefix": "wiki",
    },
    "shakespeare": {
        "prep_script": "data/prepare_shakespeare.py",
        "prefix": "shakespeare",
        "wandb_prefix": "shk",
    },
}

# Condition suffixes (appended to corpus prefix)
CONDITION_SUFFIXES = {
    "modern": {"suffix": "modern", "label": "Modern (baseline)"},
    "scriptio": {"suffix": "scriptio", "label": "Scriptio Continua"},
    "nospaces": {"suffix": "nospaces", "label": "No-Spaces (ablation)"},
    "nopunct": {"suffix": "nopunct", "label": "No-Punctuation (ablation)"},
    "uppercase": {"suffix": "uppercase", "label": "Uppercase (ablation)"},
}


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    volumes={"/root/output": volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def train_condition(condition_name: str, corpus: str = "wikitext"):
    """Train a single condition on a GPU."""
    import subprocess
    import os
    import sys

    corpus_cfg = CORPORA[corpus]
    cond = CONDITION_SUFFIXES[condition_name]
    dataset_name = f"{corpus_cfg['prefix']}_{cond['suffix']}"
    wandb_name = f"{corpus_cfg['wandb_prefix']}-{condition_name}"

    project_dir = "/root/project"
    output_dir = f"/root/output/{corpus}-{condition_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the data
    print(f"\n{'='*60}")
    print(f"Preparing data: {cond['label']} ({corpus})")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, corpus_cfg["prep_script"]],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
        raise RuntimeError("Data preparation failed")

    # Train
    print(f"\n{'='*60}")
    print(f"Training: {cond['label']} on {corpus}")
    print(f"Dataset: {dataset_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "train.py",
        f"--out_dir={output_dir}",
        f"--dataset={dataset_name}",
        "--wandb_log=True",
        "--wandb_project=scriptio-continua",
        f"--wandb_run_name={wandb_name}",
        "--compile=True",
        "--device=cuda",
        # Model architecture (same as Shakespeare configs)
        "--n_layer=6",
        "--n_head=6",
        "--n_embd=384",
        "--dropout=0.2",
        # Training
        "--batch_size=64",
        "--block_size=256",
        "--max_iters=5000",
        "--eval_interval=250",
        "--eval_iters=200",
        "--log_interval=10",
        # Learning rate
        "--learning_rate=1e-3",
        "--min_lr=1e-4",
        "--lr_decay_iters=5000",
        "--warmup_iters=100",
        "--weight_decay=1e-1",
        # Checkpointing
        "--always_save_checkpoint=True",
    ]

    result = subprocess.run(
        cmd,
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR (last 2000 chars): {result.stderr[-2000:]}")
        raise RuntimeError(f"Training failed for {condition_name}")

    # Read final metrics from the last eval line
    lines = result.stdout.strip().split("\n")
    eval_lines = [l for l in lines if "[eval]" in l]
    last_eval = eval_lines[-1] if eval_lines else "no eval found"

    volume.commit()

    return {
        "condition": condition_name,
        "corpus": corpus,
        "label": cond["label"],
        "last_eval": last_eval,
        "output_dir": output_dir,
    }


@app.local_entrypoint()
def main(
    no_ablation: bool = False,
    condition: str = "",
    corpus: str = "wikitext",
):
    """Run training across conditions, in parallel on separate GPUs."""

    if corpus not in CORPORA:
        print(f"Unknown corpus: {corpus}")
        print(f"Available: {', '.join(CORPORA.keys())}")
        return

    if condition:
        if condition not in CONDITION_SUFFIXES:
            print(f"Unknown condition: {condition}")
            print(f"Available: {', '.join(CONDITION_SUFFIXES.keys())}")
            return
        conditions = [condition]
    elif no_ablation:
        conditions = ["modern", "scriptio"]
    else:
        conditions = list(CONDITION_SUFFIXES.keys())

    print(f"\n{'='*60}")
    print(f"SCRIPTIO CONTINUA - MODAL TRAINING")
    print(f"Corpus: {corpus}")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"{'='*60}\n")

    # Launch all conditions in parallel
    results = list(train_condition.map(
        conditions,
        kwargs={"corpus": corpus},
    ))

    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"\n  {r['label']} ({r['corpus']}):")
        print(f"    {r['last_eval']}")

    print(f"\nOutputs saved to Modal volume 'scriptio-continua-vol'")
    print("Download with: modal volume get scriptio-continua-vol <path>")
