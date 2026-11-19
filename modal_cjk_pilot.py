"""
CJK Shakespeare pilot: train and evaluate every condition on Modal.

Prepare the data locally first (needs jieba, fugashi, unidic-lite):
    python data/extract_cjk_shakespeare.py
    python data/prepare_cjk_shakespeare.py

Then:
    modal run modal_cjk_pilot.py              # 8 conditions x 3 seeds
    modal run modal_cjk_pilot.py --seeds 1

Each run trains a small byte-level GPT on Acts 1-4, keeps the checkpoint with
the best validation loss, and scores it on Act 5 (held out). Because every
language's Act 5 is a translation of the same English Act 5, the key metric is
total bits on Act 5 divided by the *English* Act 5 character count:
"bits per English character". Results are written to results/cjk_pilot.json.
"""
import json
from pathlib import Path

import modal

app = modal.App("scriptio-continua-cjk-pilot")

CONDITIONS = [
    "zh_en_modern", "zh_en_nospaces", "zh_zh_natural", "zh_zh_segmented",
    "ja_en_modern", "ja_en_nospaces", "ja_ja_natural", "ja_ja_segmented",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy>=1.24.0", "wandb")
    .add_local_file("train.py", "/root/project/train.py")
    .add_local_file("model.py", "/root/project/model.py")
    .add_local_file("configurator.py", "/root/project/configurator.py")
    .add_local_dir("data", "/root/project/data",
                   ignore=["cjk_raw/**", "wikitext_*/**", "shakespeare_*/**", "**/__pycache__/**"])
)

MODEL_ARGS = [
    "--n_layer=4", "--n_head=4", "--n_embd=256", "--block_size=256",
    "--dropout=0.2", "--bias=False",
]
TRAIN_ARGS = [
    "--batch_size=32", "--gradient_accumulation_steps=1",
    "--max_iters=2500", "--lr_decay_iters=2500", "--warmup_iters=100",
    "--learning_rate=1e-3", "--min_lr=1e-4", "--weight_decay=1e-1",
    "--eval_interval=50", "--eval_iters=40", "--log_interval=100",
    "--always_save_checkpoint=False",  # keep only the best-val checkpoint
    "--compile=False", "--wandb_log=False", "--device=cuda",
]


def strided_bits(model, data, T, stride, device="cuda"):
    """Total bits over `data`, giving every scored token >= T - stride context."""
    import math
    import torch
    import torch.nn.functional as F

    total_nats, scored, prev_end = 0.0, 0, 0
    starts = list(range(0, max(len(data) - 1, 1), stride))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in starts:
            e = min(s + T, len(data) - 1)
            if e <= prev_end:
                break
            x = data[s:e].unsqueeze(0).to(device)
            y = data[s + 1:e + 1].unsqueeze(0).to(device)
            logits, _ = model(x, y)
            nll = F.cross_entropy(logits.float()[0], y[0], reduction="none")
            keep = e - prev_end  # only score targets not already scored
            total_nats += nll[-keep:].sum().item()
            scored += keep
            prev_end = e
    assert scored == len(data) - 1, (scored, len(data))
    return total_nats / math.log(2), scored


@app.function(image=image, gpu="L4", timeout=1800)
def run(condition: str, seed: int):
    import json as _json
    import re
    import subprocess
    import sys

    import numpy as np
    import torch

    project = "/root/project"
    out_dir = f"/tmp/out/{condition}_s{seed}"
    cmd = [sys.executable, "train.py", f"--out_dir={out_dir}", f"--dataset=cjk_{condition}",
           f"--seed={seed}", *MODEL_ARGS, *TRAIN_ARGS]
    res = subprocess.run(cmd, cwd=project, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{condition} s{seed} failed:\n{res.stderr[-3000:]}")

    evals = [(int(i), float(tr), float(va)) for i, tr, va in
             re.findall(r"\[eval\] iter (\d+): train loss ([\d.]+), val loss ([\d.]+)", res.stdout)]

    sys.path.insert(0, project)
    from model import GPT, GPTConfig
    ckpt = torch.load(f"{out_dir}/ckpt.pt", map_location="cuda", weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
    model.to("cuda").eval()

    data_dir = f"{project}/data/cjk_{condition}"
    test = torch.from_numpy(np.fromfile(f"{data_dir}/test.bin", dtype=np.uint16).astype(np.int64))
    bits, scored = strided_bits(model, test, T=256, stride=128)
    stats = _json.load(open(f"{data_dir}/stats.json"))

    return {
        "condition": condition, "seed": seed, "pair": stats["pair"], "lang": stats["lang"],
        "best_iter": ckpt["iter_num"], "best_val_nats_per_byte": float(ckpt["best_val_loss"]),
        "test_bits": bits, "test_bytes_scored": scored,
        "test_bytes": stats["test_bytes"], "test_chars": stats["test_chars"],
        "bits_per_byte": bits / scored,
        "evals": evals,
    }


@app.local_entrypoint()
def main(seeds: int = 3):
    jobs = [(c, 1337 + k) for c in CONDITIONS for k in range(seeds)]
    results = list(run.starmap(jobs))

    # Normalize by the English test length of the same pair (same Act 5 text).
    en_chars = {r["pair"]: r["test_chars"] for r in results if r["condition"].endswith("en_modern")}
    for r in results:
        r["bits_per_en_char"] = r["test_bits"] / en_chars[r["pair"]]
        r["bits_per_own_char"] = r["test_bits"] / r["test_chars"]

    out = Path("results/cjk_pilot.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    import statistics as st
    print(f"\n{'condition':<18} {'best it':>7} {'bits/byte':>10} {'bits/own ch':>12} {'bits/EN ch':>18}")
    for c in CONDITIONS:
        rs = [r for r in results if r["condition"] == c]
        def ms(k):
            v = [r[k] for r in rs]
            return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
        bpe, sd = ms("bits_per_en_char")
        print(f"{c:<18} {st.mean(r['best_iter'] for r in rs):>7.0f} {ms('bits_per_byte')[0]:>10.3f} "
              f"{ms('bits_per_own_char')[0]:>12.3f} {bpe:>10.3f} ± {sd:.3f}")
    print(f"\nSaved {len(results)} runs to {out}")
