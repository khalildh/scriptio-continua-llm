"""
Evaluate saved checkpoints on the full validation set, on Modal.

Unlike the sampled [eval] lines in the training log, this makes one
deterministic pass over the entire val split and reports:
  - nats/char and bits/char (BPC) on that condition's own text
  - total bits over the val set
  - bits per *modern* character (total bits / modern val length), which makes
    conditions comparable even though removing spaces/punctuation shortens text

Usage:
    modal run modal_eval.py                       # WikiText, all conditions
    modal run modal_eval.py --corpus shakespeare
"""

import modal

app = modal.App("scriptio-continua-eval")
volume = modal.Volume.from_name("scriptio-continua-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy>=1.24.0", "requests", "datasets", "tqdm")
    .add_local_file("model.py", "/root/project/model.py")
    .add_local_dir("data", "/root/project/data")
)

PREP = {
    "wikitext": ("data/prepare_wikitext.py", "wikitext"),
    "shakespeare": ("data/prepare_shakespeare.py", "shakespeare"),
}
CONDITIONS = ["modern", "uppercase", "nopunct", "nospaces", "scriptio"]


@app.function(image=image, gpu="L4", timeout=3600, volumes={"/root/output": volume})
def evaluate(corpus: str = "wikitext"):
    import glob
    import math
    import os
    import subprocess
    import sys

    import numpy as np
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root/project")
    from model import GPT, GPTConfig

    prep_script, prefix = PREP[corpus]
    subprocess.run([sys.executable, prep_script], cwd="/root/project", check=True,
                   stdout=subprocess.DEVNULL)

    def val_tokens(cond):
        path = f"/root/project/data/{prefix}_{cond}/val.bin"
        return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))

    modern_len = len(val_tokens("modern"))
    results = []

    for cond in CONDITIONS:
        data = val_tokens(cond)
        ckpts = sorted(glob.glob(f"/root/output/{corpus}-{cond}/ckpt*.pt"))
        for ckpt_path in ckpts:
            name = os.path.basename(ckpt_path)
            try:
                ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
            except Exception as e:  # e.g. a save interrupted by a timeout
                results.append({"condition": cond, "ckpt": name, "error": repr(e)[:120]})
                continue

            model = GPT(GPTConfig(**ckpt["model_args"]))
            sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
            model.load_state_dict(sd)
            model.to("cuda").eval()
            T = ckpt["model_args"]["block_size"]

            # Non-overlapping windows of length T; predict tokens 1..N-1.
            n = (len(data) - 1) // T
            X = data[: n * T].view(n, T)
            Y = data[1 : n * T + 1].view(n, T)
            total_nats = 0.0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, n, 256):
                    x, y = X[i : i + 256].cuda(), Y[i : i + 256].cuda()
                    logits, _ = model(x, y)
                    total_nats += F.cross_entropy(
                        logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="sum"
                    ).item()
            count = n * T
            total_bits = total_nats / math.log(2)
            results.append({
                "condition": cond,
                "ckpt": name,
                "iter": ckpt.get("iter_num"),
                "nats_per_char": total_nats / count,
                "bpc": total_bits / count,
                # Scale this condition's per-char rate up to its full val text,
                # then spread over the modern text's length.
                "bits_per_modern_char": (total_bits / count) * len(data) / modern_len,
                "val_chars": len(data),
            })
            del model, ckpt
            torch.cuda.empty_cache()

    return {"modern_val_chars": modern_len, "results": results}


@app.local_entrypoint()
def main(corpus: str = "wikitext"):
    import json

    out = evaluate.remote(corpus)
    print(json.dumps(out, indent=2))
    print(f"\n{'condition':<10} {'ckpt':<20} {'iter':>5} {'nats/ch':>8} {'BPC':>6} {'bits/modern ch':>15}")
    for r in out["results"]:
        if "error" in r:
            print(f"{r['condition']:<10} {r['ckpt']:<20}  ERROR {r['error']}")
            continue
        print(f"{r['condition']:<10} {r['ckpt']:<20} {r['iter']:>5} {r['nats_per_char']:>8.4f} "
              f"{r['bpc']:>6.3f} {r['bits_per_modern_char']:>15.3f}")
