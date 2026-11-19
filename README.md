# Scriptio Continua LLM

**Do language models learn differently when trained on text without spaces?**

This project investigates how removing word boundaries (spaces), punctuation, and case distinction affects language model learning. We compare models trained on:

1. **Modern text**: Standard English with spaces, punctuation, and mixed case
2. **Scriptio continua**: Ancient writing style with no spaces, no punctuation, uppercase only

Three ablation conditions (`python run_experiment.py --ablation`) each change one variable at a time: **no-spaces**, **no-punctuation**, and **uppercase**. See [Results](#results).

## Background

Historical *scriptio continua* (Latin: "continuous script") was the standard writing convention in many ancient languages, including Greek, Latin, and still used today in languages like Mandarin and Japanese (without spaces between words).

### Research Questions

1. Can transformers learn to implicitly segment words without explicit boundaries?
2. Does removing spaces affect model compression efficiency (bits per character)?
3. How do generated samples differ between conditions?
4. Does the model learn different attention patterns?

## Quick Start

### Using DevContainer (Recommended)

1. Open in VS Code with the Dev Containers extension
2. Reopen in container
3. Run the experiment:

```bash
python run_experiment.py --config config/train_shakespeare_small.py
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data
python data/prepare_shakespeare.py

# Run experiment
python run_experiment.py --config config/train_shakespeare_small.py

# Analyze results
python scripts/analyze_results.py experiments/<experiment_name>
```

## Project Structure

```
scriptio-continua-llm/
├── config/                     # Training configurations
│   ├── train_shakespeare_small.py   # ~10M params, ~20 min
│   └── train_shakespeare_medium.py  # ~25M params, ~60 min
├── data/
│   └── prepare_shakespeare.py  # Creates both dataset versions
├── experiments/                # Experiment outputs (gitignored)
├── scripts/
│   └── analyze_results.py      # Paper-quality figures & stats
├── model.py                    # GPT model (from nanoGPT)
├── train.py                    # Training loop (from nanoGPT)
├── sample.py                   # Generation script
├── run_experiment.py           # Main experiment runner
└── .devcontainer/              # ROCm GPU devcontainer config
```

## Example: Shakespeare Transformation

**Modern (Spaced):**
```
HAMLET: To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
```

**Scriptio Continua:**
```
HAMLETTOBEORNOTTOBETHATISTHEQUESTION
WHETHERTISNOBLERINTHEDEMINDTOSUFFER
THESLINGSANDARROWSOFOUTRAGEOUSFORTUNE
```

## Hardware Requirements

- **Minimum**: Any GPU with 4GB+ VRAM, or CPU with 16GB+ RAM
- **Recommended**: GPU with 8GB+ VRAM (tested on AMD Radeon 8060S)
- **Training time**: ~20 min (small), ~60 min (medium) on consumer GPU

### AMD GPU Support (Strix APUs)

The devcontainer is configured for AMD Strix Halo/Strix Point iGPUs (gfx1151) using [TheRock](https://github.com/ROCm/TheRock) nightly builds:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchvision torchaudio
```

See [TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) for other supported GPU architectures.

## Results

### Shakespeare ablation (April 2026)

Five character-level models were trained on Shakespeare on a Modal cloud GPU, one per condition, for about 1,400–1,500 iterations each. The three ablation conditions (`--ablation`) each change one variable, so the scriptio continua effect can be split into its parts.

| Condition | What changed | Vocab | Best val loss (nats/char) | BPC | Best at iter |
|---|---|---|---|---|---|
| uppercase | case only | 39 | **1.450** | 2.09 | 500 |
| modern (baseline) | nothing | 65 | 1.473 | 2.12 | 550 |
| nopunct | punctuation only | 55 | 1.488 | 2.15 | 550 |
| nospaces | spaces only | 64 | 1.825 | 2.63 | 450 |
| scriptio | spaces + punctuation + case | 26 | 1.846 | 2.66 | 450 |

**Findings**

- **Removing spaces causes almost all of the penalty.** No-spaces alone raises validation loss by about 0.35 over the baseline, which is nearly the whole gap to full scriptio continua (about 0.37).
- **Case and punctuation barely matter.** Uppercasing helped slightly (−0.02), probably because the vocabulary is smaller. Removing punctuation hurt slightly (+0.015). Both effects are within what you'd expect from noise.
- **Once spaces are gone, the other changes add almost nothing.** Scriptio continua is only about 0.02 worse than no-spaces alone.

**Caveats**

- **Every run overfit badly.** Validation loss bottomed out around iterations 450–550, then rose while training loss kept falling (for example, scriptio ended at 0.12 train loss and 3.35 val loss). The table reports each run's best checkpoint. Later runs need more dropout or early stopping, or else more data.
- **Loss per character isn't fully comparable across conditions.** Texts without spaces have fewer characters, and spaces are easy to predict, which lowers the baseline's per-character average. A fairer comparison is total bits over the validation set, normalized to the same original text length. That hasn't been computed yet.
- **Each condition was run once, with one seed.** Differences under about 0.05 shouldn't be treated as meaningful.
- **WikiText runs** (`wikitext-*`) have checkpoints on the Modal volume `scriptio-continua-vol` but no training logs, so they haven't been evaluated yet.

## Expected Outputs

After training, you'll get:

1. **Loss curves**: Training and validation loss over time
2. **BPC comparison**: Bits per character for each condition
3. **Generated samples**: Text samples from each model
4. **LaTeX table**: Paper-ready results table

## Acknowledgments

- Built on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- Inspired by discussions about tokenization efficiency in CJK languages

## Citation

If you use this work, please cite:

```bibtex
@misc{scriptio-continua-llm,
  author = {Your Name},
  title = {Scriptio Continua LLM: Learning Without Word Boundaries},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/scriptio-continua-llm}
}
```

## License

MIT License - see nanoGPT for original code license.
