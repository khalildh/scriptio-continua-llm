# Scriptio Continua LLM

**Do language models learn differently when trained on text without spaces?**

This project investigates how removing word boundaries (spaces), punctuation, and case distinction affects language model learning. We compare models trained on:

1. **Modern text**: Standard English with spaces, punctuation, and mixed case
2. **Scriptio continua**: Ancient writing style with no spaces, no punctuation, uppercase only

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

## Expected Results

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
