# Scriptio Continua LLM

**Do language models learn differently when trained on text without spaces?**

This project investigates how removing word boundaries (spaces), punctuation, and case distinction affects language model learning. We compare models trained on:

1. **Modern text**: Standard English with spaces, punctuation, and mixed case
2. **Scriptio continua**: Ancient writing style with no spaces, no punctuation, uppercase only

Three ablation conditions (`python run_experiment.py --ablation`) each change one variable at a time: **no-spaces**, **no-punctuation**, and **uppercase**. See [Results](#results).

## What the results mean

**Short version:** language models need the word boundaries a text was written with, but adding boundaries to a script that never had them barely helps.

- **Spaces are what make scriptio continua hard.** On both Shakespeare and WikiText-103, removing spaces accounts for nearly all of the extra difficulty. Uppercasing and removing punctuation barely matter.
- **Removing spaces doesn't save work; it moves it somewhere harder.** No-space text is shorter, yet the model still spends more total effort on it. Without spaces, it has to work out where each word ends before it can predict what comes next.
- **Adding spaces to Chinese or Japanese helps little or not at all.** Inserting word boundaries made no difference for Chinese and helped Japanese by about 1.4%. Those scripts mark structure in other ways (meaningful characters, switches between scripts), so spaces mostly tell the model what it already knew.
- **The effect is lopsided.** Removing spaces from English costs about 7–8%; adding them to Chinese gains nothing. Each writing system carries its own cues, and the harm comes from removing a cue the text was built around, not from spaces being universally necessary.

These are small models trained on small corpora (the CJK texts are a few plays), so treat the cross-language findings as a pilot. Details, numbers and caveats are in [Results](#results).

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

### WikiText-103 (April 2026, evaluated November 2026)

The same five conditions were trained on WikiText-103 (about 100M characters), using the same 6-layer, 384-dim model with dropout 0.2. The runs were planned for 5,000 iterations but stopped early; the last intact checkpoint is at iteration 1,000 (the final `ckpt.pt` files on the volume are truncated). Checkpoints were scored with `modal run modal_eval.py`, which makes one full, deterministic pass over each validation set.

Because removing characters shortens the text, the table reports two rates:

- **BPC**: bits per character of that condition's own text.
- **Bits per modern char**: total bits over the whole validation set, divided by the modern text's length (1,145,909 chars). This compares the cost of encoding the same passages.

| Condition | Val chars (vs modern) | BPC @ 500 | BPC @ 1000 | Bits per modern char @ 1000 |
|---|---|---|---|---|
| modern (baseline) | 1,145,909 (100%) | 1.957 | **1.628** | 1.628 |
| uppercase | 1,145,909 (100%) | 1.934 | 1.633 | 1.633 |
| nopunct | 1,108,746 (97%) | 1.999 | 1.686 | 1.632 |
| nospaces | 929,558 (81%) | 2.434 | 2.048 | 1.661 |
| scriptio | 866,320 (76%) | 2.409 | 2.056 | **1.554** |

**Findings**

- **The Shakespeare pattern holds per character.** Removing spaces raises BPC by about 0.42. Case and punctuation change it by at most about 0.06.
- **Removing spaces makes the model less efficient overall.** No-spaces text is 19% shorter and carries less information (no word boundaries), yet it still costs more total bits than the modern text (1.661 vs 1.628 per modern char). The model spends more figuring out where words end than it saves by not predicting spaces.
- **Scriptio continua costs the fewest total bits, but only because it encodes less.** It drops spaces, punctuation, case, digits and newlines, so there is simply less to predict. This isn't evidence that the model learns it better.
- **No overfitting yet.** Unlike Shakespeare, validation loss was still falling between iterations 500 and 1,000, so these models are undertrained, not overfit.

**Caveats**

- **One seed per condition**, and training stopped at iteration 1,000 of 5,000. Rankings could change with longer training.
- **The modern vocabulary is large** (thousands of Unicode characters), which is why iteration-0 loss is about 8.5 nats for every condition.

### CJK Shakespeare pilot (November 2026)

Chinese and Japanese are written without spaces, so they test the question from the other side: does *adding* word boundaries help? This pilot uses public-domain translations of the same plays:

| Pair | Plays | Translation | Source |
|---|---|---|---|
| English–Chinese | *Hamlet*, *A Midsummer Night's Dream* | Zhu Shenghao (朱生豪, d. 1944) | Chinese Wikisource |
| English–Japanese | *Romeo and Juliet* | Tsubouchi Shōyō (坪内逍遥, d. 1935) | Aozora Bunko |

English text is the Moby edition from [shakespeare.mit.edu](http://shakespeare.mit.edu).

**Setup**

- **Splits:** train on Acts 1–4 (holding out the last 10% of each for early stopping) and test on Act 5. Every language's test set is a translation of the same English Act 5.
- **Byte-level models:** every condition uses raw UTF-8 bytes (vocabulary 256), so no language gets a vocabulary advantage. The model is a 3.2M-parameter GPT (4 layers, 256 dims, dropout 0.2), keeping the checkpoint with the best validation loss.
- **Segmentation:** spaces are inserted between words by jieba for Chinese and MeCab (fugashi + unidic-lite) for Japanese.
- **Metric:** total bits on Act 5, divided by the English Act 5 character count ("bits per English char"). Lower is better.
- **Three seeds per condition**, reported as mean ± standard deviation.

| Pair | Condition | Bits per English char |
|---|---|---|
| en–zh | English | **2.351** ± 0.009 |
| en–zh | English, no spaces | 2.524 ± 0.005 |
| en–zh | Chinese (natural) | 2.772 ± 0.007 |
| en–zh | Chinese, segmented | 2.780 ± 0.013 |
| en–ja | English | **2.443** ± 0.008 |
| en–ja | English, no spaces | 2.637 ± 0.011 |
| en–ja | Japanese (natural) | 3.013 ± 0.004 |
| en–ja | Japanese, segmented | 2.972 ± 0.010 |

**Findings**

- **Removing spaces from English hurts clearly.** It costs about 0.17–0.19 bits per English char (+7–8%) in both pairs, far outside the seed-to-seed spread.
- **Adding spaces to Chinese does nothing.** Natural and segmented Chinese are within noise of each other (2.772 vs 2.780).
- **Adding spaces to Japanese helps slightly.** It saves about 0.04 bits per English char (−1.4%), a small but consistent gain across seeds.
- **The effect is asymmetric.** Taking word boundaries away from a language written with them hurts much more than adding them to a language written without them helps. This suggests that, at this scale, the cost comes from losing a cue the text was written around, not from word boundaries being universally necessary.

**Caveats**

- **Cross-language totals aren't comparable.** The translations differ from the English in length and style (Tsubouchi's Japanese test act is 42% more bytes than the English, and Zhu renders verse as prose), and CJK characters take 3 bytes each. Compare conditions *within* a language, not English against Chinese or Japanese.
- **Segmenters are imperfect.** jieba is tuned for Simplified Chinese and these texts are Traditional; MeCab's modern dictionary struggles with Tsubouchi's old-style Japanese. Better segmentation might help more.
- **The corpora are tiny** (90–240 KB of training text), and the models start overfitting after 1,000–2,400 iterations. The English no-spaces penalty here is much larger than on WikiText-103, so treat these results as a pilot, not a measurement.

**Reproduce**

```bash
python data/extract_cjk_shakespeare.py   # parse raw downloads in data/cjk_raw/
python data/prepare_cjk_shakespeare.py   # needs jieba, fugashi, unidic-lite
modal run modal_cjk_pilot.py             # 8 conditions x 3 seeds -> results/cjk_pilot.json
```

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
