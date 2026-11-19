"""
Build byte-level datasets for the CJK Shakespeare pilot.

Reads act-split text from data/cjk_text/ (made by extract_cjk_shakespeare.py)
and writes data/cjk_<condition>/{train,val,test}.bin + meta.pkl + stats.json.

Two language pairs, each on the plays available in both languages:
  pair "zh": Hamlet + A Midsummer Night's Dream (Zhu Shenghao translation)
  pair "ja": Romeo and Juliet                    (Tsubouchi Shōyō translation)

Conditions per pair:
  en_modern     English as written
  en_nospaces   English with spaces removed (newlines kept)
  zh_natural    Chinese as written (no word spaces)
  zh_segmented  Chinese with spaces between words (jieba)
  ja_natural    Japanese as written (no word spaces)
  ja_segmented  Japanese with spaces between words (MeCab via fugashi + unidic-lite)

Splits (identical across languages, so test sets are translations of each other):
  test  = Act 5 of every play
  train = Acts 1-4, minus the last 10% of each act
  val   = the last 10% of each of Acts 1-4 (for early stopping only)

Everything is UTF-8 bytes (vocab 256), so Chinese/Japanese characters are
~3 tokens each and every condition shares one vocabulary.
"""
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
TEXT = BASE / "cjk_text"

PAIRS = {
    "zh": ["hamlet", "midsummer"],
    "ja": ["romeo"],
}


def load(lang, play, act):
    return (TEXT / lang / f"{play}_act{act}.txt").read_text(encoding="utf-8")


def remove_spaces(s):
    return s.replace(" ", "")


def segment_zh(s):
    import jieba
    jieba.setLogLevel(60)
    return "\n".join(" ".join(t for t in jieba.cut(line) if t.strip()) for line in s.split("\n"))


_tagger = None


def segment_ja(s):
    global _tagger
    if _tagger is None:
        import fugashi
        _tagger = fugashi.Tagger()
    return "\n".join(" ".join(w.surface for w in _tagger(line)) for line in s.split("\n"))


TRANSFORMS = {
    "modern": lambda s: s,
    "nospaces": remove_spaces,
    "natural": lambda s: s,
    "segmented_zh": segment_zh,
    "segmented_ja": segment_ja,
}


def split_texts(lang, plays, transform):
    train, val, test = [], [], []
    for play in plays:
        for act in range(1, 5):
            t = transform(load(lang, play, act))
            cut = int(len(t) * 0.9)
            cut = t.rfind("\n", 0, cut) + 1 or cut  # split on a line boundary
            train.append(t[:cut])
            val.append(t[cut:])
        test.append(transform(load(lang, play, 5)))
    return "".join(train), "".join(val), "\n".join(test)


def write(name, train, val, test, pair, lang):
    out = BASE / f"cjk_{name}"
    out.mkdir(parents=True, exist_ok=True)
    stats = {"pair": pair, "lang": lang}
    for split, text in [("train", train), ("val", val), ("test", test)]:
        b = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16)
        b.tofile(out / f"{split}.bin")
        stats[f"{split}_bytes"] = int(len(b))
        stats[f"{split}_chars"] = len(text)
    (out / "test.txt").write_text(test, encoding="utf-8")
    with open(out / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": 256, "encoding": "utf-8 bytes"}, f)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def conditions():
    """(dataset name, pair, lang, transform key) for every pilot condition."""
    out = []
    for pair, plays in PAIRS.items():
        out += [
            (f"{pair}_en_modern", pair, "en", "modern"),
            (f"{pair}_en_nospaces", pair, "en", "nospaces"),
            (f"{pair}_{pair}_natural", pair, pair, "natural"),
            (f"{pair}_{pair}_segmented", pair, pair, f"segmented_{pair}"),
        ]
    return out


def main(only=None):
    all_stats = {}
    for name, pair, lang, tkey in conditions():
        if only and name != only:
            continue
        train, val, test = split_texts(lang, PAIRS[pair], TRANSFORMS[tkey])
        all_stats[name] = write(name, train, val, test, pair, lang)
        s = all_stats[name]
        print(f"{name:<22} train {s['train_bytes']:>8,} B  val {s['val_bytes']:>7,} B  "
              f"test {s['test_bytes']:>7,} B / {s['test_chars']:>7,} chars")
    return all_stats


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
