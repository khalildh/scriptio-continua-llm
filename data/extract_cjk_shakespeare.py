"""
Extract clean, act-split play text for the CJK Shakespeare pilot.

Reads the raw downloads in data/cjk_raw/ and writes plain UTF-8 text to
data/cjk_text/<lang>/<play>_act<N>.txt. Speaker names and stage directions
are kept (they are part of the text in every language), but site
navigation, editorial notes and markup are removed.

Sources (all public domain):
  en: MIT Shakespeare (Moby text)       http://shakespeare.mit.edu/<play>/full.html
  zh: Zhu Shenghao (朱生豪, d. 1944)    zh.wikisource.org
  ja: Tsubouchi Shōyō (坪内逍遥, d. 1935) Aozora Bunko card 42773

Raw files are fetched once by hand (see README); this script only parses.
"""
import html
import json
import re
from pathlib import Path

BASE = Path(__file__).parent
RAW = BASE / "cjk_raw"
OUT = BASE / "cjk_text"

EN_PLAYS = {"hamlet": "hamlet", "midsummer": "midsummer", "romeo": "romeo_juliet"}


def strip_tags(s: str) -> str:
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", s, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</(p|div|h\d|li|tr|blockquote)>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s)


def tidy(s: str) -> str:
    lines = [ln.strip() for ln in s.replace("\r", "").split("\n")]
    # One line per line of text, no blank lines: the sources disagree on
    # paragraph spacing, and blank lines would inflate some languages' size.
    return "\n".join(ln for ln in lines if ln) + "\n"


def extract_en():
    for play, slug in EN_PLAYS.items():
        raw = (RAW / "en" / f"{slug}.html").read_text(encoding="latin-1")
        # Acts begin at <H3>ACT I</H3> ... ; split on those headings.
        parts = re.split(r"<H3>\s*ACT\s+[IVX]+\s*</H3>", raw, flags=re.I)
        acts = parts[1:]
        assert len(acts) == 5, (play, len(acts))
        for i, a in enumerate(acts, 1):
            a = re.split(r"<table|</body>", a, flags=re.I)[0] if i == 5 else a
            write("en", play, i, tidy(strip_tags(a)))


def extract_zh():
    for play in ["hamlet", "midsummer"]:
        for i in range(1, 6):
            d = json.loads((RAW / "zh" / f"{play}_act{i}.json").read_text(encoding="utf-8"))
            h = d["parse"]["text"]
            # Drop the Wikisource header box, page-number links, footnote refs,
            # licence box and navigation templates.
            h = re.sub(r'<div[^>]*(ws-noexport|licenseContainer|header_|navbox|mw-references)[^>]*>.*?</div>',
                       "", h, flags=re.S)
            h = re.sub(r'<span[^>]*class="[^"]*(pagenum|ws-pagenum)[^"]*"[^>]*>.*?</span>', "", h, flags=re.S)
            h = re.sub(r"<sup[^>]*>.*?</sup>", "", h, flags=re.S)
            h = re.sub(r'<table[^>]*>.*?</table>', "", h, flags=re.S)
            h = re.sub(r'<span class="mw-editsection">.*?</span>\s*</span>', "", h, flags=re.S)
            h = re.sub(r'<div[^>]*id="toc"[^>]*>.*?</ul>\s*</div>', "", h, flags=re.S)
            h = re.sub(r'<ol class="references">.*?</ol>', "", h, flags=re.S)
            text = strip_tags(h)
            # Remove leftover licence lines, edit links, footnotes ("↑ 註…"),
            # inline footnote markers ("註三"), and the act heading itself.
            text = re.sub(r"\[(編輯|编辑)\]", "", text)
            text = re.sub(r"[〔（(]?註[一二三四五六七八九十]+[〕）)]?", "", text)
            text = "\n".join(
                ln for ln in text.split("\n")
                if not re.search(r"公有領域|公有领域|版權|版权|Public domain|PD-", ln)
                and not ln.strip().startswith("↑")
                and not re.fullmatch(r"\s*(目次|目錄|\d+\s+第.+場.*|第[一二三四五]幕)\s*", ln)
            )
            write("zh", play, i, tidy(text))


def extract_ja():
    raw = (RAW / "ja" / "03romeo_to_juliet.txt").read_bytes().decode("cp932", errors="replace").replace("\r\n", "\n")
    # Aozora format: header separated by a dashed line, footer starts at 底本：
    body = re.split(r"-{20,}", raw)
    body = body[2] if len(body) >= 3 else raw
    body = body.split("底本：")[0]
    body = re.sub(r"《[^》]*》", "", body)          # ruby readings
    body = body.replace("｜", "")                   # ruby base markers
    body = re.sub(r"［＃[^］]*］", "", body)         # editorial annotations
    body = body.replace("※", "")                   # placeholder for unencodable chars
    # Acts are headed 第一幕 ... 第五幕
    parts = re.split(r"\n\s*第[一二三四五]幕[^\n]*\n", body)
    acts = parts[1:]
    assert len(acts) == 5, len(acts)
    for i, a in enumerate(acts, 1):
        write("ja", "romeo", i, tidy(a))


def write(lang, play, act, text):
    p = OUT / lang / f"{play}_act{act}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    extract_en()
    extract_zh()
    extract_ja()
    for p in sorted(OUT.rglob("*.txt")):
        t = p.read_text(encoding="utf-8")
        print(f"{p.relative_to(OUT)}: {len(t):,} chars, {len(t.encode()):,} bytes")
