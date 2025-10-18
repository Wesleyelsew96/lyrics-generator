import argparse
import glob
import json
import os
import re
from typing import List, Optional, Tuple


def read_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_title_and_lines(html: str, fallback_name: str) -> Tuple[str, List[str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Try common title sources
    title_candidates = []
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title_candidates.append(h1.get_text(strip=True))
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title_candidates.append(og.get("content").strip())
    if soup.title and soup.title.string:
        title_candidates.append(soup.title.string.strip())

    title = (
        title_candidates[0]
        if title_candidates
        else os.path.splitext(os.path.basename(fallback_name))[0]
    )

    # Clean title: remove trailing site branding like " - GuitarTabs" or " | Site"
    title = re.split(r"\s[-|]\s", title)[0].strip() or title

    # Try to get main textual content
    # Some tabs/lyrics sites use <pre> or specific containers
    text_blocks = []
    for selector in ["pre", "div.lyrics", "div#lyrics", "div#content", "article", "div#song"]:
        for el in soup.select(selector):
            t = el.get_text("\n", strip=True)
            if t:
                text_blocks.append(t)

    if not text_blocks:
        body_text = soup.get_text("\n", strip=True)
        text_blocks = [body_text]

    # Choose the largest text block as the main content
    main_text = max(text_blocks, key=len)

    # Normalize newlines and split into lines
    lines = [ln.rstrip() for ln in main_text.splitlines()]
    return title, lines


CHORD_RE = re.compile(
    r"^(?:\s*[A-G](?:[#b])?(?:m|maj|min|sus|dim|aug)?\d*(?:/[A-G](?:[#b])?)?\s*)+$"
)


def is_chord_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # a line entirely composed of chord tokens
    if CHORD_RE.match(s):
        return True
    # heuristic: high ratio of chord-like tokens
    tokens = s.split()
    chordish = 0
    chord_token = re.compile(r"^[A-G](?:[#b])?(?:m|maj|min|sus|dim|aug)?\d*(?:/[A-G](?:[#b])?)?$")
    for t in tokens:
        if chord_token.match(t):
            chordish += 1
    return chordish >= max(2, int(0.6 * len(tokens)))


def split_stanzas(lines: List[str]) -> List[List[str]]:
    stanzas: List[List[str]] = []
    cur: List[str] = []
    def flush():
        if cur and any(x.strip() for x in cur):
            stanzas.append(cur.copy())
    for ln in lines:
        if not ln.strip():
            flush()
            cur = []
        else:
            cur.append(ln)
    flush()
    return stanzas


def clean_lines(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        # Drop chord-only or mostly-chord lines
        if is_chord_line(ln):
            continue
        # Remove inline bracketed chords like [C], [Am]
        ln2 = re.sub(r"\[[A-G](?:[#b])?(?:m|maj|min|sus|dim|aug)?\d*(?:/[A-G](?:[#b])?)?\]", "", ln)
        # Collapse multiple spaces
        ln2 = re.sub(r"\s{2,}", " ", ln2).strip()
        if ln2:
            out.append(ln2)
    return out


SECTION_MARKERS = (
    "[chorus", "[refrain", "chorus:", "refrain:", "(chorus", "{chorus"
)


def find_chorus(lines: List[str]) -> Optional[List[str]]:
    # 1) Try explicit markers
    lower = [ln.lower() for ln in lines]
    n = len(lines)
    for i, ln in enumerate(lower):
        if any(ln.startswith(m) or m in ln for m in SECTION_MARKERS):
            # capture until blank line or next bracketed section
            j = i + 1
            buff: List[str] = []
            while j < n:
                raw = lines[j]
                s = raw.strip()
                low = s.lower()
                if not s:
                    break
                if s.startswith("[") and s.endswith("]"):
                    break
                if low.startswith("verse") or low.startswith("bridge") or low.startswith("intro"):
                    break
                if not is_chord_line(raw):
                    buff.append(raw)
                j += 1
            buff = clean_lines(buff)
            if buff:
                return buff

    # 2) Heuristic: pick a repeated stanza
    stanzas = [clean_lines(s) for s in split_stanzas(lines)]
    norm = ["\n".join(s).strip().lower() for s in stanzas if s]
    counts = {}
    for idx, s in enumerate(norm):
        if not s:
            continue
        counts.setdefault(s, {"count": 0, "idxs": []})
        counts[s]["count"] += 1
        counts[s]["idxs"].append(idx)
    # choose the most repeated non-trivial stanza
    candidate = None
    for s, meta in counts.items():
        lines_count = s.count("\n") + 1
        if meta["count"] >= 2 and 2 <= lines_count <= 12:
            if candidate is None or meta["count"] > candidate[1]["count"] or (
                meta["count"] == candidate[1]["count"] and lines_count > candidate[0].count("\n") + 1
            ):
                candidate = (s, meta)
    if candidate is not None:
        # return original (non-lowered) stanza for readability
        idx0 = candidate[1]["idxs"][0]
        return stanzas[idx0]

    # 3) Fallback: longest stanza that looks like lyrics
    stanzas_nonempty = [s for s in stanzas if s]
    if stanzas_nonempty:
        stanzas_nonempty.sort(key=lambda s: (-len("\n".join(s)), -len(s)))
        return stanzas_nonempty[0]

    return None


def main():
    parser = argparse.ArgumentParser(description="Parse local HTML files into JSONL {title, chorus} records.")
    parser.add_argument("--input", type=str, required=True, help="Path or glob to HTML files (e.g., html/*.html)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for files processed")
    args = parser.parse_args()

    files: List[str] = []
    if os.path.isdir(args.input):
        files = [
            os.path.join(args.input, p)
            for p in os.listdir(args.input)
            if p.lower().endswith((".html", ".htm"))
        ]
    else:
        files = glob.glob(args.input)

    if args.limit:
        files = files[: args.limit]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    written = 0
    skipped = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for fp in files:
            try:
                html = read_html(fp)
                title, lines = extract_title_and_lines(html, fp)
                chorus_lines = find_chorus(lines)
                if not chorus_lines:
                    skipped += 1
                    continue
                record = {
                    "title": title,
                    "chorus": "\n".join(chorus_lines).strip(),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            except Exception:
                skipped += 1
                continue

    print(f"Wrote {written} records to {args.output}; skipped {skipped} files.")


if __name__ == "__main__":
    main()

