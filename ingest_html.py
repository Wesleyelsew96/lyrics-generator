import argparse
import glob
import json
import os
import re
from typing import List, Optional, Tuple, Dict


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


def extract_ultimate_guitar_meta(html: str, soup) -> Dict[str, Optional[str]]:
    """Extract meta from an Ultimate Guitar page if detected.
    Returns dict with possible keys: title, artist, bpm (int as str).
    This is best-effort and tolerant to missing fields.
    """
    meta: Dict[str, Optional[str]] = {"title": None, "artist": None, "bpm": None}

    # Look for JSON blobs in scripts that might contain tempo
    try:
        for sc in soup.find_all("script"):
            if not sc.string:
                # Some scripts have contents in .text
                text = sc.get_text("\n", strip=False)
            else:
                text = sc.string
            if not text:
                continue
            # Tempo/BPM patterns in embedded JSON
            for pat in [r'"tempo"\s*:\s*(\d{2,3})', r'"bpm"\s*:\s*(\d{2,3})']:
                m = re.search(pat, text)
                if m:
                    meta["bpm"] = m.group(1)
                    break
            # Artist/title patterns (very heuristic)
            if meta["artist"] is None:
                m = re.search(r'"artist_name"\s*:\s*"([^"]+)"', text)
                if m:
                    meta["artist"] = m.group(1)
            if meta["title"] is None:
                m = re.search(r'"song_name"\s*:\s*"([^"]+)"', text)
                if m:
                    meta["title"] = m.group(1)
    except Exception:
        pass

    # Fallback: parse og:title like "Song Name chords by Artist @ Ultimate-Guitar.Com"
    try:
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            content = og.get("content").strip()
            # Extract "Song Name" and "Artist" heuristically
            # Pattern examples: "Song Name chords by Artist @ Ultimate-Guitar.Com"
            m = re.match(r"(.+?)\s+(?:chords|tab|tabs)\s+by\s+(.+?)\s+@\s+Ultimate-Guitar", content, re.IGNORECASE)
            if m:
                if meta["title"] is None:
                    meta["title"] = m.group(1).strip()
                if meta["artist"] is None:
                    meta["artist"] = m.group(2).strip()
    except Exception:
        pass

    return meta


CHORD_RE = re.compile(
    r"^(?:\s*[A-G](?:[#b])?(?:m|maj|min|sus|dim|aug)?\d*(?:/[A-G](?:[#b])?)?\s*)+$"
)


def is_chord_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    tokens = s.split()
    # a line entirely composed of chord tokens
    chord_token = re.compile(r"^[A-G](?:[#b])?(?:m|maj|min|sus|dim|aug)?\d*(?:/[A-G](?:[#b])?)?$")
    if CHORD_RE.match(s):
        # If it's purely chords, treat it as such only when there aren't obvious lyric words
        lyricish = any(re.search(r"[A-Za-z]{2,}", t) and not chord_token.match(t) for t in tokens)
        if not lyricish:
            return True
    non_chord_words = [
        t for t in tokens if not chord_token.match(t) and re.search(r"[A-Za-z]{2,}", t)
    ]
    if non_chord_words:
        return False
    chordish = sum(1 for t in tokens if chord_token.match(t))
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


def extract_bpm(lines: List[str]) -> Optional[int]:
    """Best-effort BPM extraction from page text.
    Looks for patterns such as:
    - Tempo: 120, BPM: 120, 120 bpm
    - ♩=120, ♪ = 120, q = 120
    - Metronome 100
    Returns an int in a sane range if found.
    """
    text = "\n".join(lines)
    candidates: List[int] = []

    patterns = [
        r"(?i)\b(?:tempo|bpm)\s*[:=]?\s*(\d{2,3})\b",
        r"(?i)\b(\d{2,3})\s*bpm\b",
        r"[♩♪]\s*[:=]?\s*(\d{2,3})\b",
        r"(?i)\bq\s*[:=]\s*(\d{2,3})\b",
        r"(?i)\bmetronome\s*[:=]?\s*(\d{2,3})\b",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text):
            try:
                bpm = int(m.group(1))
                if 30 <= bpm <= 300:
                    candidates.append(bpm)
            except Exception:
                continue

    if candidates:
        # choose the mode-ish or first reasonable candidate
        return candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Parse local HTML files into JSONL {title, chorus} records.")
    parser.add_argument("--input", type=str, required=True, help="Path or glob to HTML files (e.g., html/*.html)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for files processed")
    parser.add_argument("--min_chorus_lines", type=int, default=2, help="Minimum number of lines for a chorus to keep")
    parser.add_argument(
        "--max_chorus_lines",
        type=int,
        default=32,
        help="Maximum number of lines for a chorus (set 0 to disable the cap)",
    )
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
    skip_reasons: Dict[str, int] = {}

    def register_skip(file_path: str, reason: str):
        nonlocal skipped
        skipped += 1
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        print(f"[skip] {file_path}: {reason}")

    with open(args.output, "w", encoding="utf-8") as out:
        for fp in files:
            try:
                html = read_html(fp)
                title, lines = extract_title_and_lines(html, fp)
                # UG-specific meta enrichment (if detected)
                artist: Optional[str] = None
                bpm_meta: Optional[int] = None
                try:
                    if "ultimate-guitar" in html.lower() or ("Ultimate-Guitar" in html):
                        from bs4 import BeautifulSoup
                        soup_tmp = BeautifulSoup(html, "lxml")
                        ugm = extract_ultimate_guitar_meta(html, soup_tmp)
                        if ugm.get("title"):
                            title = ugm["title"].strip()
                        if ugm.get("artist"):
                            artist = ugm["artist"].strip()
                        if ugm.get("bpm"):
                            try:
                                val = int(ugm["bpm"])  # type: ignore[arg-type]
                                if 30 <= val <= 300:
                                    bpm_meta = val
                            except Exception:
                                pass
                except Exception:
                    pass
                chorus_lines = find_chorus(lines)
                if not chorus_lines:
                    register_skip(fp, "no chorus detected")
                    continue
                if args.min_chorus_lines and len(chorus_lines) < args.min_chorus_lines:
                    register_skip(fp, f"chorus too short ({len(chorus_lines)} lines)")
                    continue
                if args.max_chorus_lines and args.max_chorus_lines > 0 and len(chorus_lines) > args.max_chorus_lines:
                    register_skip(fp, f"chorus too long ({len(chorus_lines)} lines)")
                    continue
                bpm = bpm_meta if bpm_meta is not None else extract_bpm(lines)
                # Full lyrics (cleaned) if desired for other tasks
                full_lyrics = "\n".join(clean_lines(lines)).strip()
                record = {
                    "title": title,
                    "chorus": "\n".join(chorus_lines).strip(),
                    "source_path": os.path.abspath(fp),
                }
                if artist:
                    record["artist"] = artist
                if bpm is not None:
                    record["bpm"] = bpm
                if full_lyrics:
                    record["lyrics"] = full_lyrics
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                print(f"[ok] {fp}: {title}")
            except Exception as exc:
                register_skip(fp, f"exception: {exc}")
                continue

    print(f"Wrote {written} records to {args.output}; skipped {skipped} files.")
    if skip_reasons:
        print("Skip reasons:")
        for reason, count in skip_reasons.items():
            print(f"- {reason}: {count}")


if __name__ == "__main__":
    main()

