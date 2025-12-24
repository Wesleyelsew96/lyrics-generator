Chorus Generator (Title + BPM -> Chorus)
========================================

This project fine-tunes a modern instruction model with LoRA/PEFT to generate a song chorus conditioned on a title and optional BPM (tempo).

IMPORTANT: Use only data you have the legal right to use. Do not scrape or train on copyrighted lyrics without permission.

What You Get
- Data format: JSONL with `{ "title": ..., "bpm": ..., "chorus": ... }` (`bpm` is optional but recommended)
- Sample synthetic dataset: `data/sample.jsonl` (invented, not copyrighted)
- Training script: `train.py` (LoRA by default; disable with `--no_lora`)
- Generation script: `generate.py` (auto-detects LoRA adapters)
- Static UI stub: `index.html` (posts to `/api/generate`; adjust the endpoint for your deployment)

Environment Setup
1) Python 3.9-3.11 recommended.
2) Install dependencies:

```
pip install -r requirements.txt

# Install PyTorch separately according to your system/cuda:
# https://pytorch.org/get-started/locally/
```

Data Format
- One JSON object per line with `title`, optional integer `bpm`, and `chorus` fields.
- Example (see `data/sample.jsonl`):

```
{"title": "Starlight Highway", "bpm": 118, "chorus": "..."}
{"title": "Coffee in the Rain", "bpm": 92, "chorus": "..."}
```

Train
```
python train.py --data data/your_dataset.jsonl --output_dir models/chorus-mistral-lora --base_model mistralai/Mistral-7B-Instruct-v0.2

# Optional: turn off LoRA if you truly want full fine-tuning (heavier)
python train.py --data data/your_dataset.jsonl --output_dir models/chorus-full --base_model mistralai/Mistral-7B-Instruct-v0.2 --no_lora
```

Generate
```
python generate.py --model_dir models/chorus-mistral-lora --title "Midnight Carousel" --bpm 120 --max_new_tokens 80
```

Where To Customize
- Prompt template: `generate.py` — change how the prompt is constructed (includes BPM when provided).
- Preprocessing: `train.py` — tune how `Title`/`BPM`/`Chorus` text is stitched together.
- Model/params: `train.py` — base model, epochs, batch size, LR, block size, LoRA knobs (`--lora_r`, `--lora_alpha`, `--lora_dropout`, `--lora_target_modules`, `--no_lora`).

Notes
- LoRA keeps training lightweight; tweak `--lora_r` or target modules to fit your GPU/VRAM.
- If you prefer smaller bases, try `microsoft/Phi-3-mini-4k-instruct`; for older experiments, you can still point to `gpt2` or `distilgpt2` with `--no_lora`.
- Be mindful that generating or distributing copyrighted lyrics without permission may infringe rights.

Ingest Local HTML Pages
- Use `ingest_html.py` to parse saved HTML pages (e.g., pages you exported from a site you have permission to use) into the JSONL format used by training. The script attempts to extract BPM from common notations on guitar tab or sheet music pages (e.g., `BPM: 120`, `Tempo: 120`, `ƒTc=120`, `q = 120`). It includes site-specific heuristics for Ultimate Guitar (best-effort parsing of embedded JSON for `tempo`, plus artist/title parsing from metadata). It also writes an optional `lyrics` field with the cleaned full text.

Example
```
python ingest_html.py --input path/to/html/*.html --output data/parsed.jsonl --min_chorus_lines 2 --max_chorus_lines 32

# Now train on parsed data
python train.py --data data/parsed.jsonl --output_dir models/chorus-mistral-lora --base_model mistralai/Mistral-7B-Instruct-v0.2
```

Heuristics
- Attempts to read title from `<h1>`, `og:title`, or `<title>`.
- Extracts main text from common containers like `<pre>`, `div.lyrics`, `article`.
- Removes chord-only lines (stricter heuristic to avoid false positives like "A love") and inline bracketed chords like `[C]`.
- Detects chorus via common markers (e.g., `[Chorus]`, `Chorus:`) or repeated stanza fallback; otherwise uses the longest plausible stanza.
- Extracts BPM from lines mentioning Tempo/BPM or using note symbols (ƒTc, ƒT¦) and simple equality forms (`q = 120`).
- Ultimate Guitar: best-effort extraction of `tempo`/`bpm` from embedded scripts, plus parsing of `artist`/`song_name` when available. Fallbacks to on-page text patterns.
- Logging: skips are reported with reasons; records include `source_path`. Use `--min_chorus_lines` / `--max_chorus_lines` to filter stanza length.

Legal Reminder
- Save and parse only content you are authorized to use (your own, public domain, or with explicit license permitting ML training). Respect each site's Terms of Service. Do not scrape or republish copyrighted lyrics without permission.

Publish to GitHub
- This repo includes a Python-focused `.gitignore` to keep large artifacts (e.g., `models/`) out of version control.
- To publish:
  1) Initialize and commit locally:
     - `git init`
     - `git add .`
     - `git commit -m "Initial commit: chorus generator"`
  2) Create a GitHub repository (via the website or `gh repo create`).
  3) Add remote and push:
     - `git branch -M main`
     - `git remote add origin https://github.com/<you>/<repo>.git`
     - `git push -u origin main`
