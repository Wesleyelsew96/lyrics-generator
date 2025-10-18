Chorus Generator (Title → Chorus)
=================================

This project fine-tunes a small language model (e.g., GPT‑2) to generate a chorus given a song title.

IMPORTANT: Use only data you have the legal right to use. Do not scrape or train on copyrighted lyrics without permission.

What You Get
- Data format: simple JSONL with `{ "title": ..., "chorus": ... }`
- Sample synthetic dataset: `data/sample.jsonl` (invented, not copyrighted)
- Training script: `train.py`
- Generation script: `generate.py`

Legal & Data Sourcing
- Preferred sources:
  - Your own lyrics (you own the rights)
  - Public domain lyrics (generally published before 1929 in the U.S.; verify jurisdiction)
  - Lyrics licensed for this purpose (e.g., explicit Creative Commons allowing text reuse)
- Avoid scraping commercial lyric sites unless you have explicit permission and comply with their Terms of Service.
- If in doubt, consult legal counsel before collecting data.

Environment Setup
1) Python 3.9–3.11 recommended.
2) Install dependencies:

```
pip install -r requirements.txt

# Install PyTorch separately according to your system/cuda:
# https://pytorch.org/get-started/locally/
```

Data Format
- One JSON object per line with `title` and `chorus` fields.
- Example (see `data/sample.jsonl`):

```
{"title": "Starlight Highway", "chorus": "..."}
{"title": "Coffee in the Rain", "chorus": "..."}
```

Train
```
python train.py --data data/your_dataset.jsonl --output_dir models/chorus-gpt2 --base_model gpt2
```

Generate
```
python generate.py --model_dir models/chorus-gpt2 --title "Midnight Carousel" --max_new_tokens 80
```

Where To Customize
- Prompt template: `generate.py:20` — change how the prompt is constructed.
- Preprocessing: `train.py:19` — tune how `Title`/`Chorus` text is stitched together.
- Model/params: `train.py:30` and `train.py:76` — base model, epochs, batch size, LR, block size.

Notes
- Small models are easier to fine-tune on modest hardware. Try `distilgpt2` if `gpt2` is heavy.
- For better results with limited data, consider LoRA/PEFT or instruction-style templates.
- Be mindful that generating or distributing copyrighted lyrics without permission may infringe rights.

Ingest Local HTML Pages
- Use `ingest_html.py` to parse saved HTML pages (e.g., pages you exported from a site you have permission to use) into the JSONL format used by training.

Example
```
python ingest_html.py --input path/to/html/*.html --output data/parsed.jsonl

# Now train on parsed data
python train.py --data data/parsed.jsonl --output_dir models/chorus-gpt2 --base_model distilgpt2
```

Heuristics
- Attempts to read title from `<h1>`, `og:title`, or `<title>`.
- Extracts main text from common containers like `<pre>`, `div.lyrics`, `article`.
- Removes chord-only lines and inline bracketed chords like `[C]`.
- Detects chorus via common markers (e.g., `[Chorus]`, `Chorus:`) or repeated stanza fallback; otherwise uses the longest plausible stanza.

Legal Reminder
- Save and parse only content you are authorized to use (your own, public domain, or with explicit license permitting ML training). Respect each site’s Terms of Service.

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
