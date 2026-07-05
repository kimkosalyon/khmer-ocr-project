# Khmer OCR Project Progress

## Current Goal

Build a Khmer OCR training pipeline with synthetic text/image generation, proper Khmer shaping, MLflow tracking, and an OCR demo.

## Completed Work

### Training Pipeline

- Refactored OCR code into shared modules under `src/`:
  - `src/model.py`
  - `src/dataset.py`
  - `src/config.py`
  - `src/decode.py`
- Added CLI training entry point at `scripts/train.py`.
- Added marimo notebook entry point at `scripts/notebook.py`.
- Added Streamlit inference demo at `scripts/serve.py`.
- Added MLflow tracking through Docker Compose.
- MLflow runs on non-default port `15005`.
- Trained full Khmer Hanuman OCR dataset to epoch 100.
- Current checkpoint:
  - `checkpoints/hanuman_full/hanumanfull_bs3072_epoch_100.safetensors`
- Final training loss around epoch 100 was about `0.0296`.

### Renderer

- Built Sone/HarfBuzz-based Khmer text renderer under `renderer/`.
- Renderer API runs on port `3456`.
- React + Vite + Tailwind frontend runs on port `3457`.
- Renderer supports:
  - `POST /render`
  - `GET /fonts`
  - PNG/JPG/WebP/PDF output
  - text color and background color
  - per-side padding
- Renderer fonts are limited to:
  - `kantumruy`
  - `moul`
  - `battambang`
  - `bayon`
  - `notosans`
  - `siemreap`
- Latin fallback font:
  - `fonts/NotoSans-Regular.ttf`
- Khmer fonts downloaded into `fonts/`.

### Synthetic Text Data

- Downloaded and extracted Khmer Wikipedia segmented data:
  - `seg_kmwiki_data.zip`
  - `kmwiki_data/`
- Source corpus stats from earlier inspection:
  - 2,481 `.txt` files
  - about 109,682 lines
  - about 43.7M characters
- Replaced character-level generation with word-level Markov generation to avoid broken Khmer clusters.
- Main generator files:
  - `scripts/generate_text.py`
  - `scripts/generate_text_csv.py`
  - `scripts/generate.py`

## Text Generation Decisions

### Why Word-Level Generation

Character-level Markov produced invalid Khmer such as:

```text
មួួយ
ខ្ញុំ្ជា
ខ្ ្
```

Current generation uses segmented words from `kmwiki_data/`, so tokens are real source words and Khmer clusters are not split.

### Space Distribution

The source corpus is word-segmented, but normal Khmer often does not contain spaces between every word. We implemented script-aware spacing.

Current default distribution:

- Khmer-Khmer word spaces: `85%` removed
- Khmer-English / Khmer-number boundary spaces: `35%` removed
- English-English spaces: `0%` removed
- Punctuation spacing is cleaned naturally
- English CamelCase joins like `CarMitsubishi` are normalized to `Car Mitsubishi`

Reasoning:

- Khmer OCR should learn mostly joined Khmer text.
- English OCR should preserve normal English spaces.
- Mixed Khmer/English boundaries should vary because real text may include either style.

### Current Generated Text CSVs

Ready-to-use 100K text distribution:

```text
generated_texts/image_text_distribution_100k.csv
```

Preview file:

```text
generated_texts/image_text_distribution_preview.csv
```

The CSVs are written with pandas using UTF-8 BOM:

```python
encoding="utf-8-sig"
```

Observed distribution for the current 100K CSV:

- Khmer-only: `81.7%`
- Mixed Khmer + English/Latin: `8.1%`
- English/Latin-only: `10.0%`
- Other/symbol-only: `0.1%`
- Average spaces per line: `2.82`

Example generated text:

```text
Level 430 Race Car Chevrolet SSR '03
Mode of transmission
ព្រះសម្មាសម្ពុទ្ធបរមសាស្តា ទ្រង់ប្រារឰនូវមច្ឆរិយកោសិយ
ការសិក្សានៅបឋមដ្ឋានភាគច្រើនលើសលុបធ្វើឡើងនៅសាលារដ្ឋ។
```

## Existing Generated Images

An earlier image dataset was generated at:

```text
generated/training_100k/
```

It contains about `99,998` image/label samples.

Note: this dataset was generated before the latest script-aware spacing cleanup. Prefer regenerating images from:

```text
generated_texts/image_text_distribution_100k.csv
```

## Useful Commands

### Start Renderer

```bash
docker compose up renderer -d --build
```

Renderer API:

```text
http://localhost:3456
```

Frontend:

```bash
cd renderer/frontend
npm run dev
```

Frontend URL:

```text
http://localhost:3457
```

### Generate Text CSV

```bash
source .venv/bin/activate
python scripts/generate_text_csv.py \
  --count 100000 \
  --mix-ratio 0.45 \
  --min-words 4 \
  --max-words 28 \
  --min-len 12 \
  --max-len 140 \
  --output generated_texts/image_text_distribution_100k.csv
```

### Tune Spacing

Fewer Khmer spaces:

```bash
python scripts/generate_text_csv.py --space-drop-prob 0.90 --output generated_texts/preview_space090.csv
```

More Khmer spaces:

```bash
python scripts/generate_text_csv.py --space-drop-prob 0.75 --output generated_texts/preview_space075.csv
```

Current recommended default:

```text
--space-drop-prob 0.85
--mixed-space-drop-prob 0.35
--latin-space-drop-prob 0.0
```

### Serve Existing Generated Images For Download

```bash
cd generated/training_100k
python3 -m http.server 8888 --bind 0.0.0.0
```

## Next Steps

1. Review `generated_texts/image_text_distribution_preview.csv`.
2. If the text distribution is acceptable, render images from `generated_texts/image_text_distribution_100k.csv`.
3. Save labels alongside generated images.
4. Optionally archive the image dataset for download.
5. Train/evaluate OCR on the new synthetic image dataset.
