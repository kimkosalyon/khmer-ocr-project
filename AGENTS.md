# AGENTS.md

## Project Overview

This repository trains, evaluates, documents, and serves Khmer OCR models. The current primary model is a ResNet34 + 2-layer BiGRU + CTC recognizer trained on a generated 200k Siemreap/Arial dataset. A lighter ResNet18 + BiGRU + CTC comparison model is also trained and uploaded to the same Hugging Face model repo.

Key components:

- `scripts/train_200k_resnet34_bigru.py` is the shared active 200k training pipeline for ResNet34 + BiGRU + CTC.
- `scripts/train_200k_resnet18_bigru.py` is the lighter ResNet18 + BiGRU + CTC comparison entry point.
- `scripts/serve.py` is the Streamlit OCR demo.
- `scripts/generate_200k_siemreap_arial.py` generates the current synthetic dataset.
- `scripts/upload_to_hf.py` uploads model checkpoints/cards to Hugging Face model repos using `.env` tokens.
- `renderer/server.js` is the Node/Sone renderer used to create Khmer text images.
- `checkpoints_200k/siemreap_arial_ocr_bs256_epoch_25.safetensors` is the current best checkpoint, with final normalized test CER around `0.18%`.
- `checkpoints_200k_resnet18_bigru/siemreap_arial_ocr_bs256_epoch_25.safetensors` is the ResNet18 comparison checkpoint, with final test CER around `0.45%`.

The older Hanuman/HF training scripts have been moved to `scripts/archive/` and should not be treated as the main workflow.

## Setup Commands

Use `uv` for Python dependency management:

```bash
uv sync
```

Run Python commands through `uv run` unless the virtualenv is already activated:

```bash
uv run python --version
```

PyTorch is configured in `pyproject.toml` to use the `pytorch-cu130` index. CUDA-only commands require a compatible NVIDIA setup. Use `--device cpu` only for small checks, not real training.

Renderer setup:

```bash
cd renderer
npm install
node server.js
```

The renderer listens on `http://localhost:3456` by default. It also supports `PORT`, for example:

```bash
cd renderer
PORT=3458 node server.js
```

## Primary Workflows

Train or resume the ResNet34 200k OCR model:

```bash
uv run python scripts/train_200k_resnet34_bigru.py \
  --data-dir generated/training_200k_siemreap_arial \
  --epochs 25 \
  --batch-size 256 \
  --lr 1e-4 \
  --height 64 \
  --val-split 0.1 \
  --test-split 0.1 \
  --device cuda \
  --num-workers 12 \
  --checkpoint-dir checkpoints_200k \
  --resume checkpoints_200k/siemreap_arial_ocr_bs256_epoch_15.safetensors
```

Important: `--epochs` is the final target epoch, not the number of extra epochs. If resuming from epoch 15 with `--epochs 25`, training runs epochs 16 through 25.

Train or resume the ResNet18 comparison model:

```bash
uv run python scripts/train_200k_resnet18_bigru.py \
  --data-dir generated/training_200k_siemreap_arial \
  --epochs 25 \
  --batch-size 256 \
  --lr 1e-4 \
  --height 64 \
  --val-split 0.1 \
  --test-split 0.1 \
  --device cuda \
  --num-workers 12 \
  --checkpoint-dir checkpoints_200k_resnet18_bigru
```

Run the Streamlit demo:

```bash
uv run streamlit run scripts/serve.py
```

Generate the current 200k synthetic dataset:

```bash
uv run python scripts/generate_200k_siemreap_arial.py \
  --count 200000 \
  --output generated/training_200k_siemreap_arial \
  --num-workers 16 \
  --render-url http://localhost:3458/render \
  --vary-colors
```

Upload the 200k dataset to Hugging Face as Parquet shards with embedded image bytes:

```bash
uv run python scripts/upload_to_hub.py \
  --dataset-dir generated/training_200k_siemreap_arial \
  --repo-id KimkosalYon/khmer-ocr-200k-siemreap-arial \
  --private \
  --max-shard-size 500MB \
  --num-proc 16
```

Upload or refresh the dataset README/card on Hugging Face:

```bash
uv run python scripts/upload_dataset_readme.py \
  --repo-id KimkosalYon/khmer-ocr-200k-siemreap-arial
```

Upload model checkpoints/cards to the existing Hugging Face model repo. The repo contains both the ResNet34 checkpoint and the ResNet18 comparison checkpoint:

```bash
uv run python scripts/upload_to_hf.py \
  --repo-id KimkosalYon/siemreap-arial-khmer-ocr \
  --checkpoint-path checkpoints_200k/siemreap_arial_ocr_bs256_epoch_25.safetensors \
  --vocab-path checkpoints_200k/vocab.json \
  --readme-path checkpoints_200k/README.md
```

Prepare a YOLO text detector dataset from generated images, without training:

```bash
uv run python scripts/prepare_yolo_text_detector.py \
  --input generated/training_200k_siemreap_arial \
  --output generated/text_det_yolo
```

## Current Project Structure

```text
training-ocr/
  src/
    model.py                  # Shared CRNN model used by Streamlit and legacy scripts
    dataset.py                # Generic dataset/collate helpers
    decode.py                 # CTC decode and CER helpers
    config.py                 # Legacy train config dataclass
    render.py                 # Python rendering helpers if needed
  scripts/
    train_200k_resnet34_bigru.py  # Active ResNet34 production/shared training script
    train_200k_resnet18_bigru.py   # ResNet18 comparison training script
    serve.py                  # Streamlit OCR demo with optional YOLO text boxes
    generate_200k_siemreap_arial.py
    upload_to_hf.py           # Safe HF model uploader/card generator, no hardcoded token
    upload_to_hub.py          # Safe HF dataset uploader, no hardcoded token
    upload_dataset_readme.py  # Uploads HF dataset card/README only
    prepare_yolo_text_detector.py
    archive/                  # Legacy training scripts
  renderer/
    server.js                 # Express API renderer
    render.js                 # CLI renderer
    fonts/                    # Khmer and Latin font files
  generated/                  # Local generated datasets; large artifact directory
  checkpoints_200k/           # Current ResNet34 model checkpoints and model card
  checkpoints_200k_resnet18_bigru/  # ResNet18 comparison checkpoints/metrics
  checkpoints/                # Older Hanuman checkpoints
  kmwiki_data/                # Local Khmer wiki text corpus
```

Generated datasets, checkpoints, archives, virtualenvs, and Node modules are artifacts. Do not refactor them as source code.

## Model and Data Details

The active ResNet34 recognizer in `scripts/train_200k_resnet34_bigru.py` uses:

- ResNet34 backbone adapted to grayscale input.
- Asymmetric stride patching in ResNet layers 3 and 4 to preserve sequence width.
- Height-agnostic vertical mean pooling with `f.mean(dim=2)`.
- 2-layer bidirectional GRU, hidden size `256`, dropout `0.2`.
- Linear classifier and CTC loss with blank index `0`.
- Dynamic-width image batching with horizontal padding.
- Input height default `64` for the 200k workflow.

The current 200k dataset is intended to contain:

- 100k Hanuman-derived text samples.
- 100k contextual wiki/Markov samples.
- Khmer-only text rendered in Siemreap.
- Mixed Khmer/English rendered with Siemreap plus Arial Latin fallback.
- English-only rows, if present from sources, rendered with Arial.
- Controlled color/background variation and tight/zero padding variation.

The Hanuman-derived half of the dataset is sourced from `seanghay/khmer-hanuman-100k`; credit the original owner/uploader `seanghay` in dataset/model cards and papers.

## Testing and Verification

There is no formal test suite yet. Use syntax checks and small smoke runs.

Syntax-check Python files:

```bash
uv run python -m py_compile scripts/train_200k_resnet34_bigru.py scripts/train_200k_resnet18_bigru.py scripts/serve.py scripts/generate_200k_siemreap_arial.py scripts/upload_to_hf.py scripts/upload_to_hub.py scripts/upload_dataset_readme.py
```

Smoke-test text generation only:

```bash
uv run python scripts/generate_200k_siemreap_arial.py \
  --count 100 \
  --output /tmp/opencode/generate_smoke \
  --texts-only
```

Smoke-test rendering with the renderer running:

```bash
uv run python scripts/generate_200k_siemreap_arial.py \
  --count 20 \
  --output /tmp/opencode/render_smoke \
  --num-workers 4 \
  --render-url http://localhost:3458/render \
  --vary-colors
```

Verify a checkpoint can load by using `scripts/serve.py` or a small local inference script. Checkpoint metadata must include `c2i` as JSON.

## Code Style and Conventions

- Prefer small, direct changes over new abstractions.
- Keep the active 200k path simple; avoid reviving legacy `scripts/archive/` workflows unless explicitly requested.
- Do not hardcode Hugging Face tokens, API keys, or local secrets. Use `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `huggingface-cli login`.
- Safetensors metadata values must be strings. Use `json.dumps(c2i)` for mappings.
- Preserve CTC blank index `0`; target labels must start at index `1`.
- Do not change `c2i` when resuming a checkpoint. If new characters are introduced, add explicit vocab expansion code or restart training.
- Keep `RandomErasing` after `ToTensor()` in transform pipelines.
- Use `torch.compile(..., dynamic=True)` only on CUDA paths where it has been validated.

## Renderer Notes

The renderer API accepts JSON:

```bash
curl -X POST http://localhost:3458/render \
  -H "Content-Type: application/json" \
  -d '{"text":"ខ្ញុំចង់ទៅសាលារៀន","font":"siemreap","fontSize":48,"color":"#000000","background":"#ffffff"}' \
  -o output.png
```

Font routing in the current renderer:

- `siemreap` is used for Khmer text.
- `arial` is available as a primary font.
- Latin fallback for mixed Siemreap text should be Arial, not Times New Roman.

If the running renderer does not list `arial`, restart a patched renderer on a free port:

```bash
cd renderer
PORT=3458 node server.js
curl http://localhost:3458/fonts
```

## Streamlit and Detection Notes

`scripts/serve.py` supports normal OCR on an uploaded image and optional YOLO text detection if a detector checkpoint exists at `runs/detect/**/weights/best.pt` or `detectors/**/*.pt`.

For detected boxes, the app expands crops by a configurable padding value before OCR. Keep this padding enabled for Khmer because superscripts, subscripts, and vowel marks are easily clipped.

The YOLO detector is not trained by default. `scripts/prepare_yolo_text_detector.py` only prepares labels from generated images.

## Security and Artifact Handling

- Never commit or paste Hugging Face tokens. Older scripts may have contained hardcoded tokens; do not reuse that pattern.
- Treat `generated/`, `checkpoints/`, `checkpoints_200k/`, `.venv/`, `renderer/node_modules/`, archives, and `.7z` files as local artifacts.
- Before copying DuckDB files, stop writers or ensure generation/training is done.
- Do not delete `generated/training_200k_siemreap_arial` until dataset uploads and any dependent training are complete.

## Common Issues

- If CTC loss looks good but CER is terrible, check `target_length <= T_steps`. CTC cannot align targets longer than the model time dimension.
- `zero_infinity=True` can hide invalid CTC batches. Use explicit checks when debugging new architectures or image widths.
- If a resumed checkpoint fails on `fc.weight` size mismatch, the active model vocab size does not match checkpoint `c2i`. Use the same script and metadata path that produced the checkpoint.
- If mixed English renders in Times New Roman, the running renderer is stale. Restart the patched `renderer/server.js`.
- If Hugging Face upload fails with `403`, the token can authenticate but lacks write/create permissions for the target namespace.

## Cleanup Guidance

Keep source changes focused on:

- `scripts/train_200k_resnet34_bigru.py`
- `scripts/train_200k_resnet18_bigru.py`
- `scripts/serve.py`
- `scripts/generate_200k_siemreap_arial.py`
- `scripts/upload_to_hub.py`
- `scripts/prepare_yolo_text_detector.py`
- `src/`
- `renderer/server.js`

Archive old experiments instead of deleting them unless the user explicitly asks for removal. Large training data can be moved outside the project, for example to `../archive/`, to keep the working tree manageable.
