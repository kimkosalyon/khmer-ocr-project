import os
import argparse
import json
import safetensors
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Load variables from .env silently, overriding any shell defaults
load_dotenv(override=True)

TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def prepare_files(args):
    print("Extracting c2i vocabulary from safetensors metadata...")
    with safetensors.safe_open(args.checkpoint_path, framework="pt") as f:
        metadata = f.metadata()
    
    if not metadata or "c2i" not in metadata:
        raise ValueError("Could not find 'c2i' metadata key in the safetensors file.")
    
    c2i = json.loads(metadata["c2i"])
    print(f"Loaded vocab with {len(c2i)} characters.")
    
    with open(args.vocab_path, "w", encoding="utf-8") as out:
        json.dump(c2i, out, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to {args.vocab_path}")
    
    print("Writing Model Card README.md...")
    readme_content = f"""---
license: mit
tags:
- ocr
- khmer
- crnn
- bigru
- text-recognition
metrics:
- cer
---

# {args.model_name}

This is a sequence-to-sequence Khmer text line OCR model trained on the 200k synthetic Khmer OCR dataset using Siemreap for Khmer text and Arial fallback for Latin text.

## Training Dataset
The model was trained on a generated 200,000-image Khmer OCR text-line dataset. Each sample contains a rendered text-line image and its ground-truth transcription.

The text corpus is built from two main sources:
* **Hanuman-derived Khmer text**: approximately 100,000 lines derived from the Hugging Face dataset [`seanghay/khmer-hanuman-100k`](https://huggingface.co/datasets/seanghay/khmer-hanuman-100k). Credit to the original dataset owner/uploader **seanghay** for publishing this Khmer text corpus.
* **Khmer Wikipedia / Markov-generated contextual text**: approximately 100,000 lines generated from a local Khmer Wikipedia text corpus using word-level sampling/Markov generation.

Rendering details:
* Khmer text is rendered with **Siemreap**.
* Latin fallback text is rendered with **Arial**.
* Images use a fixed height of 64 px and dynamic width.
* Training uses synthetic color/background variation and OCR-oriented augmentations.

This repository uploads the trained recognizer checkpoint only. The original source text corpora remain credited to their respective owners and sources.

## Model Architecture
* **Encoder**: {args.architecture} backbone adapted for 1-channel grayscale input, with modified strides in layers 3 & 4 to preserve sequence width while reducing height.
* **Vertical Pool**: Mean pooling over the vertical axis.
* **Decoder**: 2-layer Bidirectional GRU (hidden size: 256, dropout: 0.2).
* **Loss**: Connectionist Temporal Classification (CTC) loss.

## Training Highlights
* **Dataset Size**: 200,000 total synthetic text lines.
* **Epochs**: 25.
* **Batch Size**: 256.
* **GPU**: RTX 5090.
* **Augmentations**: Random rotation (±3°), Color Jitter (brightness/contrast), Gaussian Blur, and Random Erasing (ink stains).

## Evaluation Metrics
The model was evaluated on held-out validation/test splits from the 200k dataset.
* **Validation Loss**: **`{args.val_loss}`**
* **Validation CER**: **`{args.val_cer}`**
* **Exact Match Accuracy**: **`{args.exact_match}`**
* **Final Test CER**: **`{args.test_cer}`**

## Loading the Checkpoint in Python
```python
import torch
import safetensors.torch
from src.model import KhmerCRNN_BiGRU

# 1. Load Vocab
with open("vocab.json", "r", encoding="utf-8") as f:
    c2i = json.load(f)
vocab_size = len(c2i) + 1 # Include CTC blank token

# 2. Init Model
model = KhmerCRNN_BiGRU(vocab_size=vocab_size, hidden=256)

# 3. Load Safetensors weights
state_dict = safetensors.torch.load_file("{os.path.basename(args.checkpoint_path)}")

# Clean torch.compile prefixes if present
cleaned_state_dict = {{}}
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        cleaned_state_dict[k[len("_orig_mod."):]] = v
    else:
        cleaned_state_dict[k] = v

model.load_state_dict(cleaned_state_dict)
model.eval()
```
"""
    with open(args.readme_path, "w", encoding="utf-8") as out:
        out.write(readme_content)
    print(f"Saved Model Card to {args.readme_path}")

def upload(args):
    if not TOKEN:
        raise ValueError("HF_TOKEN variable is not found in .env. Please define HF_TOKEN in your .env file.")
        
    api = HfApi(token=TOKEN)
    print(f"Creating Hugging Face repository {args.repo_id} if it doesn't exist...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)
    
    print("Uploading model checkpoint file...")
    api.upload_file(
        path_or_fileobj=args.checkpoint_path,
        path_in_repo=os.path.basename(args.checkpoint_path),
        repo_id=args.repo_id,
        repo_type="model"
    )
    print("Uploading vocab file...")
    api.upload_file(
        path_or_fileobj=args.vocab_path,
        path_in_repo="vocab.json",
        repo_id=args.repo_id,
        repo_type="model"
    )
    print("Uploading README model card...")
    api.upload_file(
        path_or_fileobj=args.readme_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model"
    )
    print(f"\nUpload complete! Model is now available at https://huggingface.co/{args.repo_id}")


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a Khmer OCR model checkpoint to Hugging Face Hub")
    parser.add_argument("--repo-id", default="KimkosalYon/siemreap-arial-khmer-ocr")
    parser.add_argument("--checkpoint-path", default="checkpoints_200k/siemreap_arial_ocr_bs256_epoch_25.safetensors")
    parser.add_argument("--vocab-path", default="checkpoints_200k/vocab.json")
    parser.add_argument("--readme-path", default="checkpoints_200k/README.md")
    parser.add_argument("--model-name", default="Khmer OCR - ResNet34 + BiGRU (Siemreap & Arial)")
    parser.add_argument("--architecture", default="ResNet34")
    parser.add_argument("--val-loss", default="0.008478")
    parser.add_argument("--val-cer", default="0.1635%")
    parser.add_argument("--exact-match", default="95.695%")
    parser.add_argument("--test-cer", default="~0.18%")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    prepare_files(args)
    upload(args)
