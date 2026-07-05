# Khmer OCR: Comparing ResNet34 and ResNet18 Backbones

This repository contains the training, evaluation, and serving pipelines for Khmer Optical Character Recognition (OCR) models. We evaluate and compare two CRNN architectures: **ResNet34 + BiGRU + CTC** and **ResNet18 + BiGRU + CTC**.

## 📊 Results Summary

Both models were trained for 25 epochs on a synthetic dataset of 200,000 Khmer text-line images rendered in Siemreap and Arial fonts.

| Model | Trainable Params | Validation Loss | Validation CER | Exact Match Accuracy | Test CER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet34 CRNN** | 23.7M | 0.0085 | **0.16%** | **95.70%** | **~0.18%** |
| **ResNet18 CRNN** | **13.6M** | 0.0189 | 0.43% | 90.25% | 0.45% |

---

## 🛠️ Setup & Installation

This project uses `uv` for fast Python dependency management.

```bash
# Install dependencies
uv sync
```

For the image rendering pipeline:
```bash
cd renderer
npm install
node server.js
```

---

## 🚀 How to Run

### 1. Generate the Synthetic Dataset
With the Node.js renderer server running at `http://localhost:3458`:
```bash
uv run python scripts/generate_200k_siemreap_arial.py \
  --count 200000 \
  --output generated/training_200k_siemreap_arial \
  --num-workers 16 \
  --render-url http://localhost:3458/render \
  --vary-colors
```

### 2. Train the ResNet34 Model
```bash
uv run python scripts/train_200k_resnet34_bigru.py \
  --data-dir generated/training_200k_siemreap_arial \
  --epochs 25 \
  --batch-size 256 \
  --lr 1e-4 \
  --device cuda \
  --checkpoint-dir checkpoints_200k
```

### 3. Train the ResNet18 Comparison Model
```bash
uv run python scripts/train_200k_resnet18_bigru.py \
  --data-dir generated/training_200k_siemreap_arial \
  --epochs 25 \
  --batch-size 256 \
  --lr 1e-4 \
  --device cuda \
  --checkpoint-dir checkpoints_200k_resnet18_bigru
```

### 4. Run the Streamlit OCR Demo
```bash
uv run streamlit run scripts/serve.py
```

---

## 🙏 Acknowledgment
The authors would like to thank Dr. Dona Valy for teaching the Advanced Machine Learning course, and for his guidance and support.
