---
license: mit
language:
- km
- en
task_categories:
- image-to-text
pretty_name: Khmer OCR 200k Siemreap Arial
tags:
- ocr
- khmer
- synthetic-data
- text-recognition
- image-to-text
- ctc
size_categories:
- 100K<n<1M
---

# Khmer OCR 200k Siemreap Arial

This dataset contains **200,000 synthetic OCR text-line images** for Khmer text recognition. It was generated for training CRNN-style Khmer OCR models such as ResNet34/ResNet18 + BiGRU + CTC.

## Dataset Description

Each row contains:

- `image`: rendered text-line image
- `text`: ground-truth transcription
- optional metadata such as source, script class, font, font size, colors, and padding values

The images are rendered at a fixed height of approximately **64 px** with **dynamic width**, depending on text length.

## Text Sources and Attribution

The dataset text is built from two main sources:

1. **Hanuman-derived Khmer text**

   Approximately 100,000 samples are derived from the public Hugging Face dataset [`seanghay/khmer-hanuman-100k`](https://huggingface.co/datasets/seanghay/khmer-hanuman-100k). Credit is given to the original uploader/owner **seanghay** for publishing this Khmer text corpus.

2. **Khmer Wikipedia / Markov contextual text**

   Approximately 100,000 samples are generated from a local Khmer Wikipedia text corpus using word-level sampling and Markov-style generation to produce contextual Khmer text lines.

## Rendering Pipeline

Images are rendered using a Node.js/Canvas renderer.

- Khmer text font: **Siemreap**
- Latin fallback font: **Arial**
- Image height: **64 px**
- Image width: dynamic
- Background/text colors: varied
- Padding: varied to simulate tight and natural OCR crops

The dataset includes Khmer-only, mixed Khmer/Latin, numbers, and punctuation where present in the generated text.

## Intended Use

This dataset is intended for:

- Khmer OCR model training
- Text-line recognition
- CNN + RNN / CTC experiments
- Lightweight OCR backbone comparison
- Synthetic-to-real OCR research

It is especially suitable for models that treat OCR as sequence recognition:

```text
Image -> CNN feature extractor -> sequence model -> CTC decoder -> text
```

## Example Model Results

Models trained on this dataset include:

| Model | Validation CER | Exact Match Accuracy | Test CER |
|---|---:|---:|---:|
| ResNet34 + BiGRU + CTC | 0.1635% | 95.695% | ~0.18% |
| ResNet18 + BiGRU + CTC | 0.4325% | 90.245% | 0.4509% |

## Limitations

This is a synthetic dataset. It may not fully represent real-world OCR conditions such as:

- scanned document noise
- camera perspective distortion
- handwriting
- degraded paper
- uneven lighting
- historical fonts or rare typography

For real-world deployment, fine-tuning or evaluation on scanned/photographed Khmer documents is recommended.

## Citation / Credit

If you use this dataset, please credit this dataset and the original Hanuman-derived text source:

```text
KimkosalYon/khmer-ocr-200k-siemreap-arial
seanghay/khmer-hanuman-100k
```

## License

This dataset card is provided under the MIT license. Please also respect the licenses and terms of the original text sources used to construct the corpus.
