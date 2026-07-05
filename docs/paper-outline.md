# Research Paper Outline — Khmer OCR

> Draft your paper here before moving to LaTeX.

---

## Paper Title

**Comparing ResNet34 and ResNet18 Backbones for Khmer OCR with BiGRU-CTC**

---

## Author

Your Name
*Dept. of Computer Science, University, City, Country*
*email@domain.com*

---

## Abstract

Khmer script presents exceptional challenges for optical character recognition due
to its complex vertical-stacking morphology — 33 consonants with subscript forms,
16 dependent vowels, 14 independent vowels, and 13 diacritics that attach before,
after, above, or below the base character. These stacked graphemes span multiple
vertical zones, and Khmer lacks explicit inter-word spaces, making sequence-level
transcription the standard formulation.

This paper presents a comparative study of two CRNN architectures for Khmer OCR —
ResNet34 + BiGRU + CTC and ResNet18 + BiGRU + CTC — trained with identical
sequence modelling and data pipelines on a synthetic dataset of 200,000 images
rendered in Siemreap and Arial fonts. On a held-out validation set of 20,000
samples, the deeper ResNet34 achieves a Character Error Rate (CER) of **0.16%**
(99.84% character accuracy) and **95.70% exact match accuracy**, while the
shallower ResNet18 achieves comparable results at **1.74$\times$ fewer parameters**
(13.6M vs. 23.7M). We discuss script-specific architectural considerations —
including receptive field design, asymmetric pooling, and backbone depth effects —
and present an error analysis identifying remaining challenges with confusable
vowel diacritics and densely stacked subscript clusters.

---

## Keywords

Khmer OCR; ResNet; BiGRU; CTC loss; synthetic data; script complexity; sequence
transcription

---

## I. Introduction

### A. The Khmer Script: Complexity and Challenges

The Khmer script is exceptionally complex. It uses 33 base consonants (each
potentially having one or more *subscript* forms via the "coeng" marker),
16 dependent vowels, 14 independent vowels, and 13 diacritics. All vowels and
diacritics can attach before, after, above, or below a consonant, often creating
three-dimensional "stacked" graphemes that span multiple vertical zones. These
stacked clusters — where subscripts and superscripts carry essential phonetic
information — mean that OCR systems must preserve full image height rather than
naively cropping or resizing. As Buoy *et al.* [2] note, resizing long text lines
can "undeniably affect" the legibility of small subscripts or diacritics,
motivating approaches such as overlapping horizontal chunks that retain vertical
context.

A further challenge is word spacing: Khmer text is written *without* regular
inter-word spaces. Spaces in Khmer typically separate phrases or clauses, not
individual words. As a result, Khmer OCR is usually cast as character-sequence
transcription of full lines; word segmentation is treated as a separate NLP task
(e.g., via CRF or neural segmenters) applied after recognition. In benchmarks,
both Character Error Rate (CER) and Word Error Rate (WER) are reported, but CER
is the primary metric. For instance, the KPTMA2026 benchmark [4] shows ResNet-based
OCR achieving 4.15% CER yet ~22.8% WER on Khmer text, reflecting that a single
character error often splits or merges a word boundary.

### B. Why This Problem Needs Solving

Khmer is spoken by over 16 million people, yet digital preservation of Khmer
documents, automation of form processing, and accessibility tools all suffer from
inadequate OCR support. Existing commercial engines and open-source solutions
(e.g., Tesseract) typically perform poorly on Khmer due to insufficient training
data and inadequate modeling of the script's structural properties. Improving
Khmer OCR directly enables large-scale digitization of government gazettes,
historical manuscripts, educational materials, and everyday documents.

### C. Related Work and State of the Art

Public Khmer OCR datasets remain scarce. Notable corpora include:

- **KHOB** (Khmer Official Gazette): scanned printed gazettes; Tesseract achieves
  ~11% CER on KHOB Level-1, while deep models reach sub-1% (e.g., 0.62% [1]).
- **KhmerST**: a scene-text dataset (1,544 images) introduced by Nom *et al.* [3];
  baseline models exceed 20% CER, reflecting the difficulty of real-world Khmer.
- **KPTMA2026**: a bilingual Khmer–English scanned-document dataset; Vitou *et al.* [4]
  report ~4.1% CER for Khmer lines using CRNN architectures.

Architecturally, prior work has compared VGG-CRNN, ResNet-CRNN, and
EfficientNet-CRNN backbones. Vitou *et al.* [4] found that VGG-CRNN (7.94 M params)
gave the best overall bilingual CER (5.80%) due to its shallow, wide design that
preserves horizontal resolution, while ResNet-CRNN (14.43 M params) better
handled Khmer's tall stacked clusters — improving accuracy by ~16.7% relative on
pure Khmer scene text (KhmerST [3]). This trade-off between compact multilingual
models and maximal Khmer-only accuracy is a recurring theme in the literature.

Buoy *et al.* [1] further demonstrated that a CTC model with a Vision Transformer
encoder achieved accuracy comparable to autoregressive attention models while
offering up to ~12× faster inference, suggesting that CTC-based pipelines remain
competitive for Khmer when efficiency matters.

### D. Objectives

This work aims to:
1. Develop a synthetic data generation pipeline producing realistic Khmer text
   images with script-appropriate font, color, and geometric variation.
2. Design and compare two CRNN variants — **ResNet34 + BiGRU + CTC** (23.7M params)
   and **ResNet18 + BiGRU + CTC** (13.6M params) — using identical sequence
   modelling and training setups, isolating the effect of backbone depth.
3. Evaluate on held-out test data and analyze error patterns to guide future
   domain-adaptation and deployment strategies.

---

## II. Datasets

### A. Synthetic Dataset Construction

Given the scarcity of public Khmer OCR datasets, we constructed a synthetic
training corpus of 200,000 text-line images. The dataset is designed to cover
the full range of Khmer Unicode characters while introducing controlled variation
in appearance to improve generalization.

#### A.1 Text Sources

Text content is drawn from two complementary sources:

- **Hanuman-derived samples (100,000 images):** Extracted from the Hanuman Khmer
  dictionary and phrase corpus. These samples cover common vocabulary, isolated
  words, and grammatical phrases, ensuring broad character coverage including
  rare subscript forms.
- **Contextual wiki/Markov samples (100,000 images):** Generated using a
  Markov-chain language model trained on Khmer Wikipedia text. These samples
  produce realistic sentence-level content with natural character bigram/trigram
  distributions, helping the model learn sequential dependencies at the language
  level rather than isolated characters.

#### A.2 Rendering Pipeline

Rendering is performed by a Node.js backend (`renderer/server.js`) using the
HTML Canvas API. Key parameters:

| Parameter     | Value / Range                                    |
|---------------|--------------------------------------------------|
| Font size     | 48 pt                                            |
| Height        | 64 px (fixed)                                    |
| Width         | Variable (depends on text length)                |
| Colors        | Randomized foreground/background (with `--vary-colors`) |
| Padding       | Alternates between tight and zero-padding        |
| Font (Khmer)  | Siemreap                                         |
| Font (Latin)  | Arial (for mixed-script samples)                 |

The renderer exposes a JSON API at `POST /render` accepting `{text, font,
fontSize, color, background}` and returns a PNG image. Mixed Khmer/English text
is routed with Siemreap for Khmer characters and Arial for Latin fallback,
avoiding the common issue of Times New Roman appearing in older renderer
versions.

### B. Dataset Splits

The 200,000 images are split as follows, stratified to maintain consistent font
and source distributions across partitions:

| Split     | Size     | Proportion |
|-----------|----------|------------|
| Training  | 160,000  | 80%        |
| Validation| 20,000   | 10%        |
| Test      | 20,000   | 10%        |

### C. Relationship to Published Benchmarks

Our synthetic CER target (<1%) is consistent with the state of the art on clean
printed Khmer (e.g., ~0.62% on KHOB [1], ~4.15% on KPTMA2026 [4]). However, we
acknowledge that synthetic-only training typically underperforms on real-world
data. Vitou *et al.* [4] observed a ~3% absolute CER gap between synthetic and real
test sets using a VGG-CRNN. Closing this gap through aggressive augmentation and
fine-tuning is a focus of our ongoing work (see Section V).

---

## III. Proposed Methods

### A. Overall Architecture

We compare two CRNN variants with identical sequence modelling — a 2-layer
bidirectional GRU (hidden 256) and CTC decoder — differing only in the CNN
backbone depth. This isolates the effect of backbone capacity on Khmer OCR
performance.

| Variant         | Backbone | Trainable Params | Ratio  |
|-----------------|----------|-----------------:|--------|
| ResNet34-CRNN   | ResNet34 | 23,695,140       | 1.74×  |
| ResNet18-CRNN   | ResNet18 | 13,586,980       | 1×     |

Figure 1 shows the high-level architecture diagram. Both variants share this
structure; the only difference is the residual block depth in the backbone.

```
[Input Image: (B, 1, H, W)]
         │
         ▼
[Modified ResNet34/18] ───► Preserves width with (2,1) strides
         │
         ▼
[Feature Map: (B, 512, H', W')]
         │
         ▼
[Vertical Mean Pooling] ───► Collapses H' dimension
         │
         ▼
[Sequence Formatting] ───► Permutes to (T, B, 512)
         │
         ▼
[2-Layer BiGRU] ───► Captures bidirectional Khmer context
         │
         ▼
[Linear Classifier] ───► Maps to Vocab Size (193 logits)
         │
         ▼
[CTC Decoding/Loss] ───► Collapses blanks/repeats to final Khmer Text
```

*Figure 1: High-level architecture. Each box is a processing stage. The backbone
reads "ResNet34/18" to indicate the two variants compared in this study.*

**Key operations:**
1. **Modified ResNet:** Single-channel input; layers 3–4 use $(2,1)$ stride to
   preserve horizontal resolution, ensuring CTC time steps $T \geq$ label length.
2. **Vertical Mean Pooling:** `f.mean(dim=2)` collapses the spatial height
   dimension, producing a 1D feature sequence per time step.
3. **Sequence Formatting:** Permutes from `(B, C, H', W')` to `(T, B, C)` for
   recurrent processing.
4. **2-Layer BiGRU:** Bidirectional GRU with hidden size 256, dropout 0.2. See
   Figure 2 for the detailed internal structure.
5. **Linear Classifier:** Projects to $|V| + 1$ logits (vocabulary + CTC blank).
6. **CTC Decoding/Loss:** Best-path (greedy) decoding; no external language model.

### B. Backbone Feature Extractor

We adopt ResNet [6] variants adapted for grayscale input by modifying the first
convolutional layer from 3 channels to 1 (no pretrained weights). Both ResNet34
and ResNet18 use the same asymmetric stride and vertical pooling strategies.

**ResNet34** has 16 residual blocks (3+4+6+3) with ~23.7M total parameters. Its
deeper structure provides a larger receptive field, which better captures the
tall stacked clusters formed by Khmer consonants, subscripts, and diacritics.
On pure Khmer scene text (KhmerST [3]), ResNet-based models improve accuracy
by ~16.7% relative to VGG counterparts [4].

**ResNet18** has 8 residual blocks (2+2+2+2) with ~13.6M total parameters — a
1.74× reduction. This makes it more suitable for resource-constrained deployment
while maintaining competitive accuracy when paired with a BiGRU sequence decoder.

#### B.1 Asymmetric Stride Pooling

A critical design decision for Khmer OCR is preserving the sequence length
(horizontal resolution) after CNN feature extraction. The CTC loss requires that
the number of time steps T be at least as large as the target label length L.
Since Khmer text lines can be long and each stacked cluster occupies a narrow
horizontal span, aggressive horizontal downsampling risks violating T ≥ L.

Following Shi *et al.* [5] (CRNN), we apply **asymmetric pooling**: the first two
ResNet layers use standard 2×2 stride, while layers 3 and 4 use 2×1 stride
(stride 2 vertically, 1 horizontally). This yields the following width reduction
for an input of width W:

- Layer 1: W/2
- Layer 2: W/4
- Layer 3: W/4 (no horizontal reduction)
- Layer 4: W/4 (no horizontal reduction)

The final feature map thus maintains a sequence length of approximately W/4,
ensuring T ≥ L for typical Khmer text lines. Buoy *et al.* [1] similarly caution
that naive width downsampling makes subscripts and diacritics "blur out,"
reinforcing the importance of horizontal resolution preservation.

#### B.2 Height-Agnostic Vertical Pooling

To handle variable image heights (though we fix H=64 in our pipeline), we apply
vertical mean pooling (`f.mean(dim=2)`) after the final ResNet layer. This
collapses the height dimension into a single vector per time step, producing a
1D feature sequence of shape `(T, C)` where C = 512 channels from ResNet34.

### C. Recurrent Module: Bidirectional GRU

![BiGRU detail](figures/bigru_detail_flow.png)
*Figure 2: Detailed internal structure of the 2-layer BiGRU module. Each layer
is bidirectional — forward and backward GRU cells process the sequence in both
directions, and their outputs are concatenated before passing to the next layer.
Red-orange: linear projections; dark blue: GRU layers.*

The feature sequence is fed into a 2-layer bidirectional GRU with hidden size 256
and dropout 0.2 between layers. We choose BiGRU [8] over BiLSTM for the following
reasons:

- **Parameter efficiency:** A GRU cell has 3 gates versus LSTM's 4, yielding
  ~25% fewer parameters. This reduces overfitting on our 200k synthetic dataset.
- **Training speed:** GRUs train faster for equivalent hidden sizes.
- **Robustness:** Recent studies show CNN+BiGRU networks are more robust under
  noisy conditions compared to CNN+BiLSTM variants.
- **Empirical adequacy:** For Khmer's typical line lengths, BiGRU's capacity is
  sufficient to model the sequential dependencies; we did not observe a
  meaningful CER difference vs. BiLSTM in preliminary experiments.

The bidirectional processing ensures that each time step's hidden state
incorporates both left and right visual context, which is important for resolving
ambiguous diacritic attachments in Khmer's stacked graphemes.

### D. Classifier and CTC Decoding

The GRU output at each time step is projected through a linear layer to a vector
of size `|V| + 1`, where `|V|` is the character vocabulary size and the extra
slot is the CTC blank token (index 0).

During training, we minimize the CTC loss [7]:

$$
\mathcal{L}_{\text{CTC}} = -\ln \sum_{\pi \in \mathcal{B}^{-1}(y)}
\prod_{t=1}^{T} p(\pi_t \mid x)
$$

where the sum is over all alignments $\pi$ that collapse to the target label
sequence $y$ via the many-to-one mapping $\mathcal{B}$ (removing blanks and
merging repeated characters).

During inference, we use best-path (greedy) decoding: at each time step, we take
the argmax of the classifier output, then collapse the resulting sequence by
removing blanks and merging consecutive identical characters. No external
language model is used, keeping inference fast and self-contained.

### E. Design Rationale Summary

| Component      | Choice        | Rationale for Khmer OCR                                     |
|----------------|---------------|-------------------------------------------------------------|
| Backbone       | ResNet34/18   | Residual connections for deep feature extraction; ResNet34 offers larger receptive field, ResNet18 offers 1.74× parameter reduction |
| Pooling        | Asymmetric    | Preserves horizontal resolution (T ≥ L); prevents subscript/diacritic blur |
| RNN            | BiGRU         | Fewer parameters than BiLSTM; faster training; robust to noise |
| Decoder        | CTC (greedy)  | Fast inference; no alignment needed; competitive with attention (up to 12× faster per Buoy *et al.* [1]) |
| Loss           | CTC Loss [7]  | Standard for sequence transcription; supports class imbalance via variants (Focal CTC) |

### F. Discussion: CTC vs. Attention for Khmer

While attention-based decoders (e.g., ASTER [9], SAR [10]) can model linguistic context
at each output step and often outperform CTC on irregular or curved text, CTC
remains highly competitive for Khmer for several reasons:

- **Monotonic alignment:** Khmer text lines are left-to-right and largely
  regular, matching CTC's monotonicity assumption.
- **Inference speed:** CTC with greedy decoding is non-autoregressive, offering
  up to ~12× faster inference than attention models (Buoy *et al.* [1]).
- **Competitive accuracy:** On clean printed Khmer (KHOB [1]), CTC-based models
  achieve sub-1% CER, comparable to attention-based approaches.

Our choice of CTC is thus driven by the efficiency and accuracy requirements of
a production OCR pipeline, where throughput matters and the input is primarily
printed/scanned text rather than highly distorted scene text.

---

## IV. Experiments and Results

### A. Experimental Setup

Training was performed on the 160k/20k/20k split described in Section II. Key
hyperparameters:

#### Model Architecture

| Component          | Configuration                                  |
|--------------------|------------------------------------------------|
| Backbone           | ResNet34 (no pretrained weights, 1-channel in) |
| Encoder strides    | Layers 1–2: (2, 2); Layers 3–4: (2, 1)        |
| Vertical pooling   | Mean-pooling along height (`f.mean(dim=2)`)    |
| Recurrent decoder  | 2-layer Bidirectional GRU                      |
| GRU hidden size    | 256 (512 after bidirectional concatenation)    |
| GRU dropout        | 0.2                                            |
| Loss function      | CTC (blank index 0, `zero_infinity=True`) [7]  |

#### Optimizer & Schedule

| Parameter                | Value                        |
|--------------------------|------------------------------|
| Optimizer                | AdamW                        |
| Weight decay             | $1 \times 10^{-4}$           |
| Learning rate (epochs 1–15) | $1 \times 10^{-3}$        |
| Learning rate (epochs 16–25) | $1 \times 10^{-4}$ (resume stage) |
| Scheduler                | None (constant LR per stage) |
| Batch size               | 256 (dynamic-width, grouped batching) |
| Input height             | 64 px                        |
| Input width              | Dynamic (aspect-ratio preserved, zero-padded to batch max) |

#### Data Augmentation

| Augmentation      | Parameters                                   |
|-------------------|----------------------------------------------|
| Random rotation   | $\pm 3^\circ$                                |
| Color jitter      | Brightness 0.2, Contrast 0.2                 |
| Gaussian blur     | Kernel $3 \times 3$, $\sigma \in [0.1, 1.0]$ |
| Random erasing    | Probability 0.4, scale $[0.02, 0.08]$ (ink-stain simulation) |

All augmentations were applied after `ToTensor()`. The model was first trained
for 15 epochs at $1 \times 10^{-3}$, then resumed from the epoch-15 checkpoint
and trained for 10 additional epochs (16–25) at $1 \times 10^{-4}$ to
stabilize convergence. No learning-rate scheduler was used within either stage.

Both ResNet34-CRNN and ResNet18-CRNN were trained with **identical** hyperparameters
listed above. The only architectural difference is the CNN backbone depth.

### B. Evaluation Protocols

**Primary metric:** Normalized Character Error Rate (CER)

$$
\text{CER} = \frac{\text{edit\_distance}(\hat{y}, y)}{\max(|\hat{y}|, |y|)}
$$

where $\hat{y}$ is the predicted text and $y$ is the ground truth. CER is the
standard metric in OCR literature because it directly reflects character-level
recognition quality and is invariant to text length.

**Secondary metric:** Exact Match Accuracy (EMA)

$$
\text{EMA} = \frac{\text{\# perfectly matched samples}}{\text{total samples}}
$$

EMA measures the proportion of text lines transcribed with zero errors. For Khmer
OCR, this is a strict metric because a single missing diacritic or subscript
counts as a complete failure. EMA is complementary to CER: a low CER with
moderate EMA suggests that most errors are small (one or two characters per
line), which is the typical profile for a well-trained Khmer OCR model.

### C. Results

#### C.1 Overall Performance

The final models (epoch 25) achieved the following on the held-out validation
set of 20,000 samples:

| Metric                       | ResNet34-CRNN | ResNet18-CRNN |
|------------------------------|:-------------:|:-------------:|
| **Validation CER**           | **0.16%**     | **0.16%**     |
| **Character Accuracy**       | **99.84%**    | **99.84%**    |
| **Exact Match Accuracy**     | **95.70%**    | **95.55%**    |
| Final Validation Loss        | 0.0085        | 0.0087        |
| Trainable Parameters         | 23,695,140    | 13,586,980    |

Despite having 1.74× fewer parameters, the ResNet18-CRNN achieves nearly identical
performance to its deeper counterpart — a CER of 0.16% (99.84% character accuracy)
and an exact match accuracy of 95.55% versus 95.70%. This suggests that for
synthetic printed Khmer text at 64px height, the bottleneck is not backbone
capacity but rather the quality and diversity of the training data and
augmentation.

#### C.2 Training Dynamics

Figure 1 shows the training curves across all 25 epochs. Key observations:

- **Rapid initial convergence (epochs 1–3):** Validation loss dropped from 0.73
  to 0.06, CER fell from 17.97% to 1.11%, and accuracy jumped from 2.23% to
  70.50%. This indicates the CRNN architecture quickly learns Khmer's basic
  character shapes and sequential patterns.
- **Loss spike at epoch 6:** The validation loss briefly spiked to 0.88
  (CER 0.87%), likely due to a difficult batch or augmentation artifact.
  The model recovered within one epoch.
- **Two-stage training (epochs 1–15 at $1 \times 10^{-3}$, 16–25 at $1 \times 10^{-4}$):**
  The initial high learning rate drove fast convergence. After resuming from the
  epoch-15 checkpoint at a reduced rate, training stabilized and produced steady
  improvement: loss decreased from 0.0209 to 0.0085, CER dropped from 0.60% to
  0.16%, and EMA rose from 89.33% to 95.70%. No overfitting was observed.

![Training curves](figures/training_curves.png)
*Figure 1: Training curves across 25 epochs. (a) Validation loss; (b) Character
Error Rate (log scale); (c) Exact Match Accuracy. The dashed vertical line at
epoch 15 marks the resume point.*

#### C.3 Error Analysis

Despite the low CER, the remaining 4.30% of lines with errors exhibit consistent
patterns:

- **Confusable vowel diacritics:** Pairs such as ឹ/ឺ (short/long u) and ែ/េ
  (ae/e) are occasionally swapped, particularly in low-contrast renderings.
- **Subscript consonant omissions:** In densely stacked clusters with 3+ stacked
  characters, the model occasionally drops a medial subscript.
- **Padding sensitivity:** Lines with extreme tight padding (characters touching
  the image border) show marginally higher CER.

These errors are consistent with the known challenges of Khmer OCR reported in
the literature [1][2] and are expected to decrease with larger datasets and
real-image fine-tuning.

---

## V. Conclusion

### Summary

We presented a comparative study of two CRNN architectures for Khmer OCR —
ResNet34-CRNN (23.7M params) and ResNet18-CRNN (13.6M params) — trained with
identical sequence modelling and data pipelines. Both models achieve:

- **0.16% CER (99.84% character accuracy)** on held-out synthetic validation data
- **~95.6% exact match accuracy** (95.70% for ResNet34, 95.55% for ResNet18)
- Rapid convergence (sub-1% CER by epoch 3) and stable two-stage training

The ResNet18-CRNN achieves these results with **1.74× fewer parameters** than the
ResNet34-CRNN, demonstrating that a shallower backbone is sufficient for printed
Khmer text recognition at 64px resolution when paired with an effective BiGRU
sequence decoder and aggressive augmentation. This has practical implications for
resource-constrained deployment scenarios.

### Future Work

- [ ] Larger and more diverse datasets (real scanned documents, handwritten text)
- [ ] YOLO-based text detection for end-to-end document OCR pipeline
- [ ] Attention decoder and/or language model post-correction
- [ ] Real-world evaluation on scanned Khmer documents to measure and close the
      synthetic-to-real domain gap (~3% absolute CER as observed in [4])
- [ ] Focal CTC or class-weighted losses to address rare diacritic errors

---

## Acknowledgment

*(Optional)*

---

## References

### Khmer OCR — Datasets and Benchmarks

[1] R. Buoy, M. Iwamura, S. Srun, and K. Kise, "Toward a low-resource non-latin-complete baseline: An exploration of Khmer optical character recognition," *IEEE Access*, vol. 11, pp. 138 544–138 566, 2023. doi:10.1109/ACCESS.2023.3340160
<!-- Cited for: KHOB baseline results (Tesseract ~11% CER, deep models sub-1%), ViT+CTC vs. attention comparison, 12× speedup, subscript/diacritic blur, KHOB dataset -->

[2] R. Buoy, S. Chenda, N. Taing, and M. Kong, "Addressing the attention drift problem for Khmer long textline recognition," *International Journal on Document Analysis and Recognition (IJDAR)*, 2025. doi:10.1007/s10032-025-00554-6
<!-- Cited for: overlapping horizontal chunks to retain vertical context, naive downsampling blurring subscripts/diacritics -->

[3] V. Nom, S. Bakkali, M. M. Luqman, M. Coustaty, and J.-M. Ogier, "KhmerST: A low-resource Khmer scene text detection and recognition benchmark," in *Proc. Asian Conf. Computer Vision (ACCV)*, 2024.
<!-- Cited for: KhmerST dataset (1,544 images), scene-text baselines >20% CER -->

[4] S. Vitou and E. Koungmeng, "KPTMA2026: A real-world bilingual Khmer-English OCR benchmark with lightweight CRNN baselines," 2026.
<!-- Cited for: KPTMA2026 benchmark, 4.15% CER / 22.8% WER on Khmer, VGG vs. ResNet vs. EfficientNet CRNN comparison, ~3% synthetic-to-real gap -->

### Foundational Architectures

[5] B. Shi, X. Bai, and C. Yao, "An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 39, no. 11, pp. 2298–2304, 2017. doi:10.1109/TPAMI.2016.2646371
<!-- Cited for: CRNN architecture, asymmetric (rectangular) pooling to preserve sequence width -->

[6] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.
<!-- Cited for: ResNet34 backbone -->

[7] A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, "Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks," in *Proc. Int. Conf. Machine Learning (ICML)*, 2006, pp. 369–376.
<!-- Cited for: CTC loss function -->

[8] K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2014, pp. 1724–1734.
<!-- Cited for: GRU cell architecture -->

### Attention-Based OCR Decoders

[9] B. Shi, M. Yang, X. Wang, P. Lyu, C. Yao, and X. Bai, "ASTER: An attentional scene text recognizer with flexible rectification," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 41, no. 9, pp. 2035–2048, 2018. doi:10.1109/TPAMI.2018.2848939
<!-- Cited for: ASTER attention decoder in CTC vs. attention discussion -->

[10] H. Li, P. Wang, C. Shen, and G. Zhang, "Show, attend and read: A simple and strong baseline for irregular text recognition," in *Proc. AAAI Conf. Artificial Intelligence*, 2019, pp. 8610–8617.
<!-- Cited for: SAR attention decoder in CTC vs. attention discussion -->

### General OCR and Augmentation

[11] M. Jaderberg, K. Simonyan, A. Vedaldi, and A. Zisserman, "Synthetic data and artificial neural networks for natural scene text recognition," in *NIPS Workshop on Deep Learning*, 2014.
<!-- Cited for: MJSynth-style synthetic data; if referenced in later domain-adaptation discussion -->

---

## Summary of Missing Citations

Here is a cross-check between all citation markers in the text and the reference list:

| Citation in text | Reference # | Status |
|---|---|---|
| Buoy *et al.* (subscript/diacritic blur, ViT+CTC, 12× speedup) | [1], [2] | ✅ Added |
| Nom *et al.* (KhmerST dataset) | [3] | ✅ Added |
| Vitou *et al.* (KPTMA2026, VGG vs. ResNet, 3% synthetic-real gap) | [4] | ✅ Added |
| KPTMA2026 benchmark | [4] | ✅ Added |
| KHOB dataset & results | [1] | ✅ Added (via Buoy 2023) |
| Shi *et al.* (CRNN, asymmetric pooling) | [5] | ✅ Added |
| ResNet34 backbone | [6] | ✅ Added |
| CTC loss | [7] | ✅ Added |
| GRU | [8] | ✅ Added |
| ASTER attention decoder | [9] | ✅ Added |
| SAR attention decoder | [10] | ✅ Added |
| Focal CTC | — | ⚠️ Mentioned but no specific paper cited; add if needed |
| ICDAR 2021 augmentation study | — | ⚠️ Not yet in current sections; add when Experiments are written |
| Zhao *et al.* (domain adaptation) | — | ⚠️ Not yet in current sections; add when Experiments are written |
| Tesseract (baseline on KHOB) | [1] | ✅ Covered by Buoy 2023 results |
| MJSynth | [11] | ✅ Added as generalized reference |
