# Project Summary & Engineering Log: Khmer OCR (CRNN + BiGRU)

This document summarizes the milestones achieved, technical lessons learned, and mistakes made (along with self-corrections) during the development and optimization of the Khmer OCR training workspace.

---

## 🚀 Milestones & Key Features Added

### 1. Robust Click CLI & Dataclass Configuration
* Refactored [train.py](file:///home/kimkosal-y/yk-projects/training-ocr/train.py) from a static, hardcoded script to a modular command-line tool.
* Structured all hyperparameters and execution settings into a Python `@dataclass` (`TrainConfig`). This cleanly separates the Click parsing layer (`train_cli`) from the machine learning execution loop (`run_training`), allowing the script to be imported and called programmatically from notebooks or other scripts.

### 2. High-Performance Blackwell GPU Tuning
* Enabled Tensor Core optimizations by padding the vocabulary size (`vocab`) to the nearest multiple of 8.
* Configured TensorFloat-32 (`allow_tf32 = True`) and cuDNN benchmarking (`benchmark = True`) to maximize throughput on the NVIDIA RTX 5090.
* Integrated dynamic JIT model compilation using `torch.compile(model, dynamic=True)`.

### 3. Organized, Pickle-Free Checkpoint System (`safetensors`)
* Replaced standard Python pickle serialization (`torch.save`) with the industry-standard `.safetensors` format (`save_file`). This ensures fast, secure, and zero-copy model weight storage.
* Serialized crucial training metadata—including the last completed `epoch`, average `loss`, `vocab` size, and the character-to-index mapping dictionary (`c2i`)—directly into the **safetensors header metadata** as a JSON string.
* Automatically routes checkpoints into a clean, custom directory structure (`checkpoints/`) rather than cluttering the root workspace.

### 4. Resumable Training Pipeline
* Implemented a `--resume` flag. Passing a path to a `.safetensors` file extracts the dictionary mapping, vocabulary size, and the completed epoch count from the file header, loads the weights, and resumes training from the exact next epoch.

### 5. Production-Grade Data Augmentations
* Preprocessed text images with advanced transforms to simulate real-world document distortions:
  * **Geometry:** `RandomRotation` to handle slightly tilted text lines.
  * **Lighting:** `ColorJitter` to handle poor contrast and uneven exposure.
  * **Focus:** `GaussianBlur` applied randomly to simulate camera defocus blur.
  * **Dirt/Stains:** `RandomErasing` applied last on tensors to simulate page ink blobs, stains, or physical tears.

---

## 🧠 Lessons Learned

1. **JIT Compiler Verification vs. Eager execution:**
   * Eager execution hides compiler incompatibilities. Operations like `nn.AdaptiveAvgPool2d((1, None))` work fine in Eager Mode but crash under symbolic dynamic compilation because PyTorch Inductor cannot evaluate relational statements (e.g. comparing height to a constant) inside custom CUDA kernel tracing.
2. **Local Scopes and Multiprocessing Pickle Limits:**
   * Declaring datasets or loader collators inside local function cells in Marimo makes them unpickleable by Python's multiprocessing backend. For notebook environments, data loading must either run on the main thread (`num_workers=0`) or be imported from a separate module file.
3. **Structuring Safetensors Metadata:**
   * While `safetensors` strictly blocks saving non-tensor elements (like strings, dicts, or ints) in the main dictionary, its string-only header metadata acts as a secure, fast database to store serialized JSON objects alongside the weights.

---

## ⚠️ Mistakes & Self-Corrections

### 1. Click CLI Signature Mismatch (Parameter Bug)
* **The Mistake:** Added the `--profile` Click option but forgot to add the parameter `profile` to the `def train(...)` function arguments list, causing Click to raise a `TypeError: train() got an unexpected keyword argument 'profile'`.
* **The Correction:** Caught on the first run, viewed the signature file, added the parameter, and re-executed.

### 2. Relational Compilation Crash (Pooling Bug)
* **The Mistake:** Originally, the model used vertical adaptive average pooling. When JIT compiling with `dynamic=True` to support dynamic horizontal widths, the compiler crashed trying to trace height variables.
* **The Correction:** Replaced the pooling layer with a mathematical equivalent: taking the tensor mean along the height dimension (`f.mean(dim=2)`). This bypasses the compiler bug entirely while retaining identical functionality and weights.

### 3. Transform Type-Safety Mismatch (PIL vs. Tensor)
* **The Mistake:** Initially planned to put `RandomErasing` before `ToTensor()`.
* **The Correction:** Verified via web documentation search that `RandomErasing` is strictly a Tensor-only operation. Positioned it at the end of the transform Compose sequence immediately after `ToTensor()`, ensuring no runtime type-mismatch crashes.
