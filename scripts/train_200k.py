import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import json
import random
import argparse
import urllib.request
from collections import Counter
from io import BytesIO
from pathlib import Path


import torch
import torch.nn as nn
import torchvision.models as tv
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import duckdb
from tqdm import tqdm
from khmernormalizer import normalize as khmer_normalize

# Setup sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

# =====================================================================
# 1. Model Architecture (Height-Agnostic ResNet-34 + BiGRU)
# =====================================================================
class KhmerCRNN_BiGRU(nn.Module):
    """CRNN (ResNet34 backbone) + Bidirectional GRU model for sequence alignment.
    Uses height-agnostic mean-pooling to handle arbitrary input image heights.
    """
    def __init__(self, vocab_size, hidden=256):
        super().__init__()
        rn = tv.resnet34(weights=None)
        rn.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1, rn.layer2)
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

        # Patch stride in ResNet layers 3 and 4 to squash height but preserve width
        for b in [self.layer3, self.layer4]:
            for block in b:
                if hasattr(block, "conv1") and block.conv1.stride != (1, 1):
                    block.conv1.stride = (2, 1)
                if block.downsample is not None:
                    block.downsample[0].stride = (2, 1)
                    
        self.bigru = nn.GRU(512, hidden, num_layers=2, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x shape: [B, 1, H, W]
        f = self.stem(x)
        f = self.layer3(f)
        f = self.layer4(f)
        
        # Height-agnostic vertical mean pooling (compiles cleanly on torch.compile)
        f = f.mean(dim=2)  # [B, C, W']
        f = f.permute(2, 0, 1)  # [W', B, C]
        
        out, _ = self.bigru(f)
        return self.fc(out)


# =====================================================================
# 2. Dataset & Padding Pipeline
# =====================================================================
class Khmer200kDataset(Dataset):
    """Loads and preprocesses generated Khmer text images with aspect ratio preservation."""
    def __init__(self, records: list[dict], data_dir: str, c2i: dict, height: int = 64, augment: bool = True):
        self.ds = records
        self.data_dir = data_dir
        self.c2i = c2i
        self.height = height

        if augment:
            self.transform = T.Compose([
                T.RandomRotation(degrees=(-3, 3), fill=255),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.RandomApply([
                    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))
                ], p=0.3),
                T.Grayscale(1),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
                T.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0.5, inplace=False)
            ])
        else:
            self.transform = T.Compose([
                T.Grayscale(1),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img_path = os.path.join(self.data_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        
        # Proportional resize preserving aspect ratio
        w, h = img.size
        new_w = max(8, int(w * (self.height / h)))
        img = img.resize((new_w, self.height), Image.Resampling.BILINEAR)

        x = self.transform(img)
        t = [self.c2i[c] for c in row["text"] if c in self.c2i]
        return x, torch.tensor(t, dtype=torch.long)


def pad_collate(batch):
    """Collation utility to dynamically pad variable-width images horizontally."""
    xs, ts = zip(*batch)
    max_w = max(x.shape[2] for x in xs)
    
    padded_xs = []
    for x in xs:
        pad_w = max_w - x.shape[2]
        # Pad with 1.0 (corresponds to white background after normalization)
        padded_x = torch.nn.functional.pad(x, (0, pad_w, 0, 0), value=1.0)
        padded_xs.append(padded_x)
        
    imgs = torch.stack(padded_xs)
    tgt_lens = torch.tensor([len(t) for t in ts], dtype=torch.long)
    targets = nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
    return imgs, targets, tgt_lens


class LengthGroupedBatchSampler(Sampler):
    """Groups samples of similar widths/lengths into batches to minimize padding overhead."""
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lengths = [len(row["text"]) for row in self.dataset.ds]
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            noise = [random.uniform(-3, 3) for _ in range(len(indices))]
            indices.sort(key=lambda idx: self.lengths[idx] + noise[idx])
        else:
            indices.sort(key=lambda idx: self.lengths[idx])
            
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)
        
    def __len__(self):
        import math
        return math.ceil(len(self.dataset) / self.batch_size)


# =====================================================================
# 3. Decoding & Levenshtein CER with Khmer Normalization
# =====================================================================
def ctc_greedy_decode(log_probs, blank=0):
    best_path = log_probs.argmax(dim=-1)
    decoded = []
    prev = blank
    for idx in best_path:
        if idx != blank and idx != prev:
            decoded.append(idx.item())
        prev = idx
    return decoded


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_single_cer(pair):
    pred, tgt = pair
    if pred == tgt:
        return 0, len(tgt)
    n_pred = khmer_normalize(pred)
    dist = levenshtein_distance(n_pred, tgt)
    return dist, len(tgt)


def calculate_normalized_cer(preds: list[str], tgts: list[str], num_workers: int = 1) -> float:
    """Computes CER by applying khmernormalizer to predictions in parallel (targets are already pre-normalized)."""
    if num_workers <= 1:
        total_dist = 0
        total_len = 0
        for pred, tgt in zip(preds, tgts):
            if pred == tgt:
                total_len += len(tgt)
                continue
            n_pred = khmer_normalize(pred)
            total_dist += levenshtein_distance(n_pred, tgt)
            total_len += len(tgt)
        return total_dist / total_len if total_len > 0 else 0.0

    from multiprocessing import Pool
    pairs = list(zip(preds, tgts))
    with Pool(num_workers) as pool:
        results = pool.map(compute_single_cer, pairs)
    
    total_dist = sum(r[0] for r in results)
    total_len = sum(r[1] for r in results)
    return total_dist / total_len if total_len > 0 else 0.0




# =====================================================================
# 4. Training Loop
# =====================================================================
def run_training(args):
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Loading metadata from DuckDB database...")
    db_path = os.path.join(args.data_dir, "metadata.duckdb")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Run generate script first.")

    con = duckdb.connect(db_path)
    df = con.execute("SELECT filename, text FROM metadata").fetchdf()
    con.close()

    # Pre-normalize targets in parallel to avoid redundant computation during evaluation loops
    print(f"Pre-normalizing ground truth labels in parallel using {args.num_workers} processes...")
    from multiprocessing import Pool
    with Pool(args.num_workers) as pool:
        df["text"] = pool.map(khmer_normalize, df["text"].tolist())



    # Apply sample size constraint
    if args.samples and args.samples < len(df):
        print(f"Limiting dataset to {args.samples} samples...")
        df = df.sample(n=args.samples, random_state=args.seed).reset_index(drop=True)

    # Filter by maximum character length to prevent VRAM spikes
    if args.max_text_len:
        print(f"Filtering dataset: keeping lines <= {args.max_text_len} characters...")
        df = df[df["text"].str.len() <= args.max_text_len].reset_index(drop=True)


    records = df.to_dict(orient="records")
    print(f"Loaded {len(records)} total records.")

    # Vocab Extraction
    vocab = sorted(list(set("".join(df["text"].tolist()))))
    # Pad vocabulary to nearest multiple of 8 for Blackwell Tensor Core optimization
    while len(vocab) % 8 != 0:
        vocab.append(" ")
        
    c2i = {char: idx + 1 for idx, char in enumerate(vocab)}
    i2c = {idx + 1: char for idx, char in enumerate(vocab)}
    i2c[0] = ""
    vocab_size = len(vocab) + 1
    print(f"Vocabulary size (including blank): {vocab_size}")

    # Dataset split
    random.shuffle(records)
    n_total = len(records)
    n_val = int(n_total * args.val_split)
    n_test = int(n_total * args.test_split)
    n_train = n_total - n_val - n_test

    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:]

    print(f"Dataset split: {len(train_records)} train, {len(val_records)} val, {len(test_records)} test.")

    # Dataloaders
    train_ds = Khmer200kDataset(train_records, args.data_dir, c2i, height=args.height, augment=True)
    val_ds = Khmer200kDataset(val_records, args.data_dir, c2i, height=args.height, augment=False)
    test_ds = Khmer200kDataset(test_records, args.data_dir, c2i, height=args.height, augment=False)

    train_sampler = LengthGroupedBatchSampler(train_ds, args.batch_size, shuffle=True)
    val_sampler = LengthGroupedBatchSampler(val_ds, args.batch_size, shuffle=False)
    test_sampler = LengthGroupedBatchSampler(test_ds, args.batch_size, shuffle=False)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=pad_collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=pad_collate, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_sampler=test_sampler, collate_fn=pad_collate, num_workers=args.num_workers)

    # Initialize Model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = KhmerCRNN_BiGRU(vocab_size=vocab_size, hidden=256)
    
    start_epoch = 1
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        from safetensors.torch import load_file
        import safetensors
        
        state_dict = load_file(args.resume)
        # Strip '_orig_mod.' prefix if present from JIT compilation
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                cleaned_state_dict[k[len("_orig_mod."):]] = v
            else:
                cleaned_state_dict[k] = v
        model.load_state_dict(cleaned_state_dict)
        
        with safetensors.safe_open(args.resume, framework="pt") as f:
            metadata = f.metadata()
        if metadata and "epoch" in metadata:
            start_epoch = int(metadata["epoch"]) + 1
            print(f"Loaded weights. Resuming from Epoch {start_epoch}")

            
    model = model.to(device)

    if args.compile and device.type == "cuda":
        print("Compiling model (torch.compile)...")
        model = torch.compile(model, dynamic=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Initialize Scheduler
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    else:
        scheduler = None

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)


    # Enable mixed precision training on GPU
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Training
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            
        model.train()
        train_loss = 0.0

        
        # Batch Loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, targets, tgt_lens in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(imgs)  # [W', B, vocab_size]
                log_probs = logits.float().log_softmax(2)
                input_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long, device=device)
                loss = criterion(log_probs, targets.to(device), input_lengths, tgt_lens.to(device))

            if not torch.isnan(loss) and not torch.isinf(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Apply gradient clipping to stabilize training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_preds, val_tgts = [], []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d} Val", leave=False)
        with torch.no_grad():
            for imgs, targets, tgt_lens in val_pbar:
                imgs = imgs.to(device)

                logits = model(imgs)
                log_probs = logits.float().log_softmax(2)
                input_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long, device=device)
                loss = criterion(log_probs, targets.to(device), input_lengths, tgt_lens.to(device))
                val_loss += loss.item() if not torch.isnan(loss) else 0.0

                
                # CTC Greedy decode predictions
                logits_cpu = logits.cpu()
                targets_cpu = targets.cpu()
                for b in range(imgs.size(0)):
                    decoded = ctc_greedy_decode(logits_cpu[:, b, :], blank=0)
                    pred_txt = "".join(i2c.get(idx, "") for idx in decoded)
                    
                    tgt_indices = targets_cpu[b, :tgt_lens[b]].tolist()
                    tgt_txt = "".join(i2c.get(idx, "") for idx in tgt_indices)
                    
                    val_preds.append(pred_txt)
                    val_tgts.append(tgt_txt)

        avg_val_loss = val_loss / len(val_loader)
        val_cer = calculate_normalized_cer(val_preds, val_tgts, num_workers=args.num_workers)

        
        peak_vram_str = ""
        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_vram_str = f" | Peak VRAM: {peak_vram:.2f} GB"
            
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val CER (Normalized): {val_cer * 100:.2f}%{peak_vram_str}")

        
        # Print sample predictions for visual verification
        print("--- Validation Samples ---")
        indices = random.sample(range(len(val_preds)), min(3, len(val_preds)))
        for idx in indices:
            print(f"  Target:    {val_tgts[idx]}")
            print(f"  Predicted: {val_preds[idx]}")
            print("-" * 50)
        print()

        # Step learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # Save checkpoints according to user's pacing schedule:
        # - Save every epoch for the first 10 epochs
        # - Save every 5 epochs after epoch 10
        # - Save on the final epoch
        is_save_epoch = False
        if epoch <= 10:
            is_save_epoch = True
        elif (epoch - 10) % 5 == 0:
            is_save_epoch = True
        elif epoch == args.epochs:
            is_save_epoch = True

        if is_save_epoch:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"siemreap_arial_ocr_bs{args.batch_size}_epoch_{epoch}.safetensors")
            print(f"Saving checkpoint to {checkpoint_path}...")
            from safetensors.torch import save_file
            
            raw_model = model.module if hasattr(model, "module") else model
            state_dict = raw_model.state_dict()
            # Strip '_orig_mod.' prefix if present from JIT compilation
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    cleaned_state_dict[k[len("_orig_mod."):]] = v
                else:
                    cleaned_state_dict[k] = v
            state_dict = {k: v.cpu().contiguous() for k, v in cleaned_state_dict.items()}

            
            metadata = {
                "epoch": str(epoch),
                "loss": f"{avg_val_loss:.6f}",
                "val_loss": f"{avg_val_loss:.6f}",
                "val_cer": f"{val_cer:.8f}",
                "val_cer_percent": f"{val_cer * 100:.4f}",
                "train_loss": f"{avg_train_loss:.6f}",
                "batch_size": str(args.batch_size),
                "lr": str(args.lr),
                "height": str(args.height),
                "epochs": str(args.epochs),
                "data_dir": str(args.data_dir),
                "model_arch": str(getattr(args, "model_arch", "ResNet34_BiGRU_CTC")),
                "val_split": str(args.val_split),
                "test_split": str(args.test_split),
                "max_text_len": str(args.max_text_len),
                "vocab_size": str(vocab_size),
                "vocab": str(vocab),
                "c2i": json.dumps(c2i)
            }
            save_file(state_dict, checkpoint_path, metadata=metadata)





    # Final Testing Evaluation
    print("\nRunning final test split evaluation...")
    test_preds, test_tgts = [], []
    test_pbar = tqdm(test_loader, desc="Final Test Evaluation")
    with torch.no_grad():
        for imgs, targets, tgt_lens in test_pbar:
            imgs = imgs.to(device)

            logits = model(imgs)
            logits_cpu = logits.cpu()
            targets_cpu = targets.cpu()
            for b in range(imgs.size(0)):
                decoded = ctc_greedy_decode(logits_cpu[:, b, :], blank=0)
                pred_txt = "".join(i2c.get(idx, "") for idx in decoded)
                tgt_indices = targets_cpu[b, :tgt_lens[b]].tolist()
                tgt_txt = "".join(i2c.get(idx, "") for idx in tgt_indices)
                test_preds.append(pred_txt)
                test_tgts.append(tgt_txt)
                
    test_cer = calculate_normalized_cer(test_preds, test_tgts, num_workers=args.num_workers)

    print(f"Final Test CER (Normalized): {test_cer * 100:.2f}%")

    summary_path = os.path.join(args.checkpoint_dir, "run_summary.json")
    summary = {
        "model_arch": str(getattr(args, "model_arch", "ResNet34_BiGRU_CTC")),
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "height": args.height,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "max_text_len": args.max_text_len,
        "vocab_size": vocab_size,
        "train_records": len(train_ds),
        "val_records": len(val_ds),
        "test_records": len(test_ds),
        "test_cer": test_cer,
        "test_cer_percent": test_cer * 100,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved run summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone training script on 200k Khmer dataset")
    parser.add_argument("--data-dir", default="generated/training_200k_siemreap_arial", help="Output dir of generate script")
    parser.add_argument("--samples", type=int, default=None, help="Number of records to train on (None = all)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--height", type=int, default=64, help="Target crop height (e.g. 48 or 64)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile", dest="compile", action="store_false", help="Disable torch.compile")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--checkpoint-dir", default="checkpoints_200k")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .safetensors to resume from")
    parser.add_argument("--lr-scheduler", default="none", choices=["none", "cosine", "plateau"], help="Learning rate scheduler")
    parser.add_argument("--max-text-len", type=int, default=199, help="Maximum text character length to prevent width/memory spikes")



    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_training(args)
