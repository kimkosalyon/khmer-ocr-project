import os
import sys
import json
import time
import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from datasets import load_dataset
from tqdm import tqdm
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from src.model import KhmerCRNN_BiGRU
from src.decode import ctc_greedy_decode, calculate_cer

class KhmerHFDataset(Dataset):
    """Dataset wrapper for the Hugging Face Khmer OCR dataset."""
    def __init__(self, hf_ds, c2i, augment=True):
        self.ds = hf_ds
        self.c2i = c2i
        if augment:
            self.transform = T.Compose([
                T.Resize((48, 256)),
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
                T.Resize((48, 256)),
                T.Grayscale(1),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row['image']  # Already loaded as PIL Image by HF datasets
        x = self.transform(img.convert("RGB"))
        t = [self.c2i[c] for c in row['text'] if c in self.c2i]
        return x, torch.tensor(t, dtype=torch.long)

def fast_pad_collate(batch):
    """Pads variable-width target label sequences inside the batch."""
    xs, ts = zip(*batch)
    imgs = torch.stack(xs)
    tgt_lens = torch.tensor([len(t) for t in ts], dtype=torch.long)
    targets = nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
    return imgs, targets, tgt_lens

def run_evaluation(model, loader, c2i, device, blank=0):
    """Run evaluation and return average loss and CER."""
    model.eval()
    ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True).to(device)
    i2c = {v: k for k, v in c2i.items()}
    
    total_loss = 0.0
    preds = []
    targets = []
    
    with torch.no_grad():
        for imgs, tgts, tgt_lens in tqdm(loader, desc="Evaluating", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            
            logits = model(imgs)
            T_steps, B, V = logits.shape
            log_probs = logits.log_softmax(2)
            input_lens = torch.full((B,), T_steps, dtype=torch.long, device=device)
            
            loss = ctc_loss(log_probs, tgts, input_lens, tgt_lens)
            total_loss += loss.item()
            
            log_probs_cpu = log_probs.cpu()
            tgts_cpu = tgts.cpu()
            tgt_lens_list = tgt_lens.tolist()
            
            for b in range(B):
                decoded_indices = ctc_greedy_decode(log_probs_cpu[:, b, :], blank=blank)
                pred_text = "".join(i2c.get(idx, "") for idx in decoded_indices)
                
                tgt_indices = tgts_cpu[b, :tgt_lens_list[b]].tolist()
                tgt_text = "".join(i2c.get(idx, "") for idx in tgt_indices)
                
                preds.append(pred_text)
                targets.append(tgt_text)
                
    avg_loss = total_loss / len(loader)
    cer = calculate_cer(preds, targets)
    return avg_loss, cer

@click.command()
@click.option('--dataset-id', default='KimkosalYon/khmer-ocr-500k', help='Hugging Face dataset repo ID.')
@click.option('--token', default=os.getenv('HF_TOKEN'), help='HF Token for private datasets.')
@click.option('--epochs', default=50, type=int, help='Training epochs.')
@click.option('--batch-size', default=256, type=int, help='Batch size.')
@click.option('--lr', default=1e-3, type=float, help='Learning rate.')
@click.option('--device', default='cuda', type=str, help='cpu/cuda.')
@click.option('--compile/--no-compile', default=True, help='torch.compile model JIT.')
@click.option('--checkpoint-dir', default='checkpoints_hf', type=str, help='Directory to save checkpoints.')
def train_cli(dataset_id, token, epochs, batch_size, lr, device, compile, checkpoint_dir):
    print("Step 1: Downloading & splitting dataset...")
    full_dataset = load_dataset(dataset_id, token=token)
    ds = full_dataset['train']
    
    # 80/10/10 split
    train_testval = ds.train_test_split(test_size=0.20, seed=42)
    test_val = train_testval['test'].train_test_split(test_size=0.50, seed=42)
    
    train_ds = train_testval['train']
    val_ds = test_val['train']
    test_ds = test_val['test']
    
    print(f"Dataset split size:")
    print(f"  Train:      {len(train_ds):,} samples")
    print(f"  Validation: {len(val_ds):,} samples")
    print(f"  Test:       {len(test_ds):,} samples")
    
    # Step 2: Build vocabulary from training set
    print("Step 2: Building vocabulary mapping...")
    all_chars = sorted({ch for row in train_ds for ch in row['text']})
    c2i = {c: i+1 for i, c in enumerate(all_chars)}
    raw_vocab = len(all_chars) + 1
    vocab = raw_vocab if raw_vocab % 8 == 0 else ((raw_vocab // 8) + 1) * 8
    print(f"Unique characters: {len(all_chars)} | Model padded vocab size: {vocab}")
    
    train_dataset = KhmerHFDataset(train_ds, c2i, augment=True)
    val_dataset = KhmerHFDataset(val_ds, c2i, augment=False)
    test_dataset = KhmerHFDataset(test_ds, c2i, augment=False)
    
    num_workers = os.cpu_count() or 4 if device == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=fast_pad_collate, num_workers=num_workers,
                              pin_memory=(device == 'cuda'), persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=fast_pad_collate, num_workers=num_workers,
                            pin_memory=(device == 'cuda'), persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=fast_pad_collate, num_workers=num_workers,
                             pin_memory=(device == 'cuda'), persistent_workers=(num_workers > 0))
    
    print(f"Step 3: Initializing CRNN model on {device}...")
    model = KhmerCRNN_BiGRU(vocab_size=vocab).to(device)
    
    if compile and device == 'cuda':
        print("Compiling model for speedup...")
        model = torch.compile(model, dynamic=True)
        
    blank = 0
    ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("\nStep 4: Starting training loops...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for imgs, tgts, tgt_lens in pbar:
            imgs = imgs.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if device == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(imgs)
                    T_steps, B, V = logits.shape
                    log_probs = logits.log_softmax(2)
                    input_lens = torch.full((B,), T_steps, dtype=torch.long, device=device)
                    loss = ctc_loss(log_probs, tgts, input_lens, tgt_lens)
            else:
                logits = model(imgs)
                T_steps, B, V = logits.shape
                log_probs = logits.log_softmax(2)
                input_lens = torch.full((B,), T_steps, dtype=torch.long, device=device)
                loss = ctc_loss(log_probs, tgts, input_lens, tgt_lens)
                
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        epoch_time = time.time() - start_time
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation evaluation
        avg_val_loss, val_cer = run_evaluation(model, val_loader, c2i, device, blank=blank)
        
        print(f"Epoch {epoch:2d} finished in {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val CER: {val_cer*100:.2f}%")
        
        # Save checkpoints
        if epoch % 10 == 0 or epoch == epochs:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"ocr_hf_epoch_{epoch}.safetensors")
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            
            metadata = {
                "epoch": str(epoch),
                "loss": f"{avg_val_loss:.6f}",
                "vocab": str(vocab),
                "c2i": json.dumps(c2i)
            }
            state_dict = {k: v.cpu().contiguous() for k, v in raw_model.state_dict().items()}
            save_file(state_dict, ckpt_path, metadata=metadata)
            print(f"Saved checkpoint: {ckpt_path}")
            
    # Final evaluation on test set
    print("\nRunning final evaluation on the test split...")
    test_loss, test_cer = run_evaluation(model, test_loader, c2i, device, blank=blank)
    print(f"Test Loss: {test_loss:.4f} | Test CER: {test_cer*100:.2f}%")

if __name__ == "__main__":
    train_cli()
