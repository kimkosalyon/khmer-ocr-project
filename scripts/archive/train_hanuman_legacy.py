import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from dotenv import load_dotenv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import click
import mlflow
from safetensors import safe_open
from safetensors.torch import save_file, load_file

from src.config import TrainConfig
from src.model import KhmerCRNN_BiGRU
from src.dataset import KhmerImgDataset, fast_pad_collate, LengthGroupedBatchSampler
from src.decode import ctc_greedy_decode, calculate_cer

load_dotenv()



def run_training(cfg: TrainConfig):
    """Main training execution loop."""
    print("Step 1: Downloading/Loading dataset...")
    if cfg.dataset_dir:
        print(f"Loading local custom dataset from: {cfg.dataset_dir}")
        db_path = os.path.join(cfg.dataset_dir, "metadata.duckdb")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"DuckDB database not found at {db_path}")
            
        import duckdb
        con = duckdb.connect(db_path)
        if cfg.full:
            query = "SELECT filename, text FROM metadata"
            samples_name = "local_full"
        else:
            query = f"SELECT filename, text FROM metadata LIMIT {cfg.samples}"
            samples_name = f"local_{cfg.samples}"
            
        rows = con.execute(query).fetchall()
        con.close()
        
        ds = [
            {
                "image": os.path.join(cfg.dataset_dir, filename),
                "text": text
            }
            for filename, text in rows
        ]
        print(f"Loaded {len(ds)} samples from local database.")
        text_col = "text"
        img_col = "image"
    else:
        full_ds = load_dataset("seanghay/khmer-hanuman-100k", split="train")
        if cfg.full:
            ds = full_ds
            samples_name = "full"
        else:
            ds = full_ds.select(range(cfg.samples))
            samples_name = str(cfg.samples)
        print(f"Loaded {len(ds)} samples from Hugging Face.")
        text_col = next(c for c in ds.column_names if c in ("text", "label", "ground_truth"))
        img_col  = next(c for c in ds.column_names if c in ("image", "img", "pixel_values"))


    start_epoch = 1
    blank = 0

    if cfg.resume:
        if not os.path.exists(cfg.resume):
            raise FileNotFoundError(f"Checkpoint file not found: {cfg.resume}")

        print(f"Loading metadata from checkpoint {cfg.resume}...")
        with safe_open(cfg.resume, framework="pt") as f:
            metadata = f.metadata()

        start_epoch = int(metadata.get("epoch", 0)) + 1
        c2i = json.loads(metadata.get("c2i"))
        vocab_meta = metadata.get("vocab")
        try:
            vocab = int(vocab_meta)
        except (TypeError, ValueError):
            vocab = max(c2i.values(), default=0) + 1
        print(f"Resuming from Epoch {start_epoch} (Checkpoint vocab size: {vocab})")
    else:
        all_chars = sorted({ch for row in ds for ch in row[text_col]})
        c2i = {c: i+1 for i, c in enumerate(all_chars)}
        raw_vocab = len(all_chars) + 1
        vocab = raw_vocab if raw_vocab % 8 == 0 else ((raw_vocab // 8) + 1) * 8

    if cfg.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:15005")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("khmer-ocr")
    mlflow.pytorch.autolog()

    i2c = {v: k for k, v in c2i.items()}

    # Shuffling and splitting if val_split/test_split is set
    import random
    random.seed(42)
    
    val_ds = None
    test_ds = None
    
    if cfg.val_split > 0.0 or cfg.test_split > 0.0:
        total_split = cfg.val_split + cfg.test_split
        if total_split >= 1.0:
            raise ValueError("Sum of validation and test split must be less than 1.0")
            
        if isinstance(ds, list):
            # For local list-of-dicts dataset
            random.shuffle(ds)
            val_idx = int(len(ds) * (1 - total_split))
            test_idx = int(len(ds) * (1 - cfg.test_split))
            train_ds = ds[:val_idx]
            val_ds = ds[val_idx:test_idx] if cfg.val_split > 0.0 else None
            test_ds = ds[test_idx:] if cfg.test_split > 0.0 else None
        else:
            # For Hugging Face Dataset
            train_testval = ds.train_test_split(test_size=total_split, seed=42)
            train_ds = train_testval["train"]
            if cfg.test_split > 0.0 and cfg.val_split > 0.0:
                test_fraction = cfg.test_split / total_split
                test_val = train_testval["test"].train_test_split(test_size=test_fraction, seed=42)
                val_ds = test_val["train"]
                test_ds = test_val["test"]
            elif cfg.val_split > 0.0:
                val_ds = train_testval["test"]
                test_ds = None
            else:
                val_ds = None
                test_ds = train_testval["test"]
            
        print(f"Dataset split: {len(train_ds):,} train samples, "
              f"{len(val_ds) if val_ds else 0:,} validation samples, "
              f"{len(test_ds) if test_ds else 0:,} test samples.")
    else:
        train_ds = ds
        val_ds = None
        test_ds = None
        print(f"Training on all {len(train_ds):,} samples (no validation or test split).")

    train_dataset = KhmerImgDataset(train_ds, text_col, img_col, c2i, augment=True)
    
    actual_workers = cfg.num_workers if cfg.device == "cuda" else 0
    train_sampler = LengthGroupedBatchSampler(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=fast_pad_collate,
        num_workers=actual_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(actual_workers > 0)
    )

    if val_ds is not None:
        val_dataset = KhmerImgDataset(val_ds, text_col, img_col, c2i, augment=False)
        val_sampler = LengthGroupedBatchSampler(val_dataset, batch_size=cfg.batch_size, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=fast_pad_collate,
            num_workers=actual_workers,
            pin_memory=(cfg.device == "cuda"),
            persistent_workers=(actual_workers > 0)
        )
    else:
        val_loader = None

    if test_ds is not None:
        test_dataset = KhmerImgDataset(test_ds, text_col, img_col, c2i, augment=False)
        test_sampler = LengthGroupedBatchSampler(test_dataset, batch_size=cfg.batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            collate_fn=fast_pad_collate,
            num_workers=actual_workers,
            pin_memory=(cfg.device == "cuda"),
            persistent_workers=(actual_workers > 0)
        )
    else:
        test_loader = None




    print(f"Step 2: Initializing Model on {cfg.device}...")
    model = KhmerCRNN_BiGRU(vocab).to(cfg.device)

    if cfg.resume:
        print(f"Loading checkpoint weights from {cfg.resume}...")
        state_dict = load_file(cfg.resume)
        model.load_state_dict(state_dict)

    if cfg.compile and cfg.device == "cuda":
        print("Compiling model with torch.compile (JIT Speedup & Dynamic shapes)...")
        model = torch.compile(model, dynamic=True)

    ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs - start_epoch + 1)
    elif cfg.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", patience=5, factor=0.5)
    else:
        scheduler = None

    if cfg.profile and cfg.device == "cuda":
        print("Resetting peak memory tracker...")
        torch.cuda.reset_peak_memory_stats()

    print(f"Step 3: Starting training loop from Epoch {start_epoch}...")
    total_start_time = time.time()

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        epoch_start_time = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{cfg.epochs}", leave=True)

        for imgs, targets, tgt_lens in pbar:
            imgs = imgs.to(cfg.device, non_blocking=True)
            targets = targets.to(cfg.device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            if cfg.device == "cuda":
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(imgs)
                    T_steps, B, V = logits.shape
                    log_probs = logits.log_softmax(2)
                    input_lens = torch.full((B,), T_steps, dtype=torch.long, device=cfg.device)
                    loss = ctc_loss(log_probs, targets, input_lens, tgt_lens)
            else:
                logits = model(imgs)
                T_steps, B, V = logits.shape
                log_probs = logits.log_softmax(2)
                input_lens = torch.full((B,), T_steps, dtype=torch.long, device=cfg.device)
                loss = ctc_loss(log_probs, targets, input_lens, tgt_lens)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optim.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(loader)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        # Validation step
        val_loss = 0.0
        val_cer = 0.0
        
        if val_loader is not None:
            model.eval()
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for imgs_val, targets_val, tgt_lens_val in val_loader:
                    imgs_val = imgs_val.to(cfg.device, non_blocking=True)
                    targets_val = targets_val.to(cfg.device, non_blocking=True)
                    
                    logits_val = model(imgs_val)
                    T_steps_val, B_val, V_val = logits_val.shape
                    log_probs_val = logits_val.log_softmax(2)
                    input_lens_val = torch.full((B_val,), T_steps_val, dtype=torch.long, device=cfg.device)
                    
                    loss_val = ctc_loss(log_probs_val, targets_val, input_lens_val, tgt_lens_val)
                    val_loss += loss_val.item()
                    
                    # CTC decoding for CER
                    log_probs_val_cpu = log_probs_val.cpu()
                    targets_val_cpu = targets_val.cpu()
                    
                    tgt_lens_val_list = tgt_lens_val.tolist()
                    
                    for b in range(B_val):
                        decoded_indices = ctc_greedy_decode(log_probs_val_cpu[:, b, :], blank=blank)
                        pred_text = "".join(i2c.get(idx, "") for idx in decoded_indices)
                        
                        tgt_indices = targets_val_cpu[b, :tgt_lens_val_list[b]].tolist()
                        tgt_text = "".join(i2c.get(idx, "") for idx in tgt_indices)
                        
                        val_preds.append(pred_text)
                        val_targets.append(tgt_text)
                        
            val_loss = val_loss / len(val_loader)
            val_cer = calculate_cer(val_preds, val_targets)
            
            # Log to MLflow
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_cer", val_cer, step=epoch)

        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        if val_loader is not None:
            if cfg.profile:
                max_vram = torch.cuda.max_memory_allocated(device=cfg.device) / (1024 ** 3) if cfg.device == "cuda" else 0.0
                tqdm.write(f"Epoch {epoch:3d} finished in {epoch_time:.2f}s. Peak VRAM: {max_vram:.3f} GB | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val CER: {val_cer*100:.2f}%")
            elif epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
                tqdm.write(f"Epoch {epoch:3d} finished. Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val CER: {val_cer*100:.2f}%")
        else:
            if cfg.profile:
                max_vram = torch.cuda.max_memory_allocated(device=cfg.device) / (1024 ** 3) if cfg.device == "cuda" else 0.0
                tqdm.write(f"Epoch {epoch:3d} finished in {epoch_time:.2f}s. Peak VRAM: {max_vram:.3f} GB | Train Loss: {avg_loss:.4f}")
            elif epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
                tqdm.write(f"Epoch {epoch:3d} finished. Train Loss: {avg_loss:.4f}")


        if epoch % 10 == 0 or epoch == cfg.epochs:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(cfg.checkpoint_dir, f"hanuman{samples_name}_bs{cfg.batch_size}_epoch_{epoch}.safetensors")
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

            metadata = {
                "epoch": str(epoch),
                "loss": f"{avg_loss:.6f}",
                "vocab": str(vocab),
                "c2i": json.dumps(c2i),
                "preserve_aspect_ratio": "true"
            }

            state_dict = {k: v.cpu().contiguous() for k, v in raw_model.state_dict().items()}
            save_file(state_dict, checkpoint_path, metadata=metadata)
            tqdm.write(f"Saved safetensors checkpoint: {checkpoint_path}")

    total_time = time.time() - total_start_time
    if cfg.profile:
        print(f"Profiling finished. Total run time: {total_time:.2f} seconds.")

    # Final evaluation on test set
    if test_loader is not None:
        print("\nStep 4: Running final evaluation on the test split...")
        model.eval()
        test_loss = 0.0
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for imgs_test, targets_test, tgt_lens_test in tqdm(test_loader, desc="Testing", leave=True):
                imgs_test = imgs_test.to(cfg.device, non_blocking=True)
                targets_test = targets_test.to(cfg.device, non_blocking=True)
                
                logits_test = model(imgs_test)
                T_steps_test, B_test, V_test = logits_test.shape
                log_probs_test = logits_test.log_softmax(2)
                input_lens_test = torch.full((B_test,), T_steps_test, dtype=torch.long, device=cfg.device)
                
                loss_test = ctc_loss(log_probs_test, targets_test, input_lens_test, tgt_lens_test)
                test_loss += loss_test.item()
                
                log_probs_test_cpu = log_probs_test.cpu()
                targets_test_cpu = targets_test.cpu()
                tgt_lens_test_list = tgt_lens_test.tolist()
                
                for b in range(B_test):
                    decoded_indices = ctc_greedy_decode(log_probs_test_cpu[:, b, :], blank=blank)
                    pred_text = "".join(i2c.get(idx, "") for idx in decoded_indices)
                    
                    tgt_indices = targets_test_cpu[b, :tgt_lens_test_list[b]].tolist()
                    tgt_text = "".join(i2c.get(idx, "") for idx in tgt_indices)
                    
                    test_preds.append(pred_text)
                    test_targets.append(tgt_text)
                    
        test_loss = test_loss / len(test_loader)
        test_cer = calculate_cer(test_preds, test_targets)
        print(f"Final Test Results | Loss: {test_loss:.4f} | CER: {test_cer*100:.2f}%")
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_cer", test_cer)


@click.command()
@click.option('--samples', default=10000, type=int, help='Number of dataset samples.')
@click.option('--epochs', default=100, type=int, help='Number of training epochs.')
@click.option('--batch-size', default=256, type=int, help='Batch size for training.')
@click.option('--lr', default=1e-3, type=float, help='Learning rate.')
@click.option('--compile/--no-compile', default=True, help='Enable/disable JIT compilation.')
@click.option('--device', default='cuda', type=str, help='Compute device to run on (cuda/cpu).')
@click.option('--num-workers', default=os.cpu_count() or 4, type=int, help='Number of DataLoader worker processes.')
@click.option('--profile', is_flag=True, help='Profile execution time and peak VRAM memory usage per epoch.')
@click.option('--checkpoint-dir', default='checkpoints', type=str, help='Directory to save model checkpoints.')
@click.option('--resume', default=None, type=click.Path(exists=True), help='Path to a safetensors checkpoint to resume from.')
@click.option('--full', is_flag=True, help='Train on the full dataset (100K samples) ignoring --samples.')
@click.option('--lr-scheduler', default='none', type=click.Choice(['none', 'cosine', 'plateau']), help='LR scheduler type.')
@click.option('--dataset-dir', default=None, type=str, help='Path to local custom dataset folder (with metadata.duckdb).')
@click.option('--val-split', default=0.0, type=float, help='Fraction of dataset to use for validation (e.g. 0.1).')
@click.option('--test-split', default=0.0, type=float, help='Fraction of dataset to use for testing (e.g. 0.1).')
def train_cli(samples, epochs, batch_size, lr, compile, device, num_workers, profile, checkpoint_dir, resume, full, lr_scheduler, dataset_dir, val_split, test_split):
    cfg = TrainConfig(
        samples=samples,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        compile=compile,
        device=device,
        num_workers=num_workers,
        profile=profile,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        full=full,
        lr_scheduler=lr_scheduler,
        dataset_dir=dataset_dir,
        val_split=val_split,
        test_split=test_split
    )


    run_training(cfg)


if __name__ == '__main__':
    train_cli()
