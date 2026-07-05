import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
    import marimo as mo
    mo.md(
        """
        # Khmer OCR — CRNN + BiGRU
        This interactive notebook runs the training pipeline for Khmer text recognition using the **Khmer Hanuman 100k** dataset from HuggingFace.

        The model uses:
        - A modified **ResNet34** encoder (for 1-channel grayscale inputs, squashed vertical strides)
        - A **2-layer Bidirectional GRU** decoder
        - **CTC Loss** for sequence alignment
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from datasets import load_dataset

    from src.model import KhmerCRNN_BiGRU
    from src.dataset import KhmerImgDataset, fast_pad_collate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    mo.md(
        f"""
        ### System Status
        * **Device:** `{device}`
        * **GPU Model:** `{device_name}`
        * **TF32 Matmul:** `{"Enabled" if device == "cuda" else "Disabled/N/A"}`
        * **cuDNN Benchmark:** `{"Enabled" if device == "cuda" else "Disabled/N/A"}`
        """
    )
    return DataLoader, KhmerCRNN_BiGRU, KhmerImgDataset, device, fast_pad_collate, load_dataset, nn, torch


@app.cell
def _(load_dataset, mo):
    with mo.redirect_stderr():
        import sys
        print("Checking/loading HuggingFace dataset (seanghay/khmer-hanuman-100k)...", file=sys.stderr)
        full_ds = load_dataset("seanghay/khmer-hanuman-100k", split="train")
        print("Dataset loaded successfully.", file=sys.stderr)
    return (full_ds,)


@app.cell
def _(mo):
    samples_slider = mo.ui.slider(
        start=1000,
        stop=50000,
        step=1000,
        value=10000,
        label="Dataset Size (Samples)"
    )
    samples_slider
    return (samples_slider,)


@app.cell
def _(full_ds, mo, samples_slider):
    ds = full_ds.select(range(samples_slider.value))

    text_col = next(c for c in ds.column_names if c in ("text", "label", "ground_truth"))
    img_col  = next(c for c in ds.column_names if c in ("image", "img", "pixel_values"))

    all_chars = sorted({ch for row in ds for ch in row[text_col]})
    c2i = {c: i+1 for i, c in enumerate(all_chars)}

    raw_vocab = len(all_chars) + 1
    vocab = raw_vocab if raw_vocab % 8 == 0 else ((raw_vocab // 8) + 1) * 8

    status_card = mo.md(
        f"""
        ### Dataset Status
        * **Samples Selected:** {len(ds):,} / {len(full_ds):,}
        * **Vocabulary Size:** {raw_vocab} characters (padded to {vocab} for Tensor Core optimization)
        * **Detected Columns:** Text: `{text_col}` | Image: `{img_col}`
        """
    )
    status_card
    return c2i, ds, img_col, text_col, vocab


@app.cell
def _(ds, mo):
    index_slider = mo.ui.slider(
        start=0,
        stop=len(ds)-1,
        step=1,
        value=0,
        label="Preview Dataset Row"
    )
    return (index_slider,)


@app.cell
def _(ds, img_col, index_slider, mo, text_col, vocab):
    row = ds[index_slider.value]
    img = row[img_col]
    text = row[text_col]

    preview_ui = mo.vstack([
        mo.md(f"### Dataset Preview (Vocabulary Size: {vocab})"),
        index_slider,
        mo.md(f"**Ground Truth Text:** `{text}`"),
        img
    ])
    preview_ui
    return


@app.cell
def _(KhmerImgDataset, c2i, ds, img_col, text_col):
    dataset = KhmerImgDataset(ds, text_col, img_col, c2i)
    return (dataset,)


@app.cell
def _(mo):
    epochs_input = mo.ui.number(start=1, stop=200, value=20, label="Epochs")
    batch_size_input = mo.ui.dropdown(options=["64", "128", "256", "512", "1024", "2048"], value="256", label="Batch Size")
    lr_input = mo.ui.number(start=1e-5, stop=1e-2, value=1e-3, step=1e-5, label="Learning Rate")
    compile_input = mo.ui.checkbox(value=False, label="Use torch.compile (JIT Speedup)")

    settings_ui = mo.md(
        f"""
        ### Training Hyperparameters
        Adjust the settings below:

        {mo.hstack([epochs_input, batch_size_input, lr_input, compile_input])}
        """
    )
    settings_ui
    return batch_size_input, compile_input, epochs_input, lr_input


@app.cell
def _(
    DataLoader,
    KhmerCRNN_BiGRU,
    batch_size_input,
    compile_input,
    dataset,
    device,
    fast_pad_collate,
    lr_input,
    nn,
    torch,
    vocab,
):
    train_loader = DataLoader(
        dataset,
        batch_size=int(batch_size_input.value),
        shuffle=True,
        collate_fn=fast_pad_collate,
        num_workers=0,
        pin_memory=True
    )

    model = KhmerCRNN_BiGRU(vocab).to(device)
    if compile_input.value and device == "cuda":
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr_input.value, weight_decay=1e-4)
    return ctc_loss, model, optim, train_loader


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Start Training", kind="success")
    train_button
    return (train_button,)


@app.cell
def _(
    ctc_loss,
    device,
    epochs_input,
    mo,
    model,
    optim,
    torch,
    train_button,
    train_loader,
):
    if not train_button.value:
        mo.output.replace(mo.md("Click the **Start Training** button above to begin training the model."))
    else:
        epochs = epochs_input.value
        mo.output.replace(mo.md(f"### Training model on `{device}`..."))

        import time
        with mo.status.progress_bar(total=epochs, title="Training CRNN model") as bar:
            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = 0.0
                start_time = time.time()
                num_batches = len(train_loader)

                for step, (imgs, targets, tgt_lens) in enumerate(train_loader, 1):
                    imgs = imgs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    optim.zero_grad(set_to_none=True)

                    if device == "cuda":
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            logits = model(imgs)
                            T_steps, B, V = logits.shape
                            log_probs = logits.log_softmax(2)
                            input_lens = torch.full((B,), T_steps, dtype=torch.long, device=device)
                            loss = ctc_loss(log_probs, targets, input_lens, tgt_lens)
                    else:
                        logits = model(imgs)
                        T_steps, B, V = logits.shape
                        log_probs = logits.log_softmax(2)
                        input_lens = torch.full((B,), T_steps, dtype=torch.long, device=device)
                        loss = ctc_loss(log_probs, targets, input_lens, tgt_lens)

                    loss.backward()
                    optim.step()

                    total_loss += loss.item()

                    elapsed = time.time() - start_time
                    rate = step / elapsed if elapsed > 0 else 0
                    bar.update(
                        increment=0,
                        subtitle=f"Epoch {epoch}/{epochs} | Step {step}/{num_batches} | Loss: {loss.item():.4f} | Speed: {rate:.2f} it/s"
                    )

                avg_loss = total_loss / num_batches
                bar.update(
                    increment=1,
                    subtitle=f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f}"
                )

        mo.output.replace(
            mo.md(
                f"""
                ### Training Complete!
                * **Finished Epochs:** {epochs}
                * **Final Avg Loss:** `{avg_loss:.4f}`
                """
            )
        )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
