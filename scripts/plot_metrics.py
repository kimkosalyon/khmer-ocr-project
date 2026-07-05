"""
Generate publication-quality training metrics plots for the Khmer OCR paper.
Outputs PNG figures suitable for inclusion in LaTeX via `includegraphics`.

Usage:
    uv run python scripts/plot_metrics.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────
with open("checkpoints_200k/metrics_history.json") as f:
    raw = json.load(f)

# Separate records with and without train_loss
train_records = [r for r in raw if "train_loss" in r]
train_epochs  = [r["epoch"] for r in train_records]
train_loss    = [r["train_loss"] for r in train_records]

# Validation metrics (only records that have val_loss)
valid_records = [r for r in raw if "val_loss" in r]
epochs      = [r["epoch"] for r in valid_records]
val_loss    = [r["val_loss"] for r in valid_records]
cer         = [r["cer"] for r in valid_records]
accuracy    = [r["accuracy"] for r in valid_records]

# Build continuous epoch arrays for clean lines
all_epochs = list(range(1, 26))

# Interpolate validation metrics for missing epochs
def interpolate_missing(x_known, y_known, x_all):
    return np.interp(x_all, x_known, y_known)

val_loss_interp = interpolate_missing(epochs, val_loss, all_epochs)
cer_interp      = interpolate_missing(epochs, cer, all_epochs)
acc_interp      = interpolate_missing(epochs, accuracy, all_epochs)

# ── Style setup ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":       10,
    "axes.labelsize":  11,
    "axes.titlesize":  12,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":      200,
    "savefig.dpi":     200,
    "savefig.bbox":    "tight",
    "lines.linewidth": 1.5,
})

# Colour palette
C_TRAIN    = "#ff7f0e"
C_LOSS     = "#d62728"
C_CER      = "#1f77b4"
C_ACC      = "#2ca02c"

def _style_ax(ax):
    """Apply common axis styling."""
    ax.set_xlim(0.5, 25.5)
    ax.set_xticks([1, 3, 5, 10, 15, 20, 25])
    ax.axvline(x=15.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.text(15.5, ax.get_ylim()[1] * 0.95, "LR 1e-4",
            fontsize=7, color="gray", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax.text(1, ax.get_ylim()[1] * 0.95, "LR 1e-3",
            fontsize=7, color="gray", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

# ── 1. Combined Loss (Training + Validation) ──────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(train_epochs, train_loss, color=C_TRAIN, marker="o", ms=3,
        label="Training Loss", zorder=3)
ax.plot(all_epochs, val_loss_interp, color=C_LOSS, marker="s", ms=3,
        label="Validation Loss", zorder=3)
# Mark measured val loss points
ax.scatter(epochs, val_loss, color=C_LOSS, s=12, zorder=4)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("(a) Training and Validation Loss")
ax.legend()
_style_ax(ax)
fig.tight_layout()
fig.savefig("docs/figures/val_loss.pdf")
fig.savefig("docs/figures/val_loss.png")
plt.close(fig)
print("\u2713 docs/figures/val_loss.pdf / .png")

# ── 2. Character Error Rate ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3))
ax.semilogy(all_epochs, cer_interp, color=C_CER, marker="s", ms=3,
            label="Validation CER", zorder=3)
ax.scatter(epochs, cer, color=C_CER, s=12, zorder=4)
ax.axhline(y=1.0, color="gray", ls=":", lw=0.7, alpha=0.5)
ax.text(25.5, 1.05, "1% CER", fontsize=7, color="gray", va="bottom")
ax.set_xlabel("Epoch")
ax.set_ylabel("CER (%) \u2014 log scale")
ax.set_title("(b) Character Error Rate")
ax.legend()
_style_ax(ax)
fig.tight_layout()
fig.savefig("docs/figures/cer.pdf")
fig.savefig("docs/figures/cer.png")
plt.close(fig)
print("\u2713 docs/figures/cer.pdf / .png")

# ── 3. Exact Match Accuracy ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(all_epochs, acc_interp, color=C_ACC, marker="^", ms=3,
        label="Exact Match Accuracy", zorder=3)
ax.scatter(epochs, accuracy, color=C_ACC, s=12, zorder=4)
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("(c) Exact Match Accuracy")
ax.legend()
_style_ax(ax)
fig.tight_layout()
fig.savefig("docs/figures/accuracy.pdf")
fig.savefig("docs/figures/accuracy.png")
plt.close(fig)
print("\u2713 docs/figures/accuracy.pdf / .png")

# ── 4. Combined four-panel figure ──────────────────────────────────────────
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 5.5))

# Panel (a): Training Loss
ax1.plot(train_epochs, train_loss, color=C_TRAIN, marker="o", ms=2.5)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss")
ax1.set_title("(a) Training Loss")
ax1.set_xlim(0.5, 25.5)
ax1.axvline(x=15.5, color="gray", ls="--", lw=0.6, alpha=0.4)

# Panel (b): Validation Loss
ax2.plot(all_epochs, val_loss_interp, color=C_LOSS, marker="s", ms=2.5)
ax2.scatter(epochs, val_loss, color=C_LOSS, s=8)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Loss")
ax2.set_title("(b) Validation Loss")
ax2.set_xlim(0.5, 25.5)
ax2.axvline(x=15.5, color="gray", ls="--", lw=0.6, alpha=0.4)

# Panel (c): CER
ax3.semilogy(all_epochs, cer_interp, color=C_CER, marker="s", ms=2.5)
ax3.scatter(epochs, cer, color=C_CER, s=8)
ax3.axhline(y=1.0, color="gray", ls=":", lw=0.5, alpha=0.4)
ax3.set_xlabel("Epoch"); ax3.set_ylabel("CER (%)")
ax3.set_title("(c) Character Error Rate")
ax3.set_xlim(0.5, 25.5)
ax3.axvline(x=15.5, color="gray", ls="--", lw=0.6, alpha=0.4)

# Panel (d): Accuracy
ax4.plot(all_epochs, acc_interp, color=C_ACC, marker="^", ms=2.5)
ax4.scatter(epochs, accuracy, color=C_ACC, s=8)
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Accuracy (%)")
ax4.set_title("(d) Exact Match Accuracy")
ax4.set_xlim(0.5, 25.5)
ax4.axvline(x=15.5, color="gray", ls="--", lw=0.6, alpha=0.4)

fig.tight_layout()
fig.savefig("docs/figures/training_curves.pdf")
fig.savefig("docs/figures/training_curves.png")
plt.close(fig)
print("\u2713 docs/figures/training_curves.pdf / .png")

print("\nAll figures saved to docs/figures/")
