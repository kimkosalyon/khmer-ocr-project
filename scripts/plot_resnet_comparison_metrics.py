"""Plot ResNet34 vs ResNet18 OCR validation metrics.

Usage:
    uv run python scripts/plot_resnet_comparison_metrics.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    rows = [r for r in rows if "val_loss" in r and "cer" in r and "accuracy" in r]
    return {
        "epoch": np.array([r["epoch"] for r in rows]),
        "val_loss": np.array([r["val_loss"] for r in rows]),
        "cer": np.array([r["cer"] for r in rows]),
        "accuracy": np.array([r["accuracy"] for r in rows]),
    }


def interp(metric, all_epochs):
    return np.interp(all_epochs, metric["epoch"], metric)


resnet34 = load_metrics("checkpoints_200k/metrics_history.json")
resnet18 = load_metrics("checkpoints_200k_resnet18_bigru/metrics_history.json")
all_epochs = np.arange(1, 26)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 220,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

colors = {"ResNet34": "#1f77b4", "ResNet18": "#ff7f0e"}


def plot_one(ax, key, ylabel, title, logy=False):
    for name, data in [("ResNet34", resnet34), ("ResNet18", resnet18)]:
        y_interp = np.interp(all_epochs, data["epoch"], data[key])
        ax.plot(all_epochs, y_interp, color=colors[name], lw=1.8, label=name)
        ax.scatter(data["epoch"], data[key], color=colors[name], s=14, zorder=3)
    if logy:
        ax.set_yscale("log")
    ax.set_xlim(0.5, 25.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.45)
    ax.legend()


fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
plot_one(axes[0], "val_loss", "Validation Loss", "(a) Validation Loss", logy=True)
plot_one(axes[1], "cer", "CER (%)", "(b) Character Error Rate", logy=True)
plot_one(axes[2], "accuracy", "Exact Match Accuracy (%)", "(c) Exact Match Accuracy")
fig.tight_layout()
fig.savefig(OUT / "resnet_comparison_metrics.pdf")
fig.savefig(OUT / "resnet_comparison_metrics.png")
plt.close(fig)

print("Saved docs/figures/resnet_comparison_metrics.pdf")
print("Saved docs/figures/resnet_comparison_metrics.png")


summary_rows = [
    ["ResNet34", resnet34["val_loss"][-1], resnet34["cer"][-1], resnet34["accuracy"][-1]],
    ["ResNet18", resnet18["val_loss"][-1], resnet18["cer"][-1], resnet18["accuracy"][-1]],
]
print("\nFinal validation metrics")
for name, loss, cer, acc in summary_rows:
    print(f"{name}: val_loss={loss:.6f}, CER={cer:.4f}%, exact_match={acc:.3f}%")
