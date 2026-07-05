"""
Generate a detailed architecture diagram of the 2-layer BiGRU module
using VisualTorch's flow (3D volumetric) style.

Usage:
    uv run python scripts/plot_bigru_detail.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn as nn
import visualtorch

DPI = 150

# ── Standalone BiGRU module ────────────────────────────────────────────────

class DetailBiGRU(nn.Module):
    """A 2-layer bidirectional GRU shown as an independent module for diagram."""
    def __init__(self, input_size=512, hidden=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, input_size)
        self.bigru_forward_1 = nn.GRUCell(input_size, hidden)
        self.bigru_backward_1 = nn.GRUCell(input_size, hidden)
        self.bigru_forward_2 = nn.GRUCell(hidden, hidden)
        self.bigru_backward_2 = nn.GRUCell(hidden, hidden)
        self.output_proj = nn.Linear(hidden * 2, hidden * 2)

    def forward(self, x):
        # x shape: (T, B, input_size)
        B = x.size(1)
        T = x.size(0)

        x = self.input_proj(x)

        # Layer 1 forward
        h_f1 = torch.zeros(B, 256, device=x.device)
        h_b1 = torch.zeros(B, 256, device=x.device)

        h_fwd1 = []
        for t in range(T):
            h_f1 = self.bigru_forward_1(x[t], h_f1)
            h_fwd1.append(h_f1)
        h_rev1 = []
        for t in range(T - 1, -1, -1):
            h_b1 = self.bigru_backward_1(x[t], h_b1)
            h_rev1.insert(0, h_b1)

        # Concatenate bidirectional
        h1 = [torch.cat([f, b], dim=-1) for f, b in zip(h_fwd1, h_rev1)]
        h1 = torch.stack(h1)

        # Layer 2 forward
        h_f2 = torch.zeros(B, 256, device=x.device)
        h_b2 = torch.zeros(B, 256, device=x.device)

        h_fwd2 = []
        for t in range(T):
            h_f2 = self.bigru_forward_2(h1[t], h_f2)
            h_fwd2.append(h_f2)
        h_rev2 = []
        for t in range(T - 1, -1, -1):
            h_b2 = self.bigru_backward_2(h1[t], h_b2)
            h_rev2.insert(0, h_b2)

        h2 = [torch.cat([f, b], dim=-1) for f, b in zip(h_fwd2, h_rev2)]
        h2 = torch.stack(h2)

        return self.output_proj(h2)


class WrappedBiGRU(nn.Module):
    """Wraps a single GRU cell into a named block for VisualTorch."""
    def __init__(self, label, cell):
        super().__init__()
        self.label = label
        self.cell = cell

    def forward(self, x):
        return self.cell(x)


# ── Simpler approach: use raw nn.GRU with named wrappers ───────────────────

class BiGRUDetail(nn.Module):
    """2-layer BiGRU with explicitly named stages for clean diagram."""
    def __init__(self, input_size=512, hidden=256, dropout=0.2):
        super().__init__()
        self.input_encoding = nn.Linear(input_size, input_size)
        self.bigru_layer1 = nn.GRU(input_size, hidden, num_layers=1,
                                    bidirectional=True, batch_first=False)
        self.bigru_layer2 = nn.GRU(hidden * 2, hidden, num_layers=1,
                                    bidirectional=True, batch_first=False)
        self.output_projection = nn.Linear(hidden * 2, hidden * 2)

    def forward(self, x):
        x = self.input_encoding(x)
        x, _ = self.bigru_layer1(x)
        x, _ = self.bigru_layer2(x)
        return self.output_projection(x)


if __name__ == "__main__":
    import os
    os.makedirs("docs/figures", exist_ok=True)

    model = BiGRUDetail()
    # The model expects (T, B, 512) but VisualTorch traces with batch dim first
    # We use batch_first=False so input is (T, B, 512) — but VisualTorch
    # typically expects batch dim first. Let's use (1, 10, 512) as (B, T, C)
    # and tell VisualTorch the correct shape.

    # VisualTorch traces with (B, ...) so we use (1, 10, 512) = (B, T, input_size)
    input_shape = (1, 10, 512)

    color_map = defaultdict(dict)
    color_map[nn.Linear]["fill"] = "#D55E00"        # red-orange
    color_map[nn.GRU]["fill"] = "#0072B2"           # dark blue

    # VisualTorch colors by layer TYPE. All Linear layers get one color,
    # all GRU layers another. The legend explains.

    print("Rendering BiGRU detail diagram...")

    # --- Flow style (3D volumetric, best for showing BiGRU internals) ---
    img = visualtorch.render(
        model,
        input_shape,
        style="flow",
        color_map=color_map,
        spacing=30,
        scale_xy=3,
        show_dimension=True,
        legend=True,
        show_input=True,
    )

    fig, ax = plt.subplots(figsize=(img.width / DPI, img.height / DPI), dpi=DPI)
    ax.imshow(img)
    ax.axis("off")
    fig.savefig("docs/figures/bigru_detail_flow.pdf", dpi=DPI, bbox_inches="tight", pad_inches=0)
    fig.savefig("docs/figures/bigru_detail_flow.png", dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  \u2713 bigru_detail_flow.pdf / .png")

    # --- Graph style (box-based, shows each named stage) ---
    img2 = visualtorch.render(
        model,
        input_shape,
        style="graph",
        show_neurons=False,
        color_map=color_map,
        layer_spacing=120,
        node_size=70,
        show_dimension=True,
    )

    fig, ax = plt.subplots(figsize=(img2.width / DPI, img2.height / DPI), dpi=DPI)
    ax.imshow(img2)
    ax.axis("off")
    fig.savefig("docs/figures/bigru_detail_graph.pdf", dpi=DPI, bbox_inches="tight", pad_inches=0)
    fig.savefig("docs/figures/bigru_detail_graph.png", dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  \u2713 bigru_detail_graph.pdf / .png")

    # Print param count
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  \u2192 BiGRU detail params: {total:,}")
    print("\nDone — figures saved to docs/figures/")
