"""
Generate high-level architecture diagrams for both CRNN variants
(ResNet34 + BiGRU and ResNet18 + BiGRU) using VisualTorch.

Each major processing stage is wrapped as a named nn.Module so VisualTorch
renders at the correct abstraction level — not every atomic Conv2d/BN/ReLU.

Usage:
    uv run python scripts/plot_architecture.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn as nn
from torchvision import models as tv
import visualtorch

DPI = 150

# ── High-level wrapper modules ─────────────────────────────────────────────

class ModifiedResNet(nn.Module):
    """ResNet backbone with asymmetric stride — shown as one block in the diagram."""
    def __init__(self, backbone_fn):
        super().__init__()
        rn = backbone_fn(weights=None)
        rn.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1, rn.layer2)
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4
        for b in [self.layer3, self.layer4]:
            for block in b:
                if hasattr(block, "conv1") and block.conv1.stride != (1, 1):
                    block.conv1.stride = (2, 1)
                if block.downsample is not None:
                    block.downsample[0].stride = (2, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (B, 512, H', W')

class VerticalMeanPooling(nn.Module):
    """Collapse height dimension via mean pooling."""
    def forward(self, x):
        return x.mean(dim=2)  # (B, 512, W')

class SequenceFormatting(nn.Module):
    """Permute to (T, B, C) for RNN consumption."""
    def forward(self, x):
        return x.permute(2, 0, 1)  # (T, B, 512)

class HighLevelCRNN(nn.Module):
    """CRNN with high-level named blocks for clean VisualTorch rendering."""
    def __init__(self, backbone_fn, vocab_size=100, hidden=256):
        super().__init__()
        self.modified_resnet = ModifiedResNet(backbone_fn)
        self.vertical_mean_pooling = VerticalMeanPooling()
        self.sequence_formatting = SequenceFormatting()
        self.bigru = nn.GRU(512, hidden, num_layers=2, bidirectional=True, dropout=0.2)
        self.classifier = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        x = self.modified_resnet(x)
        x = self.vertical_mean_pooling(x)
        x = self.sequence_formatting(x)
        x, _ = self.bigru(x)
        return self.classifier(x)


# ── Generate architecture visualisations ───────────────────────────────────

def render_arch(backbone_fn, name, input_shape=(1, 1, 64, 320)):
    """Render graph and flow style for a CRNN variant at high level."""
    model = HighLevelCRNN(backbone_fn)

    color_map = defaultdict(dict)
    # High-level blocks
    color_map[ModifiedResNet]["fill"]        = "#E69F00"   # orange
    color_map[VerticalMeanPooling]["fill"]   = "#009E73"   # green
    color_map[SequenceFormatting]["fill"]    = "#56B4E9"   # blue
    color_map[nn.GRU]["fill"]                = "#0072B2"   # dark blue
    color_map[nn.Linear]["fill"]             = "#D55E00"   # red-orange

    # --- Graph style (box-based, clean for papers) ---
    img_graph = visualtorch.render(
        model,
        input_shape,
        style="graph",
        show_neurons=False,
        color_map=color_map,
        layer_spacing=100,
        node_size=70,
        show_dimension=True,
    )
    fig, ax = plt.subplots(figsize=(img_graph.width / DPI, img_graph.height / DPI), dpi=DPI)
    ax.imshow(img_graph)
    ax.axis("off")
    fig.savefig(f"docs/figures/arch_{name}_graph.pdf", dpi=DPI, bbox_inches="tight", pad_inches=0)
    fig.savefig(f"docs/figures/arch_{name}_graph.png", dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  \u2713 arch_{name}_graph.pdf / .png")

    # --- Flow style (volumetric, more compact) ---
    img_flow = visualtorch.render(
        model,
        input_shape,
        style="flow",
        color_map=color_map,
        spacing=40,
        scale_xy=2,
        show_dimension=True,
        legend=True,
    )
    fig, ax = plt.subplots(figsize=(img_flow.width / DPI, img_flow.height / DPI), dpi=DPI)
    ax.imshow(img_flow)
    ax.axis("off")
    fig.savefig(f"docs/figures/arch_{name}_flow.pdf", dpi=DPI, bbox_inches="tight", pad_inches=0)
    fig.savefig(f"docs/figures/arch_{name}_flow.png", dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  \u2713 arch_{name}_flow.pdf / .png")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    os.makedirs("docs/figures", exist_ok=True)
    input_shape = (1, 1, 64, 320)

    print("Rendering ResNet34 variant...")
    m34 = render_arch(tv.resnet34, "resnet34", input_shape)

    print("\nRendering ResNet18 variant...")
    m18 = render_arch(tv.resnet18, "resnet18", input_shape)

    print("\nDone — figures saved to docs/figures/")
