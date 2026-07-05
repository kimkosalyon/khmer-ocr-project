from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, x, y, w, h, title, subtitle="", color="#eef4ff"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.8,
        edgecolor="#1f3b63",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=13, weight="bold", color="#10233f")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=10.5, color="#42526e")


def add_arrow(ax, x, y1, y2):
    arrow = FancyArrowPatch(
        (x, y1),
        (x, y2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="#1f3b63",
    )
    ax.add_patch(arrow)


def draw_architecture(model_name, backbone_name, feature_shape, cer_text, png_name, svg_name):
    output_dir = Path("docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")

    ax.text(
        5,
        15.35,
        model_name,
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
        color="#10233f",
    )
    ax.text(
        5,
        14.9,
        "Khmer OCR line-recognition architecture",
        ha="center",
        va="center",
        fontsize=12,
        color="#42526e",
    )

    x, w, h = 1.4, 7.2, 1.05
    ys = [13.5, 11.7, 9.9, 8.1, 6.3, 4.5, 2.7, 0.9]

    blocks = [
        ("Input Image", "(B, 1, H, W)", "#f7fbff"),
        (backbone_name, "Preserves width with (2,1) strides", "#e8f0ff"),
        ("Feature Map", feature_shape, "#f7fbff"),
        ("Vertical Mean Pooling", "Collapses H' dimension", "#edf7ed"),
        ("Sequence Formatting", "Permutes to (T, B, 512)", "#fff7e6"),
        ("2-Layer BiGRU", "Captures bidirectional Khmer context", "#f3eefe"),
        ("Linear Classifier", "Maps to vocabulary size: 193 logits", "#eaf7f7"),
        ("CTC Decoding / Loss", "Collapses blanks/repeats to final Khmer text", "#fff0f0"),
    ]

    for y, (title, subtitle, color) in zip(ys, blocks):
        add_box(ax, x, y, w, h, title, subtitle, color)

    center_x = x + w / 2
    for i in range(len(ys) - 1):
        add_arrow(ax, center_x, ys[i], ys[i + 1] + h)

    ax.text(
        5,
        0.18,
        cer_text,
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        color="#1f3b63",
    )

    png_path = output_dir / png_name
    svg_path = output_dir / svg_name
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def main():
    draw_architecture(
        model_name="Model 1: ResNet34 + BiGRU + CTC",
        backbone_name="Modified ResNet34",
        feature_shape="(B, 512, H', W')",
        cer_text="Final test CER: approximately 0.18%",
        png_name="resnet34_bigru_ctc_architecture.png",
        svg_name="resnet34_bigru_ctc_architecture.svg",
    )
    draw_architecture(
        model_name="Model 2: ResNet18 + BiGRU + CTC",
        backbone_name="Modified ResNet18",
        feature_shape="(B, 512, H', W')",
        cer_text="Epoch 5 test CER: 1.42%",
        png_name="resnet18_bigru_ctc_architecture.png",
        svg_name="resnet18_bigru_ctc_architecture.svg",
    )


if __name__ == "__main__":
    main()
