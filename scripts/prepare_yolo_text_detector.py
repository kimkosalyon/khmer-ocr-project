import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def foreground_box(path: Path, margin: int) -> tuple[float, float, float, float] | None:
    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.array(gray)
        if arr.mean() < 127:
            mask = arr > 40
        else:
            mask = arr < 245
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x1 = max(0, int(xs.min()) - margin)
        y1 = max(0, int(ys.min()) - margin)
        x2 = min(img.width, int(xs.max()) + margin + 1)
        y2 = min(img.height, int(ys.max()) + margin + 1)
        cx = ((x1 + x2) / 2) / img.width
        cy = ((y1 + y2) / 2) / img.height
        w = (x2 - x1) / img.width
        h = (y2 - y1) / img.height
        return cx, cy, w, h


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO text detector labels from generated OCR images")
    parser.add_argument("--input", required=True, help="Directory containing img_*.png files")
    parser.add_argument("--output", default="generated/text_det_yolo")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    images = sorted(src.glob("img_*.png"))
    if not images:
        raise SystemExit(f"No img_*.png files found in {src}")

    random.seed(args.seed)
    random.shuffle(images)
    val_count = max(1, int(len(images) * args.val_ratio))
    splits = {"val": images[:val_count], "train": images[val_count:]}

    for split, paths in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for path in tqdm(paths, desc=f"Preparing {split}"):
            box = foreground_box(path, args.margin)
            if box is None:
                continue
            dst_img = out / "images" / split / path.name
            dst_label = out / "labels" / split / f"{path.stem}.txt"
            shutil.copy2(path, dst_img)
            dst_label.write_text("0 " + " ".join(f"{v:.6f}" for v in box) + "\n", encoding="utf-8")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: text\n",
        encoding="utf-8",
    )
    print(f"Wrote YOLO dataset to {out}")
    print(f"Train images: {len(splits['train'])}")
    print(f"Val images: {len(splits['val'])}")
    print(f"Data YAML: {data_yaml}")


if __name__ == "__main__":
    main()
