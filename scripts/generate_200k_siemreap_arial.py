import argparse
import io
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import duckdb
import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from scripts.generate import apply_spacing_augmentation, sanitize_khmer_text
from scripts.generate_text import WordMarkovGenerator, load_wikipedia_data


RENDER_URL = "http://localhost:3456/render"
KHMER_RE = re.compile(r"[\u1780-\u17ff\u19e0-\u19ff]")
LATIN_RE = re.compile(r"[A-Za-z]")
ALLOWED_RE = re.compile(
    r"^[\u1780-\u17FF\u19E0-\u19FFa-zA-Z0-9\s"
    r".,;:!?()\[\]{}_+\-*/=%$@&#|\\~<>\"'"
    r"\u201c\u201d\u2018\u2019\u2013\u2014\u2026\u00ab\u00bb\u17db\u00b0\u200b"
    r"]+$"
)


ENGLISH_WORDS = [
    "Cambodia", "Siem", "Reap", "Angkor", "temple", "school", "market", "river", "street", "history",
    "culture", "language", "technology", "student", "teacher", "family", "travel", "museum", "city", "village",
    "health", "education", "computer", "phone", "internet", "research", "project", "training", "model", "dataset",
]


def classify_script(text: str) -> str:
    has_khmer = bool(KHMER_RE.search(text))
    has_latin = bool(LATIN_RE.search(text))
    if has_khmer and has_latin:
        return "mixed"
    if has_khmer:
        return "pure_khmer"
    if has_latin:
        return "pure_english"
    return "pure_khmer"


def clean_text(text: str) -> str:
    kept = []
    for ch in text:
        is_khmer = "\u1780" <= ch <= "\u17ff" or "\u19e0" <= ch <= "\u19ff"
        is_ascii = " " <= ch <= "~"
        is_symbol = ch in "\u200b“”‘’–—…«»៛°"
        if is_khmer or is_ascii or is_symbol:
            kept.append(ch)
    text = "".join(kept)
    text = re.sub(r"\s+", " ", text).strip()
    return sanitize_khmer_text(text)


def good_text(text: str, min_len: int, max_len: int) -> bool:
    if not (min_len <= len(text) <= max_len):
        return False
    if not ALLOWED_RE.match(text):
        return False
    return bool(KHMER_RE.search(text) or LATIN_RE.search(text))


def font_for_text(text: str) -> tuple[str, str]:
    script = classify_script(text)
    if script == "pure_english":
        return "arial", "arial"
    if script == "mixed":
        return "siemreap", "arial"
    return "siemreap", "none"


def choose_colors(vary_colors: bool, rng) -> tuple[str, str]:
    if not vary_colors:
        return "#000000", "#ffffff"
    clean_pairs = [
        ("#000000", "#ffffff"),
        ("#111111", "#ffffff"),
        ("#222222", "#f8f8f8"),
        ("#333333", "#faf7f0"),
        ("#1f2937", "#f9fafb"),
        ("#3f3f46", "#ffffff"),
        ("#0f172a", "#f8fafc"),
        ("#1c1917", "#fff7ed"),
    ]
    pastel_pairs = [
        ("#111827", "#dbeafe"),  # blue
        ("#111827", "#bfdbfe"),
        ("#111827", "#fee2e2"),  # red
        ("#111827", "#fecaca"),
        ("#111827", "#dcfce7"),  # green
        ("#111827", "#bbf7d0"),
        ("#111827", "#fef3c7"),  # yellow
        ("#111827", "#fde68a"),
        ("#111827", "#ede9fe"),  # purple
        ("#111827", "#ddd6fe"),
        ("#111827", "#cffafe"),  # cyan
    ]
    dark_pairs = [
        ("#ffffff", "#1e3a8a"),  # dark blue
        ("#f8fafc", "#172554"),
        ("#ffffff", "#7f1d1d"),  # dark red
        ("#f8fafc", "#450a0a"),
        ("#ffffff", "#14532d"),  # dark green
        ("#f8fafc", "#052e16"),
        ("#ffffff", "#581c87"),  # dark purple
        ("#f8fafc", "#312e81"),
        ("#ffffff", "#78350f"),  # dark amber
    ]
    r = rng.random()
    if r < 0.70:
        return rng.choice(clean_pairs)
    if r < 0.90:
        return rng.choice(pastel_pairs)
    return rng.choice(dark_pairs)


def choose_padding(rng) -> tuple[int, int, int, int]:
    r = rng.random()
    if r < 0.15:
        # Tight crops, including true edge-touching text for real-world OCR crops.
        return (
            rng.randint(0, 4),
            rng.randint(0, 8),
            rng.randint(0, 4),
            rng.randint(0, 8),
        )
    if r < 0.35:
        return (
            rng.randint(2, 10),
            rng.randint(4, 18),
            rng.randint(2, 10),
            rng.randint(4, 18),
        )
    return (
        rng.randint(8, 22),
        rng.randint(14, 44),
        rng.randint(8, 22),
        rng.randint(14, 44),
    )



def generate_english_sentence() -> str:
    words = random.choices(ENGLISH_WORDS, k=random.randint(5, 14))
    if random.random() < 0.35:
        words.insert(random.randrange(1, len(words)), str(random.randint(1, 2030)))
    text = " ".join(words)
    if random.random() < 0.35:
        text += random.choice([".", ",", "?", "!"])
    return text


def inject_english(text: str) -> str:
    words = text.split()
    if len(words) < 3:
        return text
    for _ in range(random.randint(1, 2)):
        idx = random.randint(1, len(words) - 1)
        token = random.choice(ENGLISH_WORDS) if random.random() < 0.7 else str(random.randint(1950, 2030))
        words.insert(idx, token)
    return " ".join(words)


def build_texts(count: int, min_len: int, max_len: int, wiki_dir: str, seed: int) -> list[dict]:
    random.seed(seed)

    hanuman = load_dataset("seanghay/khmer-hanuman-100k", split="train")
    hanuman_lines = [clean_text(x) for x in hanuman["text"]]
    hanuman_lines = [x for x in hanuman_lines if good_text(x, min_len, max_len)]

    wiki_lines = [clean_text(x) for x in load_wikipedia_data(wiki_dir)]
    wiki_lines = [x for x in wiki_lines if good_text(x, min_len, max_len)]
    wiki_khmer = [x for x in wiki_lines if classify_script(x) == "pure_khmer"]
    wiki_mixed = [x for x in wiki_lines if classify_script(x) == "mixed"]

    markov = WordMarkovGenerator(order=2)
    markov.train(wiki_lines + hanuman_lines)

    hanuman_target = count // 2
    contextual_target = count - hanuman_target
    targets = {
        "hanuman": hanuman_target,
        "wiki": contextual_target // 2,
        "markov": contextual_target - (contextual_target // 2),
        "english": 0,
    }

    rows = []
    hanuman_sample = random.sample(hanuman_lines, min(targets["hanuman"], len(hanuman_lines)))
    for text in tqdm(hanuman_sample, desc="Selecting Hanuman texts"):
        rows.append({"source": "hanuman", "text": text})

    wiki_pool = wiki_khmer + wiki_mixed
    wiki_sample = random.sample(wiki_pool, min(targets["wiki"], len(wiki_pool)))
    for text in tqdm(wiki_sample, desc="Selecting wiki texts"):
        rows.append({"source": "wiki", "text": text})

    attempts = 0
    markov_count = 0
    with tqdm(total=targets["markov"], desc="Generating Markov texts") as pbar:
        while markov_count < targets["markov"] and attempts < targets["markov"] * 50:
            attempts += 1
            text = markov.generate(random.randint(5, 24))
            if random.random() < 0.18:
                text = inject_english(text)
                source = "markov_mixed"
            else:
                source = "markov_khmer"
            text = clean_text(text)
            if good_text(text, min_len, max_len):
                rows.append({"source": source, "text": text})
                markov_count += 1
                pbar.update(1)

    hanuman_count = sum(1 for r in rows if r["source"].startswith("hanuman"))

    with tqdm(total=count, initial=len(rows), desc="Filling missing rows") as pbar:
        while len(rows) < count:
            if hanuman_count < hanuman_target:
                text = random.choice(hanuman_lines)
                rows.append({"source": "hanuman_fill", "text": text})
                hanuman_count += 1
                pbar.update(1)
            else:
                text = markov.generate(random.randint(5, 24))
                if random.random() < 0.18:
                    text = inject_english(text)
                    source = "markov_mixed_fill"
                else:
                    source = "markov_khmer_fill"
                text = clean_text(text)
                if good_text(text, min_len, max_len):
                    rows.append({"source": source, "text": text})
                    pbar.update(1)


    rows = rows[:count]
    random.shuffle(rows)

    for row in tqdm(rows, desc="Applying spacing cleanup"):
        script = classify_script(row["text"])
        if script == "pure_khmer":
            row["text"] = apply_spacing_augmentation(row["text"], space_drop_prob=0.75, mixed_space_drop_prob=0.35, latin_space_drop_prob=0.0)
        elif script == "mixed":
            row["text"] = apply_spacing_augmentation(row["text"], space_drop_prob=0.55, mixed_space_drop_prob=0.20, latin_space_drop_prob=0.0)
        row["text"] = clean_text(row["text"])
        row["script_class"] = classify_script(row["text"])
    return rows


def render_one(task):
    idx, row, output_dir, resize_grayscale, render_url, vary_colors, seed = task
    
    import random as local_random
    rng = local_random.Random(seed + idx)

    text = row["text"]
    font, expected_english_font = font_for_text(text)
    font_size = rng.choice([28, 32, 36, 40, 44, 48, 52, 56])
    p_top, p_right, p_bottom, p_left = choose_padding(rng)
    text_color, bg_color = choose_colors(vary_colors, rng)
    filename = f"img_{idx:06d}.png"
    output_path = os.path.join(output_dir, filename)

    # Check if the image has already been successfully rendered on a previous attempt
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return {
            "filename": filename,
            "text": text,
            "source": row["source"],
            "script_class": row["script_class"],
            "font": font,
            "font_size": font_size,
            "text_color": text_color,
            "bg_color": bg_color,
            "padding_top": p_top,
            "padding_right": p_right,
            "padding_bottom": p_bottom,
            "padding_left": p_left,
            "expected_english_font": expected_english_font,
            "text_len": len(text),
        }

    payload = {
        "text": text,
        "font": font,
        "fontSize": font_size,
        "color": text_color,
        "background": bg_color,
        "paddingTop": p_top,
        "paddingRight": p_right,
        "paddingBottom": p_bottom,
        "paddingLeft": p_left,
    }
    req = urllib.request.Request(render_url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        img_bytes = response.read()
    if resize_grayscale:
        with Image.open(io.BytesIO(img_bytes)) as img:
            img = img.convert("L")
            img.save(output_path, format="PNG", optimize=True)
    else:
        with open(output_path, "wb") as f:
            f.write(img_bytes)
    return {
        "filename": filename,
        "text": text,
        "source": row["source"],
        "script_class": row["script_class"],
        "font": font,
        "font_size": font_size,
        "text_color": text_color,
        "bg_color": bg_color,
        "padding_top": p_top,
        "padding_right": p_right,
        "padding_bottom": p_bottom,
        "padding_left": p_left,
        "expected_english_font": expected_english_font,
        "text_len": len(text),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate 200k Siemreap/Arial OCR images from Hanuman + wiki + Markov text")
    parser.add_argument("--count", type=int, default=200_000)
    parser.add_argument("--output", default="generated/training_200k_siemreap_arial")
    parser.add_argument("--wiki-dir", default="kmwiki_data")
    parser.add_argument("--min-len", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize-grayscale", action="store_true")
    parser.add_argument("--texts-only", action="store_true")
    parser.add_argument("--render-url", default=RENDER_URL)
    parser.add_argument("--vary-colors", action="store_true", help="Use realistic dark text on light background variations")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building {args.count} text rows...")
    rows = build_texts(args.count, args.min_len, args.max_len, args.wiki_dir, args.seed)
    counts = Counter(row["script_class"] for row in rows)
    print("Script distribution:", dict(counts))

    texts_path = output_dir / "labels.csv"
    pd.DataFrame(rows).to_csv(texts_path, index=True, index_label="row_id", encoding="utf-8-sig")
    print(f"Saved labels to {texts_path}")

    if args.texts_only:
        return

    print(f"Rendering {len(rows)} images with {args.num_workers} workers...")
    tasks = [(i, row, str(output_dir), args.resize_grayscale, args.render_url, args.vary_colors, args.seed) for i, row in enumerate(rows)]

    metadata = []
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            for item in tqdm(pool.imap_unordered(render_one, tasks), total=len(tasks)):
                metadata.append(item)
    else:
        for task in tqdm(tasks):
            metadata.append(render_one(task))

    metadata.sort(key=lambda x: x["filename"])
    df = pd.DataFrame(metadata)
    db_path = output_dir / "metadata.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE metadata AS SELECT * FROM df")
    con.close()
    print(f"Saved metadata to {db_path}")


if __name__ == "__main__":
    main()
