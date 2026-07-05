import argparse
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from generate import apply_spacing_augmentation, sanitize_khmer_text
from generate_text import WordMarkovGenerator, augment_text, load_wikipedia_data


def build_samples(
    data_dir: str,
    count: int,
    mix_ratio: float,
    order: int,
    min_words: int,
    max_words: int,
    min_len: int,
    max_len: int,
    augment: bool,
    spacing_augment: bool,
    space_drop_prob: float,
    mixed_space_drop_prob: float,
    latin_space_drop_prob: float,
):
    wiki_lines = load_wikipedia_data(data_dir)
    wiki_candidates = [
        sanitize_khmer_text(line.strip())
        for line in wiki_lines
        if min_len <= len(line.strip()) <= max_len
    ]
    wiki_candidates = [line for line in wiki_candidates if line]

    num_markov = int(count * mix_ratio)
    num_wiki = count - num_markov

    rows = []

    for text in random.choices(wiki_candidates, k=num_wiki):
        if spacing_augment:
            text = apply_spacing_augmentation(
                text,
                space_drop_prob,
                mixed_space_drop_prob,
                latin_space_drop_prob,
            )
        rows.append({"source": "wiki", "text": text})

    gen = WordMarkovGenerator(order=order)
    gen.train(wiki_lines)

    while len(rows) < count:
        num_words = random.randint(min_words, max_words)
        text = gen.generate(num_words).strip()
        if augment:
            text = augment_text(text)
        text = sanitize_khmer_text(text)
        if spacing_augment:
            text = apply_spacing_augmentation(
                text,
                space_drop_prob,
                mixed_space_drop_prob,
                latin_space_drop_prob,
            )
        if len(text) >= min_len:
            rows.append({"source": "word_markov", "text": text})

    random.shuffle(rows)
    return rows[:count]


def main():
    parser = argparse.ArgumentParser(description="Generate review CSV of Khmer OCR training text")
    parser.add_argument("--data-dir", default="kmwiki_data")
    parser.add_argument("--output", default="generated_texts/preview_word_markov.csv")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--mix-ratio", type=float, default=0.5, help="0.5 = 50% word-Markov, 50% direct wiki")
    parser.add_argument("--order", type=int, default=2, help="Word n-gram order")
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=30)
    parser.add_argument("--min-len", type=int, default=20)
    parser.add_argument("--max-len", type=int, default=120)
    parser.add_argument("--no-augment", action="store_true", help="Disable symbol/number insertion")
    parser.add_argument("--no-spacing-augment", action="store_true", help="Keep original spaces")
    parser.add_argument("--space-drop-prob", type=float, default=0.85, help="Probability of removing Khmer-Khmer word-boundary spaces")
    parser.add_argument("--mixed-space-drop-prob", type=float, default=0.35, help="Probability of removing Khmer-Latin/digit boundary spaces")
    parser.add_argument("--latin-space-drop-prob", type=float, default=0.0, help="Probability of removing Latin-Latin or Latin-digit boundary spaces")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    rows = build_samples(
        data_dir=args.data_dir,
        count=args.count,
        mix_ratio=args.mix_ratio,
        order=args.order,
        min_words=args.min_words,
        max_words=args.max_words,
        min_len=args.min_len,
        max_len=args.max_len,
        augment=not args.no_augment,
        spacing_augment=not args.no_spacing_augment,
        space_drop_prob=args.space_drop_prob,
        mixed_space_drop_prob=args.mixed_space_drop_prob,
        latin_space_drop_prob=args.latin_space_drop_prob,
    )

    df = pd.DataFrame({"id": i, **row} for i, row in enumerate(rows))
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
