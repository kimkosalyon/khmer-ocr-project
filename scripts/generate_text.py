import os
import sys
import random
import json
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))


class WordMarkovGenerator:
    """Word-level Markov chain for Khmer text generation.
    Operates on segmented (space-delimited) words — guarantees
    every token is a real Khmer word, no broken coengs/vowels."""

    def __init__(self, order=2):
        self.order = order
        self.chain = defaultdict(list)
        self.starts = []

    def train(self, texts: list[str]):
        for text in texts:
            words = text.split()
            if len(words) <= self.order:
                continue
            self.starts.append(tuple(words[:self.order]))
            for i in range(len(words) - self.order):
                gram = tuple(words[i:i + self.order])
                next_word = words[i + self.order]
                self.chain[gram].append(next_word)

    def generate(self, num_words: int = 20) -> str:
        if not self.starts:
            return ""
        current = list(random.choice(self.starts))
        result = current.copy()
        for _ in range(num_words - self.order):
            gram = tuple(current[-self.order:])
            candidates = self.chain.get(gram, [])
            if not candidates:
                current = list(random.choice(self.starts))
                result.extend(current)
                continue
            next_word = random.choice(candidates)
            result.append(next_word)
            current.pop(0)
            current.append(next_word)
        return " ".join(result)


KHMER_SYMBOLS = [
    "។", "ៗ", "៕", "៚", "៙", "៖",
    "(", ")", "[", "]", "{", "}",
    "«", "»", "'", "'", '"', '"',
    "…", "–", "—", "/", "\\", "|",
]

KHMER_NUMBERS = "០១២៣៤៥៦៧៨៩0123456789"

NOISE_CHARS = [" ", " ", "  "]


def augment_text(text: str, p_space=0.01, p_symbol=0.005, p_number=0.003, p_noise=0.002) -> str:
    """Insert spaces/symbols/numbers only between tokens, not inside Khmer clusters."""
    words = text.split()
    if not words:
        return text

    result = []
    for word in words:
        result.append(word)
        r = random.random()
        if r < p_space:
            result.append(random.choice(NOISE_CHARS))
        elif r < p_space + p_symbol:
            result.append(random.choice(KHMER_SYMBOLS))
        elif r < p_space + p_symbol + p_number:
            result.append(random.choice(KHMER_NUMBERS))
        elif r < p_space + p_symbol + p_number + p_noise:
            result.append(random.choice(NOISE_CHARS))
    return " ".join(result)


def load_wikipedia_data(data_dir: str = "kmwiki_data") -> list[str]:
    """Load all text lines from kmwiki_data directory."""
    lines = []
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} not found", file=sys.stderr)
        return lines

    for fname in os.listdir(data_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) >= 10:
                    lines.append(line)
    return lines


def generate_dataset(
    data_dir: str = "kmwiki_data",
    output_dir: str = "generated_texts",
    num_samples: int = 5000,
    min_words: int = 5,
    max_words: int = 30,
    order: int = 2,
    augment: bool = True,
):
    """Generate a synthetic Khmer text dataset using word-level Markov chains."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {data_dir}...")
    lines = load_wikipedia_data(data_dir)
    print(f"Loaded {len(lines)} lines")

    if not lines:
        print("No data found!")
        return

    print(f"Training word Markov model (order={order})...")
    gen = WordMarkovGenerator(order=order)
    gen.train(lines)

    output_file = os.path.join(output_dir, "samples.txt")
    meta_file = os.path.join(output_dir, "meta.json")

    samples = []
    print(f"Generating {num_samples} samples...")
    for i in range(num_samples):
        num_words = random.randint(min_words, max_words)
        text = gen.generate(num_words)

        if augment:
            text = augment_text(text, p_space=0.01, p_symbol=0.005, p_number=0.003, p_noise=0.002)

        text = text.strip()
        if len(text) >= 10:
            samples.append(text)

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample + "\n")

    meta = {
        "source": data_dir,
        "order": order,
        "model": "word_markov",
        "num_samples": len(samples),
        "min_words": min_words,
        "max_words": max_words,
        "augment": augment,
        "training_lines": len(lines),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(samples)} samples to {output_file}")
    print(f"Saved metadata to {meta_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic Khmer text using word-level Markov chains")
    parser.add_argument("--data-dir", default="kmwiki_data", help="Directory with Khmer text files")
    parser.add_argument("--output-dir", default="generated_texts", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--min-words", type=int, default=5, help="Minimum number of words")
    parser.add_argument("--max-words", type=int, default=30, help="Maximum number of words")
    parser.add_argument("--order", type=int, default=2, help="Word n-gram order (1-4)")
    parser.add_argument("--no-augment", action="store_true", help="Disable random augmentation")
    parser.add_argument("--preview", type=int, default=0, help="Preview N samples instead of generating")
    args = parser.parse_args()

    if args.preview > 0:
        lines = load_wikipedia_data(args.data_dir)
        gen = WordMarkovGenerator(order=args.order)
        gen.train(lines)
        for i in range(args.preview):
            num_words = random.randint(args.min_words, args.max_words)
            text = gen.generate(num_words)
            text = augment_text(text)
            print(f"[{i+1}] {text[:120]}...")
    else:
        generate_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            min_words=args.min_words,
            max_words=args.max_words,
            order=args.order,
            augment=not args.no_augment,
        )
