import os
import sys
import json
import subprocess
import random
import colorsys
import urllib.request
import urllib.error
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

import io
from PIL import Image
import re


sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

RENDERER_DIR = Path(__file__).parent.parent / "renderer"
RENDER_SCRIPT = RENDERER_DIR / "render.js"
FONTS = ["kantumruy", "moul", "battambang", "bayon", "notosans", "siemreap"]


def sanitize_khmer_text(text: str) -> str:
    """
    Clean up invalid Khmer script sequences to prevent rendering anomalies
    (like dotted circles, stacked duplicate vowels, or trailing Coeng signs).
    """
    # 1. Replace multiple consecutive Coeng signs with a single one
    text = re.sub(r'\u17d2+', '\u17d2', text)
    
    # 2. Prevent duplicate consecutive vowels (codepoints \u17b6 to \u17c5)
    text = re.sub(r'([\u17b6-\u17c5])\1+', r'\1', text)
    
    # 3. Prevent duplicate consecutive diacritics (codepoints \u17c6 to \u17d3 except Coeng)
    text = re.sub(r'([\u17c6-\u17d1\u17d3])\1+', r'\1', text)

    # 4. Remove floating Coeng signs (Coeng must be followed by a consonant \u1780-\u17a2)
    # If a Coeng is at the end of the text, or followed by space/punctuation, remove it.
    text = re.sub(r'\u17d2(?![ក-អ])', '', text)
    
    # 5. Remove floating vowels, diacritics, or Coeng signs at the start of the string or after spaces
    text = re.sub(r'(^|\s)[\u17b6-\u17d3]+', r'\1', text)
    
    # 6. Ensure we don't end on a Coeng or trailing subscript marker
    text = re.sub(r'[\ue000-\uf8ff]', '', text)

    # 7. Remove repeated whitespace introduced by cleanup
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])([\u1780-\u17ff])', r'\1 \2', text)
    text = re.sub(r'([\u1780-\u17ff])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)

    text = text.strip()
    while text.endswith('\u17d2'):
        text = text[:-1].strip()
        
    return text


def _char_script(char: str) -> str:
    if re.match(r"[\u1780-\u17ff]", char):
        return "khmer"
    if re.match(r"[A-Za-z]", char):
        return "latin"
    if re.match(r"[0-9\u17e0-\u17e9]", char):
        return "digit"
    return "symbol"


def _boundary_script(token: str, reverse: bool = False) -> str:
    chars = reversed(token) if reverse else token
    for char in chars:
        script = _char_script(char)
        if script != "symbol":
            return script
    return "symbol"


def _space_drop_probability(prev_token: str, token: str, khmer_drop_prob: float, mixed_drop_prob: float,
                            latin_drop_prob: float) -> float:
    no_space_before = set(".,;:!?)]}។៕៘៙៚ៗ%")
    no_space_after = set("([{«\"'")
    if token and token[0] in no_space_before:
        return 1.0
    if prev_token and prev_token[-1] in no_space_after:
        return 1.0

    prev_script = _boundary_script(prev_token, reverse=True)
    script = _boundary_script(token)
    if prev_script == "latin" and script in {"latin", "digit"}:
        return latin_drop_prob
    if prev_script == "digit" and script == "latin":
        return latin_drop_prob
    if prev_script == "khmer" and script == "khmer":
        return khmer_drop_prob
    if "khmer" in {prev_script, script} and ({prev_script, script} & {"latin", "digit"}):
        return mixed_drop_prob
    if prev_script == "digit" and script == "digit":
        return 0.2
    return 0.5


def apply_spacing_augmentation(text: str, space_drop_prob: float = 0.85, mixed_space_drop_prob: float = 0.35,
                               latin_space_drop_prob: float = 0.0) -> str:
    """
    Randomly remove spaces between segmented words using script-aware rules.

    Khmer word spaces are mostly removed. Latin/English word spaces are kept
    because OCR should learn English spacing differently from Khmer spacing.
    """
    words = [w for w in text.split() if w]
    if not words:
        return ""

    output = [words[0]]
    for word in words[1:]:
        drop_prob = _space_drop_probability(output[-1], word, space_drop_prob, mixed_space_drop_prob,
                                            latin_space_drop_prob)
        if random.random() < drop_prob:
            output[-1] += word
        else:
            output.append(word)
    return " ".join(output)


def generate_hard_text(length: int = 40) -> str:
    """
    Generate difficult text strings containing gibberish Khmer consonant clusters,
    subscripts, independent vowels, math equations, and alphanumeric symbols.
    Forces the OCR model to learn visual features rather than just language patterns.
    """
    consonants = list("កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ")
    subscripts = [chr(0x17d2) + c for c in list("កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហអ")]
    vowels = ["", "ា", "ិ", "ី", "ឹ", "ឺ", "ុ", "ូ", "ួ", "ើ", "ឿ", "ៀ", "េ", "ែ", "ៃ", "ោ", "ៅ", "ុំ", "ំ", "ាំ", "ះ", "ុះ", "េះ", "ោះ"]
    diacritics = ["", "់", "៌", "៏", "័", "៍", "ៈ", "ំ", "ះ"]
    independent_vowels = list("ឥឦឧឨឩឪឫឬឭឮឯអៃឱឲ")
    english_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = list("០១២៣៤៥៦៧៨៩0123456789")
    symbols = ["+", "-", "*", "/", "=", "%", "$", "(", ")", "[", "]", "{", "}", "?", "!", "@", "#", "&", ",", ".", ":", "។", "ៗ", "៕"]

    parts = []
    current_len = 0
    while current_len < length:
        mode = random.choice(["gibberish", "alpha_mix", "nested_sub", "rare_vowel", "math_symbol"])
        word = ""
        if mode == "gibberish":
            c1 = random.choice(consonants)
            sub = random.choice(subscripts) if random.random() < 0.6 else ""
            v = random.choice(vowels) if random.random() < 0.8 else ""
            d = random.choice(diacritics) if random.random() < 0.3 else ""
            word = c1 + sub + v + d
        elif mode == "alpha_mix":
            word = "".join(random.choices(english_chars, k=random.randint(3, 8)))
            if random.random() < 0.5:
                word += random.choice(symbols) + "".join(random.choices(digits, k=random.randint(1, 4)))
        elif mode == "nested_sub":
            c1 = random.choice(consonants)
            sub1 = random.choice(subscripts)
            sub2 = random.choice(subscripts) if random.random() < 0.4 else ""
            v = random.choice(vowels)
            word = c1 + sub1 + sub2 + v
        elif mode == "rare_vowel":
            word = random.choice(independent_vowels)
            if random.random() < 0.5:
                word += random.choice(consonants) + random.choice(vowels)
        elif mode == "math_symbol":
            word = random.choice(symbols) + " " + "".join(random.choices(digits, k=random.randint(1, 3))) + " " + random.choice(symbols)

        parts.append(word)
        current_len += len(word) + 1

    raw_text = " ".join(parts)[:length].strip()
    # 1% chance to keep the raw text with typos/dangling scripts for real-world robustness
    if random.random() < 0.01:
        return raw_text
    return sanitize_khmer_text(raw_text)


def get_random_contrast_pair(scheme: str = "both"):
    """
    Generate a random, high-contrast text and background color pair using HSL space.
    Guarantees that the text remains fully readable by enforcing lightness separation.
    Allows a full saturation range (10-100%) for extremely vibrant colors.
    """
    mode = scheme
    if mode == "both":
        mode = random.choice(["light", "dark"])

    if mode == "light":
        # Background: light color (vibrant pastel)
        bg_h = random.randint(0, 360)
        bg_s = random.randint(10, 100)
        bg_l = random.randint(80, 98)

        # Text: dark color (rich dark shades)
        tx_h = random.randint(0, 360)
        tx_s = random.randint(10, 100)
        tx_l = random.randint(0, 25)
    else:
        # Background: dark color (rich dark background)
        bg_h = random.randint(0, 360)
        bg_s = random.randint(10, 100)
        bg_l = random.randint(0, 20)

        # Text: light color (vibrant neon/light text)
        tx_h = random.randint(0, 360)
        tx_s = random.randint(10, 100)
        tx_l = random.randint(75, 100)

    def hsl_to_hex(h, s, l):
        r, g, b = colorsys.hls_to_rgb(h / 360.0, l / 100.0, s / 100.0)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    return hsl_to_hex(tx_h, tx_s, tx_l), hsl_to_hex(bg_h, bg_s, bg_l)


def render_khmer(text: str, font: str = "notosans", font_size: int = 48,
                 color: str = "#000", background: str = "#fff",
                 padding: int = 32, padding_top: int = None,
                 padding_right: int = None, padding_bottom: int = None,
                 padding_left: int = None, output_path: str = None,
                 resize_grayscale: bool = False) -> bytes:
    """
    Render Khmer text to an image by sending a request to the running Sone renderer API server.
    Extremely fast compared to spawning separate Node processes. Falls back to CLI if server is down.
    """
    url = "http://localhost:3456/render"
    data = {
        "text": text,
        "font": font,
        "fontSize": font_size,
        "color": color,
        "background": background,
        "padding": padding,
        "paddingTop": padding_top,
        "paddingRight": padding_right,
        "paddingBottom": padding_bottom,
        "paddingLeft": padding_left
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            img_bytes = response.read()

        if output_path:
            if resize_grayscale:
                try:
                    with Image.open(io.BytesIO(img_bytes)) as img:
                        img_resized = img.resize((256, 48), Image.Resampling.LANCZOS)
                        img_gray = img_resized.convert("L")
                        img_gray.save(output_path, format="PNG", optimize=True)
                    return None
                except Exception:
                    pass
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            return None
            
        if resize_grayscale:
            try:
                with Image.open(io.BytesIO(img_bytes)) as img:
                    img_resized = img.resize((256, 48), Image.Resampling.LANCZOS)
                    img_gray = img_resized.convert("L")
                    out_io = io.BytesIO()
                    img_gray.save(out_io, format="PNG", optimize=True)
                    return out_io.getvalue()
            except Exception:
                pass
        return img_bytes
    except Exception:
        # Fallback to spawning a CLI process if the API server is unavailable
        cmd = [
            "node", str(RENDER_SCRIPT),
            "--font", font,
            "--size", str(font_size),
            "--color", color,
            "--bg", background,
            "--padding", str(padding),
            "--format", "png",
            "--output", "/dev/stdout" if output_path is None else output_path,
            text,
        ]
        if padding_top is not None:
            cmd.extend(["--padding-top", str(padding_top)])
        if padding_right is not None:
            cmd.extend(["--padding-right", str(padding_right)])
        if padding_bottom is not None:
            cmd.extend(["--padding-bottom", str(padding_bottom)])
        if padding_left is not None:
            cmd.extend(["--padding-left", str(padding_left)])

        if output_path:
            subprocess.run(cmd, check=True, capture_output=True)
            if resize_grayscale:
                try:
                    with Image.open(output_path) as img:
                        img_resized = img.resize((256, 48), Image.Resampling.LANCZOS)
                        img_gray = img_resized.convert("L")
                        img_gray.save(output_path, format="PNG", optimize=True)
                except Exception:
                    pass
            return None
        else:
            result = subprocess.run(cmd, check=True, capture_output=True)
            img_bytes = result.stdout
            if resize_grayscale:
                try:
                    with Image.open(io.BytesIO(img_bytes)) as img:
                        img_resized = img.resize((256, 48), Image.Resampling.LANCZOS)
                        img_gray = img_resized.convert("L")
                        out_io = io.BytesIO()
                        img_gray.save(out_io, format="PNG", optimize=True)
                        return out_io.getvalue()
                except Exception:
                    pass
            return img_bytes



def render_worker(task):
    """Worker function for multiprocessing pool."""
    text, font, font_size, color, background, p_top, p_right, p_bottom, p_left, output_path, resize_grayscale = task
    try:
        render_khmer(text, font=font, font_size=font_size, color=color, background=background,
                     padding_top=p_top, padding_right=p_right, padding_bottom=p_bottom, padding_left=p_left,
                     output_path=output_path, resize_grayscale=resize_grayscale)
        return True
    except Exception as e:
        return str(e)



def generate_samples(texts: list, output_dir: str, fonts: list = None,
                      font_sizes: list = None, count: int = None,
                      random_colors: bool = False, color_scheme: str = "both",
                      hard_mix_ratio: float = 0.0, min_len: int = 20, max_len: int = 100,
                      num_workers: int = 4, space_drop_prob: float = 0.85,
                      mixed_space_drop_prob: float = 0.35, latin_space_drop_prob: float = 0.0,
                      resize_grayscale: bool = False):

    """Generate training images from a list of Khmer texts with multiprocessing support."""
    os.makedirs(output_dir, exist_ok=True)

    if fonts is None:
        fonts = FONTS
    if font_sizes is None:
        font_sizes = [24, 32, 40, 48, 56, 64, 72, 80]
    if count is None:
        count = len(texts)

    tasks = []
    for i in range(count):
        if hard_mix_ratio > 0 and random.random() < hard_mix_ratio:
            text = generate_hard_text(random.randint(min_len, max_len))
        else:
            text = apply_spacing_augmentation(
                texts[i % len(texts)].strip(),
                space_drop_prob,
                mixed_space_drop_prob,
                latin_space_drop_prob,
            )

        font = random.choice(fonts)
        font_size = random.choice(font_sizes)
        
        # Randomize top/bottom padding and left/right padding independently to simulate crop offsets
        p_top = random.randint(8, 24)
        p_bottom = random.randint(8, 24)
        p_left = random.randint(16, 48)
        p_right = random.randint(16, 48)

        color, background = "#000", "#fff"
        if random_colors:
            color, background = get_random_contrast_pair(color_scheme)

        output_path = os.path.join(output_dir, f"img_{i:06d}.png")
        tasks.append((text, font, font_size, color, background, p_top, p_right, p_bottom, p_left, output_path, resize_grayscale))


    print(f"Generating {len(tasks)} images using {num_workers} parallel workers...")
    
    generated = 0
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(render_worker, tasks), total=len(tasks), desc="Generating images"))
            for r in results:
                if r is True:
                    generated += 1
                else:
                    tqdm.write(f"Error rendering: {r}")
    else:
        # Fallback to sequential for debugging
        pbar = tqdm(tasks, desc="Generating images")
        for task in pbar:
            res = render_worker(task)
            if res is True:
                generated += 1
            else:
                tqdm.write(f"Error rendering: {res}")

    print(f"Generated {generated} images in {output_dir}")

    # Write metadata to DuckDB
    print("Writing metadata to DuckDB...")
    import pandas as pd
    import duckdb
    
    metadata_records = []
    for t in tasks:
        text, font, font_size, color, background, p_top, p_right, p_bottom, p_left, output_path = t
        filename = os.path.basename(output_path)
        
        # Expected English font fallback logic
        if font in ["kantumruy", "notosans"]:
            expected_english_font = font
        else:
            expected_english_font = "arial_or_timesnewroman"
            
        metadata_records.append({
            "filename": filename,
            "text": text,
            "font": font,
            "font_size": font_size,
            "text_color": color,
            "bg_color": background,
            "padding_top": p_top,
            "padding_right": p_right,
            "padding_bottom": p_bottom,
            "padding_left": p_left,
            "expected_english_font": expected_english_font
        })
        
    df_meta = pd.DataFrame(metadata_records)
    db_path = os.path.join(output_dir, "metadata.duckdb")
    
    if os.path.exists(db_path):
        os.remove(db_path)
        
    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE metadata AS SELECT * FROM df_meta")
    con.close()
    print(f"Saved metadata records to DuckDB at {db_path}")

    return generated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Khmer text training images")
    parser.add_argument("--input", "-i", help="Text file with one Khmer line per line")
    parser.add_argument("--output", "-o", default="generated", help="Output directory")
    parser.add_argument("--text", "-t", help="Single text to render")
    parser.add_argument("--font", default=None, help="Font name (default: random)")
    parser.add_argument("--size", type=int, default=48, help="Font size in pixels")
    parser.add_argument("--count", type=int, default=None, help="Number of images to generate")
    parser.add_argument("--list-fonts", action="store_true", help="List available fonts")
    parser.add_argument("--color", default="#000", help="Text color for single rendering or default (default: #000)")
    parser.add_argument("--bg", default="#fff", help="Background color for single rendering or default (default: #fff)")
    parser.add_argument("--random-colors", action="store_true", help="Randomize text and background colors with guaranteed readability contrast")
    parser.add_argument("--color-scheme", default="both", choices=["both", "light", "dark"], help="Color scheme mode when --random-colors is active (default: both)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible generation")
    parser.add_argument("--hard-mix-ratio", type=float, default=0.0, help="Ratio of hard text/gibberish injected into the dataset (0.0=none, 1.0=all)")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4, help="Number of CPU cores/workers for parallel image generation")
    parser.add_argument("--from-markov", action="store_true", help="Generate text using Markov model first, then render")
    parser.add_argument("--markov-samples", type=int, default=1000, help="Number of Markov samples to generate")
    parser.add_argument("--markov-order", type=int, default=2, help="Word n-gram order")
    parser.add_argument("--min-words", type=int, default=5, help="Min words for Markov")
    parser.add_argument("--max-words", type=int, default=30, help="Max words for Markov")
    parser.add_argument("--min-len", type=int, default=20, help="Minimum character length for direct Wikipedia/hard samples")
    parser.add_argument("--max-len", type=int, default=120, help="Maximum character length for direct Wikipedia/hard samples")
    parser.add_argument("--mix-ratio", type=float, default=0.5, help="Ratio of Markov text vs direct Wikipedia (0.0=wiki only, 1.0=markov only)")
    parser.add_argument("--space-drop-prob", type=float, default=0.85, help="Probability of removing Khmer-Khmer word-boundary spaces")
    parser.add_argument("--mixed-space-drop-prob", type=float, default=0.35, help="Probability of removing Khmer-Latin/digit boundary spaces")
    parser.add_argument("--latin-space-drop-prob", type=float, default=0.0, help="Probability of removing Latin-Latin or Latin-digit boundary spaces")
    parser.add_argument("--resize-grayscale", action="store_true", help="Resize images to (256, 48) and convert to 8-bit grayscale to optimize size")
    args = parser.parse_args()


    if args.seed is not None:
        random.seed(args.seed)

    if args.list_fonts:
        print("Available fonts:", ", ".join(FONTS))
        sys.exit(0)

    if args.from_markov:
        from generate_text import WordMarkovGenerator, augment_text, load_wikipedia_data

        print(f"Loading Wikipedia data...")
        wiki_lines = load_wikipedia_data("kmwiki_data")
        print(f"Training word Markov model (order={args.markov_order})...")
        gen = WordMarkovGenerator(order=args.markov_order)
        gen.train(wiki_lines)

        os.makedirs(args.output, exist_ok=True)

        num_hard = int(args.markov_samples * args.hard_mix_ratio)
        num_markov = int(args.markov_samples * args.mix_ratio * (1.0 - args.hard_mix_ratio))
        num_wiki = args.markov_samples - num_markov - num_hard

        texts = []

        # 1. Wikipedia lines (with 1% chance of keeping raw typos)
        wiki_candidates = [
            l.strip() if random.random() < 0.01 else sanitize_khmer_text(l.strip())
            for l in wiki_lines if args.min_len <= len(l.strip()) <= args.max_len
        ]
        if num_wiki > 0 and wiki_candidates:
            sampled_wiki = random.choices(wiki_candidates, k=num_wiki)
            texts.extend(sampled_wiki)
            print(f"Sampled {len(sampled_wiki)} lines from Wikipedia")

        # 2. Word Markov generated text (with 1% chance of keeping raw typos)
        markov_count = 0
        while markov_count < num_markov:
            num_words = random.randint(args.min_words, args.max_words)
            raw_text = augment_text(gen.generate(num_words))
            text = raw_text if random.random() < 0.01 else sanitize_khmer_text(raw_text)
            if len(text) >= 10:
                texts.append(text)
                markov_count += 1
        if num_markov > 0:
            print(f"Generated {num_markov} Markov samples")

        # 3. Hard gibberish text lines (generate_hard_text already has 1% bypass)
        for _ in range(num_hard):
            length = random.randint(args.min_len, args.max_len)
            texts.append(generate_hard_text(length))
        if num_hard > 0:
            print(f"Generated {num_hard} hard gibberish samples")

        # Clean, apply spacing augmentation (fully joined/partially joined/as-is), and shuffle
        texts = [
            apply_spacing_augmentation(
                t.strip(),
                args.space_drop_prob,
                args.mixed_space_drop_prob,
                args.latin_space_drop_prob,
            )
            for t in texts
            if t.strip()
        ]
        random.shuffle(texts)

        labels_file = os.path.join(args.output, "labels.txt")
        with open(labels_file, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t + "\n")

        # Build parallel tasks
        fonts = [args.font] if args.font else FONTS
        tasks = []
        for i, text in enumerate(texts):
            font = random.choice(fonts)
            font_size = random.choice([24, 32, 40, 48, 56, 64, 72, 80])
            
            # Randomize top/bottom padding and left/right padding independently to simulate crop offsets
            p_top = random.randint(8, 24)
            p_bottom = random.randint(8, 24)
            p_left = random.randint(16, 48)
            p_right = random.randint(16, 48)
            
            color, background = args.color, args.bg
            if args.random_colors:
                color, background = get_random_contrast_pair(args.color_scheme)
                
            output_path = os.path.join(args.output, f"img_{i:06d}.png")
            tasks.append((text, font, font_size, color, background, p_top, p_right, p_bottom, p_left, output_path, args.resize_grayscale))


        print(f"Generating {len(tasks)} images using {args.num_workers} parallel workers...")
        generated = 0
        
        if args.num_workers > 1:
            with Pool(processes=args.num_workers) as pool:
                results = list(tqdm(pool.imap_unordered(render_worker, tasks), total=len(tasks), desc="Generating images"))
                for r in results:
                    if r is True:
                        generated += 1
                    else:
                        tqdm.write(f"Error rendering: {r}")
        else:
            pbar = tqdm(tasks, desc="Generating images")
            for task in pbar:
                res = render_worker(task)
                if res is True:
                    generated += 1
                else:
                    tqdm.write(f"Error rendering: {res}")

        print(f"Generated {generated} images + labels in {args.output}")

    elif args.text:
        output_path = os.path.join(args.output, "output.png")
        os.makedirs(args.output, exist_ok=True)
        
        color, background = args.color, args.bg
        if args.random_colors:
            color, background = get_random_contrast_pair(args.color_scheme)
            
        render_khmer(args.text, font=args.font or "notosans",
                     font_size=args.size, color=color, background=background, output_path=output_path,
                     resize_grayscale=args.resize_grayscale)
        print(f"Saved {output_path}")

    elif args.input:
        with open(args.input, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        fonts = [args.font] if args.font else None
        generate_samples(texts, args.output, fonts=fonts, count=args.count,
                         random_colors=args.random_colors, color_scheme=args.color_scheme,
                          hard_mix_ratio=args.hard_mix_ratio, min_len=args.min_len, max_len=args.max_len,
                          num_workers=args.num_workers, space_drop_prob=args.space_drop_prob,
                          mixed_space_drop_prob=args.mixed_space_drop_prob,
                          latin_space_drop_prob=args.latin_space_drop_prob,
                          resize_grayscale=args.resize_grayscale)

    else:
        parser.print_help()
