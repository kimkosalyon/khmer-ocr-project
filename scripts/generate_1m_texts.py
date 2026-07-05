import os
import sys
import random
import re
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from generate import apply_spacing_augmentation, sanitize_khmer_text
from generate_text import WordMarkovGenerator, augment_text, load_wikipedia_data

# Whitelist regex pattern for Khmer + English + basic standard symbols
ALLOWED_PATTERN = re.compile(
    r'^[\u1780-\u17FF\u19E0-\u19FFa-zA-Z0-9\s'
    r'.,;:!?\(\)\[\]\{\}_+\-\*/=%$@&#\|\\~<>\x22\x27\^'
    r'\u201c\u201d\u2018\u2019\u2013\u2014\u2026\u00ab\u00bb\u17db\u00b0\u200b'
    r']+$'
)

def clean_text_to_whitelist(text: str) -> str:
    cleaned = []
    for char in text:
        is_khmer = '\u1780' <= char <= '\u17ff' or '\u19e0' <= char <= '\u19ff'
        is_ascii_printable = '\u0020' <= char <= '\u007e'
        is_common_symbol = char in ['\u200b', '\u201c', '\u201d', '\u2018', '\u2019', '\u2013', '\u2014', '\u2026', '\u00a0', '«', '»', '៛', '°']
        if is_khmer or is_ascii_printable or is_common_symbol:
            cleaned.append(char)
    cleaned_str = "".join(cleaned)
    return re.sub(r'\s+', ' ', cleaned_str).strip()

ENGLISH_WORDS = [
    "the", "of", "to", "and", "a", "in", "is", "it", "you", "that", "he", "was", "for", "on", "are", "with", "as", "I", "his", "they",
    "be", "at", "one", "have", "this", "from", "or", "had", "by", "but", "some", "what", "there", "we", "can", "out", "other",
    "were", "all", "your", "when", "up", "use", "word", "how", "said", "an", "each", "she", "which", "do", "their", "time", "if", "will",
    "way", "about", "many", "then", "them", "write", "would", "like", "so", "these", "her", "long", "make", "thing", "see", "him", "two",
    "has", "look", "more", "day", "could", "go", "come", "did", "number", "sound", "no", "most", "people", "my", "over", "know", "water",
    "than", "call", "first", "who", "may", "down", "side", "been", "now", "find", "any", "new", "work", "part", "take", "get", "place",
    "made", "live", "where", "after", "back", "little", "only", "round", "man", "year", "came", "show", "every", "good", "me", "give",
    "under", "name", "very", "through", "just", "form", "sentence", "great", "think", "say", "help", "low", "line", "differ", "turn",
    "cause", "much", "mean", "before", "move", "right", "boy", "old", "too", "same", "tell", "does", "set", "three", "want", "air", "well",
    "also", "play", "small", "end", "put", "home", "read", "hand", "port", "large", "spell", "add", "even", "land", "here", "must", "big",
    "high", "such", "follow", "act", "why", "ask", "men", "change", "went", "light", "kind", "off", "need", "house", "picture", "try",
    "us", "again", "animal", "point", "mother", "world", "near", "build", "self", "earth", "father", "head", "stand", "own", "page",
    "should", "country", "found", "answer", "school", "grow", "study", "still", "learn", "plant", "cover", "food", "sun", "four", "between",
    "state", "keep", "eye", "never", "last", "let", "thought", "city", "tree", "cross", "farm", "hard", "start", "might", "story", "saw",
    "far", "sea", "draw", "left", "late", "run", "while", "press", "close", "night", "real", "life", "few", "north", "open", "seem",
    "together", "next", "white", "children", "begin", "got", "walk", "example", "paper", "group", "always", "music", "those", "both"
]

def classify_script(text: str) -> str:
    has_khmer = bool(re.search(r"[\u1780-\u17ff]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    
    if has_khmer and has_latin:
        return "mixed"
    elif has_khmer:
        return "pure_khmer"
    elif has_latin:
        return "pure_english"
    else:
        if re.search(r"[0-9\u17e0-\u17e9]", text):
            return "pure_khmer" if re.search(r"[\u17e0-\u17e9]", text) else "pure_english"
        return "pure_khmer"

def generate_english_sentence() -> str:
    num_words = random.randint(5, 18)
    words = random.choices(ENGLISH_WORDS, k=num_words)
    
    if random.random() < 0.2:
        words.insert(random.randint(1, len(words)-1), str(random.randint(0, 2030)))
        
    sentence = " ".join(words)
    if random.random() < 0.3:
        sentence += random.choice([".", "!", "?", "...", ", etc."])
    return sentence

def main():
    random.seed(42)
    data_dir = "kmwiki_data"
    output_dir = "generated_texts"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Wikipedia lines...")
    all_lines = load_wikipedia_data(data_dir)
    print(f"Loaded {len(all_lines)} raw candidate lines from Wikipedia.")
    
    # Filter strictly to whitelist
    raw_lines = [line.strip() for line in all_lines if ALLOWED_PATTERN.match(line.strip())]
    print(f"Filtered to {len(raw_lines)} whitelisted lines ({len(raw_lines)/len(all_lines)*100:.2f}% retained).")
    
    # Classify Wikipedia candidates
    wiki_pure_khmer = []
    wiki_mixed = []
    
    for line in raw_lines:
        s_line = sanitize_khmer_text(line)
        if not s_line or len(s_line) < 15:
            continue
        script_type = classify_script(s_line)
        if script_type == "pure_khmer":
            wiki_pure_khmer.append(s_line)
        elif script_type == "mixed":
            wiki_mixed.append(s_line)
            
    print(f"  Wiki Pure Khmer lines: {len(wiki_pure_khmer)}")
    print(f"  Wiki Mixed lines: {len(wiki_mixed)}")
    
    print("Training Word Markov generator on clean Wikipedia corpus...")
    gen = WordMarkovGenerator(order=2)
    gen.train(raw_lines)
    
    target_count = 1000000
    target_pure_khmer = int(target_count * 0.75) # 750k
    target_mixed = int(target_count * 0.20)      # 200k
    target_pure_english = int(target_count * 0.05) # 50k
    
    print(f"Generating targets:")
    print(f"  Pure Khmer: {target_pure_khmer}")
    print(f"  Mixed: {target_mixed}")
    print(f"  Pure English: {target_pure_english}")
    
    final_rows = []
    
    # 1. Pure English (50,000 rows)
    print("Generating pure English rows...")
    for _ in range(target_pure_english):
        final_rows.append({"source": "synthetic_english", "text": generate_english_sentence()})
        
    # 2. Pure Khmer (750,000 rows)
    print("Populating pure Khmer rows...")
    num_wiki_khmer = min(len(wiki_pure_khmer), int(target_pure_khmer * 0.4))
    sampled_wiki_khmer = random.sample(wiki_pure_khmer, k=num_wiki_khmer)
    for text in sampled_wiki_khmer:
        final_rows.append({"source": "wiki_pure_khmer", "text": text})
        
    remaining_pure_khmer = target_pure_khmer - len(sampled_wiki_khmer)
    print(f"  Generating {remaining_pure_khmer} pure Khmer sentences via Markov...")
    generated_khmer_count = 0
    while generated_khmer_count < remaining_pure_khmer:
        num_words = random.randint(5, 25)
        raw_txt = gen.generate(num_words)
        augmented = augment_text(raw_txt, p_space=0.01, p_symbol=0.005, p_number=0.003, p_noise=0.002)
        cleaned = clean_text_to_whitelist(augmented)
        sanitized = sanitize_khmer_text(cleaned)
        if len(sanitized) >= 15 and classify_script(sanitized) == "pure_khmer" and ALLOWED_PATTERN.match(sanitized):
            final_rows.append({"source": "markov_pure_khmer", "text": sanitized})
            generated_khmer_count += 1
            if generated_khmer_count % 150000 == 0:
                print(f"    Generated {generated_khmer_count} / {remaining_pure_khmer}...")
                
    # 3. Mixed Khmer + English (200,000 rows)
    print("Populating mixed Khmer + English/Numbers rows...")
    num_wiki_mixed = min(len(wiki_mixed), int(target_mixed * 0.4))
    sampled_wiki_mixed = random.sample(wiki_mixed, k=num_wiki_mixed)
    for text in sampled_wiki_mixed:
        final_rows.append({"source": "wiki_mixed", "text": text})
        
    remaining_mixed = target_mixed - len(sampled_wiki_mixed)
    print(f"  Generating {remaining_mixed} mixed sentences via Markov + injection...")
    generated_mixed_count = 0
    while generated_mixed_count < remaining_mixed:
        num_words = random.randint(5, 25)
        raw_txt = gen.generate(num_words)
        words = raw_txt.split()
        if len(words) > 3:
            inject_count = random.randint(1, 2)
            for _ in range(inject_count):
                idx = random.randint(1, len(words) - 1)
                if random.random() < 0.5:
                    words.insert(idx, random.choice(ENGLISH_WORDS))
                else:
                    words.insert(idx, str(random.randint(1950, 2030)))
            raw_txt = " ".join(words)
            
        augmented = augment_text(raw_txt, p_space=0.01, p_symbol=0.005, p_number=0.003, p_noise=0.002)
        cleaned = clean_text_to_whitelist(augmented)
        sanitized = sanitize_khmer_text(cleaned)
        if len(sanitized) >= 15 and classify_script(sanitized) == "mixed" and ALLOWED_PATTERN.match(sanitized):
            final_rows.append({"source": "markov_mixed", "text": sanitized})
            generated_mixed_count += 1
            if generated_mixed_count % 50000 == 0:
                print(f"    Generated {generated_mixed_count} / {remaining_mixed}...")

    # 4. Spacing augmentation & Digits optimization
    print("Applying spacing augmentations and ensuring digits standards...")
    khmer_digits = "០១២៣៤៥៦៧៨៩"
    arabic_digits = "0123456789"
    
    khmer_digit_count = 0
    arabic_digit_count = 0
    
    for i in range(len(final_rows)):
        row = final_rows[i]
        text = row["text"]
        
        text = apply_spacing_augmentation(
            text,
            space_drop_prob=0.85,
            mixed_space_drop_prob=0.35,
            latin_space_drop_prob=0.0
        )
        
        has_kd = any(c in khmer_digits for c in text)
        has_ad = any(c in arabic_digits for c in text)
        
        if not has_kd and khmer_digit_count < 50000 and i % 20 == 0:
            k_year = "".join(khmer_digits[int(d)] for d in str(random.randint(1980, 2030)))
            text = text + f" ឆ្នាំ {k_year}"
            has_kd = True
            
        if not has_ad and arabic_digit_count < 150000 and i % 6 == 0:
            a_num = str(random.randint(10, 5000))
            text = text + f" ({a_num})"
            has_ad = True
            
        if has_kd:
            khmer_digit_count += 1
        if has_ad:
            arabic_digit_count += 1
            
        # Final pass whitelist clean
        text = clean_text_to_whitelist(text)
        row["text"] = text

    print("Shuffling final dataset...")
    random.shuffle(final_rows)
    
    df = pd.DataFrame(final_rows)
    output_path = os.path.join(output_dir, "training_1m_labels.csv")
    df.to_csv(output_path, index=True, index_label="id", encoding="utf-8-sig")
    print(f"Successfully wrote 1,000,000 rows to {output_path}")
    
    # 5. Evaluate the distribution
    print("\n================ EVALUATION OF THE 1M TEXT CORPUS ================")
    total = len(df)
    
    df["script_class"] = df["text"].apply(classify_script)
    class_counts = df["script_class"].value_counts()
    source_counts = df["source"].value_counts()
    
    kd_lines = df["text"].apply(lambda x: any(c in khmer_digits for c in x)).sum()
    ad_lines = df["text"].apply(lambda x: any(c in arabic_digits for c in x)).sum()
    
    vocab = Counter()
    for text in df["text"]:
        vocab.update(text)
    
    print(f"Total Rows: {total}")
    print("\nScript/Language Distribution:")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt} ({cnt/total*100:.2f}%)")
        
    print("\nSource Distribution:")
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt} ({cnt/total*100:.2f}%)")
        
    print("\nDigit Distribution:")
    print(f"  Lines containing Khmer Digits (០-៩): {kd_lines} ({kd_lines/total*100:.2f}%) [Target: >=5.0%]")
    print(f"  Lines containing Arabic Digits (0-9): {ad_lines} ({ad_lines/total*100:.2f}%) [Target: >=15.0%]")
    
    print(f"\nVocabulary Statistics:")
    print(f"  Unique Characters: {len(vocab)}")
    print(f"  Top 10 characters: {vocab.most_common(10)}")
    
    # Check for foreign leaks
    leakage = 0
    for char, count in vocab.most_common():
        is_khmer = '\u1780' <= char <= '\u17ff' or '\u19e0' <= char <= '\u19ff'
        is_ascii_printable = '\u0020' <= char <= '\u007e'
        is_common_symbol = char in ['\u200b', '\u201c', '\u201d', '\u2018', '\u2019', '\u2013', '\u2014', '\u2026', '\u00a0', '«', '»', '៛', '°']
        if not (is_khmer or is_ascii_printable or is_common_symbol):
            leakage += count
    print(f"  Foreign character leak count: {leakage} chars")
    
    report_path = os.path.join(output_dir, "distribution_report.md")
    report_lines = [
        "# 1M Text Corpus Distribution Analysis (Strictly Whitelisted)\n",
        f"This report evaluates the distribution of the generated 1,000,000 text labels with strict whitelisting filters.\n",
        "## Summary",
        f"- **Total Rows**: {total:,}",
        f"- **Unique Characters (Vocab Size)**: {len(vocab)}",
        f"- **Foreign Character Leak Count**: {leakage} (Goal: 0)",
        "\n## Script/Language Distribution",
        "| Script Class | Count | Percentage | Target |",
        "|---|---|---|---|",
        f"| Pure Khmer | {class_counts.get('pure_khmer', 0):,} | {class_counts.get('pure_khmer', 0)/total*100:.2f}% | 75.0% |",
        f"| Mixed Khmer + English | {class_counts.get('mixed', 0):,} | {class_counts.get('mixed', 0)/total*100:.2f}% | 20.0% |",
        f"| Pure English | {class_counts.get('pure_english', 0):,} | {class_counts.get('pure_english', 0)/total*100:.2f}% | 5.0% |",
        "\n## Source Breakdown",
        "| Source | Count | Percentage |",
        "|---|---|---|",
    ]
    for src, cnt in source_counts.items():
        report_lines.append(f"| {src} | {cnt:,} | {cnt/total*100:.2f}% |")
        
    report_lines.extend([
        "\n## Digit Distribution",
        "| Digit System | Lines Count | Percentage | Target | Status |",
        "|---|---|---|---|---|",
        f"| Khmer Digits (`០-៩`) | {kd_lines:,} | {kd_lines/total*100:.2f}% | >= 5.0% | {'✅ Met' if kd_lines/total*100 >= 5.0 else '❌ Below'} |",
        f"| Arabic Digits (`0-9`) | {ad_lines:,} | {ad_lines/total*100:.2f}% | >= 15.0% | {'✅ Met' if ad_lines/total*100 >= 15.0 else '❌ Below'} |",
        "\n## Vocabulary Sample",
        f"Top 20 most frequent characters: `{' '.join(char for char, _ in vocab.most_common(20))}`"
    ])
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Wrote detailed distribution report to {report_path}")

if __name__ == "__main__":
    main()
