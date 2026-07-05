import os
import sys
import argparse
import random
import shutil
from collections import Counter
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Create a balanced 500K subset from the 1M dataset.")
    parser.add_argument("--src", default="generated/training_1m", help="Source directory containing images and metadata.duckdb")
    parser.add_argument("--dst", default="generated/training_500k", help="Destination directory for the 500K subset")
    parser.add_argument("--max-len", type=int, default=200, help="Maximum allowed text length")
    parser.add_argument("--rare-threshold", type=int, default=5000, help="Frequencies below this are considered rare")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.src):
        print(f"Error: Source directory '{args.src}' does not exist.", file=sys.stderr)
        sys.exit(1)

    db_path = os.path.join(args.src, "metadata.duckdb")
    if not os.path.exists(db_path):
        print(f"Error: Database '{db_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        import duckdb
        import pandas as pd
    except ImportError:
        print("Error: 'duckdb' and 'pandas' packages are required. Run 'uv add duckdb pandas' first.", file=sys.stderr)
        sys.exit(1)

    # 1. Connect to DuckDB and retrieve records under length threshold
    print("Loading metadata from DuckDB...")
    con = duckdb.connect(db_path)
    df = con.execute(f"SELECT * FROM metadata WHERE length(text) <= {args.max_len}").df()
    con.close()

    total_candidates = len(df)
    print(f"Total candidates (length <= {args.max_len}): {total_candidates:,}")

    if total_candidates < 500000:
        print(f"Error: Only {total_candidates:,} candidates exist. Cannot create a 500K subset.", file=sys.stderr)
        sys.exit(1)

    # 2. Count character frequencies to identify rare characters
    print("Analyzing character frequencies...")
    char_counts = Counter()
    for text in df["text"]:
        char_counts.update(text)

    rare_chars = {char for char, count in char_counts.items() if count < args.rare_threshold}
    print(f"Identified {len(rare_chars)} rare characters (appearing < {args.rare_threshold:,} times).")

    # 3. Separate records into "must-keep" (contains rare chars) and "optional"
    must_keep_indices = []
    optional_indices = []

    for idx, row in enumerate(df.itertuples()):
        has_rare = any(char in rare_chars for char in row.text)
        if has_rare:
            must_keep_indices.append(idx)
        else:
            optional_indices.append(idx)

    must_keep_count = len(must_keep_indices)
    optional_count = len(optional_indices)
    print(f"Must-keep records (with rare characters): {must_keep_count:,}")
    print(f"Optional records (common characters): {optional_count:,}")

    # 4. Sample from optional records to hit exactly 500,000 total records
    target_total = 500000
    needed_optional = target_total - must_keep_count

    if needed_optional < 0:
        print(f"Warning: Must-keep records ({must_keep_count:,}) exceed the 500K target! Keeping all must-keeps and skipping optionals.")
        selected_indices = must_keep_indices[:target_total]
    else:
        print(f"Sampling {needed_optional:,} records from the optional pool...")
        sampled_optional_indices = random.sample(optional_indices, k=needed_optional)
        selected_indices = must_keep_indices + sampled_optional_indices

    # Shuffling to mix the dataset
    random.shuffle(selected_indices)
    df_subset = df.iloc[selected_indices].copy()

    # 5. Create target directory and copy images
    os.makedirs(args.dst, exist_ok=True)
    print(f"Copying {target_total:,} images to {args.dst}...")
    
    copied = 0
    for filename in tqdm(df_subset["filename"], desc="Copying images"):
        src_file = os.path.join(args.src, filename)
        dst_file = os.path.join(args.dst, filename)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            copied += 1
        else:
            print(f"Warning: Image file '{src_file}' not found.")

    # 6. Save new subset metadata to new DuckDB
    print("Writing new metadata database...")
    new_db_path = os.path.join(args.dst, "metadata.duckdb")
    if os.path.exists(new_db_path):
        os.remove(new_db_path)

    con_new = duckdb.connect(new_db_path)
    con_new.execute("CREATE TABLE metadata AS SELECT * FROM df_subset")
    con_new.close()

    print("\nBalanced subset creation complete!")
    print(f"  Target directory:          {args.dst}")
    print(f"  Total records in subset:   {len(df_subset):,}")
    print(f"  Successfully copied files: {copied:,}")

if __name__ == "__main__":
    main()
