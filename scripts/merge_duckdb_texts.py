import os
import sys
import argparse
import pandas as pd
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from generate_1m_texts import clean_text_to_whitelist, classify_script, ALLOWED_PATTERN

def main():
    parser = argparse.ArgumentParser(description="Merge extra text lines from a DuckDB database into the 1M training labels dataset")
    parser.add_argument("--db", required=True, help="Path to the .duckdb file")
    parser.add_argument("--table", required=True, help="Table name in the database")
    parser.add_argument("--column", required=True, help="Column name containing the text lines")
    parser.add_argument("--labels-csv", default="generated_texts/training_1m_labels.csv", help="Path to current training_1m_labels.csv")
    parser.add_argument("--keep-1m", action="store_true", default=True, help="Keep total count at exactly 1M by replacing Markov rows (default: True)")
    args = parser.parse_args()

    # Try importing duckdb
    try:
        import duckdb
    except ImportError:
        print("Error: 'duckdb' package is not installed.", file=sys.stderr)
        print("Please run the following command to install it first:", file=sys.stderr)
        print("  uv add duckdb", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.db):
        print(f"Error: Database file '{args.db}' not found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.labels_csv):
        print(f"Error: Current labels CSV '{args.labels_csv}' not found.", file=sys.stderr)
        sys.exit(1)

    # 1. Connect to DuckDB and query the text data
    print(f"Connecting to DuckDB database: {args.db}")
    con = duckdb.connect(args.db)
    
    print(f"Querying table '{args.table}', column '{args.column}'...")
    try:
        df_db = con.execute(f"SELECT {args.column} FROM {args.table} WHERE {args.column} IS NOT NULL").df()
    except Exception as e:
        print(f"SQL Error: {e}", file=sys.stderr)
        con.close()
        sys.exit(1)
        
    con.close()
    
    raw_texts = df_db[args.column].astype(str).tolist()
    print(f"Retrieved {len(raw_texts):,} text rows from database.")

    # 2. Clean and filter incoming database texts
    print("Sanitizing and whitelisting incoming database texts...")
    clean_texts = []
    skipped_count = 0
    
    for txt in raw_texts:
        cleaned = clean_text_to_whitelist(txt)
        if len(cleaned) < 15:
            skipped_count += 1
            continue
            
        # Ensure it strictly conforms to the allowed character set
        if ALLOWED_PATTERN.match(cleaned):
            clean_texts.append(cleaned)
        else:
            skipped_count += 1
            
    print(f"Cleaned database lines: {len(clean_texts):,} (skipped {skipped_count:,} too-short or invalid lines).")
    
    if not clean_texts:
        print("Error: No valid whitelisted texts found in the database. Aborting.", file=sys.stderr)
        sys.exit(1)

    # 3. Load existing labels
    print(f"Loading existing labels from {args.labels_csv}...")
    df_labels = pd.read_csv(args.labels_csv)
    total_existing = len(df_labels)
    print(f"Loaded {total_existing:,} existing rows.")

    # 4. Prepare new rows
    new_rows = []
    for text in clean_texts:
        new_rows.append({
            "source": f"duckdb_{args.table}",
            "text": text,
            "script_class": classify_script(text)
        })
        
    df_new = pd.DataFrame(new_rows)

    # 5. Merge logic
    if args.keep_1m:
        print("Merge mode: Keeping total dataset size at exactly 1,000,000 rows.")
        # We will replace synthetic Markov rows to make room for high-quality database texts
        new_count = len(df_new)
        
        if new_count >= total_existing:
            print(f"Warning: Database rows ({new_count:,}) exceed 1M. Using first 900,000 database rows to preserve pure English/wiki context.")
            df_new = df_new.iloc[:900000]
            new_count = len(df_new)
            
        # Group current rows by source
        # We prefer to keep: synthetic_english (50k), wiki_pure_khmer (80k), wiki_mixed (8k)
        # We will drop from: markov_pure_khmer and markov_mixed
        df_labels["script_class"] = df_labels["text"].apply(classify_script)
        
        # Split into keep and replacement candidates
        keep_sources = ["synthetic_english", "wiki_pure_khmer", "wiki_mixed"]
        df_keep = df_labels[df_labels["source"].isin(keep_sources)]
        df_replace = df_labels[~df_labels["source"].isin(keep_sources)]
        
        needed_from_replace = total_existing - len(df_keep) - new_count
        print(f"  Keeping {len(df_keep):,} high-quality wiki/English rows.")
        print(f"  Down-sampling Markov rows from {len(df_replace):,} to {needed_from_replace:,}.")
        
        if needed_from_replace > 0:
            df_replace_sampled = df_replace.sample(n=needed_from_replace, random_state=42)
            df_merged = pd.concat([df_keep, df_replace_sampled, df_new], ignore_index=True)
        else:
            print("  Note: Database rows fully replaced all Markov rows.")
            df_merged = pd.concat([df_keep, df_new], ignore_index=True)
    else:
        print("Merge mode: Appending database rows (expanding dataset size).")
        df_labels["script_class"] = df_labels["text"].apply(classify_script)
        df_merged = pd.concat([df_labels, df_new], ignore_index=True)

    # Clean indices
    if "id" in df_merged.columns:
        df_merged = df_merged.drop(columns=["id"])
    if "Unnamed: 0" in df_merged.columns:
        df_merged = df_merged.drop(columns=["Unnamed: 0"])
        
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save back
    df_merged.to_csv(args.labels_csv, index=True, index_label="id", encoding="utf-8-sig")
    print(f"Successfully wrote merged dataset ({len(df_merged):,} rows) to {args.labels_csv}")

    # 6. Re-evaluate
    total = len(df_merged)
    class_counts = df_merged["script_class"].value_counts()
    source_counts = df_merged["source"].value_counts()
    
    khmer_digits = "០១២៣៤៥៦៧៨៩"
    arabic_digits = "0123456789"
    kd_lines = df_merged["text"].apply(lambda x: any(c in khmer_digits for c in str(x))).sum()
    ad_lines = df_merged["text"].apply(lambda x: any(c in arabic_digits for c in str(x))).sum()
    
    print("\n================ NEW MERGED DISTRIBUTION ================ ")
    print(f"Total Rows: {total}")
    print("\nScript/Language Distribution:")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt} ({cnt/total*100:.2f}%)")
    print("\nSource Distribution:")
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt} ({cnt/total*100:.2f}%)")
    print("\nDigit Distribution:")
    print(f"  Khmer Digits (០-៩): {kd_lines} ({kd_lines/total*100:.2f}%)")
    print(f"  Arabic Digits (0-9): {ad_lines} ({ad_lines/total*100:.2f}%)")

if __name__ == "__main__":
    main()
