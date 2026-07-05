import os
import sys
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Prune OCR dataset by deleting images with text length exceeding a threshold.")
    parser.add_argument("--dir", default="generated/training_1m", help="Directory containing images and metadata.duckdb")
    parser.add_argument("--max-len", type=int, default=120, help="Maximum allowed text length in characters")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without deleting files or database rows")
    args = parser.parse_args()

    db_path = os.path.join(args.dir, "metadata.duckdb")
    if not os.path.exists(db_path):
        print(f"Error: Database '{db_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        import duckdb
    except ImportError:
        print("Error: 'duckdb' package is required. Run 'uv add duckdb' first.", file=sys.stderr)
        sys.exit(1)

    con = duckdb.connect(db_path)
    
    # Check total count
    total_rows = con.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    
    # Query files exceeding limit
    long_rows = con.execute(f"SELECT filename, length(text) as len FROM metadata WHERE length(text) > {args.max_len}").fetchall()
    
    con.close()

    long_count = len(long_rows)
    if long_count == 0:
        print(f"No records found exceeding max-len of {args.max_len}.")
        return

    print(f"Total dataset size: {total_rows:,} records.")
    print(f"Found {long_count:,} records ({long_count/total_rows*100:.2f}%) with text length > {args.max_len} characters.")

    if args.dry_run:
        print("\nDry run mode active. No files or database entries were modified.")
        print("Top 5 longest text samples that would be deleted:")
        long_rows_sorted = sorted(long_rows, key=lambda x: x[1], reverse=True)
        for fname, length in long_rows_sorted[:5]:
            print(f"  {fname}: {length} characters")
        return

    # Delete files
    print("\nDeleting image files...")
    deleted_count = 0
    for fname, _ in tqdm(long_rows, desc="Deleting images"):
        file_path = os.path.join(args.dir, fname)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

    # Remove from DuckDB
    print("\nRemoving records from DuckDB...")
    con = duckdb.connect(db_path)
    con.execute(f"DELETE FROM metadata WHERE length(text) > {args.max_len}")
    con.close()

    print("\nPruning complete:")
    print(f"  Deleted image files: {deleted_count:,}")
    print(f"  Removed database records: {long_count:,}")
    print(f"  Remaining dataset size: {total_rows - long_count:,} records.")

if __name__ == "__main__":
    main()
