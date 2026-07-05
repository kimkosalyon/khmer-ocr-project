import argparse
import os
import sys
from pathlib import Path

import duckdb
from datasets import Dataset, Image
from huggingface_hub import HfApi


def get_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def table_columns(db_path: Path) -> set[str]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("DESCRIBE metadata").fetchall()
    finally:
        con.close()
    return {row[0] for row in rows}


def build_select_columns(columns: set[str]) -> list[str]:
    preferred = [
        "filename",
        "text",
        "source",
        "script_class",
        "font",
        "font_size",
        "text_color",
        "bg_color",
        "padding_top",
        "padding_right",
        "padding_bottom",
        "padding_left",
        "expected_english_font",
        "text_len",
    ]
    return [col for col in preferred if col in columns]


def main():
    parser = argparse.ArgumentParser(description="Upload generated OCR image dataset to Hugging Face Hub")
    parser.add_argument("--dataset-dir", default="generated/training_200k_siemreap_arial")
    parser.add_argument("--repo-id", required=True, help="Hub dataset repo, e.g. KimkosalYon/khmer-ocr-200k-siemreap-arial")
    parser.add_argument("--private", action="store_true", help="Create/upload as a private dataset repo")
    parser.add_argument("--max-shard-size", default="500MB")
    parser.add_argument("--num-proc", type=int, default=min(os.cpu_count() or 4, 16))
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    db_path = dataset_dir / "metadata.duckdb"
    if not db_path.exists():
        print(f"metadata.duckdb not found: {db_path}", file=sys.stderr)
        raise SystemExit(1)

    token = get_token()
    api = HfApi(token=token)
    try:
        user = api.whoami()["name"]
        print(f"Authenticated as: {user}")
    except Exception:
        if token:
            raise
        print("No HF_TOKEN found and no cached login available. Run `huggingface-cli login` or set HF_TOKEN.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Target dataset repo: {args.repo_id}")
    print(f"Dataset directory: {dataset_dir}")

    columns = table_columns(db_path)
    select_cols = build_select_columns(columns)
    if "filename" not in select_cols or "text" not in select_cols:
        raise SystemExit("metadata table must contain at least `filename` and `text` columns")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        total_rows = con.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    finally:
        con.close()
    print(f"Rows: {total_rows:,}")
    print(f"Columns: {', '.join(select_cols)}")

    num_proc = max(1, min(args.num_proc, total_rows))
    chunk_size = (total_rows + num_proc - 1) // num_proc
    starts = [i * chunk_size for i in range(num_proc)]
    ends = [min((i + 1) * chunk_size, total_rows) for i in range(num_proc)]

    db_path_str = str(db_path)
    dataset_dir_str = str(dataset_dir)
    select_sql = ", ".join(f'"{col}"' for col in select_cols)

    def gen_rows(start_indices, end_indices):
        import duckdb
        from pathlib import Path

        for start, end in zip(start_indices, end_indices):
            limit = end - start
            con = duckdb.connect(db_path_str, read_only=True)
            try:
                rows = con.execute(
                    f"SELECT {select_sql} FROM metadata ORDER BY filename LIMIT {limit} OFFSET {start}"
                ).fetchall()
            finally:
                con.close()

            for row in rows:
                item = dict(zip(select_cols, row))
                filename = item.pop("filename")
                image_path = Path(dataset_dir_str) / filename
                item = {k: (int(v) if k.startswith("padding_") or k in {"font_size", "text_len"} else v) for k, v in item.items()}
                item["image"] = str(image_path)
                yield item

    print("Building Hugging Face Dataset from local files...")
    dataset = Dataset.from_generator(
        gen_rows,
        gen_kwargs={"start_indices": starts, "end_indices": ends},
        num_proc=num_proc,
    )
    dataset = dataset.cast_column("image", Image())

    print("Creating repo if needed...")
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    print("Uploading dataset shards...")
    dataset.push_to_hub(
        repo_id=args.repo_id,
        split=args.split,
        token=token,
        private=args.private,
        max_shard_size=args.max_shard_size,
    )

    print("Upload complete:")
    print(f"https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
