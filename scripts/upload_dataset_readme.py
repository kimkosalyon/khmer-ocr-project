import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi


def main():
    load_dotenv(override=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is not set in .env")

    parser = argparse.ArgumentParser(description="Upload a README dataset card to a Hugging Face dataset repo")
    parser.add_argument("--repo-id", default="KimkosalYon/khmer-ocr-200k-siemreap-arial")
    parser.add_argument("--readme", default="docs/hf_dataset_card_200k_siemreap_arial.md")
    args = parser.parse_args()

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=args.readme,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded README.md to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
