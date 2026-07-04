#!/usr/bin/env python
"""
Download PrimeKG dataset from Harvard Dataverse.

Downloads:
  - kg.csv             (~580 MB, 8.1M rows, the main knowledge graph)
  - drug_features.csv  (optional)
  - disease_features.csv (optional)

After download, extracts a drug-disease subset by default (can be skipped
with --no-extract).

Usage:
    python scripts/download_primekb.py                          # kg.csv + extract drug-disease
    python scripts/download_primekb.py --all                    # all three files
    python scripts/download_primekb.py --no-extract             # skip subset extraction
    python scripts/download_primekb.py --extract drug-disease   # explicit subset
    python scripts/download_primekb.py --output_dir ./my_data   # custom dir
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request

DATAVERSE_FILES = {
    "kg.csv": "https://dataverse.harvard.edu/api/access/datafile/6180620",
    "drug_features.csv": "https://dataverse.harvard.edu/api/access/datafile/6180618",
    "disease_features.csv": "https://dataverse.harvard.edu/api/access/datafile/6180619",
}

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

CHUNK_SIZE = 1024 * 1024  # 1 MB


def _download_urllib(url: str, dest: str) -> None:
    """Download using urllib with a browser-like User-Agent header."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = min(100.0, downloaded / total * 100)
                mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                sys.stdout.write(f"\r  {pct:5.1f}%  {mb:.1f} / {total_mb:.1f} MB")
            else:
                mb = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  {mb:.1f} MB downloaded")
            sys.stdout.flush()
    print()


def _download_wget(url: str, dest: str) -> None:
    """Fallback: download via wget subprocess."""
    print("  (using wget)")
    subprocess.check_call(["wget", "-q", "--show-progress", "-O", dest, url])


def _download_curl(url: str, dest: str) -> None:
    """Fallback: download via curl subprocess."""
    print("  (using curl)")
    subprocess.check_call(["curl", "-L", "-o", dest, "--progress-bar", url])


def download_file(url: str, dest: str) -> None:
    """Download a file, trying urllib first then wget/curl as fallbacks."""
    print(f"Downloading {os.path.basename(dest)} ...")
    start = time.time()

    try:
        _download_urllib(url, dest)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"\n  urllib failed ({exc}), trying wget/curl ...")
        if os.path.exists(dest):
            os.remove(dest)
        if shutil.which("wget"):
            _download_wget(url, dest)
        elif shutil.which("curl"):
            _download_curl(url, dest)
        else:
            raise RuntimeError(
                f"Download failed and neither wget nor curl is available.\n"
                f"Install wget and run: wget -O {dest} {url}"
            ) from exc

    elapsed = time.time() - start
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  Saved {size_mb:.1f} MB in {elapsed:.0f}s -> {dest}")


SUBSET_FILTERS = {
    "drug-disease": {
        "x_type": "drug",
        "y_type": "disease",
        "description": "Drug-disease edges (indication, contraindication, off-label use)",
    },
}


def extract_subset(kg_path: str, output_dir: str, subset: str = "drug-disease") -> str:
    """Extract a node-type-pair subset from kg.csv and save as a separate CSV.

    Returns the path to the extracted file, or raises if the subset is unknown.
    """
    if subset not in SUBSET_FILTERS:
        raise ValueError(
            f"Unknown subset '{subset}'. Available: {list(SUBSET_FILTERS.keys())}"
        )

    filt = SUBSET_FILTERS[subset]
    out_name = f"kg_{subset.replace('-', '_')}.csv"
    out_path = os.path.join(output_dir, out_name)

    try:
        import pandas as pd
    except ImportError:
        print("  pandas is required for --extract.  pip install pandas")
        sys.exit(1)

    print(f"\nExtracting '{subset}' subset ({filt['description']}) ...")

    first_line = open(kg_path, encoding="utf-8").readline().strip()
    skiprows = 1 if not first_line.startswith("relation") else 0

    df = pd.read_csv(kg_path, skiprows=skiprows, low_memory=False)
    mask = (df["x_type"] == filt["x_type"]) & (df["y_type"] == filt["y_type"])
    filtered = df[mask]

    if filtered.empty:
        print(f"  WARNING: no rows matched {filt}. Skipping extraction.")
        return ""

    filtered.to_csv(out_path, index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    relations = sorted(filtered["display_relation"].unique())
    print(f"  {len(filtered):,} rows, {size_mb:.1f} MB -> {out_path}")
    print(f"  Relations: {relations}")
    return out_path


def summarize_kg(path: str) -> None:
    """Print a quick summary of the downloaded kg.csv."""
    try:
        import pandas as pd
    except ImportError:
        print("  (install pandas to see a dataset summary)")
        return

    print("\nDataset summary:")
    df = pd.read_csv(path, nrows=0)
    print(f"  Columns: {list(df.columns)}")

    row_count = sum(1 for _ in open(path, encoding="utf-8")) - 1
    print(f"  Total rows: {row_count:,}")

    df_sample = pd.read_csv(path, nrows=100_000)
    node_types = set(df_sample["x_type"].unique()) | set(df_sample["y_type"].unique())
    relations = set(df_sample["display_relation"].unique())
    print(f"  Node types (from first 100k rows): {sorted(node_types)}")
    print(f"  Relation types (from first 100k rows): {sorted(relations)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download PrimeKG dataset from Harvard Dataverse",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        help="Directory to save files (default: data/)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all files (kg.csv + drug_features.csv + disease_features.csv)",
    )
    parser.add_argument(
        "--skip_summary",
        action="store_true",
        help="Skip printing dataset summary after download",
    )
    parser.add_argument(
        "--extract",
        default="drug-disease",
        help="Subset to extract after download (default: drug-disease)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip subset extraction (keep only the full kg.csv)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files_to_download = ["kg.csv"]
    if args.all:
        files_to_download = list(DATAVERSE_FILES.keys())

    print("=" * 60)
    print("PrimeKG Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Files: {', '.join(files_to_download)}\n")

    for fname in files_to_download:
        dest = os.path.join(args.output_dir, fname)
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"Skipping {fname} (already exists, {size_mb:.1f} MB)")
            continue
        download_file(DATAVERSE_FILES[fname], dest)

    kg_path = os.path.join(args.output_dir, "kg.csv")
    if not args.skip_summary and os.path.exists(kg_path):
        summarize_kg(kg_path)

    # --- Subset extraction ---
    subset_path = ""
    if not args.no_extract and os.path.exists(kg_path):
        subset_path = extract_subset(kg_path, args.output_dir, subset=args.extract)

    print("\n" + "=" * 60)
    print("Done! Next steps:")

    if subset_path:
        print(f"\n  Path A — Split + evaluate (drug-disease subset):")
        print(f"    python scripts/eval/split_primekb.py --input {subset_path}")
        print(f"    python scripts/import_primekb_to_neo4j.py   # imports data/eval/train.csv")
        print(f"    python scripts/eval/generate_qa.py --test-csv data/eval/test.csv")
        print(f"    python scripts/eval/evaluate.py --qa-dataset data/eval/qa_dataset.json")

    print(f"\n  Path B — RAG pipeline (full or filtered dataset):")
    input_file = subset_path or kg_path
    print(f"    python scripts/index_rag.py --input {input_file} --format primekb \\")
    print(f"        --output_dir ./output/rag_primekb")
    print(f"    python scripts/migrate_faiss_to_neo4j.py --faiss-dir ./output/rag_primekb")
    print(f"    python scripts/demo_mcp_agent.py              # run demo")
    print(f"    python scripts/interactive_agent.py            # interactive chat")
    print(f"\n  Path C — Import into Neo4j as :Triplet nodes:")
    print(f"    python scripts/import_primekb_to_neo4j.py     # uses data/eval/train.csv")
    print(f"    python scripts/demo_mcp_agent.py              # run demo")
    print(f"    python scripts/interactive_agent.py            # interactive chat")
    print("=" * 60)


if __name__ == "__main__":
    main()
