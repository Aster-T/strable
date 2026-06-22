"""Download the CARTE and TextTabBench datasets used by the TabPFN harness.

STRABLE itself is fetched by data/download_datasets.py; this script adds the two
sibling benchmarks into the exact on-disk layout the adapters / manifest expect:

  CARTE -> data/CARTE_datasets/data_carte/<name>/{config_data.json, raw.parquet}
           (HuggingFace dataset inria-soda/carte-benchmark)
  TTB   -> data/TTB_datasets/raw/<name>/<file>
           (5 Kaggle datasets; the other TTB tables already overlap STRABLE)

Usage:
    python scripts/test/download_benchmarks.py --benchmark all
    python scripts/test/download_benchmarks.py --benchmark carte
    python scripts/test/download_benchmarks.py --benchmark ttb

CARTE needs network (and socksio if behind a SOCKS proxy). TTB needs Kaggle
credentials: set KAGGLE_USERNAME+KAGGLE_KEY (or KAGGLE_KEY alone for an API
token), or place ~/.kaggle/kaggle.json.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from configs.path_configs import path_configs  # noqa: E402

CARTE_ROOT = Path(path_configs["carte_datasets"])
TTB_RAW = Path(path_configs["ttb_datasets"]) / "raw"

# (local subdir, kaggle dataset ref, file to pull) — paths match TTB manifest.json
TTB_SPECS = [
    ("hs_cards",      "jeradrose/hearthstone-cards",                    "cards_flat.csv"),
    ("job_frauds",    "shivamb/real-or-fake-fake-jobposting-prediction", "fake_job_postings.csv"),
    ("spotify_genre", "maharshipandya/-spotify-tracks-dataset",         "dataset.csv"),
    ("airbnb",        "airbnb/seattle",                                 "listings.csv"),
    ("laptops",       "dhanushbommavaram/laptop-dataset",               "complete laptop data0.csv"),
]


def download_carte() -> None:
    from huggingface_hub import snapshot_download

    print(f"CARTE -> {CARTE_ROOT}  (HF inria-soda/carte-benchmark, data_carte/*)")
    snapshot_download(
        repo_id="inria-soda/carte-benchmark", repo_type="dataset",
        local_dir=str(CARTE_ROOT), allow_patterns=["data_carte/*", "README.md"],
    )
    n = len(list((CARTE_ROOT / "data_carte").glob("*/raw.parquet")))
    print(f"  done: {n} CARTE datasets under {CARTE_ROOT/'data_carte'}")


def download_ttb() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()  # reads KAGGLE_KEY / KAGGLE_USERNAME or ~/.kaggle/kaggle.json
    for sub, ref, fname in TTB_SPECS:
        dest = TTB_RAW / sub
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / fname).exists():
            print(f"  = {sub}/{fname} already present — skipping")
            continue
        print(f"  + {sub}: {ref} :: {fname}")
        api.dataset_download_file(ref, fname, path=str(dest))
        # Kaggle sometimes delivers a single file zipped as <fname>.zip — unpack it.
        z = dest / f"{fname}.zip"
        if z.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
            z.unlink()
        if not (dest / fname).exists():
            print(f"    ⚠ expected {dest/fname} not found after download — check Kaggle access")
    print(f"  done: TTB raw files under {TTB_RAW}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", choices=["carte", "ttb", "all"], default="all")
    args = p.parse_args()
    if args.benchmark in ("carte", "all"):
        download_carte()
    if args.benchmark in ("ttb", "all"):
        download_ttb()
    print("\nAll requested downloads complete.")


if __name__ == "__main__":
    main()
