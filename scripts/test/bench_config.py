"""Configuration for the TabPFN cloud-API benchmark over the STRABLE datasets.

Everything tunable lives here: API tokens, the test-size sweep, sub-sampling
caps and output locations.  Paths are derived from the repository root (via
``configs/path_configs.py``), so this module works unmodified on any machine —
no hard-coded ``/home/...`` paths.

Override the tokens at runtime without editing the file:

    export TABPFN_TOKENS="tabpfn_sk_aaa,tabpfn_sk_bbb"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make ``configs`` importable regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.path_configs import path_configs  # noqa: E402

# --- TabPFN cloud API tokens -------------------------------------------------
# Tokens are NEVER hard-coded here, so this file is safe to commit/publish.
# They are loaded (in priority order) from:
#   1. the TABPFN_TOKENS env var  (comma-separated)
#   2. an untracked local file    scripts/test/tokens.local  (one token per line)
# The runner rotates through them so that when one hits its usage limit it
# transparently falls back to the next.  See tokens.local.example for the format.
def _load_tokens() -> list[str]:
    env = os.environ.get("TABPFN_TOKENS", "")
    if env.strip():
        return [t.strip() for t in env.split(",") if t.strip()]
    local = Path(__file__).with_name("tokens.local")
    if local.exists():
        return [ln.strip() for ln in local.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
    return []


API_TOKENS = _load_tokens()

# --- Experiment sweep --------------------------------------------------------
# Every dataset is evaluated at each of these test fractions (train = 1 - ts).
TEST_SIZES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RANDOM_STATE = 42

# --- Task tags (the "task" field inside each dataset's config.json) ----------
CLASSIFICATION_TASKS = {"b-classification", "m-classification"}
REGRESSION_TASKS = {"regression"}

# --- Sub-sampling caps -------------------------------------------------------
# TabPFN-cloud caps the *training* set (max rows and max rows*cols cells).  We
# subsample the training split down to these caps; when FETCH_SERVER_LIMITS is
# on we additionally clamp to the live server limits so a call is never
# rejected.  The *test* split is capped purely to keep each call fast/cheap —
# the reported metric is then an estimate computed on that test sample.
MAX_TRAIN_ROWS = 10000
MAX_TRAIN_CELLS = 500_000
MAX_TEST_ROWS = 2000
FETCH_SERVER_LIMITS = True

# --- Output ------------------------------------------------------------------
RESULTS_ROOT = Path(path_configs["results"]) / "tabpfn_test"
PER_ROW_DIR = RESULTS_ROOT / "per_row"  # one CSV of predictions per run
SUMMARY_CSV = RESULTS_ROOT / "summary.csv"  # one row per run (metrics + meta)
DATA_PROCESSED = Path(path_configs["path_data_processed"])

WRITE_PER_ROW = True  # write per-row predictions (needed for bad-case drill-down)
SKIP_EXISTING = True  # skip (dataset, test_size) pairs already in summary.csv

# --- Bad-case thresholds (consumed by extract_bad_cases.py) ------------------
REG_BAD_R2 = 0.30  # a regression run with R^2 below this counts as "bad"
CLS_BAD_AUROC = 0.65  # a classification run with AUROC below this is "bad"
CLS_BAD_ACC_LIFT = 0.05  # ... or one that barely beats the majority baseline

# --- Multi-benchmark integration (STRABLE / CARTE / TextTabBench) -------------
BENCHMARKS = ["strable", "carte", "ttb"]
CARTE_ROOT = Path(
    path_configs["carte_datasets"]
)  # data/CARTE_datasets (has data_carte/<name>/)
TTB_ROOT = Path(path_configs["ttb_datasets"])  # data/TTB_datasets
TTB_MANIFEST = TTB_ROOT / "manifest.json"  # pins {file,target,task} per TTB dataset

# Datasets that are the SAME underlying table across benchmarks (verified from
# STRABLE's own preprocess scripts loading data_raw/carte/* and data_raw/textTabBench/*,
# plus the CARTE/TTB inventories). group_id -> [(benchmark, dataset_name), ...].
# Rows sharing a group are the same data (sometimes under a different task framing,
# e.g. michelin is binary in CARTE but multiclass in STRABLE) — collapse on the
# group before pooling metrics, and never split one group across a train/test boundary.
OVERLAP_GROUPS = {
    # STRABLE <-> CARTE (STRABLE re-imported these exact files from data_raw/carte/)
    "ramen_ratings": [("strable", "ramen-ratings"), ("carte", "ramen_ratings")],
    "chocolate": [
        ("strable", "chocolate-bar-ratings"),
        ("carte", "chocolate_bar_ratings"),
    ],
    "whisky": [("strable", "meta-critic_whisky"), ("carte", "whisky")],
    "mlds_salaries": [
        ("strable", "aijob_ai-ml-ds-salaries"),
        ("carte", "mlds_salaries"),
    ],
    "clear_corpus": [("strable", "clear-corpus"), ("carte", "clear_corpus")],
    "museums": [("strable", "museums"), ("carte", "museums")],
    "yelp": [("strable", "yelp_business"), ("carte", "yelp")],
    "michelin": [("strable", "michelin-ratings"), ("carte", "michelin")],
    "journal_sjr": [("strable", "journal-ranking_wide"), ("carte", "journal_sjr")],
    # STRABLE <-> TTB (STRABLE re-imported these exact files from data_raw/textTabBench/)
    # beer is a triple (STRABLE+CARTE+TTB all trace to BeerAdvocate-style ratings).
    "beer": [("strable", "beer-ratings"), ("carte", "beer_ratings"), ("ttb", "beer")],
    "california_houses": [("strable", "california-houses"), ("ttb", "calif_houses")],
    "wine": [("strable", "wine-dataset"), ("ttb", "wine")],
    "mercari": [("strable", "mercari"), ("ttb", "mercari")],
    "sf_permits": [("strable", "sf-building-permits"), ("ttb", "sf_permits")],
    "kickstarter": [("strable", "kickstarter-projects"), ("ttb", "kickstarter")],
    "osha": [("strable", "osha-accidents"), ("ttb", "osha_accidents")],
    # same upstream provider, NOT byte-identical (US CFPB complaint database)
    "cfpb_complaints": [
        ("strable", "financial-product-complaint"),
        ("ttb", "complaints"),
    ],
}
# (benchmark, name) -> group_id, for O(1) lookup when stamping specs.
OVERLAP_LOOKUP = {
    member: gid for gid, members in OVERLAP_GROUPS.items() for member in members
}
