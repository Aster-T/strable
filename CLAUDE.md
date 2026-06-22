# Claude Code instructions

This repo's agent-facing documentation lives in [AGENTS.md](AGENTS.md). Read that file before answering questions about setup, running the benchmark, reproducing paper artifacts, or extending pipelines — it contains the flag grammar, ablation-script mapping, output layout, and guardrails.

For end-user prose, examples with badges, and the TL;DR, see [README.md](README.md).

---

## Fork addition: TabPFN multi-benchmark harness (`scripts/test/`)

This fork adds a harness that evaluates **TabPFN (cloud API)** across **STRABLE + CARTE + TextTabBench** from one runner. It lives entirely under [`scripts/test/`](scripts/test/) and is independent of the paper's main pipeline. Key files:

- `tabpfn_test.py` — runner: `--benchmark {strable,carte,ttb,all}`, `--workers N` (concurrent cloud calls), test-size sweep 0.1–0.9, resumable, token rotation.
- `bench_adapters.py` — unified loader for the 3 benchmarks; `bench_config.py` — params + `OVERLAP_GROUPS`; `_common.py` — metrics/features.
- `extract_bad_cases.py` — cross-benchmark bad-case ranking; `backfill_proba.py` — re-derive per-row `predict_proba`.
- `download_benchmarks.py` — fetch CARTE + TTB data; `requirements-tabpfn.txt` — the harness's light dependency set.

### Setup on a new machine (follow these steps in order)

Python **3.12**, `uv`, and `git` must be installed. Run everything from the repo root.

**1. Create the venv and install the harness dependencies** (the *light* set — the cloud client + sklearn/pandas; do **not** install `requirements-full.txt`, the harness never runs local models):

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV="$(pwd)/.venv" uv pip install -r scripts/test/requirements-tabpfn.txt
export STRABLE_ROOT="$(pwd)"          # all paths derive from this (configs/path_configs.py)
```

`uv pip install` is uv's pip-compatible mode — it does **not** create a `uv.lock` (that only appears with `uv lock`/`uv sync`). That's expected; the repo pins deps via requirements files, not a lockfile.

**2. Provide TabPFN cloud tokens** (required; never committed). Create `scripts/test/tokens.local`, one token per line — copy from `scripts/test/tokens.local.example`:

```bash
cp scripts/test/tokens.local.example scripts/test/tokens.local
# then edit it and paste the user's TabPFN token(s), one per line
# (alternatively: export TABPFN_TOKENS="tabpfn_sk_aaa,tabpfn_sk_bbb")
```

Ask the user for the token(s) — they are not in git. Verify they load:

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'scripts/test'); import bench_config as c; print('tokens loaded:', len(c.API_TOKENS))"
```

**3. Download the data** (all datasets are gitignored — re-fetch on every machine; ~1.4 GB total):

```bash
# STRABLE (108 tables) — HuggingFace -> data/data_processed/
.venv/bin/python data/download_datasets.py

# CARTE (51 tables, HF) + TextTabBench (5 Kaggle tables) -> data/CARTE_datasets/, data/TTB_datasets/raw/
.venv/bin/python scripts/test/download_benchmarks.py --benchmark all
```

TTB needs **Kaggle credentials**: ask the user, then `export KAGGLE_KEY=...` (and `KAGGLE_USERNAME=...` if it's a user/key pair) or place `~/.kaggle/kaggle.json`. If only STRABLE+CARTE are needed, run `--benchmark carte` and skip Kaggle.

**4. SOCKS-proxy gotcha (only if downloads fail).** If this machine routes through a SOCKS proxy (e.g. `all_proxy=socks5://127.0.0.1:7890`), HuggingFace downloads fail with `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`. `socksio` is already in `requirements-tabpfn.txt`, so step 1 fixes it; if it still fails, either confirm `socksio` installed or run the command with `all_proxy=` cleared to fall back to the HTTP proxy.

**5. Smoke test** (one tiny dataset, one fold — confirms tokens + data + API all work):

```bash
.venv/bin/python scripts/test/tabpfn_test.py --benchmark strable \
    --datasets miga-issued-projects --test-sizes 0.5
```

Expect one row appended to `results/tabpfn_test/summary.csv` and a per-row CSV under `results/tabpfn_test/per_row/strable/`. If that works, scale up:

```bash
# full sweep, concurrent (cloud calls are I/O-bound; resumable — re-run to continue)
.venv/bin/python scripts/test/tabpfn_test.py --benchmark all --workers 12
# then the cross-benchmark analysis
.venv/bin/python scripts/test/extract_bad_cases.py
```

### Gotchas worth knowing

- **Datasets and `results/per_row/` are gitignored** — only small result summaries are committed. Always re-download data on a fresh clone.
- **Resumable**: the runner skips `(benchmark, dataset, test_size)` triples already in `summary.csv`; if it dies, just re-run the same command.
- **TabPFN cloud is ~deterministic but not 100%** — `backfill_proba.py` guards against the occasional prediction drift (refuses to write misaligned probabilities).
- **The runner appends to one `summary.csv`** — don't run two sweeps against it concurrently.
