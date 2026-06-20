"""Benchmark TabPFN (cloud API) across the STRABLE datasets.

For every dataset under ``data/data_processed/`` and every test fraction in
``bench_config.TEST_SIZES`` (0.1 … 0.9) this:

  1. loads ``data.parquet`` + ``config.json``,
  2. splits train/test (stratified for classification when possible),
  3. sub-samples the training split to respect TabPFN-cloud limits and caps the
     test split for speed,
  4. fits TabPFN, predicts, and computes task-appropriate metrics,
  5. writes one row to ``results/tabpfn_test/summary.csv`` and (optionally) a
     per-row prediction CSV under ``results/tabpfn_test/per_row/``.

Robustness features:
  * two free-tier tokens are rotated automatically; when one is rate-limited the
    runner falls back to the next,
  * the run is *resumable* — already-completed (dataset, test_size) pairs are
    skipped (``SKIP_EXISTING``),
  * the summary CSV is appended after every run, so progress is never lost.

Examples
--------
    # everything (108 datasets x 9 test sizes)
    python scripts/test/tabpfn_test.py

    # a smoke test on one small dataset / one fold
    python scripts/test/tabpfn_test.py --datasets meta-critic_whisky --test-sizes 0.5

    # only the classification tasks, overwriting existing results
    python scripts/test/tabpfn_test.py --tasks classification --overwrite
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tabpfn_client
from sklearn.model_selection import train_test_split
from tabpfn_client import TabPFNClassifier, TabPFNRegressor

import bench_config as cfg
import _common as common
import bench_adapters as reg

# ===== add a new cloud model only here: name -> (classifier, regressor) =====
MODEL_REGISTRY = {
    "tabpfn": (TabPFNClassifier, TabPFNRegressor),
}

# Summary columns kept in a stable order (metrics are merged in dynamically).
_META_COLS = [
    "benchmark", "dataset", "overlap_group", "task", "kind", "test_size", "model",
    "n_total", "n_train_used", "n_test_used", "n_features", "target_card",
    "fit_seconds", "predict_seconds", "token_index", "status", "error",
]
_METRIC_COLS = [
    "accuracy", "balanced_acc", "macro_f1", "auroc", "majority_acc", "acc_lift",
    "r2", "rmse", "mae", "nrmse",
]


# --------------------------------------------------------------------------- #
# TabPFN-cloud plumbing
# --------------------------------------------------------------------------- #
def fetch_server_limits():
    """Best-effort live model limits; None if unavailable (offline / pre-login)."""
    if not cfg.FETCH_SERVER_LIMITS:
        return None
    try:
        from tabpfn_client.service_wrapper import ServiceClient

        resp = ServiceClient.get_model_limits()
        return resp.max_model_limit if resp else None
    except Exception as e:  # network, auth, API change — degrade gracefully
        print(f"  (could not fetch server limits: {e!r}; using config fallbacks)")
        return None


def is_quota_error(exc: Exception) -> bool:
    """Does this exception look like a *usage/rate-limit* refusal (not a data error)?

    Deliberately narrow: data-shape rejections ("number of features exceeds the
    limit", "exceeds the maximum number of classes") also contain words like
    'limit'/'exceeded', so matching those would wrongly rotate tokens and abort
    the whole run.  We only treat unambiguous quota/rate phrasing as exhaustion.
    """
    msg = str(exc).lower()
    needles = (
        "quota", "payment", "upgrade your", "out of credits", "credits",
        "usage limit", "daily limit", "monthly limit", "insufficient",
    )
    return any(n in msg for n in needles)


def is_transient_error(exc: Exception) -> bool:
    """Rate-limit / network blips: back off and retry the SAME job (don't rotate)."""
    msg = str(exc).lower()
    needles = (
        "429", "too many requests", "rate limit", "ratelimit", "timeout",
        "timed out", "temporarily", "connection", "network", "reset by peer",
        "503", "502", "500", "unavailable", "try again", "bad gateway",
    )
    return any(n in msg for n in needles)


class SkipRun(Exception):
    """Raised to skip a dataset for a structural reason (not an API failure)."""


class TokenRotator:
    """Round-robin over the configured tokens, skipping exhausted ones."""

    def __init__(self, tokens: list[str]):
        if not tokens:
            raise SystemExit("No API tokens configured (see bench_config.API_TOKENS).")
        self.tokens = list(tokens)
        self.idx = 0
        self.exhausted: set[int] = set()
        self._activate(self.idx)

    def _activate(self, i: int) -> None:
        tabpfn_client.set_access_token(self.tokens[i])
        self.idx = i

    def rotate(self) -> bool:
        """Mark the current token exhausted and switch to the next live one.

        Returns False when every token is exhausted.
        """
        self.exhausted.add(self.idx)
        for step in range(1, len(self.tokens) + 1):
            cand = (self.idx + step) % len(self.tokens)
            if cand not in self.exhausted:
                self._activate(cand)
                print(f"  -> rotating to token #{cand}")
                return True
        return False


# --------------------------------------------------------------------------- #
# Sub-sampling
# --------------------------------------------------------------------------- #
def effective_caps(n_features: int, limits) -> tuple[int, int]:
    """Resolve (max_train_rows, max_test_rows) from config + live server limits."""
    max_train = cfg.MAX_TRAIN_ROWS
    cells_cap = cfg.MAX_TRAIN_CELLS
    max_test = cfg.MAX_TEST_ROWS
    if limits is not None:
        max_train = min(max_train, getattr(limits, "train_set_max_rows", max_train))
        cells_cap = min(cells_cap, getattr(limits, "train_set_max_cells", cells_cap))
        max_test = min(max_test, getattr(limits, "test_set_max_rows", max_test))
    # rows*cols must stay under the cell budget
    if n_features > 0:
        max_train = min(max_train, max(1, cells_cap // n_features))
    return max(1, max_train), max(1, max_test)


def _subsample(X, y, n, seed):
    if len(X) <= n:
        return X, y
    Xs = X.sample(n=n, random_state=seed)
    return Xs, y.loc[Xs.index]


# --------------------------------------------------------------------------- #
# One (dataset, test_size) run
# --------------------------------------------------------------------------- #
def build_model(model_name: str, kind: str):
    clf_cls, reg_cls = MODEL_REGISTRY[model_name]
    return clf_cls() if kind == "classification" else reg_cls()


def run_one(spec: "reg.DatasetSpec", test_size: float, model_name: str, limits) -> dict:
    rec = spec.load()                       # benchmark-specific load -> DatasetRecord
    kind = common.task_kind(rec.task)

    row = {
        "benchmark": rec.benchmark, "dataset": rec.name,
        "overlap_group": rec.overlap_group or "", "task": rec.task, "kind": kind,
        "test_size": test_size, "model": model_name, "status": "ok", "error": "",
    }

    # X/y are already cleaned (reset index, NaN-target dropped, regression cast)
    # inside the adapter's prepare_xy, so identical handling across benchmarks.
    X, y, target = rec.X, rec.y, rec.target
    row["n_total"] = len(X)
    row["n_features"] = X.shape[1]
    row["target_card"] = int(y.nunique())

    # Structural limits the server enforces but sub-sampling can't fix: skip
    # cleanly (recorded, never retried) rather than letting the API reject them.
    if limits is not None:
        if X.shape[1] > getattr(limits, "max_cols", math.inf):
            raise SkipRun(f"{X.shape[1]} cols > server max_cols {limits.max_cols}")
        if kind == "classification" and y.nunique() > getattr(limits, "max_classes", math.inf):
            raise SkipRun(f"{y.nunique()} classes > server max_classes {limits.max_classes}")

    # Stratify classification splits when every class has >= 2 members.
    stratify = None
    if kind == "classification" and y.value_counts().min() >= 2:
        stratify = y
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=cfg.RANDOM_STATE, stratify=stratify
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=cfg.RANDOM_STATE
        )

    max_train, max_test = effective_caps(X.shape[1], limits)
    X_tr, y_tr = _subsample(X_tr, y_tr, max_train, cfg.RANDOM_STATE)
    X_te, y_te = _subsample(X_te, y_te, max_test, cfg.RANDOM_STATE)
    row["n_train_used"], row["n_test_used"] = len(X_tr), len(X_te)

    model = build_model(model_name, kind)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    row["fit_seconds"] = round(time.time() - t0, 2)

    t0 = time.time()
    y_pred = model.predict(X_te)
    row["predict_seconds"] = round(time.time() - t0, 2)

    y_proba = None
    classes = None
    if kind == "classification":
        try:
            y_proba = model.predict_proba(X_te)
            classes = model.classes_
        except Exception:
            pass

    row.update(common.compute_metrics(kind, y_te, y_pred, y_proba, classes))

    if cfg.WRITE_PER_ROW:
        _write_per_row(rec.benchmark, rec.name, kind, test_size, X_te, y_te, y_pred,
                       target, y_proba, classes)
    return row


def _write_per_row(benchmark, name, kind, test_size, X_te, y_te, y_pred, target,
                   y_proba=None, classes=None):
    out_dir = cfg.PER_ROW_DIR / benchmark / name      # namespaced: same name can exist in >1 benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    res = X_te.copy()
    res[target] = y_te.to_numpy()
    res["predicted"] = y_pred
    if kind == "classification":
        res["correct"] = res[target].astype(str).to_numpy() == pd.Series(y_pred).astype(str).to_numpy()
        # Persist per-class probabilities (column-aligned to the model's classes)
        # so AUROC / calibration can be re-derived later without re-running.
        if y_proba is not None and classes is not None:
            for j, c in enumerate(np.asarray(classes).astype(str)):
                res[f"proba_{c}"] = y_proba[:, j]
    else:
        res["error"] = pd.to_numeric(res["predicted"], errors="coerce").to_numpy() - res[target].to_numpy()
    res.to_csv(out_dir / f"{kind}_{test_size}.csv", index=False)


# --------------------------------------------------------------------------- #
# Summary I/O (resumable, append-after-each-run)
# --------------------------------------------------------------------------- #
def load_done_pairs() -> set[tuple[str, str, float]]:
    if not cfg.SUMMARY_CSV.exists():
        return set()
    df = pd.read_csv(cfg.SUMMARY_CSV)
    if "benchmark" not in df.columns:        # legacy summary predates multi-benchmark
        df["benchmark"] = "strable"
    df = df[df["status"].isin(["ok", "skipped"])] if "status" in df else df
    return {(r.benchmark, r.dataset, float(r.test_size)) for r in df.itertuples()}


def _migrate_summary_if_legacy(cols: list[str]) -> None:
    """Backfill a pre-multi-benchmark summary.csv to the new schema, once."""
    if not cfg.SUMMARY_CSV.exists():
        return
    head = pd.read_csv(cfg.SUMMARY_CSV, nrows=0)
    if "benchmark" in head.columns:
        return
    old = pd.read_csv(cfg.SUMMARY_CSV)
    old.insert(0, "benchmark", "strable")
    if "overlap_group" not in old.columns:
        old["overlap_group"] = ""
    old.reindex(columns=cols).to_csv(cfg.SUMMARY_CSV, index=False)


def append_summary(row: dict) -> None:
    cfg.SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = _META_COLS + _METRIC_COLS
    _migrate_summary_if_legacy(cols)
    df = pd.DataFrame([{c: row.get(c, "") for c in cols}])
    header = not cfg.SUMMARY_CSV.exists()
    df.to_csv(cfg.SUMMARY_CSV, mode="a", header=header, index=False)


# --------------------------------------------------------------------------- #
# Concurrent execution (TabPFN cloud calls are I/O-bound -> threads scale well)
# --------------------------------------------------------------------------- #
_IO_LOCK = threading.Lock()


def _emit(row: dict, counters: dict, key: str, msg: str = "") -> None:
    """Thread-safe: append one summary row, bump a counter, print a line."""
    with _IO_LOCK:
        append_summary(row)
        counters[key] += 1
        if msg:
            print(msg)
        elif key == "ok":
            _print_result(row)


def _run_job(spec, ts, model_name, limits, rotator, counters, retries=4):
    """One (dataset, test_size) job with retry/backoff; safe to call from a thread."""
    for attempt in range(retries):
        try:
            row = run_one(spec, ts, model_name, limits)
            row["token_index"] = rotator.idx
            _emit(row, counters, "ok")
            return
        except SkipRun as e:
            _emit({"benchmark": spec.benchmark, "dataset": spec.name,
                   "overlap_group": spec.overlap_group or "", "test_size": ts,
                   "model": model_name, "status": "skipped", "error": str(e)[:300],
                   "token_index": rotator.idx},
                  counters, "skip", msg=f"  (skip {spec.benchmark}:{spec.name} ts={ts}: {e})")
            return
        except Exception as e:
            if is_quota_error(e):
                with _IO_LOCK:
                    rotated = rotator.rotate()
                if rotated:
                    continue                       # retry same job on a fresh token
            if is_transient_error(e) and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))      # linear backoff: 5s, 10s, 15s
                continue
            _emit({"benchmark": spec.benchmark, "dataset": spec.name,
                   "overlap_group": spec.overlap_group or "", "task": "", "kind": "",
                   "test_size": ts, "model": model_name, "status": "error",
                   "error": f"{type(e).__name__}: {e}"[:300], "token_index": rotator.idx},
                  counters, "err",
                  msg=f"  !! ERROR {spec.benchmark}:{spec.name} ts={ts}: {type(e).__name__}: {e}")
            return


# --------------------------------------------------------------------------- #
# CLI / main loop
# --------------------------------------------------------------------------- #
def select_datasets(args) -> list["reg.DatasetSpec"]:
    benches = cfg.BENCHMARKS if args.benchmark in (None, "all") else [args.benchmark]
    specs = reg.build_registry(benches)
    if args.datasets:
        wanted = set(args.datasets)
        specs = [s for s in specs if s.name in wanted]
        missing = wanted - {s.name for s in specs}
        if missing:
            print(f"warning: unknown dataset(s): {sorted(missing)}")
    if args.tasks:
        keep = set(args.tasks)
        specs = [s for s in specs if common.task_kind(s.task) in keep]
    return specs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="dataset folder names; default = all")
    p.add_argument("--test-sizes", nargs="*", type=float, default=None,
                   help=f"override test sizes; default = {cfg.TEST_SIZES}")
    p.add_argument("--tasks", nargs="*", default=None,
                   choices=["classification", "regression"],
                   help="restrict to a task kind")
    p.add_argument("--benchmark", choices=["strable", "carte", "ttb", "all"],
                   default="strable",
                   help="which corpus to evaluate (default keeps the legacy STRABLE run)")
    p.add_argument("--model", default="tabpfn", choices=list(MODEL_REGISTRY))
    p.add_argument("--workers", type=int, default=1,
                   help="concurrent cloud requests (cloud calls are I/O-bound; "
                        "8-16 gives a near-linear speedup). Default 1 = serial.")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even if a result already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="only the first selected dataset, test_size=0.5")
    p.add_argument("--list", action="store_true",
                   help="list the selected datasets and exit")
    args = p.parse_args()

    specs = select_datasets(args)
    if args.list:
        for s in specs:
            grp = f"  [overlap:{s.overlap_group}]" if s.overlap_group else ""
            print(f"{s.benchmark:8s} {s.name}{grp}")
        print(f"\n{len(specs)} dataset(s).")
        return
    if not specs:
        raise SystemExit("No datasets selected.")

    test_sizes = args.test_sizes or cfg.TEST_SIZES
    if args.dry_run:
        specs = specs[:1]
        test_sizes = [0.5]

    skip_existing = cfg.SKIP_EXISTING and not args.overwrite
    done = load_done_pairs() if skip_existing else set()

    rotator = TokenRotator(cfg.API_TOKENS)
    print(f"Usage check: {_safe_usage()}")
    limits = fetch_server_limits()
    if limits is not None:
        g = lambda k: getattr(limits, k, "?")
        print(f"Server limits: train<= {g('train_set_max_rows')} rows / "
              f"{g('train_set_max_cells')} cells, test<= {g('test_set_max_rows')} rows, "
              f"max_classes={g('max_classes')}, max_cols={g('max_cols')}")

    jobs = [(s, ts) for s in specs for ts in test_sizes]
    total = len(jobs)
    n_ok = n_skip = n_err = 0
    n_bench = len({s.benchmark for s in specs})
    print(f"\n{len(specs)} datasets ({n_bench} benchmark(s)) x {len(test_sizes)} test sizes = {total} runs\n")

    # ---- concurrent path: TabPFN cloud calls are I/O-bound, so threads scale ----
    if args.workers and args.workers > 1:
        os.environ.setdefault("TABPFN_CLIENT_CI_MODE", "true")   # silence per-call spinner
        pending = [(s, ts) for (s, ts) in jobs
                   if not (skip_existing and (s.benchmark, s.name, float(ts)) in done)]
        counters = {"ok": 0, "skip": total - len(pending), "err": 0}
        print(f"concurrent: {args.workers} workers, {len(pending)} pending "
              f"({counters['skip']} already done)\n")
        with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run_job, s, ts, args.model, limits, rotator, counters)
                    for (s, ts) in pending]
            for n, f in enumerate(futures.as_completed(futs), 1):
                f.result()                                       # errors handled inside _run_job
                if n % 20 == 0 or n == len(pending):
                    with _IO_LOCK:
                        print(f"  --- {n}/{len(pending)} complete "
                              f"({counters['ok']} ok, {counters['err']} err) ---")
        _final_report(counters["ok"], counters["skip"], counters["err"], total)
        return

    for i, (spec, ts) in enumerate(jobs, 1):
        tag = f"[{i}/{total}] {spec.benchmark}:{spec.name} ts={ts}"
        if skip_existing and (spec.benchmark, spec.name, float(ts)) in done:
            n_skip += 1
            print(f"{tag}  (skip, already done)")
            continue

        while True:  # retry loop for token rotation
            try:
                print(f"{tag}  running...")
                row = run_one(spec, ts, args.model, limits)
                row["token_index"] = rotator.idx
                append_summary(row)
                n_ok += 1
                _print_result(row)
                break
            except SkipRun as e:
                n_skip += 1
                append_summary({
                    "benchmark": spec.benchmark, "dataset": spec.name,
                    "overlap_group": spec.overlap_group or "",
                    "test_size": ts, "model": args.model,
                    "status": "skipped", "error": str(e)[:300], "token_index": rotator.idx,
                })
                print(f"  (skip: {e})")
                break
            except Exception as e:
                if is_quota_error(e) and rotator.rotate():
                    print(f"  quota/limit on token; retrying. ({e})")
                    continue  # same job, next token
                n_err += 1
                err_row = {
                    "benchmark": spec.benchmark, "dataset": spec.name,
                    "overlap_group": spec.overlap_group or "", "task": "", "kind": "",
                    "test_size": ts, "model": args.model, "status": "error",
                    "error": f"{type(e).__name__}: {e}"[:300],
                    "token_index": rotator.idx,
                }
                append_summary(err_row)
                print(f"  !! ERROR: {type(e).__name__}: {e}")
                if is_quota_error(e):
                    print("  All tokens exhausted — stopping. Re-run later to resume.")
                    _final_report(n_ok, n_skip, n_err, total)
                    return
                break

    _final_report(n_ok, n_skip, n_err, total)


def _print_result(row: dict) -> None:
    if row["kind"] == "classification":
        print(f"   acc={row.get('accuracy'):.3f} auroc={_fmt(row.get('auroc'))} "
              f"lift={row.get('acc_lift'):+.3f}  "
              f"(train {row['n_train_used']}, test {row['n_test_used']}, "
              f"{row['fit_seconds']}+{row['predict_seconds']}s)")
    else:
        print(f"   r2={_fmt(row.get('r2'))} rmse={_fmt(row.get('rmse'))}  "
              f"(train {row['n_train_used']}, test {row['n_test_used']}, "
              f"{row['fit_seconds']}+{row['predict_seconds']}s)")


def _fmt(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "nan"


def _safe_usage() -> str:
    try:
        return tabpfn_client.get_api_usage()
    except Exception as e:
        return f"(unavailable: {e})"


def _final_report(n_ok, n_skip, n_err, total):
    print("\n" + "-" * 50)
    print(f"done: {n_ok} ok, {n_skip} skipped, {n_err} errors  (of {total})")
    print(f"summary -> {cfg.SUMMARY_CSV}")


if __name__ == "__main__":
    main()
