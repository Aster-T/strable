"""Backfill per-class probabilities into existing per-row CSVs.

The original sweep saved predictions but not ``predict_proba``.  Because every
split is seeded (``RANDOM_STATE``) and TabPFN-cloud inference is deterministic,
re-running a classification (dataset, test_size) reproduces the *exact* same
test rows in the same order — so the recomputed probabilities align row-for-row
with the already-stored predictions.  This script re-runs each classification
job and adds ``proba_<class>`` columns to its per-row CSV (only when the stored
predictions match, as an alignment guard).

It is idempotent: a per-row CSV that already has ``proba_*`` columns is skipped
unless ``--overwrite``.  Re-runs cost one cloud call each.

    python scripts/test/backfill_proba.py                 # all classification per-row CSVs
    python scripts/test/backfill_proba.py --datasets ramen-ratings yelp_business
    python scripts/test/backfill_proba.py --benchmark carte --workers 12
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import threading

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import bench_config as cfg
import _common as common
import bench_adapters as reg
import tabpfn_test as T

_LOCK = threading.Lock()


def _reproduce_split(spec, test_size, limits):
    """Reproduce run_one's split + subsample exactly -> (X_te, y_te, X_tr, y_tr, kind)."""
    rec = spec.load()
    kind = common.task_kind(rec.task)
    X, y = rec.X, rec.y
    stratify = y if (kind == "classification" and y.value_counts().min() >= 2) else None
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=cfg.RANDOM_STATE, stratify=stratify)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=cfg.RANDOM_STATE)
    max_train, max_test = T.effective_caps(X.shape[1], limits)
    X_tr, y_tr = T._subsample(X_tr, y_tr, max_train, cfg.RANDOM_STATE)
    X_te, y_te = T._subsample(X_te, y_te, max_test, cfg.RANDOM_STATE)
    return X_tr, y_tr, X_te, y_te, kind, rec.target


def backfill_one(spec, test_size, limits, overwrite, counters):
    name, bench = spec.name, spec.benchmark
    csv = cfg.PER_ROW_DIR / bench / name / f"classification_{test_size}.csv"
    if not csv.exists():
        return
    stored = pd.read_csv(csv)
    if any(c.startswith("proba_") for c in stored.columns) and not overwrite:
        with _LOCK:
            counters["skip"] += 1
        return
    try:
        X_tr, y_tr, X_te, y_te, kind, target = _reproduce_split(spec, test_size, limits)
        if kind != "classification":
            return
        model = T.build_model("tabpfn", kind)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        pred = np.asarray(model.predict(X_te)).astype(str)
        classes = np.asarray(model.classes_).astype(str)

        # alignment guards: same length + identical stored predictions
        if len(stored) != len(pred):
            raise RuntimeError(f"row mismatch {len(stored)} vs {len(pred)}")
        agree = (stored["predicted"].astype(str).to_numpy() == pred).mean()
        if agree < 0.999:
            raise RuntimeError(f"prediction disagreement {agree:.3f} (split drift)")

        out = stored.copy()
        for j, c in enumerate(classes):
            out[f"proba_{c}"] = proba[:, j]
        out.to_csv(csv, index=False)
        with _LOCK:
            counters["ok"] += 1
            print(f"  + {bench}:{name} ts={test_size}  ({len(classes)} classes, agree={agree:.3f})")
    except Exception as e:
        with _LOCK:
            counters["err"] += 1
            print(f"  !! {bench}:{name} ts={test_size}: {type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", choices=["strable", "carte", "ttb", "all"], default="all")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--test-sizes", nargs="*", type=float, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true", help="recompute even if proba_* exist")
    args = p.parse_args()

    os.environ.setdefault("TABPFN_CLIENT_CI_MODE", "true")
    benches = cfg.BENCHMARKS if args.benchmark == "all" else [args.benchmark]
    specs = reg.build_registry(benches)
    specs = [s for s in specs if common.task_kind(s.task) == "classification"]
    if args.datasets:
        specs = [s for s in specs if s.name in set(args.datasets)]
    test_sizes = args.test_sizes or cfg.TEST_SIZES

    rotator = T.TokenRotator(cfg.API_TOKENS)
    limits = T.fetch_server_limits()
    jobs = [(s, ts) for s in specs for ts in test_sizes]
    counters = {"ok": 0, "skip": 0, "err": 0}
    print(f"{len(specs)} classification datasets x {len(test_sizes)} = {len(jobs)} per-row CSVs "
          f"to backfill ({args.workers} workers)\n")

    with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        fs = [ex.submit(backfill_one, s, ts, limits, args.overwrite, counters)
              for (s, ts) in jobs]
        for f in futures.as_completed(fs):
            f.result()

    print(f"\ndone: {counters['ok']} backfilled, {counters['skip']} skipped, {counters['err']} errors")


if __name__ == "__main__":
    main()
