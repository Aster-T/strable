"""Evaluate SAP-RPT-1-OSS (sap-rpt-oss) on STRABLE/CARTE/TTB datasets, using the
EXACT same protocol as tabpfn_test.py so scores are directly comparable to the
TabPFN summary:

  - same bench_adapters loading (prepare_xy),
  - same train/test split (random_state=42, stratified for classification),
  - same sub-sample caps (MAX_TRAIN_ROWS / MAX_TRAIN_CELLS / MAX_TEST_ROWS),
  - same _common.compute_metrics.

Only the model differs (RPT instead of TabPFN cloud). Results go to a SEPARATE
results/rpt_test/ so the TabPFN files are never touched. Resumable: already-done
(benchmark, dataset, test_size) triples are skipped.

Run with the sap-rpt-oss venv (Python 3.11; has sap_rpt_oss + torch). Examples:
  # default: the 5 free-text-dominant STRABLE tables + ramen + clear
  python scripts/test/rpt_test.py
  # everything TabPFN covers (164 datasets) for a full head-to-head
  python scripts/test/rpt_test.py --benchmark all
  # one corpus
  python scripts/test/rpt_test.py --benchmark strable
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(os.environ.get("STRABLE_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(REPO / "scripts" / "test"))
sys.path.insert(0, str(REPO))

import bench_config as cfg          # noqa: E402
import bench_adapters as reg        # noqa: E402
import _common as common           # noqa: E402
from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor  # noqa: E402

# Default selection (used when neither --benchmark nor --datasets is given):
# the 5 free-text-dominant STRABLE tables + 2 user additions.
DEFAULT_DATASETS = [
    "managed-care-enrollment", "beer-ratings", "historical-earthquake-locations",
    "child-adult-healthcare-quality", "grant", "ramen-ratings", "clear-corpus",
]
MODEL_NAME = "sap-rpt-oss"

OUT = REPO / "results" / "rpt_test"
SUMMARY = OUT / "summary.csv"
PER_ROW = OUT / "per_row"

_META = ["benchmark", "dataset", "overlap_group", "task", "kind", "test_size", "model",
         "n_total", "n_train_used", "n_test_used", "n_features", "target_card",
         "fit_seconds", "predict_seconds", "status", "error"]
_METRIC = ["accuracy", "balanced_acc", "macro_f1", "auroc", "majority_acc", "acc_lift",
           "r2", "rmse", "mae", "nrmse"]

# Structural limits RPT cannot handle (record + skip, never crash the sweep).
RPT_MAX_COLS = 500       # SAP_RPT_OSS_Estimator.MAX_NUM_COLUMNS
RPT_MAX_CLASSES = 160    # parity with TabPFN's server cap; huge label sets are infeasible


def effective_caps(n_features):
    """Identical to tabpfn_test.effective_caps with limits=None (cfg caps only)."""
    max_train, cells, max_test = cfg.MAX_TRAIN_ROWS, cfg.MAX_TRAIN_CELLS, cfg.MAX_TEST_ROWS
    if n_features > 0:
        max_train = min(max_train, max(1, cells // n_features))
    return max(1, max_train), max(1, max_test)


def _subsample(X, y, n, seed):
    if len(X) <= n:
        return X, y
    Xs = X.sample(n=n, random_state=seed)
    return Xs, y.loc[Xs.index]


def done_set():
    if not SUMMARY.exists():
        return set()
    d = pd.read_csv(SUMMARY)
    if "benchmark" not in d:
        d["benchmark"] = "strable"
    return {(str(r.benchmark), str(r.dataset), float(r.test_size)) for r in d.itertuples()}


def append(row):
    OUT.mkdir(parents=True, exist_ok=True)
    cols = _META + _METRIC
    df = pd.DataFrame([{c: row.get(c, "") for c in cols}])
    df.to_csv(SUMMARY, mode="a", header=not SUMMARY.exists(), index=False)


def write_per_row(benchmark, name, kind, ts, X_te, y_te, y_pred, target, y_proba=None, classes=None):
    d = PER_ROW / benchmark / name
    d.mkdir(parents=True, exist_ok=True)
    res = X_te.copy()
    res[target] = y_te.to_numpy()
    res["predicted"] = list(y_pred)
    if kind == "classification":
        res["correct"] = res[target].astype(str).to_numpy() == pd.Series(y_pred).astype(str).to_numpy()
        if y_proba is not None and classes is not None:
            for j, c in enumerate(np.asarray(classes).astype(str)):
                res[f"proba_{c}"] = y_proba[:, j]
    else:
        res["error"] = (pd.to_numeric(res["predicted"], errors="coerce").to_numpy()
                        - pd.to_numeric(res[target], errors="coerce").to_numpy())
    res.to_csv(d / f"{kind}_{ts}.csv", index=False)


def select_specs(args):
    """Resolve which DatasetSpecs to run from --benchmark / --datasets."""
    if args.benchmark:
        benches = cfg.BENCHMARKS if args.benchmark == "all" else [args.benchmark]
    else:
        benches = cfg.BENCHMARKS if args.datasets else ["strable"]
    specs = reg.build_registry(benches)
    if args.datasets:
        want = set(args.datasets)
        specs = [s for s in specs if s.name in want]
        missing = want - {s.name for s in specs}
        if missing:
            print("WARNING missing datasets:", sorted(missing))
    elif not args.benchmark:
        specs = [s for s in specs if s.name in set(DEFAULT_DATASETS)]
    return specs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["strable", "carte", "ttb", "all"], default=None,
                   help="run a whole corpus (or all). Omit to use --datasets or the default set.")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--test-sizes", nargs="*", type=float, default=cfg.TEST_SIZES)
    p.add_argument("--max-context", type=int, default=2048)
    p.add_argument("--bagging", default="auto")
    args = p.parse_args()

    bag = args.bagging
    if isinstance(bag, str) and bag.isdigit():
        bag = int(bag)

    import torch
    print(f"device: {'cuda:'+torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} | "
          f"max_context={args.max_context} bagging={bag}")

    specs = select_specs(args)
    test_sizes = args.test_sizes
    done = done_set()
    models = {}  # kind -> reused model (avoids reloading weights every run)

    def get_model(kind):
        if kind not in models:
            cls = SAP_RPT_OSS_Classifier if kind == "classification" else SAP_RPT_OSS_Regressor
            models[kind] = cls(max_context_size=args.max_context, bagging=bag)
        return models[kind]

    total = len(specs) * len(test_sizes)
    n_bench = len({s.benchmark for s in specs})
    print(f"{len(specs)} datasets ({n_bench} benchmark(s)) x {len(test_sizes)} test sizes = {total} runs")
    i = n_ok = n_skip = n_err = 0
    for spec in specs:
        rec = None
        for ts in test_sizes:
            i += 1
            key = (spec.benchmark, spec.name, float(ts))
            tag = f"[{i}/{total}] {spec.benchmark}:{spec.name} ts={ts}"
            if key in done:
                n_skip += 1
                continue
            if rec is None:                      # lazy-load once per dataset
                try:
                    rec = spec.load()
                except Exception as e:
                    print(tag, "LOAD ERROR", type(e).__name__, str(e)[:160])
                    break
            kind = common.task_kind(rec.task)
            X, y, target = rec.X, rec.y, rec.target
            row = {"benchmark": spec.benchmark, "dataset": spec.name,
                   "overlap_group": rec.overlap_group or "", "task": rec.task, "kind": kind,
                   "test_size": ts, "model": MODEL_NAME, "n_total": len(X),
                   "n_features": X.shape[1], "target_card": int(pd.Series(y).nunique()),
                   "status": "ok", "error": ""}
            # Structural skips (parity with TabPFN's SkipRun): record + move on.
            if X.shape[1] > RPT_MAX_COLS:
                row["status"] = "skipped"; row["error"] = f"{X.shape[1]} cols > RPT_MAX_COLS {RPT_MAX_COLS}"
                append(row); n_skip += 1; print(tag, "skip", row["error"]); continue
            if kind == "classification" and y.nunique() > RPT_MAX_CLASSES:
                row["status"] = "skipped"; row["error"] = f"{y.nunique()} classes > {RPT_MAX_CLASSES}"
                append(row); n_skip += 1; print(tag, "skip", row["error"]); continue

            stratify = y if (kind == "classification" and y.value_counts().min() >= 2) else None
            try:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=ts, random_state=cfg.RANDOM_STATE, stratify=stratify)
            except ValueError:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=ts, random_state=cfg.RANDOM_STATE)
            mtr, mte = effective_caps(X.shape[1])
            Xtr, ytr = _subsample(Xtr, ytr, mtr, cfg.RANDOM_STATE)
            Xte, yte = _subsample(Xte, yte, mte, cfg.RANDOM_STATE)
            row["n_train_used"], row["n_test_used"] = len(Xtr), len(Xte)
            try:
                model = get_model(kind)
                t = time.time(); model.fit(Xtr, ytr); row["fit_seconds"] = round(time.time() - t, 2)
                t = time.time()
                ypred = model.predict(Xte)
                yproba = classes = None
                if kind == "classification":
                    try:
                        yproba = model.predict_proba(Xte)
                        classes = model.classes_
                    except Exception:
                        pass
                row["predict_seconds"] = round(time.time() - t, 2)
                row.update(common.compute_metrics(kind, yte, ypred, yproba, classes))
                write_per_row(spec.benchmark, spec.name, kind, ts, Xte, yte, ypred, target, yproba, classes)
                append(row); n_ok += 1
                if kind == "classification":
                    print(tag, f"acc={row.get('accuracy'):.3f} auroc={row.get('auroc')}")
                else:
                    print(tag, f"r2={row.get('r2'):.3f} rmse={row.get('rmse'):.3f}")
            except Exception as e:
                models.pop(kind, None)  # reset reused model in case its state was corrupted
                row["status"] = "error"; row["error"] = f"{type(e).__name__}: {e}"[:300]
                append(row); n_err += 1
                print(tag, "ERROR", type(e).__name__, str(e)[:200])

    print(f"\nDONE: {n_ok} ok, {n_skip} skipped, {n_err} errors (of {total})")


if __name__ == "__main__":
    main()
