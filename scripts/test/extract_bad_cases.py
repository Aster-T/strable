"""Find the datasets where TabPFN does poorly — and why.

Reads ``results/tabpfn_test/summary.csv`` (produced by ``tabpfn_test.py``),
ranks datasets by a task-appropriate quality score, flags the "bad cases", and
correlates performance with structural properties of each table (size, string
cardinality, free-text content, missingness, …) to surface *patterns*.

Outputs
-------
  results/tabpfn_test/dataset_features.csv     structural fingerprint per dataset
  results/tabpfn_test/bad_cases_per_dataset.csv  per-dataset scores + features + is_bad
  results/tabpfn_test/bad_cases_report.txt     the human-readable report (also printed)

Quality score
-------------
  regression     -> R^2          (higher better; <0 means worse than the mean)
  classification -> AUROC        (falls back to accuracy when AUROC is absent)
A dataset is "bad" when its median quality across test sizes is below the
threshold in bench_config (REG_BAD_R2 / CLS_BAD_AUROC / CLS_BAD_ACC_LIFT).

    python scripts/test/extract_bad_cases.py            # full report
    python scripts/test/extract_bad_cases.py --top 15   # show 15 worst per kind
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import bench_config as cfg
import _common as common
import bench_adapters as reg

# Empty-slice means are expected (e.g. a kind with no "bad" datasets yet).
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Structural features we correlate against quality.
_FEATURE_COLS = [
    "n_rows", "n_features", "n_object_cols", "n_numeric_cols", "frac_object",
    "frac_missing", "target_card", "target_imbalance",
    "mean_card_ratio", "max_card_ratio", "frac_highcard_cols",
    "mean_str_tokens", "frac_freetext_cols", "mean_str_chars", "frac_datetime_cols",
]


# --------------------------------------------------------------------------- #
# Load + score
# --------------------------------------------------------------------------- #
def load_summary() -> pd.DataFrame:
    if not cfg.SUMMARY_CSV.exists():
        raise SystemExit(f"No summary at {cfg.SUMMARY_CSV}. Run tabpfn_test.py first.")
    df = pd.read_csv(cfg.SUMMARY_CSV)
    if "benchmark" not in df.columns:        # legacy single-benchmark summary
        df["benchmark"] = "strable"
    if "overlap_group" not in df.columns:
        df["overlap_group"] = ""
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise SystemExit("summary.csv has no successful runs yet.")
    # De-dup re-runs (e.g. --overwrite appends): keep the most recent row.
    df = df.drop_duplicates(subset=["benchmark", "dataset", "test_size"], keep="last")
    # Per-run quality is a SINGLE metric per task kind — never mix AUROC and
    # accuracy within one dataset's aggregate (the accuracy fall-back is applied
    # per-dataset later, only when AUROC is absent for every fold).
    auroc = pd.to_numeric(df.get("auroc"), errors="coerce")
    r2 = pd.to_numeric(df.get("r2"), errors="coerce")
    df["quality"] = np.where(df["kind"] == "classification", auroc, r2)
    df["accuracy"] = pd.to_numeric(df.get("accuracy"), errors="coerce")
    df["train_fraction"] = 1.0 - df["test_size"]
    return df


def aggregate_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (benchmark, name), g in df.groupby(["benchmark", "dataset"]):
        g = g.sort_values("test_size")
        q = g["quality"].astype(float)
        # Classification dataset with no AUROC anywhere -> fall back to accuracy
        # (whole-dataset, so the aggregate is never a mix of two metrics).
        if g["kind"].iloc[0] == "classification" and q.isna().all():
            q = g["accuracy"].astype(float)
            g = g.assign(quality=q)
        at_05 = g.loc[g["test_size"] == 0.5, "quality"]
        # slope of quality vs train fraction (positive = more data helps)
        slope = np.nan
        if g["train_fraction"].nunique() >= 2:
            slope = float(np.polyfit(g["train_fraction"], q.fillna(q.mean()), 1)[0])
        # Accuracy aggregates (classification only; NaN for regression).
        is_cls = g["kind"].iloc[0] == "classification"
        acc = pd.to_numeric(g.get("accuracy"), errors="coerce") if is_cls else pd.Series(dtype=float)
        bacc = pd.to_numeric(g.get("balanced_acc"), errors="coerce") if is_cls else pd.Series(dtype=float)
        acc_05 = g.loc[g["test_size"] == 0.5, "accuracy"] if is_cls and "accuracy" in g else pd.Series(dtype=float)
        rows.append({
            "benchmark": benchmark,
            "dataset": name,
            "overlap_group": g["overlap_group"].iloc[0] if "overlap_group" in g else "",
            "kind": g["kind"].iloc[0],
            "task": g["task"].iloc[0],
            "n_runs": len(g),
            "quality_median": float(q.median()),
            "quality_mean": float(q.mean()),
            "quality_min": float(q.min()),
            "quality_max": float(q.max()),
            "quality_at_0.5": float(at_05.iloc[0]) if len(at_05) else float("nan"),
            "quality_slope_vs_trainfrac": slope,
            # accuracy alongside AUROC for classification (NaN for regression)
            "acc_median": float(acc.median()) if len(acc) else float("nan"),
            "balanced_acc_median": float(bacc.median()) if len(bacc) else float("nan"),
            "acc_at_0.5": float(pd.to_numeric(acc_05, errors="coerce").iloc[0]) if len(acc_05) else float("nan"),
            "acc_lift_median": float(pd.to_numeric(g.get("acc_lift"), errors="coerce").median()),
        })
    return pd.DataFrame(rows)


def flag_bad(agg: pd.DataFrame) -> pd.DataFrame:
    def _bad(r):
        if r["kind"] == "regression":
            return r["quality_median"] < cfg.REG_BAD_R2
        return (r["quality_median"] < cfg.CLS_BAD_AUROC) or (
            pd.notna(r["acc_lift_median"]) and r["acc_lift_median"] < cfg.CLS_BAD_ACC_LIFT
        )
    agg = agg.copy()
    agg["is_bad"] = agg.apply(_bad, axis=1)
    return agg


# --------------------------------------------------------------------------- #
# Structural features
# --------------------------------------------------------------------------- #
def build_features(pairs: list[tuple[str, str]], refresh: bool) -> pd.DataFrame:
    """Structural features for each (benchmark, dataset), via the unified registry."""
    cache = cfg.RESULTS_ROOT / "dataset_features.csv"
    if cache.exists() and not refresh:
        cached = pd.read_csv(cache)
        if "benchmark" in cached.columns and set(pairs).issubset(
            set(zip(cached["benchmark"], cached["dataset"]))
        ):
            return cached
    specs = {(s.benchmark, s.name): s for s in reg.build_registry(cfg.BENCHMARKS)}
    rows = []
    for (benchmark, name) in pairs:
        s = specs.get((benchmark, name))
        if s is None:
            continue
        try:
            rec = s.load()
            df = rec.X.copy()
            df[rec.target] = rec.y.values          # reattach target for dataset_features
            feat = common.dataset_features(df, {"target_name": rec.target, "task": rec.task})
            rows.append({"benchmark": benchmark, "dataset": name, **feat})
        except Exception:
            continue
    out = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


# --------------------------------------------------------------------------- #
# Pattern finding
# --------------------------------------------------------------------------- #
def correlations(merged: pd.DataFrame, kind: str) -> pd.Series:
    sub = merged[merged["kind"] == kind]
    cols = [c for c in _FEATURE_COLS if c in sub and sub[c].nunique() > 1]
    if len(sub) < 4 or not cols:
        return pd.Series(dtype=float)
    corr = sub[cols + ["quality_median"]].corr(method="spearman")["quality_median"]
    return corr.drop("quality_median").sort_values()


def group_compare(merged: pd.DataFrame, kind: str) -> pd.DataFrame:
    sub = merged[merged["kind"] == kind]
    cols = [c for c in _FEATURE_COLS if c in sub]
    if sub["is_bad"].nunique() < 2:
        return pd.DataFrame()
    good = sub[~sub["is_bad"]][cols].mean()
    bad = sub[sub["is_bad"]][cols].mean()
    return pd.DataFrame({"bad_mean": bad, "good_mean": good,
                         "ratio_bad_over_good": bad / good.replace(0, np.nan)})


# --------------------------------------------------------------------------- #
# Drill-down into the worst rows
# --------------------------------------------------------------------------- #
def drill_examples(benchmark: str, name: str, kind: str, n: int = 3) -> str:
    folder = cfg.PER_ROW_DIR / benchmark / name
    files = sorted(folder.glob(f"{kind}_*.csv")) if folder.exists() else []
    if not files:
        return "    (no per-row CSV)"
    # pick the run with the most rows (usually highest test_size)
    df = pd.read_csv(max(files, key=lambda p: p.stat().st_size))
    str_cols = [c for c in df.columns
                if c not in ("predicted", "correct", "error") and df[c].dtype == object][:2]
    lines = []
    if kind == "regression" and "error" in df:
        worst = df.reindex(df["error"].abs().sort_values(ascending=False).index).head(n)
    elif "correct" in df:
        worst = df[~df["correct"].astype(str).isin(["True", "true"])].head(n)
    else:
        worst = df.head(n)
    for _, r in worst.iterrows():
        ctx = "  ".join(f"{c}={str(r[c])[:30]!r}" for c in str_cols)
        tail = f"err={r['error']:.3g}" if "error" in df else f"pred={r['predicted']}"
        lines.append(f"    - {ctx}  {tail}")
    return "\n".join(lines) if lines else "    (no clear bad rows)"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report(merged: pd.DataFrame, top: int, drill: bool) -> str:
    L = []
    w = L.append
    n_bad = int(merged["is_bad"].sum())
    w("=" * 78)
    w(f"TabPFN bad-case analysis  —  {len(merged)} datasets, {n_bad} flagged bad")
    w("=" * 78)

    for kind in ("regression", "classification"):
        sub = merged[merged["kind"] == kind].sort_values("quality_median")
        if sub.empty:
            continue
        is_cls = kind == "classification"
        metric = "R^2" if kind == "regression" else "AUROC"
        # Classification shows accuracy + balanced accuracy next to AUROC.
        acc_h = f" {'acc':>6s} {'bacc':>6s}" if is_cls else ""
        w(f"\n### {kind.upper()}  ({len(sub)} datasets, quality = {metric}"
          f"{', + acc/bacc' if is_cls else ''})")
        w(f"{'benchmark':8s} {'dataset':36s} {'q_med':>7s}{acc_h} {'q@0.5':>7s} {'slope':>7s} bad")
        for _, r in sub.head(top).iterrows():
            ov = f" (={r['overlap_group']})" if r.get("overlap_group") else ""
            acc_c = (f" {r.get('acc_median', float('nan')):6.3f}"
                     f" {r.get('balanced_acc_median', float('nan')):6.3f}") if is_cls else ""
            w(f"{r['benchmark']:8s} {r['dataset']:36s} {r['quality_median']:7.3f}{acc_c} "
              f"{r['quality_at_0.5']:7.3f} {r['quality_slope_vs_trainfrac']:7.3f} "
              f"{'  <-- BAD' if r['is_bad'] else ''}{ov}")

        corr = correlations(merged, kind)
        if not corr.empty:
            w(f"\n  Spearman corr(structural feature, quality_median):")
            for k, v in pd.concat([corr.head(4), corr.tail(4)]).items():
                w(f"    {k:24s} {v:+.3f}")

        cmp = group_compare(merged, kind)
        if not cmp.empty:
            w(f"\n  Bad vs good group means (features that differ most):")
            cmp = cmp.assign(absdiff=(cmp["bad_mean"] - cmp["good_mean"]).abs())
            cmp = cmp.sort_values("absdiff", ascending=False).head(6)
            for k, r in cmp.iterrows():
                w(f"    {k:24s} bad={r['bad_mean']:.3f}  good={r['good_mean']:.3f}")

        if drill:
            w(f"\n  Worst-row examples (top {min(3, top)} bad datasets):")
            for _, r in sub[sub["is_bad"]].head(3).iterrows():
                w(f"  [{r['benchmark']}:{r['dataset']}]  q_med={r['quality_median']:.3f}")
                w(drill_examples(r["benchmark"], r["dataset"], kind))

    _append_imbalance_section(merged, w, top)
    return "\n".join(L)


def _append_imbalance_section(merged: pd.DataFrame, w, top: int) -> None:
    """Rank classification datasets by Acc/AUROC vs balanced-accuracy divergence.

    A large gap = the model leans on the majority class: raw accuracy looks fine
    while balanced accuracy (and often AUROC) reveals it barely separates classes.
    `acc_gap` = acc_median - balanced_acc_median (high -> majority-class crutch).
    """
    cls = merged[merged["kind"] == "classification"].copy()
    if cls.empty or "balanced_acc_median" not in cls:
        return
    cls["acc_gap"] = cls["acc_median"] - cls["balanced_acc_median"]
    cls["auroc_vs_acc"] = cls["quality_median"] - cls["acc_median"]
    w("\n### CLASSIFICATION — metric divergence "
      "(acc inflated by class imbalance?)")
    w("  acc_gap = acc - balanced_acc  (high => model rides the majority class)")
    w(f"{'benchmark':8s} {'dataset':36s} {'auroc':>6s} {'acc':>6s} {'bacc':>6s} "
      f"{'acc_gap':>8s} {'imbal':>6s}")
    ranked = cls.sort_values("acc_gap", ascending=False).head(top)
    for _, r in ranked.iterrows():
        imb = r.get("target_imbalance", float("nan"))
        flag = "  <-- majority-class crutch" if r["acc_gap"] >= 0.25 else ""
        w(f"{r['benchmark']:8s} {r['dataset']:36s} "
          f"{r['quality_median']:6.3f} {r['acc_median']:6.3f} "
          f"{r['balanced_acc_median']:6.3f} {r['acc_gap']:8.3f} "
          f"{imb:6.3f}{flag}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=20, help="rows to show per kind")
    p.add_argument("--refresh-features", action="store_true",
                   help="recompute structural features (ignore cache)")
    p.add_argument("--no-drill", action="store_true", help="skip worst-row examples")
    args = p.parse_args()

    df = load_summary()
    agg = flag_bad(aggregate_per_dataset(df))
    feats = build_features(list(zip(agg["benchmark"], agg["dataset"])), args.refresh_features)
    merged = agg.merge(feats.drop(columns=["kind", "task"], errors="ignore"),
                       on=["benchmark", "dataset"], how="left")

    out = cfg.RESULTS_ROOT / "bad_cases_per_dataset.csv"
    merged.sort_values(["kind", "quality_median"]).to_csv(out, index=False)

    report = build_report(merged, args.top, drill=not args.no_drill)
    print(report)
    report_path = cfg.RESULTS_ROOT / "bad_cases_report.txt"
    report_path.write_text(report)
    print(f"\nwrote:\n  {out}\n  {report_path}\n  {cfg.RESULTS_ROOT/'dataset_features.csv'}")


if __name__ == "__main__":
    main()
