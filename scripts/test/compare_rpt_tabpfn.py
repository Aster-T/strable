"""Compare RPT (results/rpt_test/summary.csv) vs TabPFN on the same datasets,
aligned by (dataset, test_size). Main metric = R2 (regression) / AUROC (classif).

Usage:
  python scripts/test/compare_rpt_tabpfn.py [TABPFN_SUMMARY_CSV]
TABPFN_SUMMARY_CSV defaults to results/tabpfn_test/summary.csv.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("STRABLE_ROOT") or Path(__file__).resolve().parents[2])
RPT = REPO / "results" / "rpt_test" / "summary.csv"
TAB = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "tabpfn_test" / "summary.csv"
OUT = REPO / "results" / "rpt_test" / "rpt_vs_tabpfn.csv"


def load(p):
    d = pd.read_csv(p)
    d = d[d["status"] == "ok"].copy()
    d = d.drop_duplicates(subset=["dataset", "test_size"], keep="last")
    for c in ["r2", "auroc", "accuracy", "balanced_acc", "rmse"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["test_size"] = pd.to_numeric(d["test_size"], errors="coerce")
    return d


rpt, tab = load(RPT), load(TAB)
datasets = list(dict.fromkeys(rpt["dataset"]))  # preserve order

rows = []
for ds in datasets:
    r = rpt[rpt["dataset"] == ds].set_index("test_size")
    t = tab[tab["dataset"] == ds].set_index("test_size")
    kind = r["kind"].iloc[0]
    col = "r2" if kind == "regression" else "auroc"
    for ts in sorted(set(r.index) & set(t.index)):
        rows.append({
            "dataset": ds, "kind": kind, "test_size": ts,
            "RPT": float(r.loc[ts, col]), "TabPFN": float(t.loc[ts, col]),
            "delta": float(r.loc[ts, col]) - float(t.loc[ts, col]),
            # secondary metric for context
            "RPT_2nd": float(r.loc[ts, "accuracy"]) if kind != "regression" else float(r.loc[ts, "rmse"]),
            "TabPFN_2nd": float(t.loc[ts, "accuracy"]) if kind != "regression" else float(t.loc[ts, "rmse"]),
        })
cmp = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
cmp.to_csv(OUT, index=False)

print("=" * 84)
print("RPT vs TabPFN  —  main metric: R2 (regression) / AUROC (classification)")
print(f"TabPFN baseline: {TAB}")
print("=" * 84)

for ds in datasets:
    sub = cmp[cmp["dataset"] == ds].sort_values("test_size")
    if sub.empty:
        continue
    kind = sub["kind"].iloc[0]
    mn = "R2" if kind == "regression" else "AUROC"
    wins = int((sub["delta"] > 0).sum())
    print(f"\n### {ds}  ({kind}, metric={mn})")
    print("  test_size :  " + " ".join(f"{ts:>6.1f}" for ts in sub["test_size"]))
    print("  RPT       :  " + " ".join(f"{v:6.3f}" for v in sub["RPT"]))
    print("  TabPFN    :  " + " ".join(f"{v:6.3f}" for v in sub["TabPFN"]))
    print("  delta     :  " + " ".join(f"{v:+6.3f}" for v in sub["delta"]))
    print(f"  --> mean RPT={sub['RPT'].mean():.3f}  TabPFN={sub['TabPFN'].mean():.3f}  "
          f"mean delta={sub['delta'].mean():+.3f}  RPT wins {wins}/{len(sub)}")

print("\n" + "=" * 84)
print("OVERALL (mean over the 9 test sizes)")
print("=" * 84)
g = cmp.groupby("dataset", sort=False).agg(
    kind=("kind", "first"), mean_RPT=("RPT", "mean"),
    mean_TabPFN=("TabPFN", "mean"), mean_delta=("delta", "mean")).round(3)
print(g.to_string())
n_better = int((g["mean_delta"] > 0).sum())
print(f"\nRPT better (mean main metric) on {n_better}/{len(g)} datasets.")
print(f"wrote: {OUT}")
