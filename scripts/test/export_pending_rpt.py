"""Export the RPT runs still pending (not yet ok in results/rpt_test/summary.csv),
so the remaining work can be moved to a bigger-GPU server and resumed.

The full set of (benchmark, dataset, test_size) triples is taken from the TabPFN
summary (the comparison baseline), so RPT's pending list aligns 1:1 with TabPFN.
"""
import os
from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("STRABLE_ROOT") or Path(__file__).resolve().parents[2])
TEST_SIZES = [round(x / 10, 1) for x in range(1, 10)]  # 0.1 .. 0.9

rpt = pd.read_csv(REPO / "results" / "rpt_test" / "summary.csv")

tab_path = Path(os.environ.get("TEMP", "")) / "tabpfn_summary_backup_20260622.csv"
if not tab_path.exists():
    tab_path = REPO / "results" / "tabpfn_test" / "summary.csv"
tab = pd.read_csv(tab_path)

ds_pairs = list(tab[["benchmark", "dataset"]].drop_duplicates().itertuples(index=False, name=None))
full = [(b, d, ts) for (b, d) in ds_pairs for ts in TEST_SIZES]

done_ok = {(r.benchmark, r.dataset, round(float(r.test_size), 1))
           for r in rpt[rpt["status"] == "ok"].itertuples()}

pending = [(b, d, ts) for (b, d, ts) in full if (b, d, round(ts, 1)) not in done_ok]
out = (pd.DataFrame(pending, columns=["benchmark", "dataset", "test_size"])
       .sort_values(["benchmark", "dataset", "test_size"]))
p = REPO / "results" / "rpt_test" / "pending_runs.csv"
out.to_csv(p, index=False)

print(f"baseline (TabPFN) datasets : {len(ds_pairs)}")
print(f"full triples               : {len(full)}")
print(f"RPT done (ok)              : {len(done_ok)}")
print(f"pending                    : {len(pending)}  -> {p}")
print("\npending per benchmark:")
print(out.groupby("benchmark").size().to_string())

# per-dataset completion buckets
done_count = {}
for (b, d, _ts) in done_ok:
    done_count[(b, d)] = done_count.get((b, d), 0) + 1
all_ds = set(ds_pairs)
complete = [bd for bd in all_ds if done_count.get(bd, 0) >= 9]
partial = [bd for bd in all_ds if 0 < done_count.get(bd, 0) < 9]
not_started = [bd for bd in all_ds if done_count.get(bd, 0) == 0]
print(f"\ndatasets complete (9/9): {len(complete)}")
print(f"datasets partial       : {len(partial)}")
print(f"datasets not started   : {len(not_started)}")
