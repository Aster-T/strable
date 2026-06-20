import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("/home/amax/al/strable/results")


def evaluate(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "correct" in df.columns:
        # 分类：target 是 "predicted" 前面那一列
        pred_idx = df.columns.get_loc("predicted")
        target_col = df.columns[pred_idx - 1]
        acc = (df[target_col].astype(str) == df["predicted"].astype(str)).mean()
        return "classification", {"acc": acc}
    elif "error" in df.columns:
        # 回归：error = predicted - actual
        pred_idx = df.columns.get_loc("predicted")
        target_col = df.columns[pred_idx - 1]
        y_true = df[target_col].to_numpy(dtype=float)
        err = df["error"].to_numpy(dtype=float)
        rmse = np.sqrt((err**2).mean())
        ss_res = (err**2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return "regression", {"rmse": rmse, "r2": r2}
    else:
        return "unknown", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths", nargs="*", help="指定结果 csv；不传则扫描 results/ 下全部"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="在每个数据集文件夹下生成 summary.csv（完整扫描该文件夹）",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths] or sorted(RESULTS_DIR.glob("*/tabpfn_*.csv"))
    if not paths:
        print(f"No result files found in {RESULTS_DIR}")
        return

    # 要写 summary 的话，把涉及文件夹下的全部结果文件都纳入评估，保证不漏
    files_to_eval = set(paths)
    if args.summary:
        for folder in {p.parent for p in paths}:
            files_to_eval.update(folder.glob("tabpfn_*.csv"))

    cache = {p: evaluate(p) for p in sorted(files_to_eval)}  # 每个文件只算一次

    # 打印（仅打印请求的 paths）
    for p in paths:
        task, metrics = cache[p]
        label = f"{p.parent.name}/{p.name}"
        if metrics is None:
            print(f"{label}: 无法判断任务类型（缺少 correct/error 列）")
        else:
            metric_str = "  ".join(f"{k.upper()}={v:.4f}" for k, v in metrics.items())
            print(f"{label}: {metric_str}  ({task})")

    # 写 summary：每个文件夹完整扫描
    if args.summary:
        for folder in sorted({p.parent for p in paths}):
            rows = []
            for csv_path in sorted(folder.glob("tabpfn_*.csv")):
                task, metrics = cache.get(csv_path) or evaluate(csv_path)
                if metrics is None:
                    continue
                rows.append({"name": csv_path.stem, **metrics})
            if rows:
                out = folder / "summary.csv"
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"  -> {out}  ({len(rows)} 行)")


if __name__ == "__main__":
    main()
