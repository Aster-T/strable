"""Shared helpers for the TabPFN benchmark: dataset discovery, metric
computation and structural feature extraction.

Imported by both ``tabpfn_test.py`` (the runner) and ``extract_bad_cases.py``
(the analyzer).  When either script is launched with ``python scripts/test/<x>.py``
Python puts ``scripts/test/`` on ``sys.path[0]``, so the plain ``import
bench_config`` below resolves.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import bench_config as cfg

# Non-dataset entries that live alongside the dataset folders.
_SKIP_NAMES = {"README.md", ".gitattributes", ".cache"}


# --------------------------------------------------------------------------- #
# Dataset discovery / loading
# --------------------------------------------------------------------------- #
def list_dataset_dirs(root: Path | None = None) -> list[Path]:
    """Every folder under data_processed that has both config.json + data.parquet."""
    root = root or cfg.DATA_PROCESSED
    out = []
    for d in sorted(root.iterdir()):
        if d.name in _SKIP_NAMES or not d.is_dir():
            continue
        if (d / "config.json").exists() and (d / "data.parquet").exists():
            out.append(d)
    return out


def load_config(dataset_dir: Path) -> dict:
    with open(dataset_dir / "config.json") as f:
        return json.load(f)


# Map other benchmarks' task vocab onto STRABLE's tags.
TASK_ALIASES = {
    "binary-classification": "b-classification",
    "multiclass-classification": "m-classification",
    "regression": "regression",
    "classification": "classification",  # coarse (CARTE) — refined to b/m at load time
    "b-classification": "b-classification",
    "m-classification": "m-classification",
}


def task_kind(task: str) -> str:
    """Map any task tag (STRABLE, CARTE or TTB vocab) to 'classification'/'regression'."""
    t = TASK_ALIASES.get(task, task)
    if t in cfg.CLASSIFICATION_TASKS or t == "classification":
        return "classification"
    if t in cfg.REGRESSION_TASKS:
        return "regression"
    raise ValueError(f"Unknown task tag: {task!r}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(kind, y_true, y_pred, y_proba=None, classes=None) -> dict:
    """Return a flat dict of metrics appropriate to the task kind.

    Classification -> accuracy, balanced_accuracy, macro_f1, auroc,
                      majority_acc, acc_lift.
    Regression     -> r2, rmse, mae, nrmse (rmse / std(y_true)).
    Any metric that cannot be computed is returned as NaN rather than raising.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
    )

    m: dict[str, float] = {}
    if kind == "classification":
        y_true = np.asarray(y_true).astype(str)
        y_pred = np.asarray(y_pred).astype(str)
        m["accuracy"] = float(accuracy_score(y_true, y_pred))
        m["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        m["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
        # Majority-class baseline measured on the test labels.
        vals, counts = np.unique(y_true, return_counts=True)
        majority = float(counts.max() / counts.sum())
        m["majority_acc"] = majority
        m["acc_lift"] = m["accuracy"] - majority
        # AUROC (best-effort; aligned to the model's class order).
        m["auroc"] = float("nan")
        if y_proba is not None and classes is not None:
            try:
                classes = np.asarray(classes).astype(str)
                if len(classes) == 2:
                    pos = classes[1]
                    m["auroc"] = float(
                        roc_auc_score((y_true == pos).astype(int), y_proba[:, 1])
                    )
                else:
                    m["auroc"] = float(
                        roc_auc_score(
                            y_true,
                            y_proba,
                            multi_class="ovr",
                            average="weighted",
                            labels=classes,
                        )
                    )
            except Exception:
                pass
    elif kind == "regression":
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        m["r2"] = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        m["rmse"] = rmse
        m["mae"] = float(mean_absolute_error(y_true, y_pred))
        std = float(np.std(y_true))
        m["nrmse"] = rmse / std if std > 0 else float("nan")
    else:
        raise ValueError(kind)
    return m


# --------------------------------------------------------------------------- #
# Structural feature extraction (for pattern finding in the bad-case analyzer)
# --------------------------------------------------------------------------- #
def _looks_datetime(series: pd.Series) -> bool:
    """True if most non-null string values parse as dates."""
    s = series.dropna().astype(str).head(200)
    if s.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence dateutil "could not infer format"
        parsed = pd.to_datetime(s, errors="coerce")
    return parsed.notna().mean() > 0.8


def dataset_features(df: pd.DataFrame, config: dict, sample: int = 5000) -> dict:
    """Structural fingerprint of a dataset, used to correlate with performance.

    Captures size, column dtypes, string cardinality/length and a free-text
    proxy — the kind of signals that plausibly explain *why* TabPFN struggles
    on a given table.
    """
    target = config["target_name"]
    kind = task_kind(config["task"])
    n_rows = len(df)
    if n_rows > sample:
        df = df.sample(sample, random_state=cfg.RANDOM_STATE)

    feat_cols = [c for c in df.columns if c != target]
    obj_cols = [c for c in feat_cols if df[c].dtype == object]
    num_cols = [c for c in feat_cols if c not in obj_cols]

    out: dict[str, float] = {
        "task": config["task"],
        "kind": kind,
        "n_rows": n_rows,
        "n_features": len(feat_cols),
        "n_object_cols": len(obj_cols),
        "n_numeric_cols": len(num_cols),
        "frac_object": len(obj_cols) / max(len(feat_cols), 1),
        "frac_missing": float(df[feat_cols].isna().mean().mean()) if feat_cols else 0.0,
    }

    # Target characteristics.
    if kind == "classification":
        vc = df[target].value_counts(normalize=True)
        out["n_classes"] = int(df[target].nunique())
        out["target_card"] = out["n_classes"]
        out["target_imbalance"] = float(vc.iloc[0]) if len(vc) else float("nan")
    else:
        out["n_classes"] = 0
        out["target_card"] = int(df[target].nunique())
        out["target_imbalance"] = float("nan")

    # String-column statistics.
    if obj_cols:
        card_ratios, tok_means, char_means, n_datetime = [], [], [], 0
        for c in obj_cols:
            s = df[c].dropna().astype(str)
            n_nonnull = max(len(s), 1)
            card_ratios.append(df[c].nunique() / n_nonnull)
            tok_means.append(s.str.split().str.len().mean() if len(s) else 0.0)
            char_means.append(s.str.len().mean() if len(s) else 0.0)
            if _looks_datetime(df[c]):
                n_datetime += 1
        card_ratios = np.array(card_ratios, dtype=float)
        tok_means = np.array(tok_means, dtype=float)
        out["mean_card_ratio"] = float(np.nanmean(card_ratios))
        out["max_card_ratio"] = float(np.nanmax(card_ratios))
        out["frac_highcard_cols"] = float(np.mean(card_ratios > 0.5))
        out["mean_str_tokens"] = float(np.nanmean(tok_means))
        out["frac_freetext_cols"] = float(np.mean(tok_means > 3))
        out["mean_str_chars"] = float(np.nanmean(char_means))
        out["frac_datetime_cols"] = n_datetime / len(obj_cols)
    else:
        for k in (
            "mean_card_ratio",
            "max_card_ratio",
            "frac_highcard_cols",
            "mean_str_tokens",
            "frac_freetext_cols",
            "mean_str_chars",
            "frac_datetime_cols",
        ):
            out[k] = 0.0
    return out
