"""Unified dataset registry over STRABLE, CARTE and TextTabBench (TTB).

Each benchmark stores tables differently:
  - STRABLE: data/data_processed/<name>/{data.parquet, config.json}
  - CARTE:   data/CARTE_datasets/data_carte/<name>/{raw.parquet, config_data.json}
  - TTB:     data/TTB_datasets/<files>  described by a checked-in manifest.json

This module normalizes all three into one ``DatasetRecord(benchmark, name, X, y,
task, ...)`` so the TabPFN runner can treat them identically.  STRABLE and CARTE
carry their own per-dataset config (target + task); TTB has none, so a manifest
pins {file, target, task} for it.  ``overlap_group`` tags datasets that are the
same underlying table across benchmarks (see bench_config.OVERLAP_GROUPS).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

import bench_config as cfg
import _common as common


@dataclass
class DatasetRecord:
    benchmark: str            # "strable" | "carte" | "ttb"
    name: str
    X: pd.DataFrame
    y: pd.Series
    task: str                 # normalized: regression | b-classification | m-classification
    target: str
    source_path: str
    overlap_group: Optional[str] = None


@dataclass
class DatasetSpec:
    """A lazy handle: list/filter datasets without reading the table."""
    benchmark: str
    name: str
    task: str                 # may be coarse ('classification') until loaded
    target: str
    path: Path
    _load: Callable[[], DatasetRecord]
    overlap_group: Optional[str] = None

    def load(self) -> DatasetRecord:
        rec = self._load()
        rec.overlap_group = self.overlap_group
        return rec


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _refine_task(raw_task: str, y) -> str:
    """Normalize a benchmark's task tag to STRABLE's vocabulary, inferring
    binary vs multiclass from the target when the source only says
    'classification'."""
    t = common.TASK_ALIASES.get(raw_task, raw_task)
    if t in ("b-classification", "m-classification", "regression"):
        return t
    if t == "classification":
        n = pd.Series(y).nunique()
        return "b-classification" if n <= 2 else "m-classification"
    return t  # unknown -> task_kind will raise downstream


def prepare_xy(df: pd.DataFrame, target: str, kind: str):
    """Drop NaN targets (coerce regression to float) and split into X, y.

    Single-sourced so every benchmark gets identical target handling.
    """
    df = df.reset_index(drop=True)
    if target not in df.columns:
        raise KeyError(f"target {target!r} not in columns {list(df.columns)[:25]}")
    df = df[df[target].notna()]
    if kind == "regression":
        df = df.copy()
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df[df[target].notna()]
    else:
        # Classification: cast labels to discrete strings. A classification
        # target stored as float (e.g. yelp 'stars' = 1.0..5.0) otherwise makes
        # TabPFNClassifier raise "Unknown label type: continuous".
        df = df.copy()
        df[target] = df[target].astype(str)
    return df.drop(columns=[target]), df[target]


def _read_table(path: Path, read: str = "csv", sep=None, encoding="latin1") -> pd.DataFrame:
    if read in ("csv", "tsv"):
        return pd.read_csv(
            path, sep="\t" if read == "tsv" else (sep or ","),
            on_bad_lines="skip", engine="python", encoding=encoding,
        )
    if read == "parquet":
        return pd.read_parquet(path)
    if read == "json":
        return pd.read_json(path)
    if read == "jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"unknown read mode: {read}")


# --------------------------------------------------------------------------- #
# Per-benchmark spec builders
# --------------------------------------------------------------------------- #
def strable_specs() -> list[DatasetSpec]:
    """Wrap STRABLE's existing folder layout (config.json + data.parquet)."""
    out = []
    for d in common.list_dataset_dirs():
        c = common.load_config(d)
        target, task = c["target_name"], c["task"]

        def _mk(d=d, target=target, task=task):
            df = pd.read_parquet(d / "data.parquet")
            X, y = prepare_xy(df, target, common.task_kind(task))
            return DatasetRecord("strable", d.name, X, y, task, target, str(d))

        out.append(DatasetSpec("strable", d.name, task, target, d, _mk))
    return out


def carte_specs() -> list[DatasetSpec]:
    """CARTE ships one folder per dataset with config_data.json + raw.parquet."""
    base = cfg.CARTE_ROOT / "data_carte"
    if not base.is_dir():
        return []
    out = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        cfgp, raw = d / "config_data.json", d / "raw.parquet"
        if not (cfgp.exists() and raw.exists()):
            continue
        c = json.loads(cfgp.read_text())
        target, raw_task = c["target_name"], c["task"]

        def _mk(d=d, raw=raw, target=target, raw_task=raw_task):
            df = pd.read_parquet(raw)
            y_ref = df[target] if target in df.columns else pd.Series([], dtype=object)
            task = _refine_task(raw_task, y_ref)
            X, y = prepare_xy(df, target, common.task_kind(task))
            return DatasetRecord("carte", d.name, X, y, task, target, str(raw))

        out.append(DatasetSpec("carte", d.name, raw_task, target, d, _mk))
    return out


def manifest_specs(benchmark: str, root: Path, manifest_path: Path) -> list[DatasetSpec]:
    """Generic manifest-driven loader (used for TTB).

    Manifest format: {"datasets": [{name, file, target, task,
                                    read?, sep?, encoding?,
                                    filter_col?, filter_values?, rename?}]}
    ``file`` is relative to ``root``.
    """
    if not manifest_path.exists():
        return []
    entries = json.loads(manifest_path.read_text())["datasets"]
    out = []
    for e in entries:
        fpath = root / e["file"]
        target, raw_task = e["target"], e["task"]

        def _mk(e=e, fpath=fpath, target=target, raw_task=raw_task):
            df = _read_table(fpath, e.get("read", "csv"), e.get("sep"), e.get("encoding", "latin1"))
            if e.get("rename"):
                df = df.rename(columns=e["rename"])
            if e.get("filter_col") and e.get("filter_values") is not None:
                df = df[df[e["filter_col"]].isin(e["filter_values"])]
            if e.get("drop_cols"):
                df = df.drop(columns=[c for c in e["drop_cols"] if c in df.columns])
            if e.get("clean_target") == "currency" and target in df.columns:
                # strip currency symbols / thousands separators -> numeric string
                df[target] = (df[target].astype(str)
                              .str.replace(r"[^0-9.]", "", regex=True).replace("", None))
            y_ref = df[target] if target in df.columns else pd.Series([], dtype=object)
            task = _refine_task(raw_task, y_ref)
            X, y = prepare_xy(df, target, common.task_kind(task))
            return DatasetRecord(benchmark, e["name"], X, y, task, target, str(fpath))

        out.append(DatasetSpec(benchmark, e["name"], raw_task, target, fpath, _mk))
    return out


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def build_registry(benchmarks: list[str]) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    if "strable" in benchmarks:
        specs += strable_specs()
    if "carte" in benchmarks:
        specs += carte_specs()
    if "ttb" in benchmarks:
        specs += manifest_specs("ttb", cfg.TTB_ROOT, cfg.TTB_MANIFEST)
    for s in specs:                       # stamp cross-benchmark provenance
        s.overlap_group = cfg.OVERLAP_LOOKUP.get((s.benchmark, s.name))
    return specs
