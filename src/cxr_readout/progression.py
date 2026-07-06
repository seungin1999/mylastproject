from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .metrics import LABEL_TO_ID, add_label_id

DEFAULT_FINDINGS = ["Consolidation", "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax"]

FINDING_TO_MSCXRT_COL = {
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Pleural Effusion": "pleural_effusion",
    "Pneumothorax": "pneumothorax",
    "Pneumonia": "pneumonia",
}


def image_path(raw: object) -> str:
    s = str(raw)
    return s if s.endswith(".jpg") else f"{s}.jpg"


def resolve_path(img_roots: Sequence[Path], rel_or_abs: str) -> Path:
    path = Path(str(rel_or_abs))
    if path.is_absolute():
        return path
    for root in img_roots:
        candidate = Path(root) / path
        if candidate.exists():
            return candidate
    return Path(img_roots[0]) / path


def _binary_label(value: object) -> int:
    if pd.isna(value):
        return 0
    try:
        return int(float(value) == 1.0)
    except Exception:
        return 0


def build_mscxrt_instances(
    *,
    mscxrt_csv: Path,
    manifest_csv: Path,
    img_roots: Sequence[Path],
    findings: Sequence[str] = DEFAULT_FINDINGS,
    check_image_files: bool = True,
) -> pd.DataFrame:
    """Build ordered MS-CXR-T progression pairs in the common readout schema."""
    ms = pd.read_csv(mscxrt_csv)
    manifest = pd.read_csv(manifest_csv)
    by_path = manifest.drop_duplicates("path").set_index("path")
    rows = []
    for row in ms.to_dict(orient="records"):
        current_path = image_path(row["dicom_id"])
        previous_path = image_path(row["previous_dicom_id"])
        if current_path not in by_path.index or previous_path not in by_path.index:
            continue
        if check_image_files and (
            not resolve_path(img_roots, current_path).exists()
            or not resolve_path(img_roots, previous_path).exists()
        ):
            continue
        current_meta = by_path.loc[current_path]
        previous_meta = by_path.loc[previous_path]
        for finding in findings:
            key = FINDING_TO_MSCXRT_COL[finding]
            progression = row.get(f"{key}_progression")
            if pd.isna(progression) or str(progression).strip() == "":
                continue
            progression = str(progression).strip().lower()
            if progression not in LABEL_TO_ID:
                continue
            current_bin = _binary_label(current_meta.get(finding))
            previous_bin = _binary_label(previous_meta.get(finding))
            rows.append(
                {
                    "finding": finding,
                    "subject_id": int(row["subject_id"]),
                    "study_id": int(row["study_id"]),
                    "previous_study_id": int(row["previous_study_id"]),
                    "current_dicom_id": str(row["dicom_id"]),
                    "previous_dicom_id": str(row["previous_dicom_id"]),
                    "path": current_path,
                    "previous_path": previous_path,
                    "label": progression,
                    "label_id": LABEL_TO_ID[progression],
                    "changed": int(progression in {"improving", "worsening"}),
                    "endpoint_pattern": f"{previous_bin}->{current_bin}",
                    "both_positive": bool(previous_bin == 1 and current_bin == 1),
                }
            )
    return pd.DataFrame(rows)


def load_gold_finding(
    *,
    mscxrt_csv: Path,
    manifest_csv: Path,
    img_root: Path,
    finding: str,
    findings: Sequence[str] = DEFAULT_FINDINGS,
    check_image_files: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_instances = build_mscxrt_instances(
        mscxrt_csv=mscxrt_csv,
        manifest_csv=manifest_csv,
        img_roots=[img_root],
        findings=findings,
        check_image_files=check_image_files,
    )
    one = all_instances[all_instances["finding"].eq(str(finding))].copy()
    one["split"] = "gold"
    return all_instances.reset_index(drop=True), one.reset_index(drop=True)


def load_silver_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["split"].isin(["train", "valid", "test"])].copy()
    if "label_id" not in df.columns:
        df = add_label_id(df)
    if "path" not in df.columns and "current_path" in df.columns:
        df["path"] = df["current_path"].astype(str)
    if "previous_path" in df.columns:
        df["previous_path"] = df["previous_path"].astype(str)
    return df.sort_values(["split", "subject_id", "previous_path", "path"]).reset_index(drop=True)


def subset_gold(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "full":
        return df.reset_index(drop=True)
    if subset == "both_present":
        return df[df["both_positive"]].reset_index(drop=True)
    if subset == "presence_flip":
        return df[df["endpoint_pattern"].isin(["0->1", "1->0"])].reset_index(drop=True)
    raise ValueError(f"Unknown gold subset: {subset}")


def ordered_pair_keys(df: pd.DataFrame, *, level: str) -> pd.Series:
    """Return exact ordered prior/current pair keys. Order is not symmetrized."""
    if level == "dicom":
        prior_candidates = ["previous_dicom_id", "prior_dicom_id"]
        current_candidates = ["current_dicom_id", "dicom_id"]
    elif level == "study":
        prior_candidates = ["previous_study_id", "prior_study_id"]
        current_candidates = ["current_study_id", "study_id"]
    elif level == "path":
        prior_candidates = ["previous_path", "prior_path"]
        current_candidates = ["path", "current_path"]
    else:
        raise ValueError(f"Unsupported pair level: {level}")

    prior_col = next((c for c in prior_candidates if c in df.columns), None)
    current_col = next((c for c in current_candidates if c in df.columns), None)
    if prior_col is None or current_col is None:
        raise KeyError(f"Cannot build {level} pair keys from columns: {list(df.columns)}")
    return df[prior_col].astype(str) + "||" + df[current_col].astype(str)


@dataclass(frozen=True)
class OverlapAudit:
    level: str
    split: str
    silver_pairs: int
    gold_pairs: int
    overlap_pairs: int
    overlap_rows: int


def audit_train_test_overlap(
    silver: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    level: str = "path",
    silver_split: str = "train",
) -> tuple[OverlapAudit, pd.DataFrame]:
    silver_part = silver[silver["split"].eq(silver_split)].copy()
    silver_keys = ordered_pair_keys(silver_part, level=level)
    gold_keys = ordered_pair_keys(gold, level=level)
    overlap_key_set = set(silver_keys).intersection(set(gold_keys))
    overlap_rows = silver_part[silver_keys.isin(overlap_key_set)].copy()
    audit = OverlapAudit(
        level=level,
        split=str(silver_split),
        silver_pairs=int(silver_keys.nunique()),
        gold_pairs=int(gold_keys.nunique()),
        overlap_pairs=int(len(overlap_key_set)),
        overlap_rows=int(len(overlap_rows)),
    )
    return audit, overlap_rows


def remove_split_overlap(
    silver: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    level: str = "path",
    silver_split: str = "train",
) -> tuple[pd.DataFrame, OverlapAudit]:
    audit, _ = audit_train_test_overlap(silver, gold, level=level, silver_split=silver_split)
    silver = silver.copy()
    split_mask = silver["split"].eq(silver_split)
    split_keys = ordered_pair_keys(silver.loc[split_mask], level=level)
    gold_keys = set(ordered_pair_keys(gold, level=level))
    keep_split = ~split_keys.isin(gold_keys)
    out = pd.concat(
        [silver.loc[~split_mask], silver.loc[split_mask].loc[keep_split.to_numpy()]],
        ignore_index=True,
    ).sort_values(["split", "subject_id", "previous_path", "path"])
    return out.reset_index(drop=True), audit


def remove_train_overlap(
    silver: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    level: str = "path",
    train_split: str = "train",
) -> tuple[pd.DataFrame, OverlapAudit]:
    return remove_split_overlap(silver, gold, level=level, silver_split=train_split)
