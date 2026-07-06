from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_readout.progression import (
    audit_train_test_overlap,
    load_gold_finding,
    load_silver_manifest,
    subset_gold,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit exact ordered silver-train/gold-test pair overlap.")
    ap.add_argument("--silver-manifest", type=Path, required=True)
    ap.add_argument("--mscxrt-csv", type=Path, required=True)
    ap.add_argument("--gold-manifest", type=Path, required=True)
    ap.add_argument("--img-root", type=Path, required=True)
    ap.add_argument("--finding", type=str, required=True)
    ap.add_argument("--gold-subset", choices=["full", "both_present", "presence_flip"], default="both_present")
    ap.add_argument("--pair-level", choices=["path", "dicom", "study"], default="path")
    ap.add_argument("--silver-split", choices=["train", "valid", "test"], default="train")
    ap.add_argument("--skip-image-check", action="store_true")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--overlap-csv", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    silver = load_silver_manifest(args.silver_manifest)
    _, gold_one = load_gold_finding(
        mscxrt_csv=args.mscxrt_csv,
        manifest_csv=args.gold_manifest,
        img_root=args.img_root,
        finding=args.finding,
        check_image_files=not args.skip_image_check,
    )
    gold_eval = subset_gold(gold_one, args.gold_subset)
    audit, overlap_rows = audit_train_test_overlap(silver, gold_eval, level=args.pair_level, silver_split=args.silver_split)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(asdict(audit), indent=2) + "\n")
    if args.overlap_csv is not None:
        args.overlap_csv.parent.mkdir(parents=True, exist_ok=True)
        overlap_rows.to_csv(args.overlap_csv, index=False)


if __name__ == "__main__":
    main()
