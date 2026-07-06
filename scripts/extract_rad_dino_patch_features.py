from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_readout.features import extract_rad_dino_patch_grids


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract pooled RAD-DINO patch-token grids.")
    ap.add_argument("--manifest-csv", type=Path, required=True)
    ap.add_argument("--img-root", type=Path, required=True)
    ap.add_argument("--img-fallback-root", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model-name", type=str, default="microsoft/rad-dino")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--splits", type=str, default="train,validate,test")
    ap.add_argument("--pool-size", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--write-pooled-features", action="store_true")
    ap.add_argument("--pooled-feature-dir", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    extract_rad_dino_patch_grids(
        manifest_csv=args.manifest_csv,
        img_root=args.img_root,
        img_fallback_root=args.img_fallback_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        splits=[x.strip() for x in args.splits.split(",") if x.strip()],
        pool_size=args.pool_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
        overwrite=args.overwrite,
        write_pooled_features=args.write_pooled_features,
        pooled_feature_dir=args.pooled_feature_dir,
    )


if __name__ == "__main__":
    main()
