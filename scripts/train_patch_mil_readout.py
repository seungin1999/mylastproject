from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_readout.features import PatchGridSource
from src.cxr_readout.metrics import metrics_from_probs
from src.cxr_readout.progression import load_gold_finding, load_silver_manifest, remove_split_overlap, subset_gold
from src.cxr_readout.readouts import PatchMILReadout, make_patch_loader, predict_patch, resolve_device, set_seed
from src.cxr_readout.training import train_patch_model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a patch-level MIL readout on frozen patch-token grids.")
    ap.add_argument("--silver-manifest", type=Path, required=True)
    ap.add_argument("--patch-dir", type=Path, action="append", required=True)
    ap.add_argument("--mscxrt-csv", type=Path, required=True)
    ap.add_argument("--gold-manifest", type=Path, required=True)
    ap.add_argument("--img-root", type=Path, required=True)
    ap.add_argument("--finding", type=str, required=True)
    ap.add_argument("--gold-subset", choices=["full", "both_present", "presence_flip"], default="both_present")
    ap.add_argument("--de-leak", action="store_true")
    ap.add_argument("--de-leak-valid", action="store_true")
    ap.add_argument("--pair-level", choices=["path", "dicom", "study"], default="path")
    ap.add_argument("--skip-image-check", action="store_true")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--change-loss-weight", type=float, default=0.25)
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--feature-set", choices=["full", "no_abs", "no_signed", "endpoints"], default="full")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    silver = load_silver_manifest(args.silver_manifest)
    _, gold_one = load_gold_finding(
        mscxrt_csv=args.mscxrt_csv,
        manifest_csv=args.gold_manifest,
        img_root=args.img_root,
        finding=args.finding,
        check_image_files=not args.skip_image_check,
    )
    gold_eval = subset_gold(gold_one, args.gold_subset)
    overlap = {"train": None, "valid": None}
    if args.de_leak:
        silver, overlap_train = remove_split_overlap(silver, gold_eval, level=args.pair_level, silver_split="train")
        overlap["train"] = overlap_train.__dict__
        if args.de_leak_valid:
            silver, overlap_valid = remove_split_overlap(silver, gold_eval, level=args.pair_level, silver_split="valid")
            overlap["valid"] = overlap_valid.__dict__

    source = PatchGridSource(args.patch_dir)
    train_df = silver[silver["split"].eq("train")].reset_index(drop=True)
    valid_df = silver[silver["split"].eq("valid")].reset_index(drop=True)
    train_loader = make_patch_loader(train_df, source=source, batch_size=args.batch_size, shuffle=True)
    valid_loader = make_patch_loader(valid_df, source=source, batch_size=args.batch_size)

    model = PatchMILReadout(
        token_dim=source.token_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        align=not args.no_align,
        feature_set=args.feature_set,
    )
    model, train_info = train_patch_model(
        model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        change_loss_weight=args.change_loss_weight,
    )
    gold_loader = make_patch_loader(gold_eval, source=source, batch_size=args.batch_size)
    probs, y = predict_patch(model, gold_loader, device)
    summary = {
        "finding": args.finding,
        "gold_subset": args.gold_subset,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_gold": int(len(gold_eval)),
        "de_leak": bool(args.de_leak),
        "de_leak_valid": bool(args.de_leak_valid),
        "overlap": overlap,
        "metrics": metrics_from_probs(probs, y),
        "best_score": float(train_info["best_score"]),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args)}, args.checkpoint)


if __name__ == "__main__":
    main()
