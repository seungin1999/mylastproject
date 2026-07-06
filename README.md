# Longitudinal CXR Readout Utilities

Minimal code for frozen-feature longitudinal chest X-ray readout experiments.

The repository contains implementation code only. It does not redistribute clinical data, feature caches, model checkpoints, or generated outputs.

## Contents

- `src/cxr_readout/features.py`: frozen encoder feature-cache helpers and RAD-DINO patch-grid extraction
- `src/cxr_readout/progression.py`: MS-CXR-T progression pair loading and exact ordered-pair overlap auditing
- `src/cxr_readout/readouts.py`: patch-level temporal MIL readout and pooled endpoint-vector MLP baseline
- `src/cxr_readout/metrics.py`: 3-way macro accuracy, Changed AUC, and Direction AUC metrics
- `scripts/`: small command-line wrappers for feature extraction, overlap audit, and readout training
- `configs/paths.example.env`: local path configuration template

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example Commands

Extract pooled RAD-DINO patch-token grids:

```bash
python scripts/extract_rad_dino_patch_features.py \
  --manifest-csv /path/to/manifest.csv \
  --img-root /path/to/mimic-cxr-jpg/files \
  --output-dir outputs/rad_dino_patch_grids \
  --write-pooled-features \
  --device cuda:0
```

`--write-pooled-features` writes `features_train.npz`, `features_validate.npz`, and `features_test.npz` by mean-pooling the same frozen patch-token grids used by the patch MIL readout.

Audit exact ordered silver-train/gold-test pair overlap:

```bash
python scripts/audit_pair_overlap.py \
  --silver-manifest /path/to/silver_progression_pairs.csv \
  --mscxrt-csv /path/to/MS_CXR_T_temporal_image_classification_v1.0.0.csv \
  --gold-manifest /path/to/mimic_manifest.csv \
  --img-root /path/to/mimic-cxr-jpg/files \
  --finding "Pleural Effusion" \
  --gold-subset both_present \
  --pair-level path \
  --output-json outputs/overlap_audit.json
```

Train patch MIL readout:

```bash
python scripts/train_patch_mil_readout.py \
  --silver-manifest /path/to/silver_progression_pairs.csv \
  --patch-dir outputs/rad_dino_patch_grids \
  --mscxrt-csv /path/to/MS_CXR_T_temporal_image_classification_v1.0.0.csv \
  --gold-manifest /path/to/mimic_manifest.csv \
  --img-root /path/to/mimic-cxr-jpg/files \
  --finding "Pleural Effusion" \
  --gold-subset both_present \
  --de-leak \
  --de-leak-valid \
  --output-json outputs/patch_mil_summary.json
```

Train pooled endpoint-vector MLP baseline:

```bash
python scripts/train_pooled_mlp_readout.py \
  --silver-manifest /path/to/silver_progression_pairs.csv \
  --feature-file outputs/rad_dino_patch_grids/features_train.npz \
  --feature-file outputs/rad_dino_patch_grids/features_validate.npz \
  --feature-file outputs/rad_dino_patch_grids/features_test.npz \
  --mscxrt-csv /path/to/MS_CXR_T_temporal_image_classification_v1.0.0.csv \
  --gold-manifest /path/to/mimic_manifest.csv \
  --img-root /path/to/mimic-cxr-jpg/files \
  --finding "Pleural Effusion" \
  --gold-subset both_present \
  --de-leak \
  --de-leak-valid \
  --output-json outputs/pooled_mlp_summary.json
```
