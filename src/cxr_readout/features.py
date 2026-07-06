from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.lib import format as np_format
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resolve_image_path(img_root: Path, rel_or_abs: str, fallback_root: Path | None = None) -> Path:
    path = Path(str(rel_or_abs))
    if path.is_absolute():
        return path
    primary = img_root / path
    if primary.exists():
        return primary
    if fallback_root is not None:
        fallback = fallback_root / path
        if fallback.exists():
            return fallback
    return primary


class ImagePathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: Path, fallback_root: Path | None = None):
        self.paths = df["path"].astype(str).to_numpy(copy=False)
        self.img_root = Path(img_root)
        self.fallback_root = Path(fallback_root) if fallback_root is not None else None

    def __len__(self) -> int:
        return int(len(self.paths))

    def __getitem__(self, idx: int):
        path = resolve_image_path(self.img_root, str(self.paths[idx]), self.fallback_root)
        return int(idx), Image.open(path).convert("RGB")


def pil_collate(batch):
    idxs, imgs = zip(*batch)
    return np.asarray(idxs, dtype=np.int64), list(imgs)


@torch.no_grad()
def pooled_patch_tokens(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    pool_size: int,
    use_amp: bool,
) -> torch.Tensor:
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(use_amp) and inputs["pixel_values"].is_cuda,
    ):
        hidden = model(**inputs).last_hidden_state
    patch = hidden[:, 1:, :]
    n_patch = int(patch.shape[1])
    side = int(math.isqrt(n_patch))
    if side * side != n_patch:
        raise ValueError(f"Patch token count is not square: {n_patch}")
    bsz, _, dim = patch.shape
    grid = patch.reshape(bsz, side, side, dim).permute(0, 3, 1, 2).contiguous()
    pooled = F.adaptive_avg_pool2d(grid, output_size=(pool_size, pool_size))
    return pooled.permute(0, 2, 3, 1).reshape(bsz, pool_size * pool_size, dim)


def extract_rad_dino_patch_grids(
    *,
    manifest_csv: Path,
    img_root: Path,
    output_dir: Path,
    model_name: str = "microsoft/rad-dino",
    device: str = "cuda:0",
    splits: Sequence[str] = ("train", "validate", "test"),
    pool_size: int = 7,
    batch_size: int = 32,
    num_workers: int = 0,
    img_fallback_root: Path | None = None,
    amp: bool = False,
    overwrite: bool = False,
    write_pooled_features: bool = False,
    pooled_feature_dir: Path | None = None,
) -> None:
    manifest = pd.read_csv(manifest_csv)
    if "split" not in manifest.columns or "path" not in manifest.columns:
        raise ValueError("Manifest must contain `split` and `path` columns.")
    manifest["split"] = manifest["split"].astype(str).str.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_plan = []
    for split in splits:
        split_l = str(split).lower()
        aliases = {split_l}
        if split_l in {"val", "validate"}:
            aliases = {"val", "validate"}
            split_l = "validate"
        pooled_dir = output_dir if pooled_feature_dir is None else Path(pooled_feature_dir)
        split_plan.append(
            {
                "split": split_l,
                "aliases": aliases,
                "patch_file": output_dir / f"patch_grid_{split_l}.npz",
                "pooled_file": pooled_dir / f"features_{split_l}.npz",
                "pooled_dir": pooled_dir,
            }
        )

    needs_model = any(overwrite or not item["patch_file"].exists() for item in split_plan)
    torch_device = None
    processor = None
    model = None
    if needs_model:
        torch_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(torch_device).eval()

    for item in split_plan:
        out_file = item["patch_file"]
        pooled_file = item["pooled_file"]
        pooled_dir = item["pooled_dir"]
        if out_file.exists() and not overwrite:
            if write_pooled_features and not pooled_file.exists():
                z = np.load(out_file, allow_pickle=True)
                pooled_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    pooled_file,
                    path=z["path"].astype(str),
                    row_index=z["row_index"].astype(np.int64),
                    features=z["tokens"].astype(np.float32).mean(axis=1),
                )
            continue
        assert processor is not None and model is not None and torch_device is not None
        df = manifest[manifest["split"].isin(item["aliases"])].copy().reset_index(drop=True)
        dataset = ImagePathDataset(df, img_root=img_root, fallback_root=img_fallback_root)
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            collate_fn=pil_collate,
        )
        tokens = None
        for idxs, imgs in loader:
            inputs = processor(images=imgs, return_tensors="pt").to(torch_device)
            batch_tokens = pooled_patch_tokens(model, inputs, pool_size=pool_size, use_amp=amp).cpu().numpy().astype(np.float32)
            if tokens is None:
                tokens = np.empty((len(df), batch_tokens.shape[1], batch_tokens.shape[2]), dtype=np.float32)
            tokens[idxs] = batch_tokens
        if tokens is None:
            tokens = np.zeros((0, pool_size * pool_size, 0), dtype=np.float32)
        np.savez(
            out_file,
            path=df["path"].astype(str).to_numpy(),
            row_index=df.index.to_numpy(dtype=np.int64),
            tokens=tokens,
        )
        if write_pooled_features:
            pooled_dir.mkdir(parents=True, exist_ok=True)
            features = tokens.mean(axis=1) if tokens.shape[0] else np.zeros((0, tokens.shape[-1]), dtype=np.float32)
            np.savez(
                pooled_file,
                path=df["path"].astype(str).to_numpy(),
                row_index=df.index.to_numpy(dtype=np.int64),
                features=features.astype(np.float32, copy=False),
            )


def _npy_payload_offset(npz_path: Path, member: str) -> tuple[int, tuple[int, ...], np.dtype, bool]:
    with ZipFile(npz_path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != 0:
            raise ValueError(f"{npz_path}:{member} is compressed; use uncompressed np.savez caches.")
        with open(npz_path, "rb") as fh:
            fh.seek(info.header_offset)
            local = fh.read(30)
            if len(local) != 30 or local[:4] != b"PK\x03\x04":
                raise ValueError(f"Invalid local zip header in {npz_path}:{member}")
            name_len, extra_len = struct.unpack("<HH", local[26:30])
            data_offset = info.header_offset + 30 + name_len + extra_len
            fh.seek(data_offset)
            version = np_format.read_magic(fh)
            shape, fortran_order, dtype = np_format._read_array_header(fh, version)
            payload_offset = fh.tell()
    return payload_offset, tuple(shape), np.dtype(dtype), bool(fortran_order)


def memmap_npz_member(npz_path: Path, member: str) -> np.memmap:
    offset, shape, dtype, fortran_order = _npy_payload_offset(npz_path, member)
    order = "F" if fortran_order else "C"
    return np.memmap(npz_path, mode="r", dtype=dtype, shape=shape, offset=offset, order=order)


@dataclass(frozen=True)
class TokenRef:
    file_key: str
    row: int


class PatchGridSource:
    """Read pooled patch-token grids from one or more extraction directories."""

    def __init__(self, patch_dirs: Sequence[Path]):
        self.tokens: dict[str, np.memmap] = {}
        self.index: dict[str, TokenRef] = {}
        for patch_dir in patch_dirs:
            patch_dir = Path(patch_dir)
            view_name = patch_dir.name
            for split in ("train", "validate", "test"):
                npz_path = patch_dir / f"patch_grid_{split}.npz"
                if not npz_path.exists():
                    continue
                key = f"{view_name}:{split}"
                z = np.load(npz_path, allow_pickle=True)
                self.tokens[key] = memmap_npz_member(npz_path, "tokens.npy")
                for row, path in enumerate(z["path"].astype(str)):
                    stem = Path(path).stem
                    if stem in self.index:
                        raise ValueError(f"Duplicate DICOM/image stem in patch cache: {stem}")
                    self.index[stem] = TokenRef(key, int(row))
        if not self.tokens:
            raise ValueError(f"No patch_grid_*.npz files found in: {list(patch_dirs)}")

    @property
    def token_dim(self) -> int:
        arr = next(iter(self.tokens.values()))
        return int(arr.shape[-1])

    def get(self, path_or_stem: str) -> np.ndarray:
        stem = Path(str(path_or_stem)).stem
        ref = self.index[stem]
        return np.asarray(self.tokens[ref.file_key][ref.row], dtype=np.float32)


class FeatureVectorSource:
    """Read pooled/global feature vectors from .npz caches containing path(s) and features."""

    def __init__(self, files: Sequence[Path]):
        feats = []
        paths = []
        for file in files:
            z = np.load(file, allow_pickle=True)
            key = "path" if "path" in z.files else "paths"
            paths.extend([str(x) for x in z[key].astype(str)])
            feats.append(z["features"].astype(np.float32))
        if not feats:
            raise ValueError("No feature files provided.")
        self.features = np.concatenate(feats, axis=0)
        self.index = {str(path): i for i, path in enumerate(paths)}
        self.stem_index = {Path(path).stem: i for i, path in enumerate(paths)}

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])

    def get(self, path_or_stem: str) -> np.ndarray:
        key = str(path_or_stem)
        if key in self.index:
            return self.features[self.index[key]]
        return self.features[self.stem_index[Path(key).stem]]
