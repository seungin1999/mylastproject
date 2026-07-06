from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .features import FeatureVectorSource, PatchGridSource
from .metrics import LABEL_TO_ID


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def resolve_device(device: str) -> torch.device:
    return torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")


def assign_shuffled_prior(df: pd.DataFrame, seed: int) -> dict[tuple[str, str], str]:
    pairs = (
        df[["path", "previous_path", "subject_id"]]
        .drop_duplicates(["path", "previous_path"])
        .sort_values(["subject_id", "previous_path", "path"])
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(pairs))
    prev_paths = pairs["previous_path"].to_numpy(dtype=str)[order].copy()
    prev_subjects = pairs["subject_id"].to_numpy(dtype=np.int64)[order].copy()
    subjects = pairs["subject_id"].to_numpy(dtype=np.int64)
    if len(pairs) > 1:
        for i in range(len(pairs)):
            if prev_subjects[i] != subjects[i]:
                continue
            for shift in range(1, len(pairs)):
                j = (i + shift) % len(pairs)
                if prev_subjects[j] != subjects[i]:
                    prev_paths[i], prev_paths[j] = prev_paths[j], prev_paths[i]
                    prev_subjects[i], prev_subjects[j] = prev_subjects[j], prev_subjects[i]
                    break
    return {
        (str(row.path), str(row.previous_path)): str(prev_paths[i])
        for i, row in enumerate(pairs.itertuples(index=False))
    }


def mode_paths(row, mode: str, shuffled_prior: dict[tuple[str, str], str] | None = None) -> tuple[str | None, str]:
    current = str(row.path)
    previous = str(row.previous_path)
    if mode == "pair":
        return previous, current
    if mode == "x2_only":
        return None, current
    if mode == "identical":
        return current, current
    if mode == "shuffled_prior":
        if shuffled_prior is None:
            raise ValueError("shuffled_prior mode requires a shuffled_prior mapping.")
        return shuffled_prior[(current, previous)], current
    raise ValueError(f"Unsupported mode: {mode}")


class PairPatchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        source: PatchGridSource,
        mode: str = "pair",
        shuffled_prior: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.source = source
        self.mode = str(mode)
        self.shuffled_prior = shuffled_prior
        self.y = self.df["label_id"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        prev, cur = mode_paths(row, self.mode, self.shuffled_prior)
        t2 = self.source.get(cur).astype(np.float32, copy=False)
        t1 = np.zeros_like(t2, dtype=np.float32) if prev is None else self.source.get(prev).astype(np.float32, copy=False)
        return t1, t2, np.int64(self.y[int(idx)])


class PatchMILReadout(nn.Module):
    """Patch-level temporal MIL readout over frozen patch-token grids."""

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        align: bool = True,
        temperature: float = 0.07,
        feature_set: str = "full",
    ) -> None:
        super().__init__()
        self.align = bool(align)
        self.temperature = float(temperature)
        self.feature_set = str(feature_set)
        if self.feature_set == "full":
            in_dim = 4 * token_dim
        elif self.feature_set in {"no_abs", "no_signed"}:
            in_dim = 3 * token_dim
        elif self.feature_set == "endpoints":
            in_dim = 2 * token_dim
        else:
            raise ValueError(f"Unsupported feature_set: {self.feature_set}")
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.patch_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def _align_prior(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        if not self.align:
            return t1
        q = F.normalize(t2, p=2, dim=-1)
        k = F.normalize(t1, p=2, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / self.temperature, dim=-1)
        return torch.matmul(attn, t1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        t1 = self.norm1(t1.float())
        t2 = self.norm2(t2.float())
        base = self._align_prior(t1, t2)
        delta = t2 - base
        if self.feature_set == "full":
            patch_feat = torch.cat([t2, base, delta, delta.abs()], dim=-1)
        elif self.feature_set == "no_abs":
            patch_feat = torch.cat([t2, base, delta], dim=-1)
        elif self.feature_set == "no_signed":
            patch_feat = torch.cat([t2, base, delta.abs()], dim=-1)
        elif self.feature_set == "endpoints":
            patch_feat = torch.cat([t2, base], dim=-1)
        else:
            raise ValueError(f"Unsupported feature_set: {self.feature_set}")
        h = self.patch_mlp(patch_feat)
        a = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        pooled = torch.sum(h * a.unsqueeze(-1), dim=1)
        return self.classifier(pooled)


class PooledMLPReadout(nn.Module):
    """Capacity-matched MLP over pooled prior/current endpoint vectors."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def endpoint_feature_matrix(
    df: pd.DataFrame,
    *,
    source: FeatureVectorSource,
    mode: str = "pair",
    shuffled_prior: dict[tuple[str, str], str] | None = None,
    include_delta: bool = True,
) -> np.ndarray:
    rows = []
    zero = np.zeros(source.dim, dtype=np.float32)
    for row in df.itertuples(index=False):
        prev, cur = mode_paths(row, mode, shuffled_prior)
        f2 = source.get(cur).astype(np.float32, copy=False)
        f1 = zero if prev is None else source.get(prev).astype(np.float32, copy=False)
        if include_delta:
            delta = f2 - f1
            rows.append(np.concatenate([f1, f2, delta, np.abs(delta)], axis=0))
        else:
            rows.append(np.concatenate([f1, f2], axis=0))
    return np.stack(rows, axis=0).astype(np.float32)


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y.astype(np.int64), minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def changed_pos_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    changed = (y != LABEL_TO_ID["stable"]).astype(np.float32)
    pos = float(changed.sum())
    neg = float(len(changed) - pos)
    return torch.as_tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)


def progression_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    weights: torch.Tensor,
    changed_weight: torch.Tensor,
    change_loss_weight: float = 0.25,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, y, weight=weights)
    if change_loss_weight <= 0:
        return ce
    changed_target = (y != LABEL_TO_ID["stable"]).float()
    changed_logit = (
        torch.logsumexp(logits[:, [LABEL_TO_ID["improving"], LABEL_TO_ID["worsening"]]], dim=1)
        - logits[:, LABEL_TO_ID["stable"]]
    )
    bce = F.binary_cross_entropy_with_logits(changed_logit, changed_target, pos_weight=changed_weight)
    return ce + float(change_loss_weight) * bce


@torch.no_grad()
def predict_patch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    ys = []
    for t1, t2, y in loader:
        logits = model(t1.to(device), t2.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32))
        ys.append(y.numpy().astype(np.int64))
    return np.concatenate(probs, axis=0), np.concatenate(ys, axis=0)


@torch.no_grad()
def predict_vectors(model: nn.Module, x: np.ndarray, y: np.ndarray, *, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    for start in range(0, len(x), int(batch_size)):
        xb = torch.as_tensor(x[start:start + int(batch_size)], dtype=torch.float32, device=device)
        probs.append(torch.softmax(model(xb), dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(probs, axis=0), y.astype(np.int64)


def make_patch_loader(
    df: pd.DataFrame,
    *,
    source: PatchGridSource,
    batch_size: int,
    mode: str = "pair",
    shuffled_prior: dict[tuple[str, str], str] | None = None,
    shuffle: bool = False,
) -> DataLoader:
    dataset = PairPatchDataset(df, source=source, mode=mode, shuffled_prior=shuffled_prior)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0, drop_last=False)


def make_vector_loader(x: np.ndarray, y: np.ndarray, *, batch_size: int, shuffle: bool = False) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0, drop_last=False)
