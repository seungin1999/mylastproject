from __future__ import annotations

import copy

import numpy as np
import torch

from .metrics import metrics_from_probs
from .readouts import (
    changed_pos_weight,
    class_weights,
    predict_patch,
    predict_vectors,
    progression_loss,
)


def train_patch_model(
    model: torch.nn.Module,
    *,
    train_loader,
    valid_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    change_loss_weight: float,
) -> tuple[torch.nn.Module, dict]:
    train_y = np.asarray(train_loader.dataset.y, dtype=np.int64)
    weights = class_weights(train_y, device)
    changed_weight = changed_pos_weight(train_y, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    history = []

    model.to(device)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for t1, t2, y in train_loader:
            y = y.to(device)
            loss = progression_loss(
                model(t1.to(device), t2.to(device)),
                y,
                weights=weights,
                changed_weight=changed_weight,
                change_loss_weight=change_loss_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        probs, y_valid = predict_patch(model, valid_loader, device)
        metrics = metrics_from_probs(probs, y_valid)
        score = float(metrics["balanced_accuracy"]) + float(metrics["changed_auc"] or 0.0)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {"best_score": best_score, "history": history}


def train_vector_model(
    model: torch.nn.Module,
    *,
    train_loader,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    train_y: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    change_loss_weight: float,
    eval_batch_size: int,
) -> tuple[torch.nn.Module, dict]:
    weights = class_weights(np.asarray(train_y, dtype=np.int64), device)
    changed_weight = changed_pos_weight(np.asarray(train_y, dtype=np.int64), device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    history = []

    model.to(device)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            y = y.to(device)
            loss = progression_loss(
                model(x.to(device)),
                y,
                weights=weights,
                changed_weight=changed_weight,
                change_loss_weight=change_loss_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        probs, y_valid = predict_vectors(model, valid_x, valid_y, device=device, batch_size=eval_batch_size)
        metrics = metrics_from_probs(probs, y_valid)
        score = float(metrics["balanced_accuracy"]) + float(metrics["changed_auc"] or 0.0)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {"best_score": best_score, "history": history}
