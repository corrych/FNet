"""Checkpoint persistence and training-resume helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch


def _extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_training_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_path,
    *,
    resume_optimizer: bool,
    device,
):
    if not checkpoint_path:
        return 0, float("-inf")

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(_extract_model_state(checkpoint), strict=True)

    start_epoch = 0
    best_iou = float("-inf")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_iou = float(checkpoint.get("best_iou", float("-inf")))

        if resume_optimizer:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(
        f"loaded checkpoint from {path} | "
        f"start epoch: {start_epoch + 1} | best iou: {best_iou:.5f}"
    )
    return start_epoch, best_iou


def save_training_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_path,
    *,
    epoch: int,
    best_iou: float,
    config,
):
    path = Path(checkpoint_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_iou": float(best_iou),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": asdict(config),
        },
        path,
    )
