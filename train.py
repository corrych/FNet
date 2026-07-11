from dataclasses import dataclass

import torch
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp

from checkpoint_utils import load_training_checkpoint, save_training_checkpoint
from data_setup import train_loader, test_loader
from edgelossfunc import seg_loss_fn
from model_builder import FNet


_METRIC_KEYS = ("tp", "fp", "fn", "tn")


@dataclass
class TrainConfig:
    epochs: int = 80
    lr: float = 5e-5
    min_lr: float = 4e-5

    encoder_checkpoint: str | None = "weights/tu-caformer_s36_imagenet_encoder.pth"
    save_path: str = "checkpoints/testmodel.pth"
    resume_path: str | None = None
    resume_optimizer: bool = True


def get_final_mask(model, images):
    outputs = model(images)
    if not isinstance(outputs, dict):
        raise RuntimeError("FNet must return a dict containing 'final_mask'.")

    final_mask = outputs.get("final_mask")
    if not isinstance(final_mask, torch.Tensor):
        raise RuntimeError("FNet output key 'final_mask' must be a tensor.")

    return final_mask


def compute_loss(final_mask, gt_mask):
    gt_mask = (gt_mask > 0.5).float()
    return seg_loss_fn(final_mask, gt_mask)


def create_metric_totals(device):
    return {
        key: torch.zeros((), dtype=torch.long, device=device)
        for key in _METRIC_KEYS
    }


def collect_seg_metrics(pred_mask, gt_mask, totals):
    pred_mask = pred_mask.detach().clamp(0.0, 1.0)
    gt_mask = (gt_mask > 0.5).long()
    tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, gt_mask, mode="binary", threshold=0.5)
    for key, value in zip(_METRIC_KEYS, (tp, fp, fn, tn)):
        totals[key].add_(value.sum())


def compute_seg_metrics(totals):
    tp, fp, fn, tn = (totals[key].reshape(1, 1) for key in _METRIC_KEYS)
    return {
        "iou": float(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")),
        "f1": float(smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")),
        "precision": float(smp.metrics.precision(tp, fp, fn, tn, reduction="micro")),
        "recall": float(smp.metrics.recall(tp, fp, fn, tn, reduction="micro")),
    }


def print_seg_metrics(prefix, loss, metrics):
    print(
        f"{prefix} loss: {loss:.5f} | "
        f"iou: {metrics['iou']:.5f} | "
        f"f1: {metrics['f1']:.5f} | "
        f"precision: {metrics['precision']:.5f} | "
        f"recall: {metrics['recall']:.5f}"
    )


def run_epoch(model, dataloader, device, *, optimizer=None, description):
    if len(dataloader) == 0:
        raise ValueError(f"{description} dataloader has no batches.")

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    metric_totals = create_metric_totals(device)
    grad_context = torch.enable_grad() if is_training else torch.inference_mode()

    with grad_context:
        for images, masks in tqdm(dataloader, desc=description, leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, dtype=torch.float32, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            final_mask = get_final_mask(model, images)
            loss = compute_loss(final_mask, masks)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            collect_seg_metrics(final_mask, masks, metric_totals)

    return {
        "loss": total_loss / len(dataloader),
        "metrics": compute_seg_metrics(metric_totals),
    }


def train(model, train_loader, test_loader, cfg, device):
    if cfg.epochs <= 0:
        raise ValueError("epochs must be a positive integer")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.min_lr,
    )
    start_epoch, best_iou = load_training_checkpoint(
        model,
        optimizer,
        scheduler,
        cfg.resume_path,
        resume_optimizer=cfg.resume_optimizer,
        device=device,
    )

    if start_epoch >= cfg.epochs:
        print(
            f"checkpoint epoch is already {start_epoch}; "
            f"increase cfg.epochs above {start_epoch} to continue training."
        )
        return

    for epoch in range(start_epoch, cfg.epochs):
        print(f"epoch {epoch + 1}/{cfg.epochs}")
        learning_rate = optimizer.param_groups[0]["lr"]
        train_result = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            description="Train",
        )
        test_result = run_epoch(model, test_loader, device, description="Test")
        scheduler.step()

        print(
            f"train loss: {train_result['loss']:.5f} | "
            f"lr: {learning_rate:.8f}"
        )
        print_seg_metrics("train", train_result["loss"], train_result["metrics"])
        print_seg_metrics("test", test_result["loss"], test_result["metrics"])

        test_iou = test_result["metrics"]["iou"]
        if test_iou > best_iou:
            best_iou = test_iou
            save_training_checkpoint(
                model,
                optimizer,
                scheduler,
                cfg.save_path,
                epoch=epoch,
                best_iou=best_iou,
                config=cfg,
            )
            print(f"save best model, iou: {best_iou:.5f}")
        print("-" * 120)


if __name__ == "__main__":
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FNet(encoder_checkpoint=cfg.encoder_checkpoint).to(device)

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
    )
