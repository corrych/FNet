import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def morphological_edge(mask, kernel_size=3):
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
    return dilated - eroded


class MorphologicalEdgeExtractor(nn.Module):
    def __init__(self, kernel_size=3):
        super(MorphologicalEdgeExtractor, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, mask):
        return morphological_edge(mask, kernel_size=self.kernel_size)


mee = MorphologicalEdgeExtractor()


class WeightedEdgeLoss(nn.Module):
    def __init__(self, pos_weight=5.0, dice_weight=1.0, bce_weight=1.0):
        super(WeightedEdgeLoss, self).__init__()
        self.dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=False)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight

    def forward(self, y_pred, y_true):
        loss_dice = self.dice_loss_fn(y_pred, y_true)
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
        bce_loss = -(
            self.pos_weight * y_true * torch.log(y_pred)
            + (1 - y_true) * torch.log(1 - y_pred)
        )
        loss_bce = torch.mean(bce_loss)

        return self.dice_weight * loss_dice + self.bce_weight * loss_bce


criterion_edge = WeightedEdgeLoss(pos_weight=5.0, dice_weight=0.6, bce_weight=0.4)
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=False)


def seg_loss_fn(pred_mask, gt_mask):
    pred_edge = mee(pred_mask)
    gt_edge = mee(gt_mask)
    edge_loss = criterion_edge(pred_edge, gt_edge)
    mask_loss = dice_loss_fn(pred_mask, gt_mask)
    edge_weight = 0.4
    mask_weight = 0.6

    return mask_weight * mask_loss + edge_weight * edge_loss
