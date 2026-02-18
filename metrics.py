
from typing import Tuple

import torch


def compute_iou_and_accuracy(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Tuple[float, float]:
    """Compute mean IoU and pixel accuracy for a batch.

    Args:
        logits: model outputs before sigmoid, shape [B, 1, H, W]
        masks: ground truth binary masks, shape [B, 1, H, W]
        threshold: threshold for converting probabilities to binary predictions
    Returns:
        (mean_iou, pixel_accuracy)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    masks = (masks > 0.5).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = (preds + masks).clamp(0, 1).sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)

    correct = (preds == masks).float().sum(dim=(1, 2, 3))
    total = masks[0].numel() * masks.shape[0]
    acc = correct.sum() / (total + eps)

    return iou.mean().item(), acc.item()
