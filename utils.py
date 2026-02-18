
import os
from typing import Optional

import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
import numpy as np


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = (targets > 0.5).float()

        num = 2 * (probs * targets).sum(dim=(1, 2, 3))
        den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.eps
        dice = num / den
        return 1 - dice.mean()


def save_sample_predictions(
    model: nn.Module,
    dataloader,
    device: torch.device,
    output_dir: str,
    num_batches: int = 1,
) -> None:
    """Save a few sample predictions (input, GT mask, predicted mask).

    This is useful for qualitative inspection in the report.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    saved = 0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Create a grid: [input, GT, pred] for first few images
            for i in range(min(4, images.size(0))):
                img = images[i].cpu()
                gt = masks[i].cpu()
                pr = preds[i].cpu()

                # Normalize image for visualization
                grid = vutils.make_grid(
                    torch.stack([img, gt.repeat(3, 1, 1), pr.repeat(3, 1, 1)]),
                    nrow=3,
                    normalize=True,
                )
                vutils.save_image(
                    grid,
                    os.path.join(output_dir, f"sample_{saved:03d}.png"),
                )
                saved += 1

            if batch_idx + 1 >= num_batches:
                break
