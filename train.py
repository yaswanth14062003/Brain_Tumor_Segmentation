
import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainTumorSegmentationDataset
from model import UNet
from metrics import compute_iou_and_accuracy
from utils import DiceLoss, save_sample_predictions


def get_transforms():
    import random

    def _transform(image, mask):
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
        return image, mask

    def transform_batch(images, masks):
        out_imgs = []
        out_masks = []
        for i in range(images.size(0)):
            img, m = _transform(images[i], masks[i])
            out_imgs.append(img)
            out_masks.append(m)
        return torch.stack(out_imgs, dim=0), torch.stack(out_masks, dim=0)

    return transform_batch


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_bce: nn.Module,
    criterion_dice: nn.Module,
    device: torch.device,
    epoch: int,
):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss_bce = criterion_bce(logits, masks)
        loss_dice = criterion_dice(logits, masks)
        loss = loss_bce + loss_dice

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())

    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss_bce = criterion_bce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss = loss_bce + loss_dice

            iou, acc = compute_iou_and_accuracy(logits, masks)

            total_loss += loss.item() * images.size(0)
            total_iou += iou * images.size(0)
            total_acc += acc * images.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_iou / n, total_acc / n


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor MRI Segmentation Training")
    parser.add_argument("--data_root", type=str, default="./dataset",
                        help="Path to dataset root containing train/valid/test folders (unzipped dataset.zip)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device(args.device)

    # Datasets and loaders
    train_dataset = BrainTumorSegmentationDataset(
        root=args.data_root,
        split="train",
        transform=None,
        img_size=args.img_size,
    )
    valid_dataset = BrainTumorSegmentationDataset(
        root=args.data_root,
        split="valid",
        transform=None,
        img_size=args.img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = UNet(n_channels=3, n_classes=1)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    best_iou = 0.0
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_bce, criterion_dice, device, epoch
        )
        val_loss, val_iou, val_acc = evaluate(model, valid_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best model based on IoU
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": val_iou,
                    "val_acc": val_acc,
                },
                best_model_path,
            )
            print(f"-> New best model saved with IoU {best_iou:.4f} at epoch {epoch}")

    # Save a few qualitative samples using the best model
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model from epoch {ckpt['epoch']} with IoU {ckpt['val_iou']:.4f}")

    sample_output_dir = os.path.join(args.checkpoint_dir, "sample_predictions")
    save_sample_predictions(model, valid_loader, device, sample_output_dir, num_batches=2)
    print(f"Sample predictions saved to: {sample_output_dir}")
if __name__ == "__main__":
    main()
