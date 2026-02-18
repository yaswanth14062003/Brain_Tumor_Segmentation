
import argparse
import os
import torch
from torch.utils.data import DataLoader

from dataset import BrainTumorSegmentationDataset
from model import UNet

from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for test set")
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="test_predictions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)

    # Load test dataset (no labels)
    test_dataset = BrainTumorSegmentationDataset(
        root=args.data_root,
        split="test",
        transform=None,
        img_size=args.img_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Load model
    model = UNet(n_channels=3, n_classes=1)
    model.to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Prediction loop
    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Save each predicted mask
            for i in range(preds.size(0)):
                mask = preds[i]                    # shape [1, H, W], float
                mask_np = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

                # Ensure the filename ends with .png
                base = filenames[i]
                if not base.lower().endswith((".png", ".jpg", ".jpeg")):
                    base = os.path.splitext(base)[0] + ".png"

                out_path = os.path.join(args.output_dir, base)
                Image.fromarray(mask_np).save(out_path)

    print(f"Test predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
