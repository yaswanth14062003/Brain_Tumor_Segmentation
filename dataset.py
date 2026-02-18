

import json
import os
from typing import Callable, Optional, Tuple, List, Dict

from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class BrainTumorSegmentationDataset(Dataset):
    """Brain tumor MRI dataset using COCO-style polygon annotations.

    This dataset:
    - Reads images from `<root>/<split>` where split in {"train", "valid", "test"}
    - For train/valid: reads polygon annotations from `<root>/<split>/_annotations.coco.json`
    - Generates a binary mask on-the-fly where tumor pixels = 1, background = 0.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        img_size: int = 256,
    ) -> None:
        super().__init__()
        assert split in {"train", "valid", "test"}, f"Invalid split: {split}"
        self.root = root
        self.split = split
        self.transform = transform
        self.img_size = img_size

        self.image_dir = "C:\\Users\\yaswa\\Downloads\\dataset\\train"

        # Collect image file names (jpg / png)
        exts = (".jpg", ".jpeg", ".png")
        self.image_files: List[str] = sorted(
            [f for f in os.listdir(self.image_dir) if f.lower().endswith(exts)]
        )
        if len(self.image_files) == 0:
            raise RuntimeError(f"No image files found in {self.image_dir}")

        self.has_labels = split in {"train", "valid"}
        self.annotations_by_filename: Dict[str, List[dict]] = {}

        if self.has_labels:
            ann_path = os.path.join(self.image_dir, "_annotations.coco.json")
            if not os.path.exists(ann_path):
                raise FileNotFoundError(f"Annotation file not found: {ann_path}")
            with open(ann_path, "r") as f:
                coco = json.load(f)

            # Build map filename -> list of annotation dicts
            self.image_info_by_id = {img["id"]: img for img in coco["images"]}
            self.filename_to_id = {img["file_name"]: img["id"] for img in coco["images"]}

            ann_by_id: Dict[int, List[dict]] = {}
            for ann in coco["annotations"]:
                img_id = ann["image_id"]
                ann_by_id.setdefault(img_id, []).append(ann)

            for fname in self.image_files:
                img_id = self.filename_to_id.get(fname)
                if img_id is None:
                    # Image present in folder but not in annotations
                    self.annotations_by_filename[fname] = []
                else:
                    self.annotations_by_filename[fname] = ann_by_id.get(img_id, [])

    def __len__(self) -> int:
        return len(self.image_files)

    def _generate_mask(self, filename: str, img_size: Tuple[int, int]) -> Image.Image:
        """Generate a binary mask (PIL Image in mode 'L') from polygon annotations.

        img_size: (width, height)
        """
        w, h = img_size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        anns = self.annotations_by_filename.get(filename, [])
        for ann in anns:
            segs = ann.get("segmentation", [])
            # COCO format: list of polygons, each polygon is [x0, y0, x1, y1, ...]
            for seg in segs:
                if len(seg) < 6:
                    continue
                # Convert flat list to list of (x, y)
                poly = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                draw.polygon(poly, outline=1, fill=1)

        return mask

    def __getitem__(self, idx: int):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)

        # Load image
        img = Image.open(img_path).convert("RGB")
        original_size = img.size  # (w, h)

        if self.has_labels:
            mask = self._generate_mask(filename, original_size)
        else:
            mask = None

        # Resize image (and mask) to target size
        img = F.resize(img, [self.img_size, self.img_size])
        img_tensor = F.to_tensor(img)  # [0,1] float tensor

        if mask is not None:
            mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask, dtype="float32")).unsqueeze(0)
            mask_tensor = (mask_tensor > 0.5).float()  # ensure binary 0/1
        else:
            mask_tensor = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)

        if self.transform is not None:
            # Custom transforms must take (image, mask) and return (image, mask)
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        if self.has_labels:
            return img_tensor, mask_tensor
        else:
            return img_tensor, filename
