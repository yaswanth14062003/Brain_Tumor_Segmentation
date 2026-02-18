

# Brain Tumor MRI Semantic Segmentation

This project implements a U-Net based convolutional neural network for **binary semantic segmentation** of brain tumor MRI images.

## Dataset

Expected folder structure after unzipping `dataset.zip`:

```text
dataset/
    train/
        <train image files>.jpg
        _annotations.coco.json
    valid/
        <valid image files>.jpg
        _annotations.coco.json
    test/
        <test image files>.jpg
```

The `*_annotations.coco.json` files are in COCO polygon format. The code generates binary masks on-the-fly:
- Tumor pixels: 1
- Background: 0

## Training

```bash
pip install -r requirements.txt

python train.py --data_root /path/to/dataset --epochs 30 --batch_size 4 --img_size 256
```

The best model (by validation IoU) is saved in `checkpoints/best_model.pth`.  
During training, the script prints **IoU** and **Pixel Accuracy** for the validation set each epoch.

## Test Prediction

To generate predicted masks for the test set:

```bash
python predict_test.py --data_root /path/to/dataset --checkpoint checkpoints/best_model.pth
```

Predicted masks (single-channel 0/255 PNG) will be written to `test_predictions/`.
=======
# Brain_Tumor_Segmentation
>>>>>>> 24c27aec6062f2ea4607777e72b8560fe39a45bf
