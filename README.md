# Cat and dog head detection with Faster R-CNN

Fine-tune a pretrained Faster R-CNN MobileNetV3 FPN model in PyTorch to localize cat and dog heads. This uses the tight Pascal VOC head-region boxes supplied by Oxford-IIIT Pet; it is not a full-body detector. The small backbone and 320-pixel input are selected to fit a 4 GB laptop GPU.

## Data and preparation

Run the ViT project's `prepare_data.py` once. It creates the shared image manifests and downloads the original annotations under `portfolio_projects/data/oxford-iiit-pet`. Oxford's Pascal VOC head-box XMLs cover the trainval images, while the official classification test list has no head boxes. This project therefore makes its own deterministic, breed-stratified 70/15/15 train/validation/test split from the images that have both a file and an XML box. The test set is held out from detector training and checkpoint selection. See [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/).

## Run

```powershell
portfolio_projects\.venv\Scripts\python.exe portfolio_projects\expanded\faster-rcnn-pet-detection\train.py --epochs 3
```

Training keeps the pretrained feature backbone frozen and updates the region proposal and classification/regression heads to fit available GPU memory. The script writes the exact split manifest and audit under `portfolio_projects/data/oxford-iiit-pet/manifests/faster_rcnn_*`. It selects a checkpoint on validation mAP@0.50 and reports held-out test mAP@0.50, mAP@[0.50:0.95], class-level AP, and localization/species failure cases. MLflow records each epoch and test metrics locally under `outputs/mlflow.db`.

## Measured evaluation

Three epochs ran on an RTX 3050 Laptop GPU. The deterministic, breed-stratified split contains 2,569 train, 551 validation, and 551 held-out test images across all 37 breeds; all 3,671 images have Oxford XML head boxes. The best checkpoint was selected using validation mAP@0.50. The separate Oxford classification test list has no XML box annotations, so this is a custom held-out detector split rather than the official classification test split.

| Held-out metric | Result |
| --- | ---: |
| mAP@0.50 | 0.9590 |
| mAP@0.50:0.95 | 0.6419 |
| Cat AP@0.50 | 0.9647 |
| Dog AP@0.50 | 0.9533 |
| Head box + species correct at IoU ≥ 0.50 | 522 / 551 (94.7%) |

The large drop at stricter IoU thresholds shows that many detections find the right head but do not tightly match the annotated box; mAP@0.50:0.95 is therefore more informative than mAP@0.50 alone. The model detects annotated head regions, not full-body instances. Outputs include the threshold-wise AP, validation history, and image-level failure cases under `outputs/`.

