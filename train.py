"""Fine-tune Faster R-CNN for cat/dog head localization on Oxford-IIIT Pet."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou


SEED = 42
CLASS_NAMES = {1: "cat", 2: "dog"}


class PetHeadDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame):
        self.rows = manifest.reset_index(drop=True)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def read_box(xml_path: str, width: int, height: int) -> list[float]:
        root = ET.parse(xml_path).getroot()
        box = root.find(".//object/bndbox")
        if box is None:
            raise ValueError(f"No Pascal VOC object/bndbox in {xml_path}")
        xmin = max(0, int(float(box.findtext("xmin", "1"))) - 1)
        ymin = max(0, int(float(box.findtext("ymin", "1"))) - 1)
        xmax = min(width, int(float(box.findtext("xmax", str(width)))))
        ymax = min(height, int(float(box.findtext("ymax", str(height)))))
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid bounding box in {xml_path}")
        return [float(xmin), float(ymin), float(xmax), float(ymax)]

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            tensor = self.to_tensor(image)
        box = self.read_box(str(row.xml_path), width, height)
        label = 1 if row.species == "cat" else 2
        target = {
            "boxes": torch.tensor([box], dtype=torch.float32),
            "labels": torch.tensor([label], dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor([(box[2] - box[0]) * (box[3] - box[1])], dtype=torch.float32),
            "iscrowd": torch.zeros(1, dtype=torch.int64),
        }
        return tensor, target, str(row.image_id)


def collate_fn(batch):
    images, targets, image_ids = zip(*batch)
    return list(images), list(targets), list(image_ids)


def make_detection_splits(data_root: Path):
    """Create a reproducible 70/15/15 split from images with actual XML boxes.

    Oxford supplies Pascal VOC head boxes for its trainval images; the official
    classification test list has no XML boxes. A detector-specific held-out
    split is therefore required to evaluate localization without empty labels.
    """
    manifest_dir = data_root / "manifests"
    manifest = pd.read_csv(manifest_dir / "manifest.csv")
    manifest = manifest.dropna(subset=["image_path", "xml_path"])
    has_image = manifest["image_path"].map(lambda value: Path(value).is_file())
    has_box = manifest["xml_path"].map(lambda value: Path(value).is_file())
    annotated = manifest.loc[has_image & has_box].copy().reset_index(drop=True)
    if annotated.empty:
        raise RuntimeError("No images with both pixels and Oxford XML head boxes were found")

    train_val, test = train_test_split(
        annotated,
        test_size=0.15,
        random_state=SEED,
        stratify=annotated["breed_id"],
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        random_state=SEED,
        stratify=train_val["breed_id"],
    )
    splits = {"train": train, "validation": validation, "test": test}
    for name, frame in splits.items():
        frame.to_csv(manifest_dir / f"faster_rcnn_{name}.csv", index=False)
    audit = {
        "dataset": "Oxford-IIIT Pet annotated head boxes",
        "split_method": "deterministic breed-stratified 70/15/15; seed 42",
        "source_rows_with_image_and_xml": int(len(annotated)),
        "split_counts": {name: int(len(frame)) for name, frame in splits.items()},
        "breeds": int(annotated["breed_id"].nunique()),
        "official_test_has_head_boxes": False,
    }
    (manifest_dir / "faster_rcnn_split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return splits["train"].copy(), splits["validation"].copy(), splits["test"].copy()


def build_model(device):
    weights = models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes=3)
    model.transform.min_size = (320,)
    model.transform.max_size = 480
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    return model.to(device)


def average_precision_by_threshold(y_true, y_pred, thresholds) -> dict[float, dict[str, object]]:
    """Greedy per-class AP, computing each detection's IoU only once."""
    prepared = {}
    for label, name in CLASS_NAMES.items():
        # The caller supplies (image_id, box, label) records in these flat lists.
        gt_by_image: dict[str, list[list[float]]] = {}
        for image_id, box, gt_label in y_true:
            if gt_label == label:
                gt_by_image.setdefault(image_id, []).append(box)
        n_gt = sum(len(boxes) for boxes in gt_by_image.values())
        detections = [row for row in y_pred if row["label"] == label]
        detections.sort(key=lambda row: row["score"], reverse=True)
        overlaps_by_detection = []
        for detection in detections:
            boxes = gt_by_image.get(detection["image_id"], [])
            if not boxes:
                overlaps_by_detection.append((detection["image_id"], -1, 0.0))
                continue
            left_top = np.maximum(np.asarray(detection["box"][:2]), np.asarray(boxes)[:, :2])
            right_bottom = np.minimum(np.asarray(detection["box"][2:]), np.asarray(boxes)[:, 2:])
            intersection = np.prod(np.maximum(0.0, right_bottom - left_top), axis=1)
            detection_area = np.prod(np.maximum(0.0, np.asarray(detection["box"][2:]) - detection["box"][:2]))
            gt_areas = np.prod(np.maximum(0.0, np.asarray(boxes)[:, 2:] - np.asarray(boxes)[:, :2]), axis=1)
            overlaps = intersection / np.maximum(detection_area + gt_areas - intersection, 1e-12)
            best_index = int(overlaps.argmax())
            overlaps_by_detection.append((detection["image_id"], best_index, float(overlaps[best_index])))
        prepared[label] = (name, n_gt, len(detections), overlaps_by_detection)

    results = {}
    for threshold in thresholds:
        class_results = {}
        aps = []
        for _label, (name, n_gt, detection_count, overlaps) in prepared.items():
            # Each image in Oxford-IIIT Pet has one annotated head box, but this
            # set-based matching also handles multiple boxes if the data changes.
            tp, fp = [], []
            matched: dict[str, set[int]] = {}
            for image_id, best_index, overlap in overlaps:
                if (best_index >= 0 and overlap >= threshold
                        and best_index not in matched.setdefault(image_id, set())):
                    tp.append(1.0)
                    fp.append(0.0)
                    matched[image_id].add(best_index)
                else:
                    tp.append(0.0)
                    fp.append(1.0)
            if n_gt == 0:
                ap = float("nan")
            elif not tp:
                ap = 0.0
            else:
                tp_sum = np.cumsum(tp)
                fp_sum = np.cumsum(fp)
                recall = tp_sum / n_gt
                precision = tp_sum / np.maximum(tp_sum + fp_sum, 1e-12)
                ap = float(np.mean([precision[recall >= point].max() if np.any(recall >= point) else 0.0
                                    for point in np.linspace(0, 1, 101)]))
            class_results[name] = {"ap": ap, "ground_truth_objects": int(n_gt), "detections": detection_count}
            if not math.isnan(ap):
                aps.append(ap)
        results[float(threshold)] = {"classes": class_results, "map": float(np.mean(aps)) if aps else 0.0}
    return results


def _collect_records(targets, image_ids):
    return [
        (image_id, box.tolist(), int(label))
        for target, image_id in zip(targets, image_ids)
        for box, label in zip(target["boxes"], target["labels"])
    ]


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    model.eval()
    ground_truth, predictions, errors = [], [], []
    for batch_index, (images, targets, image_ids) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        outputs = model([image.to(device) for image in images])
        ground_truth.extend(_collect_records(targets, image_ids))
        for output, image_id, target in zip(outputs, image_ids, targets):
            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()
            selected = scores >= 0.05
            for box, score, label in zip(boxes[selected], scores[selected], labels[selected]):
                predictions.append(
                    {"image_id": image_id, "box": box.tolist(), "score": float(score), "label": int(label)}
                )
            best = int(scores.argmax()) if len(scores) else -1
            best_correct = False
            if best >= 0 and len(target["boxes"]):
                overlap = box_iou(boxes[best : best + 1], target["boxes"])[0].max().item()
                best_correct = int(labels[best]) == int(target["labels"][0]) and overlap >= 0.50
            if not best_correct:
                errors.append(
                    {
                        "image_id": image_id,
                        "ground_truth_species": CLASS_NAMES[int(target["labels"][0])],
                        "best_prediction_species": CLASS_NAMES.get(int(labels[best]), "background") if best >= 0 else "none",
                        "best_score": float(scores[best]) if best >= 0 else 0.0,
                        "best_iou": float(overlap) if best >= 0 and len(target["boxes"]) else 0.0,
                    }
                )
    thresholds = np.arange(0.50, 0.96, 0.05)
    raw_by_threshold = average_precision_by_threshold(ground_truth, predictions, thresholds)
    map_by_threshold = {f"ap{threshold:.2f}": result for threshold, result in raw_by_threshold.items()}
    map50 = map_by_threshold["ap0.50"]
    map50_95 = float(np.mean([result["map"] for result in map_by_threshold.values()]))
    return {
        "map50": map50["map"],
        "map50_95": map50_95,
        "ap_by_class_at_iou_0_50": map50["classes"],
        "ap_by_threshold": {key: value["map"] for key, value in map_by_threshold.items()},
        "ground_truth_objects": len(ground_truth),
        "predictions_at_score_0_05": len(predictions),
        "failure_cases": errors,
        "confusion_counts": {
            "head_detection_success_iou50": len(ground_truth) - len(errors),
            "head_detection_failure_or_wrong_species": len(errors),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    project = Path(__file__).resolve().parent
    data_default = project.parents[1] / "data" / "oxford-iiit-pet"
    parser.add_argument("--data-root", type=Path, default=data_default)
    parser.add_argument("--output-dir", type=Path, default=project / "outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0, help="Optional short debug run; 0 uses all data")
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}; CUDA available: {torch.cuda.is_available()}")
    if device.type == "cpu":
        torch.set_num_threads(4)

    train_frame, val_frame, test_frame = make_detection_splits(args.data_root)

    train_loader = DataLoader(
        PetHeadDataset(train_frame), batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        PetHeadDataset(val_frame), batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        PetHeadDataset(test_frame), batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )
    model = build_model(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.002, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs - 1), gamma=0.3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(args.output_dir / 'mlflow.db').resolve().as_posix()}")
    mlflow.set_experiment("oxford-pet-faster-rcnn-head-detection")
    best_val = -1.0
    best_path = args.output_dir / "fasterrcnn_best.pt"
    history = []
    with mlflow.start_run(run_name="mobilenetv3-faster-rcnn"):
        mlflow.log_params(
            {
                "architecture": "Faster R-CNN MobileNetV3 Large 320 FPN",
                "target": "one Pascal VOC head box per pet; cat/dog species class",
                "train_samples": len(train_frame),
                "validation_samples": len(val_frame),
                "test_samples": len(test_frame),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "device": str(device),
                "seed": SEED,
            }
        )
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_losses = []
            started = time.perf_counter()
            for batch_index, (images, targets, _ids) in enumerate(train_loader):
                if args.max_batches and batch_index >= args.max_batches:
                    break
                images = [image.to(device) for image in images]
                targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                optimizer.zero_grad(set_to_none=True)
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite detector loss at epoch {epoch}: {loss_dict}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            scheduler.step()
            val_metrics = evaluate(model, validation_loader, device, args.max_batches)
            record = {
                "epoch": epoch,
                "mean_training_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                "validation_map50": val_metrics["map50"],
                "validation_map50_95": val_metrics["map50_95"],
                "seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(record)
            mlflow.log_metrics(
                {key: float(value) for key, value in record.items() if key not in {"epoch", "seconds"} and value is not None},
                step=epoch,
            )
            if val_metrics["map50"] > best_val:
                best_val = val_metrics["map50"]
                torch.save(model.state_dict(), best_path)
        (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        test_metrics = evaluate(model, test_loader, device, args.max_batches)
        test_metrics.update(
            {
                "split": "deterministic breed-stratified 15% holdout from XML-annotated images; seed 42; checkpoint selected by validation mAP@0.50",
                "validation_best_map50": best_val,
                "test_samples": int(len(test_frame)),
                "target_box": "annotated tight head region, not full-body instance mask",
            }
        )
        (args.output_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        pd.DataFrame(test_metrics["failure_cases"]).to_csv(args.output_dir / "failure_cases.csv", index=False)
        mlflow.log_metrics({"test_map50": test_metrics["map50"], "test_map50_95": test_metrics["map50_95"]})
        mlflow.log_artifact(str(args.output_dir / "metrics.json"))
        mlflow.log_artifact(str(best_path))
    print(json.dumps({key: value for key, value in test_metrics.items() if key != "failure_cases"}, indent=2))


if __name__ == "__main__":
    main()

