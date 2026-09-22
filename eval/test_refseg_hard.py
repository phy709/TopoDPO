import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASETS = {
    "aerial-d": "aerial-d/val/annotations.json",
    "risors": "risors/val/annotations.json",
}
METRIC_KEYS = (
    "iou", "f1_score", "precision", "recall", "betti0_error",
    "betti1_error", "topology_accuracy", "boundary_f1", "corner_recall",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic full-set RefSeg evaluation v2")
    parser.add_argument("--model")
    parser.add_argument("--data-root", default="/data0/data/Aerial_R1_Dataset")
    parser.add_argument("--dataset", choices=("aerial-d", "risors", "both"), default="both")
    parser.add_argument("--force-cuda", action="store_true")
    parser.add_argument("--work-dir", default="./results/full_eval_v2/manual")
    parser.add_argument("--mask-thr", type=float, default=0.5)
    parser.add_argument("--boundary-tolerance", type=int, default=3)
    parser.add_argument("--hard-manifest", default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def topology_signature(mask):
    mask = (mask > 0).astype(np.uint8)
    n_fg, _ = cv2.connectedComponents(mask, connectivity=8)
    background = (1 - mask).astype(np.uint8)
    n_bg, labels = cv2.connectedComponents(background, connectivity=4)
    holes = 0
    for label in range(1, n_bg):
        touches_border = (
            np.any(labels[0, :] == label) or np.any(labels[-1, :] == label)
            or np.any(labels[:, 0] == label) or np.any(labels[:, -1] == label)
        )
        holes += int(not touches_border)
    return int(n_fg - 1), int(holes)


def mask_boundary(mask):
    mask = (mask > 0).astype(np.uint8)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    return (mask != eroded).astype(np.uint8)


def boundary_f1(pred, gt, tolerance):
    pred_b, gt_b = mask_boundary(pred), mask_boundary(gt)
    pred_n, gt_n = int(pred_b.sum()), int(gt_b.sum())
    if pred_n == 0 and gt_n == 0:
        return 1.0
    if pred_n == 0 or gt_n == 0:
        return 0.0
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    gt_d = cv2.dilate(gt_b, kernel, iterations=1)
    pred_d = cv2.dilate(pred_b, kernel, iterations=1)
    precision = float(np.logical_and(pred_b > 0, gt_d > 0).sum()) / pred_n
    recall = float(np.logical_and(gt_b > 0, pred_d > 0).sum()) / gt_n
    return 2.0 * precision * recall / (precision + recall + 1e-12)


def macro_corners(mask):
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    corners = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue
        epsilon = 0.01 * cv2.arcLength(contour, True)
        corners.extend(point[0] for point in cv2.approxPolyDP(contour, epsilon, True))
    return np.asarray(corners) if corners else None


def corner_recall(pred, gt, tolerance=15.0):
    gt_corners, pred_corners = macro_corners(gt), macro_corners(pred)
    if gt_corners is None:
        return 1.0
    if pred_corners is None:
        return 0.0
    matched = sum(
        np.min(np.linalg.norm(pred_corners - corner, axis=1)) <= tolerance
        for corner in gt_corners
    )
    return float(matched) / len(gt_corners)


def calculate_metrics(pred, gt, boundary_tolerance):
    pred, gt = pred > 0, gt > 0
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    gt_b0, gt_b1 = topology_signature(gt)
    pred_b0, pred_b1 = topology_signature(pred)
    return {
        "iou": float(iou),
        "f1_score": float(f1_score),
        "precision": float(precision),
        "recall": float(recall),
        "betti0_error": float(abs(gt_b0 - pred_b0)),
        "betti1_error": float(abs(gt_b1 - pred_b1)),
        "topology_accuracy": float((gt_b0, gt_b1) == (pred_b0, pred_b1)),
        "boundary_f1": float(boundary_f1(pred, gt, boundary_tolerance)),
        "corner_recall": float(corner_recall(pred, gt)),
        "gt_betti0": gt_b0,
        "pred_betti0": pred_b0,
        "gt_betti1": gt_b1,
        "pred_betti1": pred_b1,
    }


def resize_prediction(prediction, height, width, threshold):
    if hasattr(prediction, "detach"):
        array = prediction.detach().float().cpu().numpy()
    else:
        array = np.asarray(prediction)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"prediction mask must be 2D after squeeze, got {array.shape}")
    if array.shape != (height, width):
        array = cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)
    return (array > threshold).astype(np.uint8)


def first_prediction(result):
    if not isinstance(result, dict):
        raise TypeError(f"predict_forward returned {type(result).__name__}, expected dict")
    predictions = result.get("prediction_masks")
    if predictions is None:
        raise KeyError("prediction_masks missing")
    if isinstance(predictions, torch.Tensor):
        if predictions.numel() == 0:
            raise ValueError("prediction_masks tensor is empty")
        return predictions[0] if predictions.ndim >= 3 else predictions
    if len(predictions) == 0:
        raise ValueError("prediction_masks is empty")
    return predictions[0]


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def evaluate_dataset(model, tokenizer, dataset_name, args):
    annotation_path = Path(args.data_root) / DATASETS[dataset_name]
    with annotation_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if args.hard_manifest:
        with open(args.hard_manifest, "r", encoding="utf-8") as manifest_handle:
            hard_ids = {item["id"] for item in json.load(manifest_handle)}
        records = [item for item in records if item.get("id") in hard_ids]
    output_dir = Path(args.work_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {key: [] for key in METRIC_KEYS}
    details, failures = [], []

    for index, item in enumerate(tqdm(records, desc=f"Evaluating {dataset_name}")):
        image_path = resolve_path(args.data_root, item["image"])
        mask_path = resolve_path(args.data_root, item["mask"])
        try:
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"missing image or mask: {image_path} | {mask_path}")
            conversations = item.get("conversations", [])
            prompt = (
                conversations[0]["value"]
                if conversations and conversations[0].get("from") == "human"
                else "<image>\nPlease segment it."
            )
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                raise ValueError(f"cv2 could not load {mask_path}")
            gt = (gt > 127).astype(np.uint8)
            with torch.inference_mode():
                result = model.predict_forward(image=image, text=prompt, tokenizer=tokenizer)
            pred = resize_prediction(first_prediction(result), height, width, args.mask_thr)
            row = calculate_metrics(pred, gt, args.boundary_tolerance)
            for key in METRIC_KEYS:
                metrics[key].append(row[key])
            details.append({"index": index, "image": str(item["image"]), **row})
        except Exception as exc:
            failure = {
                "index": index,
                "image": str(item.get("image", "")),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
            failures.append(failure)
            details.append({**failure, "failed": True})

    means = {key: (float(np.mean(values)) if values else None) for key, values in metrics.items()}
    payload = {
        "dataset": dataset_name,
        "annotation_file": str(annotation_path),
        "total_records": len(records),
        "successful_records": len(records) - len(failures),
        "failed_records": len(failures),
        "mask_threshold": args.mask_thr,
        "boundary_tolerance": args.boundary_tolerance,
        "mean_metrics": means,
        "failures": failures,
        "details": details,
    }
    with (output_dir / "metrics_full.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def run_self_test():
    gt = np.zeros((64, 64), np.uint8)
    gt[8:56, 8:56] = 1
    assert topology_signature(gt) == (1, 0)
    assert calculate_metrics(gt.copy(), gt, 3)["topology_accuracy"] == 1.0
    holed = gt.copy()
    holed[24:40, 24:40] = 0
    assert topology_signature(holed) == (1, 1)
    disconnected = gt.copy()
    disconnected[:, 31:33] = 0
    assert topology_signature(disconnected)[0] == 2
    assert 0.0 <= boundary_f1(disconnected, gt, 3) <= 1.0
    print("self-test: PASS")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if not args.model:
        raise ValueError("--model is required unless --self-test is used")
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    if args.force_cuda and not torch.cuda.is_available():
        raise RuntimeError("--force-cuda requested but CUDA is unavailable")
    device_map = {"": 0} if args.force_cuda else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=device_map,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    if args.force_cuda:
        parameter_devices = {parameter.device.type for parameter in model.parameters()}
        if parameter_devices != {"cuda"}:
            raise RuntimeError(f"forced CUDA placement failed: {sorted(parameter_devices)}")
        print("device-placement: PASS (all model parameters on CUDA)", flush=True)
    names = list(DATASETS) if args.dataset == "both" else [args.dataset]
    outputs = {name: evaluate_dataset(model, tokenizer, name, args) for name in names}
    summary = {
        "model": str(Path(args.model).resolve()),
        "datasets": {
            name: {
                "total_records": value["total_records"],
                "successful_records": value["successful_records"],
                "failed_records": value["failed_records"],
                "mean_metrics": value["mean_metrics"],
            }
            for name, value in outputs.items()
        },
    }
    with (Path(args.work_dir) / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    failures = sum(value["failed_records"] for value in outputs.values())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
