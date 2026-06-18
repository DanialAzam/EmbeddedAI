"""Torch-free mAP evaluation for .onnx models - runs on the Raspberry Pi.

Computes COCO-style mAP50 / mAP50-95 (+ precision / recall) using ONLY
onnxruntime + opencv + numpy, by running the SAME decode + NMS the deployed
detector uses (detect.Detector) over a labelled YOLO val split.

It mirrors ultralytics' validator where it matters - low conf threshold to
trace the full PR curve, per-class greedy IoU matching over 10 IoU thresholds
(0.50:0.95), and the same 101-point AP interpolation - so the numbers are
close to (but, being a separate implementation, not identical to) the PC's
ultralytics mAP. Use it for on-device accuracy when PyTorch isn't installed.

    python eval_native.py --model models/01_trained_fp32_static.onnx
    python eval_native.py --all
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from detect import Detector  # torch-free: onnxruntime + cv2 + numpy

BASE = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IOUV = np.linspace(0.5, 0.95, 10)  # COCO IoU sweep
# np.trapz was renamed np.trapezoid in NumPy 2.0; support both (Pi has NumPy<2).
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _val_images(data_yaml: str) -> list[Path]:
    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(d.get("path") or Path(data_yaml).parent)
    if not root.is_absolute():
        root = (Path(data_yaml).parent / root).resolve()
    val = d.get("val", "images/val")
    val_dir = (root / val).resolve()
    return sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def _label_path(img_path: Path) -> Path:
    """YOLO convention: .../images/<split>/x.jpg -> .../labels/<split>/x.txt."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    s = str(img_path)
    s = sb.join(s.rsplit(sa, 1)) if sa in s else s
    return Path(s).with_suffix(".txt")


def _load_gt(label_path: Path, w: int, h: int, vehicle_ids: set[int]):
    boxes, classes = [], []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if vehicle_ids and c not in vehicle_ids:
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
            classes.append(c)
    return (np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(classes, dtype=int))


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between [N,4] and [M,4] xyxy boxes -> [N,M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Average precision via 101-point interpolation (same as ultralytics)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))   # precision envelope
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x))


def evaluate_onnx(model_path, data_yaml: str, imgsz: int, vehicle_ids,
                  conf: float = 0.001, iou: float = 0.7, verbose: bool = True) -> dict:
    """Return {model, mAP50, mAP50_95, precision, recall} for one .onnx model."""
    vids = set(int(c) for c in vehicle_ids)
    det = Detector(Path(model_path), conf=conf, iou=iou, class_filter=vids)
    images = _val_images(data_yaml)
    if not images:
        raise SystemExit(f"no val images found via {data_yaml} (expected an images/val split)")

    classes = sorted(vids)
    acc = {c: {"scores": [], "tp": [], "npos": 0} for c in classes}
    for k, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        pb_all, ps_all, pc_all = det(img)                       # xyxy in original coords
        gtb, gtc = _load_gt(_label_path(img_path), w, h, vids)
        for c in classes:
            gm = gtc == c
            acc[c]["npos"] += int(gm.sum())
            pm = pc_all == c
            if not pm.any():
                continue
            pb, ps = pb_all[pm], ps_all[pm]
            order = np.argsort(-ps)
            pb, ps = pb[order], ps[order]
            tp = np.zeros((len(pb), len(IOUV)), dtype=bool)
            gb = gtb[gm]
            if len(gb):
                ious = _iou_matrix(pb, gb)
                for ti, t in enumerate(IOUV):
                    matched = np.zeros(len(gb), dtype=bool)
                    for i in range(len(pb)):
                        j = int(np.argmax(ious[i]))
                        if ious[i, j] >= t and not matched[j]:
                            tp[i, ti] = True
                            matched[j] = True
            acc[c]["scores"].append(ps)
            acc[c]["tp"].append(tp)
        if verbose and (k + 1) % 25 == 0:
            print(f"[eval-native]   {k + 1}/{len(images)} images", flush=True)

    ap = np.zeros((len(classes), len(IOUV)))
    p50, r50 = [], []
    for ci, c in enumerate(classes):
        if not acc[c]["scores"]:
            continue
        scores = np.concatenate(acc[c]["scores"])
        tp = np.concatenate(acc[c]["tp"], axis=0)
        order = np.argsort(-scores)
        tp = tp[order]
        tpc = np.cumsum(tp, axis=0)
        fpc = np.cumsum(~tp, axis=0)
        recall = tpc / (acc[c]["npos"] + 1e-9)
        precision = tpc / (tpc + fpc + 1e-9)
        for ti in range(len(IOUV)):
            ap[ci, ti] = _ap(recall[:, ti], precision[:, ti])
        f1 = 2 * precision[:, 0] * recall[:, 0] / (precision[:, 0] + recall[:, 0] + 1e-9)
        if len(f1):
            bi = int(np.argmax(f1))
            p50.append(precision[bi, 0])
            r50.append(recall[bi, 0])

    valid = [ci for ci, c in enumerate(classes) if acc[c]["npos"] > 0]
    if not valid:
        raise SystemExit("no vehicle ground-truth labels found in the val split.")
    ap = ap[valid]
    return {
        "model": Path(model_path).name,
        "mAP50": round(float(ap[:, 0].mean()), 4),
        "mAP50_95": round(float(ap.mean()), 4),
        "precision": round(float(np.mean(p50)) if p50 else 0.0, 4),
        "recall": round(float(np.mean(r50)) if r50 else 0.0, 4),
    }


def main() -> None:
    from common import MODELS_DIR, RESULTS_DIR, ensure_dirs, load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="one .onnx model to evaluate")
    ap.add_argument("--all", action="store_true", help="every .onnx in models/")
    ap.add_argument("--data", default=None, help="data.yaml (default: config dataset.data_yaml)")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_config()
    data_yaml = args.data or str(BASE / cfg["dataset"]["data_yaml"])
    if not Path(data_yaml).exists():
        raise SystemExit(f"dataset not found: {data_yaml}")
    imgsz = args.imgsz or cfg["train"]["imgsz"]
    vehicle_ids = cfg["detect"]["vehicle_class_ids"]

    if args.all:
        models = sorted(p for p in MODELS_DIR.glob("*.onnx")
                        if not p.name.endswith(".prep.onnx"))
    elif args.model:
        models = [Path(args.model)]
    else:
        raise SystemExit("pass --model <path.onnx>  or  --all")
    models = [m for m in models if m.exists()]

    print(f"[eval-native] data: {Path(data_yaml).name} | classes {vehicle_ids} "
          "| torch-free onnxruntime evaluator")
    print(f"\n{'model':30s} {'mAP50':>7s} {'mAP50-95':>9s} {'precision':>10s} {'recall':>8s}")
    print("-" * 70)
    rows = []
    for mp in models:
        try:
            r = evaluate_onnx(mp, data_yaml, imgsz, vehicle_ids, args.conf, args.iou)
        except Exception as e:  # noqa: BLE001
            print(f"{mp.name:30s}  FAILED: {str(e)[:38]}")
            continue
        rows.append(r)
        print(f"{r['model']:30s} {r['mAP50']:7.4f} {r['mAP50_95']:9.4f} "
              f"{r['precision']:10.4f} {r['recall']:8.4f}")

    if rows:
        out = RESULTS_DIR / "accuracy.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[eval-native] -> {out.relative_to(BASE)}")


if __name__ == "__main__":
    main()
