"""Check model ACCURACY (mAP / precision / recall) on the labeled validation set.

Unlike 06_benchmark.py - which reports an in-context "avg vehicles/frame" proxy -
this runs proper detection validation against ground-truth labels and reports
COCO-style mAP. It works on BOTH trainable .pt and deployed .onnx models, and
scores only the vehicle classes (car / motorcycle / bus / truck), matching the
deployed detector.

    python evaluate.py --model models/0trained_static.onnx   # one model
    python evaluate.py --all                                  # every model in models/
    python evaluate.py --model models/10pruned_fp32.pt --data dataset/vehicles_coco/data.yaml

Outputs:
    results/accuracy.csv   (model, mAP50, mAP50-95, precision, recall)

Metrics:
    mAP50      - mean Average Precision at IoU 0.50            (lenient overlap)
    mAP50-95   - mAP averaged over IoU 0.50:0.95               (the headline metric)
    precision  - of the boxes it predicted, how many were right
    recall     - of the real vehicles, how many it found
Requires torch + ultralytics + the labeled dataset -> runs on the PC, not the Pi.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import BASE, MODELS_DIR, RESULTS_DIR, ensure_dirs, load_config

try:
    import train_utils  # noqa: F401  - registers C2f_v2 so pruned .pt checkpoints unpickle
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False   # the Raspberry Pi (edge) install has no torch / ultralytics


def evaluate_one(model_path: Path, data_yaml: str, imgsz: int,
                 vehicle_ids: list[int], conf: float, iou: float) -> dict:
    """Validate one model and return its accuracy metrics."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    m = model.val(
        data=data_yaml, imgsz=imgsz, classes=vehicle_ids,
        conf=conf, iou=iou, verbose=False, plots=False,
        project=str(BASE / "runs"), name=f"val_{model_path.stem}", exist_ok=True,
    )
    return {
        "model": model_path.name,
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="path to one .pt or .onnx model to evaluate")
    ap.add_argument("--all", action="store_true",
                    help="evaluate every .pt and .onnx in models/")
    ap.add_argument("--data", default=None,
                    help="dataset data.yaml (default: config dataset.data_yaml)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="validation image size (default: config train.imgsz)")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="confidence threshold for mAP (low = standard, default 0.001)")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU for validation")
    ap.add_argument("--native", action="store_true",
                    help="force the torch-free onnxruntime evaluator (.onnx only)")
    args = ap.parse_args()

    use_native = args.native or not _HAVE_TORCH
    if use_native and not _HAVE_TORCH:
        print("[eval] PyTorch not found - using the torch-free onnxruntime evaluator "
              "(.onnx only; .pt models can only be scored on the PC).")

    ensure_dirs()
    cfg = load_config()
    data_yaml = args.data or str(BASE / cfg["dataset"]["data_yaml"])
    if not Path(data_yaml).exists():
        raise SystemExit(f"dataset not found: {data_yaml}\n"
                         "Point --data at a data.yaml with a labelled val split.")
    imgsz = args.imgsz or cfg["train"]["imgsz"]
    vehicle_ids = cfg["detect"]["vehicle_class_ids"]

    if args.all:
        models = sorted({p for p in MODELS_DIR.glob("*.onnx") if not p.name.endswith(".prep.onnx")}
                        | set(MODELS_DIR.glob("*.pt")))
    elif args.model:
        models = [Path(args.model)]
    else:
        raise SystemExit("pass --model <path>  or  --all")
    models = [m for m in models if m.exists()]
    if not models:
        raise SystemExit("no models found to evaluate (check the path / build some first).")

    print(f"[eval] data: {Path(data_yaml).name}  | vehicle classes: {vehicle_ids}  | imgsz {imgsz}")
    print(f"\n{'model':30s} {'mAP50':>7s} {'mAP50-95':>9s} {'precision':>10s} {'recall':>8s}")
    print("-" * 70)
    rows: list[dict] = []
    for mp in models:
        try:
            if use_native:
                if mp.suffix != ".onnx":
                    print(f"{mp.name:30s}  skipped (.pt needs PyTorch - score it on the PC)")
                    continue
                import eval_native
                r = eval_native.evaluate_onnx(mp, data_yaml, imgsz, vehicle_ids, conf=args.conf)
            else:
                r = evaluate_one(mp, data_yaml, imgsz, vehicle_ids, args.conf, args.iou)
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
        print(f"\n[eval] accuracy -> {out.relative_to(BASE)}")
        best = max(rows, key=lambda r: r["mAP50_95"])
        print(f"[eval] best mAP50-95: {best['model']}  ({best['mAP50_95']:.4f})")


if __name__ == "__main__":
    main()
