"""STEP 1 - Simple (baseline) model.

Fine-tunes COCO-pretrained YOLOv8n on the dataset, exports it to ONNX, and
benchmarks it. Every later optimization (pruning, quantization, NAS) is
measured against THIS model, so run it first.

    python 01_simple_model.py                                   # train per config.yaml
    python 01_simple_model.py --epochs 0                         # just export pretrained
    python 01_simple_model.py --data dataset/vehicles_coco/data.yaml --epochs 30 --out simple_visdrone

Outputs:
    models/<out>.pt   (prunable source)
    models/<out>.onnx (deployable / detect / quantize source)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import BASE, MODELS_DIR, benchmark_onnx, ensure_dirs, load_config, print_benchmark_row
from train_utils import export_onnx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="dataset data.yaml (default: config dataset.data_yaml)")
    ap.add_argument("--weights", default=None,
                    help="trainable .pt to start FROM (default: pretrained yolov8n). "
                         "ONNX can't be trained - pass a .pt.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override config (0 = skip training, use pretrained as-is)")
    ap.add_argument("--imgsz", type=int, default=None, help="override config train.imgsz")
    ap.add_argument("--out", default="simple",
                    help="output name stem -> models/<out>.pt + .onnx")
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_config()
    tcfg = cfg["train"]
    epochs = tcfg["epochs"] if args.epochs is None else args.epochs
    imgsz = tcfg["imgsz"] if args.imgsz is None else args.imgsz
    data_yaml = args.data if args.data else cfg["dataset"]["data_yaml"]
    if not Path(data_yaml).is_absolute():
        data_yaml = str(BASE / data_yaml)

    base_weights = MODELS_DIR / "base" / "yolov8n.pt"
    src_weights = Path(args.weights) if args.weights else base_weights
    if src_weights.suffix.lower() != ".pt":
        raise SystemExit(f"Can't train from {src_weights.name} - training needs a .pt "
                         "(ONNX is frozen). Pick a .pt source, e.g. simple.pt.")
    if not src_weights.exists():
        raise SystemExit(f"{src_weights} missing - run download.py (base) or train it first.")

    out_pt = MODELS_DIR / f"{args.out}.pt"

    from ultralytics import YOLO

    if epochs > 0:
        print(f"[01] training from {src_weights.name} for {epochs} epochs @ {imgsz}px on "
              f"{Path(data_yaml).parent.name}  (CPU: expect a few minutes per epoch)")
        yolo = YOLO(str(src_weights))
        yolo.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=tcfg["batch"],
            name=args.out,
            project=str(BASE / "runs"),
            exist_ok=True,
        )
        best = Path(yolo.trainer.best)
        shutil.copy2(best, out_pt)
        metrics = yolo.val(data=data_yaml, imgsz=imgsz, verbose=False)
        print(f"[01] val mAP50-95 = {metrics.box.map:.4f}   mAP50 = {metrics.box.map50:.4f}")
    else:
        print(f"[01] --epochs 0: exporting {src_weights.name} directly (no fine-tune)")
        shutil.copy2(src_weights, out_pt)

    print(f"[01] saved -> {out_pt.relative_to(BASE)}")

    onnx_path = export_onnx(out_pt, MODELS_DIR / f"{args.out}.onnx", imgsz)

    bench = benchmark_onnx(onnx_path,
                           runs=cfg["benchmark"]["runs"],
                           threads=cfg["benchmark"]["threads"])
    print(f"\n[01] {args.out} benchmark (laptop CPU):")
    print_benchmark_row(bench, prefix="     ")
    print(f"\nNext:  prune/quantize it via the GUI, or 02_pruned_model.py --weights models/{args.out}.pt")


if __name__ == "__main__":
    main()
