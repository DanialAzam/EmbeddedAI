"""Build and compare EVERY quantization level from one FP32 ONNX model.

From models/simple.onnx (override with --onnx) this produces:

  quantized_int8_full.onnx   INT8, every layer quantized. Smallest + fastest,
                             but on YOLO the detection head loses too much
                             precision and accuracy usually collapses. Kept as
                             the "why selective matters" baseline.

  quantized_static.onnx      INT8 with the detection head (model.<last>) left in
                             FP32. ~3-4x smaller, fast on ARM, accuracy mostly
                             preserved. THIS is what detect.py's 'quantized'
                             variant loads - the real deployment model.

  quantized_fp16.onnx        FP16 half precision. ~2x smaller, accuracy almost
                             identical to FP32, little CPU speedup (I/O kept
                             float32 so detect.py needs no changes).

Each level is benchmarked (size + CPU latency) and given a detection sanity
check (average vehicles/frame over a short clip) so you can SEE which levels
still actually detect. Run:

    python quantize_levels.py
    python quantize_levels.py --onnx models/pruned.onnx --prefix pruned_quant
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import onnx

from common import (
    BASE, MODELS_DIR, benchmark_onnx, ensure_dirs, load_config,
    onnx_input_size, _CalibReader,  # noqa: PLC2701  (intentional internal reuse)
)


def head_nodes_to_exclude(onnx_path: Path) -> list[str]:
    """Names of all nodes in the YOLO Detect head (the last /model.<N>/ module).
    Quantizing these to INT8 is what collapses YOLO accuracy, so we skip them."""
    model = onnx.load(str(onnx_path))
    idxs = set()
    for n in model.graph.node:
        m = re.search(r"/model\.(\d+)/", n.name or "")
        if m:
            idxs.add(int(m.group(1)))
    if not idxs:
        return []
    head = max(idxs)
    return [n.name for n in model.graph.node
            if f"/model.{head}/" in (n.name or "") and n.name]


def calib_images(cfg) -> list[Path]:
    cal_dir = BASE / cfg["dataset"]["calibration_dir"]
    imgs = sorted(p for p in cal_dir.glob("**/*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    return imgs[: cfg["quant"]["calibration_images"]]


def build_int8(src: Path, dst: Path, cfg, imgsz: int, exclude: list[str]) -> None:
    """Static INT8 (U8S8 scheme, per-channel). `exclude` keeps named nodes FP32.
    Calibration method from config.quant.calibration_method (Entropy clips
    activation outliers for better int8 scales), with a MinMax fallback."""
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationMethod, QuantFormat, QuantType, quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prep = src.with_suffix(".prep.onnx")
    try:
        quant_pre_process(str(src), str(prep))
        model_for_quant = prep
    except Exception as e:  # noqa: BLE001
        print(f"   (pre-process skipped: {e})")
        model_for_quant = src

    sess = ort.InferenceSession(str(model_for_quant), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    imgs = calib_images(cfg)

    table = {"entropy": CalibrationMethod.Entropy, "percentile": CalibrationMethod.Percentile,
             "minmax": CalibrationMethod.MinMax}
    primary = table.get(str(cfg.get("quant", {}).get("calibration_method", "entropy")).lower(),
                        CalibrationMethod.Entropy)
    order = [primary] + ([CalibrationMethod.MinMax] if primary != CalibrationMethod.MinMax else [])

    last_err: Exception | None = None
    for method in order:
        try:
            quantize_static(
                model_input=str(model_for_quant),
                model_output=str(dst),
                calibration_data_reader=_CalibReader(imgs, input_name, imgsz),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,   # U8 activations -> fast ARM/CPU path
                weight_type=QuantType.QInt8,        # S8 weights  (U8S8 scheme)
                per_channel=True,
                nodes_to_exclude=exclude or None,
                calibrate_method=method,
            )
            print(f"   (calibration: {method.name} on {len(imgs)} images)")
            if model_for_quant != src:
                model_for_quant.unlink(missing_ok=True)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"   ({method.name} calibration failed: {str(e)[:70]})")
    if model_for_quant != src:
        model_for_quant.unlink(missing_ok=True)
    raise last_err if last_err else RuntimeError("quantize_static produced no model")


def build_fp16(src: Path, dst: Path) -> None:
    # The external onnxconverter-common converter mangles YOLOv8's Resize/Concat
    # neck and yields a graph onnxruntime refuses to load. onnxruntime ships its
    # OWN float16 converter (in the transformers tools) that handles this model
    # correctly. keep_io_types leaves input/output FP32 so detect.py is unchanged.
    import onnxruntime as ort
    from onnxruntime.transformers.onnx_model import OnnxModel

    om = OnnxModel(onnx.load(str(src)))
    om.convert_float_to_float16(keep_io_types=True)
    om.save_model_to_file(str(dst))
    # Validate it actually loads before we depend on it.
    ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])


def build_fp16_head(selective: Path, dst: Path) -> None:
    """Mixed precision: take the selective-INT8 model (int8 body, FP32 head) and
    convert ONLY the detection head to FP16. Smaller than the FP32-head model,
    ~same accuracy (FP16 has ample range for the head). On the Pi this is a size
    win, not a speed one (ORT's CPU casts FP16->FP32)."""
    import onnxruntime as ort
    from onnxruntime.transformers.onnx_model import OnnxModel

    model = onnx.load(str(selective))
    head = set(head_nodes_to_exclude(selective))           # head node names (still FP32)
    if not head:
        raise RuntimeError("no detection-head nodes found to convert")
    body_block = [n.name for n in model.graph.node if n.name and n.name not in head]
    om = OnnxModel(model)
    # convert everything NOT in the block list -> only the head becomes FP16,
    # with Cast nodes auto-inserted at the int8/FP16 boundary.
    om.convert_float_to_float16(keep_io_types=True, node_block_list=body_block)
    om.save_model_to_file(str(dst))
    ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])  # validate


def detect_avg_vehicles(onnx_path: Path, video: Path, cfg, frames: int = 120) -> float:
    """Average vehicles/frame over the first `frames` - a quick accuracy proxy."""
    from detect import Detector  # local import; defines no side effects on import

    dcfg = cfg["detect"]
    det = Detector(str(onnx_path), conf=dcfg["conf_threshold"],
                   iou=dcfg["iou_threshold"], class_filter=set(dcfg["vehicle_class_ids"]))
    cap = cv2.VideoCapture(str(video))
    counts: list[int] = []
    while len(counts) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, _scores, _ids = det(frame)
        counts.append(len(boxes))
    cap.release()
    return float(np.mean(counts)) if counts else 0.0


def run(onnx_path: str | Path = MODELS_DIR / "simple.onnx",
        prefix: str = "quantized",
        verify_video: str | Path | None = None) -> None:
    """Build every quantization level from `onnx_path`.

    Importable so the numbered pipeline (03_quantized_model.py) and the GUI
    share this exact head-excluded logic instead of re-implementing the
    (broken) full quantization.
    """
    ensure_dirs()
    cfg = load_config()
    src = Path(onnx_path)
    if not src.exists():
        raise SystemExit(f"{src} missing - run 01_simple_model.py first.")
    imgsz = onnx_input_size(src)

    if verify_video is not None:
        verify_video = Path(verify_video)
    else:
        verify_video = BASE / "dataset" / "demo" / "highway-busy.mp4"
    if not verify_video.exists():
        verify_video = next((BASE / "dataset" / "demo").glob("*.mp4"), None)

    full = MODELS_DIR / f"{prefix}_int8_full.onnx"
    selective = MODELS_DIR / f"{prefix}_static.onnx"
    fp16head = MODELS_DIR / f"{prefix}_static_fp16head.onnx"
    fp16 = MODELS_DIR / f"{prefix}_fp16.onnx"

    head = head_nodes_to_exclude(src)
    print(f"[levels] source: {src.name}  ({imgsz}px)")
    print(f"[levels] detection-head nodes kept FP32 in selective INT8: {len(head)}")

    built: list[tuple[str, Path]] = [("FP32 baseline", src)]

    print("\n[levels] 1/4 INT8 full (aggressive, every layer)...")
    try:
        build_int8(src, full, cfg, imgsz, exclude=[])
        built.append(("INT8 full (aggressive)", full))
        print(f"         -> {full.name}")
    except Exception as e:  # noqa: BLE001
        print(f"         FAILED: {e}")

    print("[levels] 2/4 INT8 selective (head kept FP32)...")
    try:
        build_int8(src, selective, cfg, imgsz, exclude=head)
        built.append(("INT8 selective (head FP32)", selective))
        print(f"         -> {selective.name}")
    except Exception as e:  # noqa: BLE001
        print(f"         FAILED: {e}")

    print("[levels] 3/4 INT8 body + FP16 head (mixed precision)...")
    try:
        if not selective.exists():
            raise RuntimeError("needs the selective INT8 model (step 2 failed)")
        build_fp16_head(selective, fp16head)
        built.append(("INT8 body + FP16 head", fp16head))
        print(f"         -> {fp16head.name}")
    except Exception as e:  # noqa: BLE001
        print(f"         FAILED: {e}")

    print("[levels] 4/4 FP16 half precision...")
    try:
        build_fp16(src, fp16)
        built.append(("FP16 half", fp16))
        print(f"         -> {fp16.name}")
    except Exception as e:  # noqa: BLE001
        print(f"         FAILED: {e}")

    # ---- compare (each row isolated so one bad model can't kill the table) ----
    print("\n" + "=" * 78)
    print(f"{'level':28s} {'MB':>7s} {'laptop ms':>10s} {'FPS':>7s} {'avg veh/frame':>14s}")
    print("-" * 78)
    for label, path in built:
        try:
            b = benchmark_onnx(path, runs=cfg["benchmark"]["runs"],
                               threads=cfg["benchmark"]["threads"])
            avg = detect_avg_vehicles(path, verify_video, cfg) if verify_video else float("nan")
            flag = "  <-- BROKEN (0 detections)" if avg < 0.05 else ""
            print(f"{label:28s} {b['size_mb']:7.2f} {b['mean_ms']:10.1f} {b['fps']:7.1f} "
                  f"{avg:14.2f}{flag}")
        except Exception as e:  # noqa: BLE001
            print(f"{label:28s}  could not load/run: {e}")
    print("=" * 78)
    print(f"detection check ran on: {verify_video.name if verify_video else 'n/a'}")
    print(f"\nThe GUI 'quantized' variant uses {selective.name} (the working one).")
    print("View any level:  python detect.py --variant int8_full | quantized | fp16 "
          "--source dataset/demo/highway-busy.mp4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(MODELS_DIR / "simple.onnx"))
    ap.add_argument("--prefix", default="quantized")
    ap.add_argument("--verify-video", default=None,
                    help="clip for the detection sanity check (default: busiest demo)")
    args = ap.parse_args()
    run(args.onnx, args.prefix, args.verify_video)


if __name__ == "__main__":
    main()
