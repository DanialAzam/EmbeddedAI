"""Shared helpers that do NOT require torch/ultralytics.

This module must stay importable on the Raspberry Pi where only
onnxruntime + opencv + numpy + pyyaml are installed. Anything that needs
torch lives in train_utils.py instead.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"
DATASET_DIR = BASE / "dataset"


def load_config() -> dict:
    with open(BASE / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for d in (MODELS_DIR, RESULTS_DIR, DATASET_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Image preprocessing (shared by detect.py and quantization calibration)
# ---------------------------------------------------------------------------

def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio, pad to (size, size) with gray.
    Returns (padded_image, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    out[top : top + nh, left : left + nw] = resized
    return out, s, left, top


def image_to_blob(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR frame -> 1x3xHxW float32 RGB blob in [0,1] (letterboxed)."""
    lb, _, _, _ = letterbox(img_bgr, size)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None]


# ---------------------------------------------------------------------------
# ONNX benchmarking
# ---------------------------------------------------------------------------

def onnx_input_size(onnx_path: str | Path, fallback: int = 320) -> int:
    """Read the (static) spatial input size from an ONNX model."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    shape = sess.get_inputs()[0].shape  # e.g. [1, 3, 320, 320]
    last = shape[-1]
    return int(last) if isinstance(last, int) else fallback


def benchmark_onnx(
    onnx_path: str | Path,
    imgsz: int | None = None,
    runs: int = 30,
    threads: int = 4,
) -> dict:
    """Measure CPU latency of an ONNX model. Returns dict with mean/std/fps/size."""
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    if imgsz is None:
        imgsz = onnx_input_size(onnx_path)

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(5):  # warmup
        sess.run(None, {name: x})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(times)
    return {
        "model": onnx_path.name,
        "size_mb": onnx_path.stat().st_size / 1e6,
        "mean_ms": mean_ms,
        "std_ms": statistics.pstdev(times),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "imgsz": imgsz,
    }


# ---------------------------------------------------------------------------
# ONNX INT8 quantization (onnxruntime only - used by scripts 03 and 04)
# ---------------------------------------------------------------------------

class _CalibReader:
    """CalibrationDataReader feeding letterboxed images from a folder."""

    def __init__(self, image_paths: list[Path], input_name: str, imgsz: int) -> None:
        self._iter = iter(image_paths)
        self._name = input_name
        self._imgsz = imgsz

    def get_next(self):
        try:
            p = next(self._iter)
        except StopIteration:
            return None
        img = cv2.imread(str(p))
        if img is None:
            return self.get_next()
        return {self._name: image_to_blob(img, self._imgsz)}


def quantize_onnx(
    src: str | Path,
    dst_static: str | Path,
    dst_dynamic: str | Path,
    calibration_dir: str | Path,
    imgsz: int,
    max_images: int = 100,
) -> dict:
    """Produce dynamic and (if calibration images exist) static INT8 models.
    Returns {'dynamic': path, 'static': path|None}."""
    import onnxruntime as ort
    from onnxruntime.quantization import (
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    src = Path(src)
    out: dict = {"dynamic": None, "static": None}

    # --- dynamic: weights-only, needs nothing extra ---
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst_dynamic),
        weight_type=QuantType.QInt8,
    )
    out["dynamic"] = Path(dst_dynamic)
    print(f"[quant] dynamic INT8 -> {dst_dynamic} "
          f"({Path(dst_dynamic).stat().st_size / 1e6:.2f} MB)")

    # --- static: quantizes activations too - the real speedup on ARM ---
    cal_dir = Path(calibration_dir)
    images = sorted(
        p for p in cal_dir.glob("**/*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[:max_images]
    if not images:
        print(f"[quant] no calibration images in {cal_dir} - skipped static INT8. "
              "Run download.py to populate it.")
        return out

    # Pre-process (shape inference) improves static quantization reliability.
    model_for_quant = src
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        prep = src.with_suffix(".prep.onnx")
        quant_pre_process(str(src), str(prep))
        model_for_quant = prep
    except Exception as e:  # noqa: BLE001 - non-fatal, fall back to raw model
        print(f"[quant] pre-process skipped ({e})")

    sess = ort.InferenceSession(str(model_for_quant), providers=["CPUExecutionProvider"])
    reader = _CalibReader(images, sess.get_inputs()[0].name, imgsz)

    quantize_static(
        model_input=str(model_for_quant),
        model_output=str(dst_static),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    if model_for_quant != src:
        model_for_quant.unlink(missing_ok=True)
    out["static"] = Path(dst_static)
    print(f"[quant] static INT8 ({len(images)} calib images) -> {dst_static} "
          f"({Path(dst_static).stat().st_size / 1e6:.2f} MB)")
    return out


def print_benchmark_row(r: dict, prefix: str = "") -> None:
    print(f"{prefix}{r['model']:38s} {r['size_mb']:7.2f} MB  "
          f"{r['mean_ms']:8.2f} ms  {r['fps']:6.1f} FPS  @{r['imgsz']}px")
