"""Verify the numeric precision of each model by reading the ACTUAL tensor
data types stored in its ONNX file. Proves FP32 (original YOLO) vs FP16 vs
INT8 are genuinely different - concrete evidence for the report.

    python inspect_precision.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import onnx
from onnx import TensorProto

from common import MODELS_DIR

# map ONNX dtype enum -> readable name  (1->FLOAT, 10->FLOAT16, 3->INT8, ...)
DT = {getattr(TensorProto, k): k for k in dir(TensorProto)
      if isinstance(getattr(TensorProto, k), int) and k.isupper()}


def weight_dtypes(path: Path) -> tuple[Counter, Counter]:
    """Return (count per dtype, total elements per dtype) over all initializers."""
    model = onnx.load(str(path))
    counts: Counter = Counter()
    elems: Counter = Counter()
    for init in model.graph.initializer:
        name = DT.get(init.data_type, str(init.data_type))
        n = 1
        for d in init.dims:
            n *= d
        counts[name] += 1
        elems[name] += n
    return counts, elems


def verdict(counts: Counter) -> str:
    if counts.get("FLOAT16", 0):
        return "FP16  (16-bit float)"
    if counts.get("INT8", 0) or counts.get("UINT8", 0):
        return "INT8  (8-bit integer)"
    if counts.get("FLOAT", 0):
        return "FP32  (32-bit float)"
    return "unknown"


def main() -> None:
    order = [
        ("simple.onnx",              "<- ORIGINAL YOLO (baseline)"),
        ("quantized_fp16.onnx",      "<- FP16 conversion"),
        ("quantized_static.onnx",    "<- INT8 selective (deployment)"),
        ("quantized_int8_full.onnx", "<- INT8 full (broken baseline)"),
        ("pruned.onnx",              ""),
        ("nas.onnx",                 ""),
    ]
    print(f"\n{'model':28s} {'MB':>6s}  {'precision':22s}  note")
    print("-" * 84)
    for fname, note in order:
        p = MODELS_DIR / fname
        if not p.exists():
            continue
        counts, elems = weight_dtypes(p)
        mb = p.stat().st_size / 1e6
        print(f"{fname:28s} {mb:6.2f}  {verdict(counts):22s}  {note}")

    # Detailed dtype breakdown for the three precision levels.
    print("\nweight-tensor dtype breakdown (count of tensors per type):")
    for fname in ("simple.onnx", "quantized_fp16.onnx", "quantized_static.onnx"):
        p = MODELS_DIR / fname
        if not p.exists():
            continue
        counts, _ = weight_dtypes(p)
        detail = ", ".join(f"{k}={v}" for k, v in counts.most_common())
        print(f"  {fname:26s} {detail}")
    print("\n(INT64 tensors in every model are just shape/index constants, not weights.)")


if __name__ == "__main__":
    main()
