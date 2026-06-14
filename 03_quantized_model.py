"""STEP 3 - Quantized model (INT8 + FP16 levels).

Builds every quantization level from the FP32 baseline and verifies each by
actually detecting on a video (not just benchmarking):

  models/quantized_static.onnx       INT8, detection head kept FP32  <- DEPLOY THIS
  models/quantized_int8_full.onnx    INT8 everything (breaks YOLO - baseline)
  models/quantized_fp16.onnx         FP16 half precision

This delegates to quantize_levels.run() so the numbered pipeline and the GUI
use the SAME head-excluded logic. (Quantizing YOLO's detection head to INT8
collapses accuracy to zero - see quantize_levels.py / the README.)

    python 03_quantized_model.py
    python 03_quantized_model.py --onnx models/nas.onnx --prefix nas_quant
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import MODELS_DIR
from quantize_levels import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(MODELS_DIR / "simple.onnx"))
    ap.add_argument("--prefix", default="quantized")
    args = ap.parse_args()
    run(Path(args.onnx), args.prefix)
    print("\nNext:  python 04_pruned_quantized_model.py")


if __name__ == "__main__":
    main()
