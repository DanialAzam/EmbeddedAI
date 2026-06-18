"""Live vehicle detection + congestion demo. Runs on PC AND Raspberry Pi 5
(needs only onnxruntime + opencv + numpy + pyyaml - see requirements_edge.txt).

    python detect.py                                  # config defaults (webcam)
    python detect.py --variant pruned_quantized       # pick an optimized model
    python detect.py --source dataset/demo/car-detection.mp4
    python detect.py --source rtsp://192.168.1.100:8554/cam1
    python detect.py --no-display                     # headless (prints stats)

Variants map to files in models/:
    simple            -> simple.onnx
    pruned            -> pruned.onnx
    quantized         -> quantized_static.onnx  (falls back to dynamic)
    pruned_quantized  -> pruned_quantized.onnx  (falls back to dynamic)
    nas               -> nas.onnx
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from common import (MODELS_DIR, fit_window_to_screen, image_to_blob, letterbox,
                    load_config, onnx_input_size)

# Optional system-resource readout (CPU/RAM, Task-Manager style). Degrades if absent.
try:
    import psutil
    psutil.cpu_percent(None)   # prime the system-wide counter (first call returns 0.0)
except Exception:  # noqa: BLE001
    psutil = None

VARIANTS = {
    "simple": ["simple.onnx"],
    "pruned": ["pruned.onnx"],
    "int8": ["quantized_static.onnx", "quantized_dynamic.onnx"],   # selective INT8 (recommended)
    "quantized": ["quantized_static.onnx", "quantized_dynamic.onnx"],  # alias for int8
    "fp16": ["quantized_fp16.onnx"],                 # half precision
    "int8_full": ["quantized_int8_full.onnx"],       # broken baseline - CLI only, for the report
    "pruned_quantized": ["pruned_quantized.onnx", "pruned_quantized_dynamic.onnx"],
    "nas": ["nas.onnx"],
}

LEVEL_COLORS = {"FREE": (60, 200, 60), "MODERATE": (40, 180, 230), "HEAVY": (40, 40, 220)}


def resolve_model(variant: str | None, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"model not found: {p}")
        return p
    for name in VARIANTS.get(variant or "simple", []):
        p = MODELS_DIR / name
        if p.exists():
            return p
    raise SystemExit(
        f"No model file found for variant '{variant}'. Run the pipeline scripts "
        f"first, or pass --model path\\to\\model.onnx"
    )


class Detector:
    def __init__(self, onnx_path: Path, conf: float, iou: float,
                 class_filter: set[int], threads: int = 4) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.imgsz = onnx_input_size(onnx_path)
        self.conf = conf
        self.iou = iou
        self.class_filter = class_filter

    def __call__(self, frame: np.ndarray):
        h0, w0 = frame.shape[:2]
        _, scale, pad_x, pad_y = letterbox(frame, self.imgsz)
        blob = image_to_blob(frame, self.imgsz)

        pred = self.sess.run(None, {self.input_name: blob})[0]
        pred = np.squeeze(pred, 0).T  # (anchors, 4+nc)

        scores_all = pred[:, 4:]
        class_ids = np.argmax(scores_all, axis=1)
        scores = scores_all[np.arange(len(scores_all)), class_ids]

        keep = scores >= self.conf
        if self.class_filter:
            keep &= np.isin(class_ids, list(self.class_filter))
        boxes, scores, class_ids = pred[keep, :4], scores[keep], class_ids[keep]
        if len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

        cx, cy, bw, bh = boxes.T
        x1, y1 = cx - bw / 2, cy - bh / 2
        idx = cv2.dnn.NMSBoxes(
            np.stack([x1, y1, bw, bh], 1).tolist(), scores.tolist(), self.conf, self.iou
        )
        idx = np.array(idx).flatten()
        x1, y1, bw, bh = x1[idx], y1[idx], bw[idx], bh[idx]
        scores, class_ids = scores[idx], class_ids[idx]

        # letterbox coords -> original frame coords
        x1 = np.clip((x1 - pad_x) / scale, 0, w0 - 1)
        y1 = np.clip((y1 - pad_y) / scale, 0, h0 - 1)
        x2 = np.clip(x1 + bw / scale, 0, w0 - 1)
        y2 = np.clip(y1 + bh / scale, 0, h0 - 1)
        return np.stack([x1, y1, x2, y2], 1), scores, class_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="simple", choices=list(VARIANTS))
    ap.add_argument("--model", default=None, help="explicit .onnx path (overrides variant)")
    ap.add_argument("--source", default=None, help="override config detect.source")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    cfg = load_config()["detect"]
    model_path = resolve_model(args.variant, args.model)
    print(f"[detect] model : {model_path.name}")

    det = Detector(
        model_path,
        conf=cfg["conf_threshold"], iou=cfg["iou_threshold"],
        class_filter=set(cfg["vehicle_class_ids"]),
    )
    class_names = {int(k): v for k, v in cfg["class_names"].items()}
    ccfg = cfg["congestion"]
    history: deque[float] = deque(maxlen=ccfg["smoothing_frames"])

    source = args.source if args.source is not None else cfg["source"]
    cap_src = int(source) if str(source).isdigit() else str(source)
    print(f"[detect] source: {cap_src}")
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        raise SystemExit("Could not open source. Check the path/URL/webcam index.")

    times: deque[float] = deque(maxlen=30)
    n, t_last = 0, 0.0
    res_last, cpu_pct, ram_used, ram_total, ram_pct = 0.0, 0.0, 0.0, 0.0, 0.0
    max_fps = float(cfg.get("max_fps", 60) or 0)     # cap processing rate (0 = uncapped)
    min_dt = (1.0 / max_fps) if max_fps > 0 else 0.0
    prev_period = min_dt or 0.02
    can_show = not args.no_display   # may flip to False if OpenCV has no GUI backend
    if can_show and not fit_window_to_screen("vehicle detection (q to quit)"):
        can_show = False
        print("[detect] OpenCV has no GUI backend - running headless (no window).")
    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                # loop video files; bail on dead streams after a short wait
                if isinstance(cap_src, str) and Path(cap_src).exists():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()
            boxes, scores, class_ids = det(frame)
            times.append(time.perf_counter() - t0)

            h, w = frame.shape[:2]
            density = len(boxes) / (h * w) * 100_000
            history.append(density)
            smoothed = float(np.mean(history))
            if smoothed < ccfg["free_below"]:
                level = "FREE"
            elif smoothed < ccfg["moderate_below"]:
                level = "MODERATE"
            else:
                level = "HEAVY"

            infer_ms = float(np.mean(times)) * 1000.0
            fps = 1.0 / max(prev_period, 1e-6)   # actual frame rate (capped at max_fps)
            type_counts = Counter(class_names.get(int(c), str(int(c))) for c in class_ids)
            types_str = "  ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))

            # Whole-machine usage, like Task Manager (CPU% + RAM used/total).
            # Sampled at most twice a second (cheap + steady).
            if psutil is not None and time.time() - res_last >= 0.5:
                res_last = time.time()
                cpu_pct = psutil.cpu_percent(None)               # whole machine, 0-100%
                vm = psutil.virtual_memory()
                ram_used, ram_total, ram_pct = vm.used / 1e9, vm.total / 1e9, vm.percent
            res_str = (f"CPU {cpu_pct:.0f}%   RAM {ram_used:.1f}/{ram_total:.1f} GB "
                       f"({ram_pct:.0f}%)   {infer_ms:.0f} ms"
                       if psutil is not None else f"{infer_ms:.0f} ms/frame")

            n += 1
            if time.time() - t_last >= 1.0:
                t_last = time.time()
                print(f"[detect] frame {n}: {len(boxes)} vehicles "
                      f"({types_str or 'none'}) | density {smoothed:.2f} | "
                      f"{level} | {fps:.1f} FPS | {res_str}")

            if can_show:
                color = LEVEL_COLORS[level]
                for box, s, c in zip(boxes.astype(int), scores, class_ids):
                    name = class_names.get(int(c), str(int(c)))
                    cv2.rectangle(frame, box[:2], box[2:], (255, 200, 80), 2)
                    cv2.putText(frame, f"{name} {s:.2f}", (box[0], box[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 80), 1)
                cv2.rectangle(frame, (0, 0), (w, 88), (20, 20, 20), -1)
                cv2.putText(
                    frame,
                    f"{model_path.stem} | vehicles {len(boxes)} | {level} | {fps:.1f} FPS",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
                )
                cv2.putText(
                    frame, types_str or "no vehicles in frame",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                )
                cv2.putText(
                    frame, res_str, (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 255), 1,
                )
                try:
                    cv2.imshow("vehicle detection (q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                except cv2.error:
                    can_show = False
                    print("[detect] OpenCV has no GUI backend - running headless "
                          "(no window). On a Pi: sudo apt install python3-opencv.")

            # Cap the processing/display rate at max_fps (don't run faster than needed).
            if min_dt:
                slack = min_dt - (time.perf_counter() - loop_start)
                if slack > 0:
                    time.sleep(slack)
            prev_period = time.perf_counter() - loop_start
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass   # headless OpenCV build has no GUI to tear down


if __name__ == "__main__":
    main()
