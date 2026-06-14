"""STEP 6 - Benchmark every model variant side by side.

For EVERY .onnx in models/ it measures:
  - file size (MB)
  - raw CPU latency (ms) + estimated Raspberry Pi 5 FPS
  - a fixed-FPS pass (config detect.max_fps) that measures, per model:
      * accuracy proxy  = average vehicles/frame
      * CPU %           = CPU needed to hold the fixed FPS
      * RAM MB          = peak process memory at the fixed FPS
    This pass ALWAYS runs (so CPU/RAM are always filled); --show just also
    opens a window so you can watch each model.

Outputs:
    results/benchmark.csv
    results/comparison.png   <- 2x2 bar charts (each as % of max): model size,
                                avg vehicles/frame, CPU usage, RAM usage.

    python 06_benchmark.py            # measure (headless) + chart
    python 06_benchmark.py --show     # same, but also watch each model on video
    python 06_benchmark.py --map      # also true mAP50-95 for .pt variants (slow)
    python 06_benchmark.py --no-accuracy   # size/latency only (CPU/RAM/acc blank)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import BASE, MODELS_DIR, RESULTS_DIR, benchmark_onnx, ensure_dirs, load_config


def measure_pass(onnx_files, video, cfg, seconds_per: float, show: bool) -> dict:
    """Run each model on the clip for `seconds_per` at the SAME capped FPS
    (detect.max_fps), measuring accuracy + CPU% + RAM. Works headless; if
    `show`, also draws a window (q=next model, ESC=stop showing).
    Returns {model_name: {"acc","cpu","ram"}}."""
    import gc
    import time
    import cv2
    import numpy as np
    from detect import Detector

    try:
        import psutil
        proc = psutil.Process()
        cores = psutil.cpu_count() or 1
        proc.cpu_percent(None)
    except Exception:  # noqa: BLE001
        proc = None
        cores = 1

    dcfg = cfg["detect"]
    max_fps = float(dcfg.get("max_fps", 60) or 0)
    min_dt = (1.0 / max_fps) if max_fps > 0 else 0.0
    win = "Benchmark - each model @ fixed FPS   (q = next, ESC = stop showing)"
    out: dict = {}
    print(f"[06] {'showing' if show else 'measuring'} {len(onnx_files)} models on "
          f"{video.name} at {int(max_fps)} FPS (~{seconds_per:.0f}s each)...")
    aborted = False
    for f in onnx_files:
        if aborted:
            break
        try:
            det = Detector(str(f), conf=dcfg["conf_threshold"], iou=dcfg["iou_threshold"],
                           class_filter=set(dcfg["vehicle_class_ids"]))
        except Exception as e:  # noqa: BLE001
            print(f"   (skipped {f.name}: {e})")
            continue
        cap = cv2.VideoCapture(str(video))
        if proc is not None:
            proc.cpu_percent(None)              # reset CPU baseline for this model
        t_end = time.time() + seconds_per
        cs, cpu_s, ram_s = [], [], []
        cur_cpu, last_s, prev = 0.0, time.time(), time.perf_counter()
        showing = show
        while time.time() < t_end:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            boxes, _scores, _ids = det(frame)
            cs.append(len(boxes))
            now = time.perf_counter()
            inst = 1.0 / max(now - prev, 1e-6)
            prev = now
            fps = min(inst, max_fps) if max_fps > 0 else inst
            if proc is not None and time.time() - last_s >= 0.4:
                last_s = time.time()
                cur_cpu = proc.cpu_percent(None) / cores
                cpu_s.append(cur_cpu)
                ram_s.append(proc.memory_info().rss / 1e6)
            if showing:
                try:
                    for b in boxes.astype(int):
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 200, 80), 2)
                    h, w = frame.shape[:2]
                    extra = (f"  |  CPU {cur_cpu:.0f}%  |  RAM {(ram_s[-1] if ram_s else 0):.0f}MB"
                             if proc is not None else "")
                    cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)
                    cv2.putText(frame, f"{f.stem}  |  veh {len(boxes)}  |  {fps:.0f} FPS"
                                f" (target {int(max_fps)}){extra}", (10, 27),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 220, 255), 2)
                    cv2.imshow(win, frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == 27:
                        aborted = True
                        break
                except cv2.error:
                    showing = False  # no display - keep measuring headlessly
            if min_dt:
                slack = min_dt - (time.perf_counter() - loop_start)
                if slack > 0:
                    time.sleep(slack)
        cap.release()
        out[f.name] = {
            "acc": round(float(np.mean(cs)), 2) if cs else 0.0,
            "cpu": round(float(np.mean(cpu_s)), 1) if cpu_s else "",
            "ram": round(float(np.max(ram_s)), 0) if ram_s else "",
        }
        del det
        gc.collect()                            # free this model before the next
    if show:
        cv2.destroyAllWindows()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", action="store_true",
                    help="also evaluate true mAP50-95 for simple/pruned/nas .pt files")
    ap.add_argument("--no-accuracy", action="store_true",
                    help="skip the fixed-FPS pass (leaves accuracy/CPU/RAM blank)")
    ap.add_argument("--show", action="store_true",
                    help="also open a window to watch each model during the pass")
    ap.add_argument("--show-seconds", type=float, default=5.0,
                    help="seconds per model in the fixed-FPS pass")
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames", type=int, default=100)  # kept for compatibility
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_config()
    scale = cfg["nas"]["pi_scale_factor"]
    fixed_fps = int(cfg["detect"].get("max_fps", 60) or 0)

    onnx_files = sorted(MODELS_DIR.glob("*.onnx"))
    if not onnx_files:
        raise SystemExit("No .onnx files in models/ - build some in the Train/Optimize tabs first.")

    verify_video = Path(args.video) if args.video else (BASE / "dataset" / "demo" / "highway-busy.mp4")
    if not verify_video.exists():
        verify_video = next((BASE / "dataset" / "demo").glob("*.mp4"), None)

    # Always run the fixed-FPS pass (accuracy + CPU + RAM) unless explicitly skipped.
    vis: dict = {}
    if not args.no_accuracy and verify_video is not None:
        vis = measure_pass(onnx_files, verify_video, cfg, args.show_seconds, show=args.show)

    maps: dict[str, float] = {}
    if args.map:
        import train_utils  # noqa: F401
        from ultralytics import YOLO
        data_yaml = str(BASE / cfg["dataset"]["data_yaml"])
        for stem in ("simple", "pruned", "nas"):
            pt = MODELS_DIR / f"{stem}.pt"
            if pt.exists():
                print(f"[06] evaluating mAP: {pt.name} (slow on CPU)...")
                try:
                    m = YOLO(str(pt)).val(data=data_yaml, imgsz=cfg["train"]["imgsz"], verbose=False)
                    maps[stem] = round(float(m.box.map), 4)
                except Exception as e:  # noqa: BLE001
                    print(f"[06]   skipped ({e})")

    cpu_key = f"cpu_pct_at_{fixed_fps}fps"
    rows = []
    print(f"\n{'model':28s} {'MB':>7s} {'estPiFPS':>9s} {'veh/frm':>8s} {'CPU%':>6s} {'RAM MB':>7s}")
    print("-" * 70)
    for f in onnx_files:
        r = benchmark_onnx(f, runs=cfg["benchmark"]["runs"], threads=cfg["benchmark"]["threads"])
        est_ms = r["mean_ms"] * scale
        stem = f.stem.replace("_static", "").replace("_dynamic", "")
        info = vis.get(f.name, {})
        acc, cpu, ram = info.get("acc", ""), info.get("cpu", ""), info.get("ram", "")
        rows.append({
            "model": f.name,
            "size_mb": round(r["size_mb"], 2),
            "laptop_ms": round(r["mean_ms"], 1),
            "est_pi_fps": round(1000.0 / est_ms, 1),
            "veh_per_frame": acc,
            cpu_key: cpu,
            "ram_mb": ram,
            "mAP50_95": maps.get(stem, ""),
        })
        rr = rows[-1]
        print(f"{rr['model']:28s} {rr['size_mb']:7.2f} {rr['est_pi_fps']:9.1f} "
              f"{str(acc):>8s} {str(cpu):>6s} {str(ram):>7s}")

    csv_path = RESULTS_DIR / "benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n[06] csv -> {csv_path.relative_to(BASE)}")

    # ---- 2x2 bar charts, each normalized to % of its own max (max = 100%) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["model"].replace(".onnx", "") for r in rows]

        def col(key):
            return [float(r[key]) if r[key] not in ("", None) else 0.0 for r in rows]

        size, acc, cpu, ram = col("size_mb"), col("veh_per_frame"), col(cpu_key), col("ram_mb")
        # (title, values, colour, unit)
        panels = [
            ("Model size (MB)  —  lower = smaller", size, "#2563EB", "MB"),
            ("Accuracy: avg vehicles / frame  —  higher = better", acc, "#7C3AED", "veh/f"),
            (f"CPU usage @ {fixed_fps} FPS (%)  —  lower = lighter", cpu, "#EA580C", "%"),
            ("RAM usage (MB) @ fixed FPS  —  lower = lighter", ram, "#0891B2", "MB"),
        ]
        height = max(9.0, 0.62 * len(names) + 3.0)
        fig, axes = plt.subplots(2, 2, figsize=(16, height))
        for ax, (title, vals, color, unit) in zip(axes.flatten(), panels):
            mx = max(vals) if vals and max(vals) > 0 else 1.0
            pct = [v / mx * 100.0 for v in vals]
            bars = ax.barh(names, pct, color=color)
            ax.set_title(title, fontweight="bold", fontsize=11)
            ax.set_xlabel("% of max  (max = 100%)")
            ax.set_xlim(0, 122)
            ax.invert_yaxis()
            ax.tick_params(axis="y", labelsize=8)
            # label shows the real value WITH ITS UNIT, then the % of max
            labels = [f"{v:g} {unit}  ({p:.0f}%)" if v else "0" for v, p in zip(vals, pct)]
            ax.bar_label(bars, labels=labels, padding=3, fontsize=7.5)
        fig.suptitle(f"Model comparison — all variants @ fixed {fixed_fps} FPS  (bars = % of max)",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        png_path = RESULTS_DIR / "comparison.png"
        fig.savefig(png_path, dpi=150)
        print(f"[06] chart -> {png_path.relative_to(BASE)}  (2x2: size, accuracy, CPU, RAM as %)")
    except Exception as e:  # noqa: BLE001
        print(f"[06] chart skipped ({e})")


if __name__ == "__main__":
    main()
