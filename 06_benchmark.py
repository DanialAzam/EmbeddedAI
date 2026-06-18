"""STEP 6 - Benchmark every model variant side by side, on THIS device.

For EVERY .onnx in models/ it measures, on whatever machine runs it (laptop or
Raspberry Pi 5 - the DEVICE NAME is recorded in the results; there is NO
cross-device simulation):
  - file size (MB)
  - raw CPU latency (ms) and max achievable FPS (run flat-out, no FPS cap)
  - a short pass on a demo clip measuring, per model:
      * accuracy proxy  = average vehicles/frame
      * CPU %           = average CPU load while running at full speed
      * RAM MB          = peak process memory
  - (optional, --map) true mAP50-95 on the labelled validation set

Outputs:
    results/benchmark.csv
    results/comparison.png   <- 2 rows x 3 cols bar charts (each as % of max):
                                size, accuracy, max FPS, CPU, RAM, mAP.

    python 06_benchmark.py            # measure (headless) + chart
    python 06_benchmark.py --show     # also watch each model on video
    python 06_benchmark.py --map      # also true mAP50-95 (needs ultralytics + data)
    python 06_benchmark.py --device "Raspberry Pi 5"   # name the device in results
"""
from __future__ import annotations

import argparse
import csv
import platform
from pathlib import Path

from common import (BASE, MODELS_DIR, RESULTS_DIR, benchmark_onnx, ensure_dirs,
                    fit_window_to_screen, load_config)


def device_name(cfg) -> str:
    """Human name for the machine running the benchmark (no Pi simulation)."""
    name = cfg.get("benchmark", {}).get("device")
    if name:
        return str(name)
    m = platform.machine().lower()
    if m in ("aarch64", "arm64", "armv7l", "armv6l"):
        return "Raspberry Pi 5"
    return "Laptop CPU"          # set benchmark.device in config.yaml for a specific name


def measure_pass(onnx_files, video, cfg, seconds_per: float, show: bool) -> dict:
    """Run each model on the clip for `seconds_per` at FULL speed (no FPS cap),
    measuring accuracy + CPU% + RAM. Headless-safe; if `show`, also draws a
    window (q=next model, ESC=stop showing). Returns {name: {acc,cpu,ram}}."""
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
    win = "Benchmark - each model at max speed   (q = next, ESC = stop showing)"
    out: dict = {}
    print(f"[06] {'showing' if show else 'measuring'} {len(onnx_files)} models on "
          f"{video.name} at max speed (~{seconds_per:.0f}s each)...")
    aborted = False
    if show:
        fit_window_to_screen(win)
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
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            boxes, _scores, _ids = det(frame)
            cs.append(len(boxes))
            now = time.perf_counter()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
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
                    cv2.putText(frame, f"{f.stem}  |  veh {len(boxes)}  |  {fps:.0f} FPS{extra}",
                                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 220, 255), 2)
                    cv2.imshow(win, frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == 27:
                        aborted = True
                        break
                except cv2.error:
                    showing = False  # no display - keep measuring headlessly
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
                    help="also evaluate true mAP50-95 on the labelled val set (needs ultralytics)")
    ap.add_argument("--no-accuracy", action="store_true",
                    help="skip the detection pass (leaves accuracy/CPU/RAM blank)")
    ap.add_argument("--show", action="store_true",
                    help="also open a window to watch each model during the pass")
    ap.add_argument("--show-seconds", type=float, default=5.0,
                    help="seconds per model in the detection pass")
    ap.add_argument("--device", default=None, help="name THIS machine for the results")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_config()
    device = args.device or device_name(cfg)

    onnx_files = sorted(f for f in MODELS_DIR.glob("*.onnx")
                        if not f.name.endswith(".prep.onnx"))   # skip quantizer intermediates
    if not onnx_files:
        raise SystemExit("No .onnx files in models/ - build some in the Train/Optimize tabs first.")

    verify_video = Path(args.video) if args.video else (BASE / "dataset" / "demo" / "highway-busy.mp4")
    if not verify_video.exists():
        verify_video = next((BASE / "dataset" / "demo").glob("*.mp4"), None)

    print(f"[06] device: {device}")
    vis: dict = {}
    if not args.no_accuracy and verify_video is not None:
        vis = measure_pass(onnx_files, verify_video, cfg, args.show_seconds, show=args.show)

    maps: dict[str, float] = {}
    if args.map:
        data_yaml = str(BASE / cfg["dataset"]["data_yaml"])
        vehicle_ids = cfg["detect"]["vehicle_class_ids"]
        mimg = cfg["train"]["imgsz"]
        if not Path(data_yaml).exists():
            print(f"[06] mAP skipped - dataset not found: {data_yaml}")
        else:
            try:                                    # exact path: ultralytics (PC)
                import train_utils  # noqa: F401    (registers C2f_v2 so pruned .pt unpickle)
                from ultralytics import YOLO
                for f in onnx_files:
                    print(f"[06] mAP: {f.name} (ultralytics val, slow)...")
                    try:
                        m = YOLO(str(f)).val(data=data_yaml, imgsz=mimg, classes=vehicle_ids,
                                             verbose=False, plots=False)
                        maps[f.name] = round(float(m.box.map), 4)
                    except Exception as e:  # noqa: BLE001
                        print(f"[06]   skipped {f.name} ({str(e)[:40]})")
            except ImportError:                     # Pi: torch-free onnxruntime evaluator
                import eval_native
                print("[06] no PyTorch - mAP via the on-device onnxruntime evaluator (approximate)")
                for f in onnx_files:
                    print(f"[06] mAP: {f.name} (on-device val, slow)...")
                    try:
                        maps[f.name] = eval_native.evaluate_onnx(
                            f, data_yaml, mimg, vehicle_ids, verbose=False)["mAP50_95"]
                    except Exception as e:  # noqa: BLE001
                        print(f"[06]   skipped {f.name} ({str(e)[:40]})")

    rows = []
    print(f"\n{'model':28s} {'MB':>7s} {'ms':>7s} {'maxFPS':>7s} {'veh/frm':>8s} {'CPU%':>6s} {'RAM MB':>7s}")
    print("-" * 78)
    for f in onnx_files:
        r = benchmark_onnx(f, runs=cfg["benchmark"]["runs"], threads=cfg["benchmark"]["threads"])
        info = vis.get(f.name, {})
        acc, cpu, ram = info.get("acc", ""), info.get("cpu", ""), info.get("ram", "")
        rows.append({
            "model": f.name,
            "size_mb": round(r["size_mb"], 2),
            "latency_ms": round(r["mean_ms"], 1),     # measured ON THIS device
            "max_fps": round(r["fps"], 1),            # raw achievable FPS on this device
            "veh_per_frame": acc,
            "cpu_pct": cpu,
            "ram_mb": ram,
            "mAP50_95": maps.get(f.name, ""),
            "device": device,
        })
        rr = rows[-1]
        print(f"{rr['model']:28s} {rr['size_mb']:7.2f} {rr['latency_ms']:7.1f} {rr['max_fps']:7.1f} "
              f"{str(acc):>8s} {str(cpu):>6s} {str(ram):>7s}")

    csv_path = RESULTS_DIR / "benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n[06] csv -> {csv_path.relative_to(BASE)}  (device: {device})")

    # ---- formatted Excel workbook (optional; skipped if openpyxl is missing) ----
    try:
        import export_results
        export_results.main()
    except Exception as e:  # noqa: BLE001
        print(f"[06] xlsx skipped ({e})  -  pip install openpyxl")

    # ---- 2 rows x 3 cols bar charts, each normalized to % of its own max ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["model"].replace(".onnx", "") for r in rows]

        def col(key):
            return [float(r[key]) if r.get(key) not in ("", None) else 0.0 for r in rows]

        has_map = any(r.get("mAP50_95") not in ("", None) for r in rows)
        # (title, values, colour, unit)
        panels = [
            ("Model size (MB)  —  lower = smaller", col("size_mb"), "#2563EB", "MB"),
            ("Accuracy: avg vehicles / frame  —  higher = better", col("veh_per_frame"), "#7C3AED", "veh/f"),
            ("Max FPS (raw inference)  —  higher = faster", col("max_fps"), "#16A34A", "FPS"),
            ("CPU usage at max speed (%)  —  lower = lighter", col("cpu_pct"), "#EA580C", "%"),
            ("RAM usage (MB)  —  lower = lighter", col("ram_mb"), "#0891B2", "MB"),
            (("Accuracy: mAP50-95  —  higher = better" if has_map
              else "mAP50-95  —  not measured (enable accuracy / --map)"),
             col("mAP50_95"), "#DB2777", ""),
        ]
        per_row_h = max(4.0, 0.40 * len(names) + 1.6)
        fig, axes = plt.subplots(2, 3, figsize=(19, per_row_h * 2))
        for ax, (title, vals, color, unit) in zip(axes.flatten(), panels):
            mx = max(vals) if vals and max(vals) > 0 else 1.0
            pct = [v / mx * 100.0 for v in vals]
            bars = ax.barh(names, pct, color=color)
            ax.set_title(title, fontweight="bold", fontsize=11)
            ax.set_xlabel("% of max  (max = 100%)")
            ax.set_xlim(0, 122)
            ax.invert_yaxis()
            ax.tick_params(axis="y", labelsize=7.5)
            labels = [f"{v:g} {unit} ({p:.0f}%)".replace("  ", " ") if v else "0"
                      for v, p in zip(vals, pct)]
            ax.bar_label(bars, labels=labels, padding=3, fontsize=7)
        fig.suptitle(f"Model comparison on {device}  —  bars = % of max",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        png_path = RESULTS_DIR / "comparison.png"
        fig.savefig(png_path, dpi=150)
        print(f"[06] chart -> {png_path.relative_to(BASE)}  "
              "(2 rows x 3 cols: size, veh/f, max FPS, CPU, RAM, mAP)")
    except Exception as e:  # noqa: BLE001
        print(f"[06] chart skipped ({e})")


if __name__ == "__main__":
    main()
