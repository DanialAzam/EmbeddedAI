"""Build a DOMAIN-MATCHED INT8 calibration set from the demo road videos.

Static INT8 quantization learns each layer's activation range from calibration
images. Calibrating on the SAME kind of road scenes the model runs on (rather
than random COCO images) gives better ranges -> better INT8 accuracy.

This samples frames evenly across dataset/demo/*.mp4 into the calibration folder
the quantizer reads (config dataset.calibration_dir).

    python make_calibration.py            # ~250 frames from the demo clips
    python make_calibration.py --n 300
    python make_calibration.py --keep     # add to the folder instead of clearing it
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import BASE, ensure_dirs, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="total calibration frames to extract")
    ap.add_argument("--keep", action="store_true",
                    help="add to the calibration folder instead of clearing it first")
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_config()
    cal_dir = BASE / cfg["dataset"]["calibration_dir"]
    cal_dir.mkdir(parents=True, exist_ok=True)
    demo = BASE / "dataset" / "demo"
    videos = sorted(demo.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no demo videos in {demo} - run download_videos.py or add clips.")

    if not args.keep:
        for p in cal_dir.glob("*.jpg"):
            p.unlink()

    per = max(1, args.n // len(videos))
    saved = 0
    for vi, v in enumerate(videos):
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, total // per) if total > 0 else 1
        got = 0
        fi = 0
        while got < per:
            if total > 0:
                if fi >= total:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(cal_dir / f"cal_{vi:02d}_{got:04d}.jpg"), frame)
            got += 1
            saved += 1
            fi += step
        cap.release()
        print(f"[calib] {v.name}: {got} frames")
    print(f"[calib] {saved} road frames -> {cal_dir.relative_to(BASE)}")


if __name__ == "__main__":
    main()
