"""Download everything the project needs: base model, dataset, calibration
images, and an optional demo traffic video.

    python download.py                     # default: coco128 dataset (no account needed)
    python download.py --roboflow KEY      # upgrade: UA-DETRAC traffic-camera dataset
    python download.py --skip-video        # don't fetch the demo video

What lands where:
    models/base/yolov8n.pt        COCO-pretrained starting weights
    dataset/coco128/              128-image dataset (auto, zero credentials)
    dataset/coco128/data_local.yaml   training config pointing at it
    dataset/calibration/          images for static INT8 quantization
    dataset/demo/car-detection.mp4    sample traffic video for detect.py

Why coco128 by default: it needs no account, downloads in seconds, and its
class IDs match the COCO-pretrained model (car=2, motorcycle=3, bus=5,
truck=7) - so the whole pipeline runs end-to-end out of the box. For real
accuracy numbers, rerun with --roboflow to fetch UA-DETRAC (actual traffic
camera footage) - the script prints the config changes that swap requires.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

from common import BASE, DATASET_DIR, MODELS_DIR, ensure_dirs

YOLOV8N_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
DEMO_VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"

COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _fetch(url: str, dest: Path, label: str) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {label} already present: {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {label}\n       {url}")

    def hook(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            sys.stdout.write(f"\r       {pct:3d}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest, reporthook=hook)
        print(f"\r       done ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"\r[FAIL] {label}: {e}")
        dest.unlink(missing_ok=True)
        return False


def get_base_model() -> None:
    dest = MODELS_DIR / "base" / "yolov8n.pt"
    if not _fetch(YOLOV8N_URL, dest, "YOLOv8n pretrained weights"):
        raise SystemExit("Could not download yolov8n.pt - check your connection.")


def get_coco128() -> Path:
    """Download + extract coco128, write a self-contained data yaml for it."""
    ds_root = DATASET_DIR / "coco128"
    zip_path = DATASET_DIR / "coco128.zip"
    if not (ds_root / "images").exists():
        if not _fetch(COCO128_URL, zip_path, "coco128 dataset (128 images)"):
            raise SystemExit("Could not download coco128 - check your connection.")
        print("[unzip] extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATASET_DIR)  # zip contains a top-level coco128/ folder
        zip_path.unlink(missing_ok=True)
    else:
        print("[skip] coco128 already extracted")

    data_yaml = ds_root / "data_local.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write("# Auto-written by download.py - self-contained coco128 config.\n")
        yaml.safe_dump(
            {
                "path": str(ds_root.resolve().as_posix()),
                "train": "images/train2017",
                "val": "images/train2017",
                "names": {i: n for i, n in enumerate(COCO80_NAMES)},
            },
            f, sort_keys=False,
        )
    print(f"[ok  ] dataset config -> {data_yaml.relative_to(BASE)}")
    return ds_root


def fill_calibration(ds_root: Path, n: int = 150) -> None:
    """Copy sample images into dataset/calibration for static INT8."""
    cal_dir = DATASET_DIR / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    existing = list(cal_dir.glob("*.jpg"))
    if len(existing) >= 50:
        print(f"[skip] calibration folder already has {len(existing)} images")
        return
    images = sorted(ds_root.glob("images/**/*.jpg"))[:n]
    for p in images:
        shutil.copy2(p, cal_dir / p.name)
    print(f"[ok  ] {len(images)} calibration images -> dataset/calibration/")


def get_demo_video() -> None:
    dest = DATASET_DIR / "demo" / "car-detection.mp4"
    if _fetch(DEMO_VIDEO_URL, dest, "demo traffic video"):
        print(f"[ok  ] demo video -> {dest.relative_to(BASE)}")
    else:
        print("[note] demo video unavailable - detect.py still works with webcam/RTSP")


def get_uadetrac(api_key: str) -> None:
    """Optional upgrade: real traffic-camera dataset via Roboflow."""
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("pip install roboflow   (then rerun)")

    dest = DATASET_DIR / "ua_detrac"
    print(f"[get ] UA-DETRAC via Roboflow -> {dest}")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace("vehicle-detection-loakn").project("ua-detrac-10k-sample")
    ds = proj.version(1).download("yolov8", location=str(dest), overwrite=True)

    data_yaml = next(Path(ds.location).glob("**/data.yaml"), None)
    if data_yaml is None:
        raise SystemExit("Download finished but no data.yaml found - check the folder.")

    with open(data_yaml, "r", encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    print("\n" + "=" * 64)
    print("UA-DETRAC READY - paste these into config.yaml:")
    print("=" * 64)
    print("dataset:")
    print(f'  data_yaml: "{data_yaml.relative_to(BASE).as_posix()}"')
    print("\n# AFTER retraining (01) + re-exporting on this dataset, also set:")
    print("detect:")
    print(f"  vehicle_class_ids: {list(range(len(names)))}")
    print("  class_names:")
    for i, n in enumerate(names):
        print(f'    {i}: "{n}"')
    print("=" * 64)
    print("Why: your custom model's class IDs start at 0, unlike COCO's 2/3/5/7.")
    print("Skipping this = the detector filters for IDs that never occur = 0 boxes.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roboflow", metavar="API_KEY", default=None,
                    help="also download UA-DETRAC (real traffic cams) via Roboflow")
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    get_base_model()
    ds_root = get_coco128()
    fill_calibration(ds_root)
    if not args.skip_video:
        get_demo_video()
    if args.roboflow:
        get_uadetrac(args.roboflow)

    print("\nAll set. Next:  python 01_simple_model.py")


if __name__ == "__main__":
    main()
