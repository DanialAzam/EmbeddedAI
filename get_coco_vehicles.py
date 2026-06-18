"""Download a class-matched VEHICLE subset of COCO into dataset/coco_vehicles/.

Real street/road images, YOLO format, vehicle classes at their project indices
(car=2, motorcycle=3, bus=5, truck=7 - COCO80). No account/API key needed.
Memory-safe: parses only the small val annotations (works in ~1 GB free RAM).
"""
from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from random import Random

from common import BASE
from download import COCO80_NAMES

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
CATMAP = {3: 2, 4: 3, 6: 5, 8: 7}     # COCO category_id -> COCO80 index (car/moto/bus/truck)
CAP = 1800
VAL_FRAC = 0.1


def fetch(url: str, dest: Path) -> None:
    def hook(b, bs, t):
        if t > 0:
            print(f"\r  {min(100, b * bs * 100 // t):3d}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()


def main() -> None:
    out = BASE / "dataset" / "coco_vehicles"
    tmp = BASE / "dataset" / "_coco_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    zip_path = tmp / "ann.zip"
    if not zip_path.exists():
        print("[coco] downloading val annotations (~241 MB, one time)...")
        fetch(ANN_URL, zip_path)

    member = "annotations/instances_val2017.json"
    with zipfile.ZipFile(zip_path) as z:
        z.extract(member, tmp)                 # extract ONLY the small val json
    ann = json.loads((tmp / member).read_text(encoding="utf-8"))
    images = {im["id"]: im for im in ann["images"]}

    per_img: dict[int, list] = {}
    for a in ann["annotations"]:
        if a["category_id"] in CATMAP:
            per_img.setdefault(a["image_id"], []).append(a)

    ids = sorted(per_img, key=lambda i: len(per_img[i]), reverse=True)[:CAP]
    Random(0).shuffle(ids)
    n_val = int(len(ids) * VAL_FRAC)
    splits = {"val": ids[:n_val], "train": ids[n_val:]}
    for s in splits:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    def grab(iid: int, split: str) -> bool:
        im = images[iid]
        url = im.get("coco_url") or f"http://images.cocodataset.org/val2017/{im['file_name']}"
        ip = out / "images" / split / im["file_name"]
        try:
            if not ip.exists():
                urllib.request.urlretrieve(url, ip)
        except Exception:  # noqa: BLE001
            return False
        W, H = im["width"], im["height"]
        lines = []
        for a in per_img[iid]:
            x, y, w, h = a["bbox"]
            lines.append(f"{CATMAP[a['category_id']]} {(x + w / 2) / W:.6f} "
                         f"{(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        (out / "labels" / split / f"{Path(im['file_name']).stem}.txt").write_text("\n".join(lines))
        return True

    total = sum(len(v) for v in splits.values())
    print(f"[coco] downloading {total} vehicle images (24 parallel)...")
    done = ok = 0
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = [ex.submit(grab, iid, s) for s, lst in splits.items() for iid in lst]
        for f in as_completed(futs):
            done += 1
            ok += 1 if f.result() else 0
            if done % 100 == 0:
                print(f"\r  {done}/{total}", end="", flush=True)
    print(f"\r  {done}/{total}")

    (out / "data.yaml").write_text(
        "# Auto-written by get_coco_vehicles.py - COCO vehicle subset (YOLO format).\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(COCO80_NAMES)),
        encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[coco] {ok}/{total} saved -> {out.relative_to(BASE)}  "
          f"(train {len(splits['train'])}, val {len(splits['val'])})")


if __name__ == "__main__":
    main()
