# EmbeddedAI — Vehicle Detection & Traffic-Congestion Monitoring on Raspberry Pi 5

Optimize a **YOLOv8n** vehicle detector for the **CPU-only edge** with **structured pruning** and
**INT8 / FP16 quantization**, benchmark every variant on a laptop *and* on the Pi itself, and deploy the
winner to a **Raspberry Pi 5** — for a **Safe City** adaptive traffic-signal application.

**Repository:** https://github.com/DanialAzam/EmbeddedAI

```bash
git clone https://github.com/DanialAzam/EmbeddedAI.git
```

> Course: **Embedded AI** · Student: **Engr Danial Azam (PHDEE25003)** · Instructor: **Prof Rehan Hafiz**

---

## What it does

Each junction camera runs the detector on a Raspberry Pi 5 at the edge, counts vehicles per lane, and
classifies congestion (**FREE / MODERATE / HEAVY**) — which a Safe City system feeds to the traffic-signal
controllers so green/red timings adapt in real time. Doing detection **on the Pi at the camera** keeps
bandwidth low and scales across a city.

```
        PC (training + optimization)                 Raspberry Pi 5 (inference)
┌────────────────────────────────────────┐       ┌──────────────────────────────┐
│ download.py   -> dataset + base weights │       │  detect.py                   │
│ 01 train      -> 0trained (baseline)    │ copy  │  (onnxruntime, 4x Cortex-A76)│
│ 02 prune      -> 10/20/30 pruned        │ .onnx │  vehicles + congestion       │
│ 03 quantize   -> _fp16 / _static (INT8) │ ────> │  from a local demo video     │
│ 06 benchmark  -> comparison table/chart │       │  (no camera / LAN needed)    │
└────────────────────────────────────────┘       └──────────────────────────────┘
```

## Model naming (`models/*.onnx`)

- **`0trained`** — baseline YOLOv8n fine-tuned **20 epochs** on the **548-image `vehicles_coco`** set (0 % pruned)
- **`10pruned` / `20pruned` / `30pruned`** — baseline with 10/20/30 % of channels pruned + recovery fine-tune
- **suffix = precision:** `_fp32` (full), `_fp16` (FP16), `_static` (INT8, static)
- e.g. `30pruned_static.onnx` = 30 %-pruned **then** INT8-quantized → **12 ONNX variants** in total

## Quick start

```bash
git clone https://github.com/DanialAzam/EmbeddedAI.git
cd EmbeddedAI
```

**Windows PC (training + optimization):**
```bat
python install.py            REM creates .venv and installs everything (or double-click install.bat)
```

**Raspberry Pi 5 (inference only):**
```bash
sudo apt install -y python3-opencv        # GUI-capable OpenCV (see "Deploy" below)
python3 install.py --edge
```

### Or drive everything from the GUI
```bash
python gui.py        # gui.bat on Windows · bash gui.sh on the Pi
```
One window: **Install · Download · Benchmark**, a live-detection launcher (model + video), **Train /
Optimize** tabs (PC only — auto-hidden on the Pi), a live **Log**, and a **Results** tab with the chart.

## Pipeline

| # | Command | Does | Output |
|---|---------|------|--------|
| 0 | `python download.py` | base weights + COCO128 + 150 calibration images + demo video | `models/base/`, `dataset/` |
| 1 | `python 01_simple_model.py` | fine-tune 20 epochs on `vehicles_coco` (548 imgs) + export the FP32 baseline | `0trained_fp32.pt/.onnx` |
| 2 | `python 02_pruned_model.py --amount 0.3` | structured channel prune + recovery fine-tune | `30pruned_fp32.pt/.onnx` |
| 3 | `python 03_quantized_model.py` | FP16 + INT8 (selective static) | `*_fp16.onnx`, `*_static.onnx` |
| 6 | `python 06_benchmark.py` | compare all variants → CSV + 2×2 chart | `results/benchmark.csv`, `comparison.png` |

Live demo:
```bash
python detect.py --model models/0trained_static.onnx --source dataset/demo/highway-busy.mp4
```

## Optimization techniques

**Structured pruning.** Removes the lowest grouped-L2 convolution channels (and every dependent weight)
via `torch-pruning`, then a **gentle-LR (0.0005)** recovery fine-tune — the default 0.01 erases the
pretrained features and detection drops to zero. The detection head is left intact.

**Quantization.** **FP16** (≈lossless, 2× smaller, no CPU speed-up) and **INT8 static** (calibrated,
per-channel, QDQ, opset 13). The **detection head (66 nodes `/model.22/`) is kept FP32** — quantizing it
collapses detections to zero. INT8 runs **~1.5× faster than FP32** on the Pi's Arm NEON path.

> Hardware-aware NAS was **investigated but not deployed** — its per-candidate retraining needs a GPU.

## Results — measured on the Raspberry Pi 5 (fixed 15 FPS)

| Model | Size (MB) | Pi infer (ms) | Accuracy (veh/f) | CPU % @15 FPS | RAM (MB) |
|---|---|---|---|---|---|
| `0trained_fp32` | 12.72 | 36.6 | 8.16 | 92.2 | 276 |
| `0trained_fp16` | 6.40 | 30.1 | 8.21 | 87.7 | 268 |
| **`0trained_static` (INT8)** | **6.16** | **24.6** | **8.32** | **78.2** | **267** |
| `30pruned_static` | 4.28 | 20.5 | 3.55 | 72.2 | 265 |

**Winner: `0trained_static`** — full accuracy, half the size, lowest CPU and RAM. Pruning shrinks the
model further but trades accuracy steeply. Full data in `results/` (laptop) and `pi_results/` (Pi);
see **`REPORT.html`** / **`REPORT.docx`**.

## Deploy to the Raspberry Pi 5 (local demo video — no camera/LAN)

1. Copy the folder to the Pi (skip `.venv`), e.g. into your home directory.
2. Install GUI-capable OpenCV + edge dependencies:
   ```bash
   sudo apt install -y python3-opencv
   cd EmbeddedAI
   python3 install.py --edge
   ```
   On a Pi, `install.py --edge` builds the venv with `--system-site-packages` and uses the **system**
   OpenCV — pip's Arm `opencv-python` is headless and can't open the video window.
3. Run:
   ```bash
   bash run_on_pi.sh                                   # default INT8 model + busy-highway clip
   bash run_on_pi.sh 0trained_static.onnx highway-busy.mp4
   ```
   Over SSH with no screen it auto-switches to headless (stats only).

## Report

`REPORT.html` (interactive, self-contained) and `REPORT.docx` follow the required **14-section** academic
structure: Abstract → Problem Statement → Dataset → Pre-processing → Baseline Model & Performance →
Training/Inference → Hardware → Optimization → Comparative Results → Target Device → Deployment Challenges
→ Conclusion → Repository → References. Regenerate with `_docx_build/build_report_html.py` (HTML) and
`_docx_build/build_docx.js` (Word).

## Repository layout

```
EmbeddedAI/
├─ gui.py                 control panel (drives the whole project)
├─ detect.py             live detection + congestion + resource overlay
├─ 01_simple_model.py    train / export the baseline (0trained)
├─ 02_pruned_model.py    structured pruning + recovery fine-tune
├─ 03_quantized_model.py / quantize_levels.py   FP16 + INT8 (selective)
├─ 06_benchmark.py       compare every variant -> CSV + chart
├─ download.py           base model + COCO128 + calibration + demo video
├─ install.py            cross-platform venv installer (--edge for the Pi)
├─ run_on_pi.sh          headless/desktop Pi launcher
├─ common.py / train_utils.py   shared inference + training helpers
├─ config.yaml           paths, prune ratio, classes, FPS cap
├─ models/               .onnx + .pt variants
├─ dataset/  results/  pi_results/   data, laptop + Pi benchmarks
└─ REPORT.html  REPORT.docx          the project report
```

## Troubleshooting

| Problem | Fix |
|---|---|
| No video window on the Pi | Install `python3-opencv` (pip's Arm OpenCV is headless); `install.py --edge` wires it into the venv |
| Benchmark chart not generated | `matplotlib` was missing — re-run `install.py --edge` |
| Full INT8 detects nothing | Expected — keep the detection head FP32 (selective INT8, the default) |
| Pruned model detects nothing | Recovery fine-tune needs a gentle LR (0.0005) — already set |
| `pip install torch` fails | Use Python 3.10–3.12 and upgrade pip first: `python -m pip install -U pip` |

---

*Engr Danial Azam (PHDEE25003) · Embedded AI · Prof Rehan Hafiz*
