# Semester Project — Vehicle Detection Model Optimization for Raspberry Pi 5

Compress a YOLOv8n vehicle detector with **pruning**, **quantization**, and
**hardware-aware NAS**, compare every variant, and deploy the winner to a
Raspberry Pi 5 that reads a LAN camera stream and reports traffic congestion.

```
            PC (this folder)                                Raspberry Pi 5
┌─────────────────────────────────────────┐          ┌────────────────────────┐
│ download.py  -> dataset + base weights  │          │                        │
│ 01 simple    -> baseline model          │  copy    │  detect.py             │
│ 02 pruned    -> fewer channels          │  .onnx   │  (onnxruntime, 4 cores)│
│ 03 quantized -> INT8                    │ ───────> │  vehicles + congestion │
│ 04 pruned+quantized -> both             │          │  from RTSP LAN feed    │
│ 05 hardware-aware NAS -> best arch      │          │                        │
│ 06 benchmark -> comparison table/chart  │          └────────────────────────┘
└─────────────────────────────────────────┘
```

## 1. Install

**Windows PC (training + optimization):** double-click `install.bat`
(or run it in a terminal). It finds a working Python (skipping the fake
Microsoft Store stub), then runs `install.py` which creates `.venv` and
installs everything. You can also run the installer directly:
```
py install.py          (or: python install.py)
```
If no Python exists yet, install it first:
`winget install -e --id Python.Python.3.12` — then reopen the terminal.

**Raspberry Pi 5 (inference only):**
```bash
bash install.sh --edge        # or: python3 install.py --edge
```

Then activate the environment every session:
- Windows: `.venv\Scripts\activate`
- Pi/Linux: `source .venv/bin/activate`

## 1b. Or drive everything from the GUI

```
python gui.py        (or double-click gui.bat)
```

One window for the whole project: a button per pipeline step with done/running
markers, live console output of the running step (pip installs, training
epochs, pruning stats, NAS leaderboard...), a Stop button, "Run remaining
steps" to chain everything pending, a detection launcher (variant + source),
and a Results tab that renders `results/benchmark.csv` as a table. The GUI is
pure standard library - it works before installing anything and can run the
installer itself (step 0a).

## More traffic videos

The bundled `car-detection.mp4` is light traffic. For busier scenes (and to
see MODERATE/HEAVY congestion trigger):
```
python download_videos.py           # busy highway clips into dataset/demo/
python detect.py --source dataset/demo/highway-busy.mp4
```

## 2. Download data + base model

```
python download.py
```
Fetches YOLOv8n pretrained weights, the coco128 dataset (no account needed,
class IDs match COCO so everything works out of the box), 150 calibration
images for quantization, and a demo traffic video.

**Optional upgrade to real traffic-camera data (UA-DETRAC):**
`python download.py --roboflow YOUR_API_KEY` — free key from
roboflow.com -> Settings -> API. The script prints the two config changes the
swap requires (dataset path + class IDs — custom models output IDs starting
at 0, not COCO's 2/3/5/7; forget this and you get zero detections).

## 3. Run the pipeline (in order)

| # | Command | What it does | Output in `models/` |
|---|---------|--------------|---------------------|
| 1 | `python 01_simple_model.py` | Fine-tune + export baseline | `simple.pt/.onnx` |
| 2 | `python 02_pruned_model.py` | Remove conv channels (L2 group importance, ratio in `config.yaml: prune.amount`), fine-tune to recover | `pruned.pt/.onnx` |
| 3 | `python 03_quantized_model.py` | FP32 -> INT8 (dynamic + calibrated static) | `quantized_*.onnx` |
| 4 | `python 04_pruned_quantized_model.py` | INT8-quantize the pruned model | `pruned_quantized.onnx` |
| 5 | `python 05_hardware_nas.py` | Search depth/width grid under a measured-latency budget | `nas.pt/.onnx` |
| 6 | `python 06_benchmark.py` | Compare all variants, write CSV + chart | `results/` |

Live demo with any variant:
```
python detect.py --variant pruned_quantized --source dataset/demo/car-detection.mp4
```

### CPU time expectations (no GPU)

Rough numbers at the default `imgsz: 320`, coco128: step 1 ~15 min,
step 2 ~25 min, steps 3-4 ~2 min each, step 5 ~1-2 h (6 candidates trained
briefly + 1 final training). Shrink `nas.depth_multipliers`/`width_multipliers`
or `search_epochs` in `config.yaml` if that's too long. With a CUDA GPU
everything is 10-20x faster.

## 4. The three techniques in one paragraph each (for the report)

**Pruning (structured).** Convolution channels with the smallest grouped L2
norm contribute least to the output; `torch-pruning` removes them *and* every
dependent weight across skip connections, then short fine-tunes recover the
lost accuracy. Result: fewer parameters and MACs. On CPUs the speedup is
sub-linear (SIMD lanes get less saturated), so pruning's main wins here are
model size and making the network friendlier to quantize.

**Quantization (INT8).** Replaces FP32 weights (dynamic) and also activations
(static, needs ~100 calibration images) with 8-bit integers. ARM CPUs like
the Pi 5's Cortex-A76 execute INT8 natively, so static quantization is where
the largest real-time gain comes from — typically 2-3x over FP32 — at a small
accuracy cost. QDQ format, per-channel scales.

### Quantization levels (`python quantize_levels.py`)

Builds every level from `simple.onnx` and runs a detection sanity check
(avg vehicles/frame on `highway-busy.mp4`) so accuracy is measured, not assumed:

| Level | Size | avg veh/frame | Verdict |
|---|---|---|---|
| FP32 baseline | 12.7 MB | 4.95 | reference |
| **INT8 full** (every layer) | 3.5 MB | **0.00** | ❌ accuracy collapses — DO NOT deploy |
| **INT8 selective** (head kept FP32) | 6.2 MB | 4.24 | ✅ the deployment model (`quantized_static.onnx`) |
| FP16 half | 6.4 MB | 4.96 | ✅ ~perfect accuracy, but no CPU speedup |

**The key finding:** naive full-INT8 quantization of YOLOv8 detects *nothing* —
8-bit can't hold the dynamic range of the detection head's box/score outputs, so
every prediction falls below threshold. Excluding the 66 head nodes
(`/model.22/*`) from quantization recovers ~86% of detections while keeping most
of the size/speed win. This "which layers to quantize" decision is the whole
game for INT8 on detectors. FP16 keeps full accuracy but, lacking INT8's native
CPU path, gives no speedup — its only benefit on CPU is halved file size.

Use `onnxruntime`'s built-in float16 converter (in `onnxruntime.transformers`),
not `onnxconverter-common` — the latter produces an invalid graph on YOLOv8's
Resize/Concat neck. View any level live:
`python detect.py --variant int8_full | quantized | fp16 --source dataset/demo/highway-busy.mp4`

**Hardware-aware NAS.** Instead of one hand-picked architecture, we search
YOLOv8's depth/width grid. Each candidate is briefly trained, then **timed on
ONNX Runtime — the actual deployment runtime — rather than scored by FLOPs**
(FLOPs are a poor latency proxy on CPUs). Laptop latency is scaled to a Pi-5
estimate and a hard latency budget (`config.yaml: nas.latency_budget_ms`)
eliminates infeasible candidates; the highest-mAP feasible architecture wins
and is trained fully.

## 5. Results table (fill from `results/benchmark.csv`)

| Model | Size (MB) | Laptop ms | Est. Pi FPS | mAP50-95 |
|---|---|---|---|---|
| simple (baseline) | | | | |
| pruned | | | | |
| quantized (static) | | | | |
| pruned + quantized | | | | |
| NAS winner | | | | |

`results/comparison.png` has the same data as bar charts.
For mAP numbers add `--map`: `python 06_benchmark.py --map` (slow on CPU).

## 6. Deploy to the Raspberry Pi 5

1. Copy the whole `semester project` folder to the Pi (or just:
   `models/pruned_quantized.onnx`, `detect.py`, `common.py`, `config.yaml`,
   `requirements_edge.txt`, `install.sh`).
2. `bash install.sh --edge && source .venv/bin/activate`
3. Point it at your LAN camera (or the PC's virtual camera from the parent
   project's `pc_tools/`):
   ```bash
   python detect.py --variant pruned_quantized --source rtsp://192.168.1.100:8554/cam1
   ```
   Headless over SSH: add `--no-display` (stats print once per second).

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails on torch | Use Python 3.10-3.12; upgrade pip first (`python -m pip install -U pip`) |
| Step 2 crashes loading `pruned.pt` elsewhere | Always run scripts from this folder — the checkpoint references `train_utils.C2f_v2` |
| Static quantization skipped | `dataset/calibration/` is empty — run `python download.py` |
| Zero detections after switching to UA-DETRAC model | Update `detect.vehicle_class_ids`/`class_names` (download.py printed the block) |
| NAS too slow | Reduce grid/epochs in `config.yaml: nas` |
| Demo video missing | Source it yourself and pass `--source path\to\video.mp4` — any traffic clip works |
