"""Training-side helpers (require torch + ultralytics). PC only - never the Pi.

Key contents:
  C2f_v2 + replace_c2f_with_c2f_v2  - rewrite YOLOv8's C2f blocks so their
      chunk() op becomes two explicit convs. torch-pruning's dependency graph
      handles convs reliably; chunk() can confuse it. This is the established
      recipe from the official torch-pruning YOLOv8 example.
  finetune_inplace  - fine-tune an in-memory (pruned) model WITHOUT letting
      Ultralytics rebuild it from its yaml (a rebuild would undo the pruning,
      silently dropping every pruned weight at shape-mismatch load time).
  export_onnx       - uniform PT -> ONNX export.

NOTE: pruned checkpoints pickle the C2f_v2 class by reference
('train_utils.C2f_v2'), so any script that loads models/pruned.pt must
`import train_utils` first and be run from this folder.
"""
from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, C2f, Conv, Detect

BASE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# C2f -> C2f_v2 conversion (from the official torch-pruning YOLOv8 example)
# ---------------------------------------------------------------------------

def infer_shortcut(bottleneck: Bottleneck) -> bool:
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, "add") and bottleneck.add


class C2f_v2(nn.Module):
    """C2f with chunk() replaced by two parallel 1x1 convs (cv0 + cv1)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def _transfer_weights(c2f: C2f, c2f_v2: C2f_v2) -> None:
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # c2f's cv1 output gets chunked in half; split those weights into cv0/cv1.
    old_weight = state_dict["cv1.conv.weight"]
    half = old_weight.shape[0] // 2
    state_dict_v2["cv0.conv.weight"] = old_weight[:half]
    state_dict_v2["cv1.conv.weight"] = old_weight[half:]
    for bn_key in ("weight", "bias", "running_mean", "running_var"):
        old_bn = state_dict[f"cv1.bn.{bn_key}"]
        state_dict_v2[f"cv0.bn.{bn_key}"] = old_bn[:half]
        state_dict_v2[f"cv1.bn.{bn_key}"] = old_bn[half:]

    for key in state_dict:
        if not key.startswith("cv1."):
            state_dict_v2[key] = state_dict[key]

    # Preserve ultralytics layer bookkeeping attrs (i, f, type, np...).
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and "_" not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, C2f):
            shortcut = infer_shortcut(child.m[0]) if len(child.m) > 0 else False
            v2 = C2f_v2(
                child.cv1.conv.in_channels,
                child.cv2.conv.out_channels,
                n=len(child.m),
                shortcut=shortcut,
                g=child.m[0].cv2.conv.groups if len(child.m) > 0 else 1,
                e=child.c / child.cv2.conv.out_channels,
            )
            _transfer_weights(child, v2)
            setattr(module, name, v2)
        else:
            replace_c2f_with_c2f_v2(child)


# ---------------------------------------------------------------------------
# Fine-tuning an in-memory model (safe for pruned architectures)
# ---------------------------------------------------------------------------

def finetune_inplace(
    nn_model: nn.Module,
    data_yaml: str,
    epochs: int,
    imgsz: int,
    batch: int,
    run_name: str,
) -> Path:
    """Train an already-built model in place; returns path to best checkpoint.

    Why not YOLO(...).train()? Because that path rebuilds the network from its
    yaml spec and transfers only shape-matching weights - for a pruned model
    that means every pruned layer silently reverts. Assigning trainer.model
    directly makes Ultralytics skip the rebuild (setup_model() returns early
    for nn.Module instances).
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    overrides = dict(
        model="yolov8n.pt",  # placeholder for arg validation; not loaded
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
        project=str(BASE / "runs"),
        exist_ok=True,
        verbose=False,
        plots=False,
        # Recovery fine-tune, NOT full training: use a LOW learning rate so the
        # pruned model gently re-tunes its surviving channels and KEEPS the COCO
        # knowledge that detects real vehicles. The default lr0=0.01 is a
        # full-training rate - on a small/biased recovery set it drifts the model
        # away from the pretrained features and accuracy collapses.
        lr0=0.0005,
        lrf=0.1,
        warmup_epochs=0.0,
        mosaic=0.0,          # heavy augmentation also hurts short recovery runs
        close_mosaic=0,
    )
    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = nn_model.to(trainer.device)
    trainer.train()
    best = Path(trainer.best)
    return best if best.exists() else Path(trainer.last)


def load_weights_into(nn_model: nn.Module, ckpt_path: Path) -> None:
    """Copy trained weights from a checkpoint into the same architecture."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = ckpt.get("ema") or ckpt["model"]
    nn_model.load_state_dict(src.float().state_dict())


def save_model_ckpt(nn_model: nn.Module, dest: Path) -> None:
    """Save a YOLO()-loadable checkpoint of an in-memory model."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    m = deepcopy(nn_model).half()
    torch.save({"model": m, "train_args": {}}, dest)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(weights_pt: Path, dest_onnx: Path, imgsz: int) -> Path:
    """Export a .pt checkpoint to ONNX at models/<name>.onnx."""
    yolo = YOLO(str(weights_pt))
    # opset >= 13 is required: per-channel INT8 quantization (QDQ) writes an
    # 'axis' attribute on DequantizeLinear that opset 12 does not allow.
    exported = Path(
        yolo.export(format="onnx", imgsz=imgsz, opset=13, dynamic=False, simplify=True)
    )
    dest_onnx.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != dest_onnx.resolve():
        shutil.copy2(exported, dest_onnx)
        exported.unlink(missing_ok=True)
    print(f"[export] {dest_onnx.name}  ({dest_onnx.stat().st_size / 1e6:.2f} MB)")
    return dest_onnx


def detect_head_layers(model: nn.Module) -> list[nn.Module]:
    """Layers pruning must never touch (the Detect head's channel counts are
    fixed by the number of classes)."""
    return [m for m in model.modules() if isinstance(m, Detect)]
