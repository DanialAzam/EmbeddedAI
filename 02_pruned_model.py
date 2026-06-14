"""STEP 2 - Pruned model (structured channel pruning).

Removes the least-important convolution channels (Group L2-norm importance,
via torch-pruning), in small steps with a recovery fine-tune between steps.
The Detect head is left untouched (its channel count is fixed by the number
of classes). C2f blocks are first rewritten to C2f_v2 so the dependency
graph is clean - see train_utils.py.

    python 02_pruned_model.py                              # prune simple -> pruned
    python 02_pruned_model.py --weights models/nas.pt --out nas_pruned
    python 02_pruned_model.py --out pruned_30   (with prune.amount tweaked in config)

Inputs : models/<weights>.pt   (default models/simple.pt, from step 1)
Outputs: models/<out>.pt, models/<out>.onnx   (default out = "pruned")
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import BASE, MODELS_DIR, benchmark_onnx, ensure_dirs, load_config, print_benchmark_row
from train_utils import (
    detect_head_layers,
    export_onnx,
    finetune_inplace,
    load_weights_into,
    replace_c2f_with_c2f_v2,
    save_model_ckpt,
)


def make_importance(tp):
    """Group-aware L2 magnitude importance, across torch-pruning versions.
    (<=1.5 calls it GroupNormImportance, 1.6+ GroupMagnitudeImportance.)"""
    for name in ("GroupNormImportance", "GroupMagnitudeImportance", "MagnitudeImportance"):
        cls = getattr(tp.importance, name, None)
        if cls is not None:
            print(f"[02] importance: tp.importance.{name}(p=2)")
            return cls(p=2)
    raise SystemExit("torch-pruning has no magnitude importance class - "
                     "check its version (pip show torch-pruning)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(MODELS_DIR / "simple.pt"),
                    help="source .pt to prune (default: models/simple.pt)")
    ap.add_argument("--out", default="pruned",
                    help="output name stem -> models/<out>.pt + .onnx")
    ap.add_argument("--amount", type=float, default=None,
                    help="fraction of channels to prune, 0-1 (override config prune.amount)")
    ap.add_argument("--onnx-only", action="store_true",
                    help="drop the .pt, keep only the .onnx")
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_config()
    pcfg = cfg["prune"]
    tcfg = cfg["train"]
    amount = pcfg["amount"] if args.amount is None else args.amount
    data_yaml = str(BASE / cfg["dataset"]["data_yaml"])
    imgsz = tcfg["imgsz"]

    src_pt = Path(args.weights)
    if not src_pt.exists():
        raise SystemExit(f"{src_pt} missing - train it first (01_simple_model.py / 05_hardware_nas.py).")
    print(f"[02] pruning source: {src_pt.name}  ->  models/{args.out}.onnx")

    import torch_pruning as tp
    from ultralytics import YOLO

    yolo = YOLO(str(src_pt))
    model = yolo.model.cpu().float()
    model.eval()

    print("[02] rewriting C2f -> C2f_v2 (prunable form)...")
    replace_c2f_with_c2f_v2(model)
    model.eval()

    example = torch.randn(1, 3, imgsz, imgsz)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example)
    print(f"[02] baseline: {base_params/1e6:.2f}M params, {base_macs/1e9:.2f} GMACs")

    print(f"[02] pruning ratio: {amount:.2f}  ({amount*100:.0f}% of channels)")
    pruner = tp.pruner.GroupNormPruner(
        model,
        example,
        importance=make_importance(tp),
        pruning_ratio=amount,
        iterative_steps=pcfg["iterative_steps"],
        ignored_layers=detect_head_layers(model),
    )

    steps = pcfg["iterative_steps"]
    for i in range(steps):
        pruner.step()
        macs, params = tp.utils.count_ops_and_params(model, example)
        print(f"[02] step {i+1}/{steps}: {params/1e6:.2f}M params "
              f"({100*params/base_params:.0f}%), {macs/1e9:.2f} GMACs")

        print(f"[02] recovery fine-tune ({pcfg['finetune_epochs_per_step']} epochs)...")
        best = finetune_inplace(
            model, data_yaml,
            epochs=pcfg["finetune_epochs_per_step"],
            imgsz=imgsz, batch=tcfg["batch"],
            run_name=f"prune_step{i+1}",
        )
        load_weights_into(model, best)
        model.cpu().float().eval()

    print(f"[02] final fine-tune ({pcfg['final_finetune_epochs']} epochs)...")
    best = finetune_inplace(
        model, data_yaml,
        epochs=pcfg["final_finetune_epochs"],
        imgsz=imgsz, batch=tcfg["batch"],
        run_name="prune_final",
    )
    load_weights_into(model, best)
    model.cpu().float().eval()

    # A pruned model is still a real (smaller) PyTorch model, so we keep BOTH:
    #   <out>.pt   - trainable / re-prunable source (a valid 'Start from' model)
    #   <out>.onnx - frozen, deployable result
    pruned_pt = MODELS_DIR / f"{args.out}.pt"
    save_model_ckpt(model, pruned_pt)
    onnx_path = export_onnx(pruned_pt, MODELS_DIR / f"{args.out}.onnx", imgsz)

    if args.onnx_only:
        pruned_pt.unlink(missing_ok=True)
        print(f"[02] output: models/{args.out}.onnx  (.pt dropped, --onnx-only)")
    else:
        print(f"[02] outputs: models/{args.out}.pt (trainable) + models/{args.out}.onnx")

    print("\n[02] pruned-model benchmark (laptop CPU):")
    bench = benchmark_onnx(onnx_path,
                           runs=cfg["benchmark"]["runs"],
                           threads=cfg["benchmark"]["threads"])
    print_benchmark_row(bench, prefix="     ")
    print(f"\nNext:  quantize it ->  python 03_quantized_model.py "
          f"--onnx models/{args.out}.onnx --prefix {args.out}")


if __name__ == "__main__":
    main()
