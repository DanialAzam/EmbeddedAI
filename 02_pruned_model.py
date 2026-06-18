"""STEP 2 - Pruned model (structured channel pruning).

Removes the least-important convolution channels (Group L2-norm importance,
via torch-pruning), in small steps with a recovery fine-tune between steps.
The Detect head is left untouched (its channel count is fixed by the number
of classes). C2f blocks are first rewritten to C2f_v2 so the dependency
graph is clean - see train_utils.py.

Iterative: prune `step` of the channels, fine-tune `epochs` to recover, and
repeat until the `amount` total is reached (rounds = round(amount / step)).
Configure in config.yaml under prune.{amount, step, finetune_epochs}, or
override per run:

    python 02_pruned_model.py --weights models/0trained_fp32.pt --out 10pruned \
        --amount 0.10 --step 0.01 --epochs 1     # 10 rounds of 1% + 1 epoch each

Inputs : models/<weights>.pt   (a trainable baseline / pruned .pt)
Outputs: models/<out>.pt, models/<out>.onnx
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
    ap.add_argument("--weights", default=str(MODELS_DIR / "0trained_fp32.pt"),
                    help="source .pt to prune (a trainable baseline / pruned model)")
    ap.add_argument("--out", default="pruned",
                    help="output name stem -> models/<out>.pt + .onnx")
    ap.add_argument("--amount", type=float, default=None,
                    help="TOTAL fraction of channels to prune, 0-1 (override prune.amount)")
    ap.add_argument("--step", type=float, default=None,
                    help="step-size fraction per round, 0-1 (override prune.step); rounds = amount/step")
    ap.add_argument("--epochs", type=int, default=None,
                    help="fine-tune epochs after EACH prune step (override prune.finetune_epochs)")
    ap.add_argument("--onnx-only", action="store_true",
                    help="drop the .pt, keep only the .onnx")
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_config()
    pcfg = cfg["prune"]
    tcfg = cfg["train"]
    amount = pcfg["amount"] if args.amount is None else args.amount
    step = args.step if args.step is not None else pcfg.get("step")
    ft_epochs = args.epochs if args.epochs is not None else pcfg.get("finetune_epochs", 1)
    # step is the per-round fraction. Fall back to a single full-amount round when it is
    # missing / non-positive, and clamp (with a notice) when it exceeds the total.
    if not step or step <= 0:
        step = amount
    elif step > amount:
        print(f"[02] step {step:.3f} > total {amount:.3f}; using a single {amount*100:.0f}% round")
        step = amount
    rounds = max(1, round(amount / step))        # prune -> finetune rounds
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

    print(f"[02] iterative prune: {amount*100:.0f}% total in {rounds} round(s) of "
          f"~{amount/rounds*100:.1f}% each, {ft_epochs} fine-tune epoch(s) per round")
    pruner = tp.pruner.GroupNormPruner(
        model,
        example,
        importance=make_importance(tp),
        pruning_ratio=amount,
        iterative_steps=rounds,
        ignored_layers=detect_head_layers(model),
    )

    for i in range(rounds):
        pruner.step()                            # prune ~amount/rounds of the channels
        macs, params = tp.utils.count_ops_and_params(model, example)
        print(f"[02] round {i+1}/{rounds}: pruned to {params/1e6:.2f}M params "
              f"({100*params/base_params:.0f}% of original), {macs/1e9:.2f} GMACs")
        if ft_epochs > 0:
            print(f"[02] recovery fine-tune ({ft_epochs} epoch(s))...")
            best = finetune_inplace(
                model, data_yaml,
                epochs=ft_epochs,
                imgsz=imgsz, batch=tcfg["batch"],
                run_name=f"prune_step{i+1}",
            )
            load_weights_into(model, best)
            model.cpu().float().eval()

    # A pruned model is still a real (smaller) PyTorch model, so we keep BOTH:
    #   <out>.pt   - trainable / re-prunable source (a valid 'Start from' model)
    #   <out>.onnx - frozen, deployable result
    pruned_pt = MODELS_DIR / f"{args.out}.pt"
    if pruned_pt.resolve() == src_pt.resolve():
        raise SystemExit(f"[02] --out '{args.out}' would overwrite the source model "
                         f"{src_pt.name}; choose a different --out name.")
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
