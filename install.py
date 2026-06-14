
"""Cross-platform installer for the semester project.

Creates a .venv next to this file and installs all required libraries.
Works on Windows (PC), Linux, and Raspberry Pi OS.

    python install.py            # full install (training + optimization, PC)
    python install.py --edge     # minimal install (inference only, Raspberry Pi)
    python install.py --recreate # delete .venv and start fresh

If 'python' is not recognized on Windows, try:
    py install.py
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
VENV = BASE / ".venv"
IS_WINDOWS = os.name == "nt"
# Raspberry Pi / ARM-Linux: pip's opencv-python is HEADLESS (no cv2.imshow window),
# so on the Pi we use Debian's system OpenCV (python3-opencv) via --system-site-packages.
IS_PI = (not IS_WINDOWS) and platform.machine().lower() in (
    "aarch64", "arm64", "armv7l", "armv6l")


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def run(cmd: list[str], step: str) -> None:
    """Run a command, streaming output; exit with a clear message on failure."""
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        fail(f"{step} failed (exit code {result.returncode}). "
             "Scroll up for the actual error message.")


def fail(msg: str) -> None:
    print(f"\n[ERROR] {msg}")
    pause()
    sys.exit(1)


def pause() -> None:
    """Keep the window open when launched by double-click on Windows."""
    if IS_WINDOWS and "--no-pause" not in sys.argv:
        try:
            input("\nPress Enter to close...")
        except EOFError:
            pass


def check_python() -> None:
    v = sys.version_info
    print(f"[1/5] Python {v.major}.{v.minor}.{v.micro}  ({sys.executable})")
    if v < (3, 9):
        fail("Python 3.9+ required. Install from https://www.python.org/downloads/ "
             "and tick 'Add python.exe to PATH'.")
    if "WindowsApps" in sys.executable:
        # Extremely unlikely (the stub can't run scripts), but just in case.
        fail("This is the Microsoft Store Python stub, not a real install.\n"
             "Install Python from https://www.python.org/downloads/ "
             "(tick 'Add python.exe to PATH'), reopen the terminal, retry.")


def make_venv(recreate: bool, edge: bool) -> None:
    if recreate and VENV.exists():
        print("[2/5] Removing old .venv ...")
        shutil.rmtree(VENV)
    if venv_python().exists():
        print("[2/5] .venv already exists - reusing it (use --recreate to rebuild)")
        return
    print("[2/5] Creating virtual environment (.venv) ...")
    cmd = [sys.executable, "-m", "venv", str(VENV)]
    if edge and IS_PI:
        # Expose Debian's GUI-capable system OpenCV to the venv.
        cmd.append("--system-site-packages")
    result = subprocess.run(cmd)
    if result.returncode != 0 or not venv_python().exists():
        hint = ("On Raspberry Pi / Debian first run:  sudo apt install python3-venv"
                if not IS_WINDOWS else
                "Your Python install may be missing the venv module - reinstall "
                "from python.org.")
        fail(f"Could not create the virtual environment. {hint}")


def install_packages(edge: bool) -> None:
    vp = venv_python()
    print(f"[3/5] Upgrading pip ...")
    run([vp, "-m", "pip", "install", "--upgrade", "pip"], "pip upgrade")
    if edge and IS_PI:
        # OpenCV is provided by the system (python3-opencv) for a working GUI window;
        # pip's ARM opencv-python is headless. Install everything else here.
        pkgs = ["onnxruntime", "numpy<2", "pyyaml", "psutil", "matplotlib"]
        print("[4/5] Installing inference deps "
              "(OpenCV comes from system python3-opencv) ...")
        run([vp, "-m", "pip", "install", *pkgs], "library install")
        return
    reqs = BASE / ("requirements_edge.txt" if edge else "requirements.txt")
    if not reqs.exists():
        fail(f"{reqs.name} not found next to install.py")
    print(f"[4/5] Installing from {reqs.name} "
          f"({'inference-only' if edge else 'full - can take 5-15 minutes'}) ...")
    run([vp, "-m", "pip", "install", "-r", str(reqs)], "library install")


def verify(edge: bool) -> None:
    print("[5/5] Verifying imports ...")
    if edge and IS_PI:
        code = ("import onnxruntime, numpy, yaml, psutil, matplotlib, cv2; "
                "print('  OK  opencv', cv2.__version__, '(system)', "
                "'| onnxruntime', onnxruntime.__version__)")
        result = subprocess.run([str(venv_python()), "-c", code])
        if result.returncode != 0:
            fail("Imports failed. If the error mentions cv2/OpenCV, install the "
                 "GUI build, then re-run this installer:\n"
                 "    sudo apt install -y python3-opencv")
        return
    mods = "cv2, onnxruntime, numpy, yaml" + ("" if edge else ", torch, ultralytics")
    code = (f"import {mods}; import onnxruntime, cv2; "
            f"print('  OK  opencv', cv2.__version__, '| onnxruntime', onnxruntime.__version__)")
    run([venv_python(), "-c", code], "import verification")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", action="store_true",
                    help="minimal inference-only install (Raspberry Pi)")
    ap.add_argument("--recreate", action="store_true",
                    help="delete and rebuild .venv")
    ap.add_argument("--no-pause", action="store_true",
                    help="don't wait for Enter at the end (for scripts)")
    args = ap.parse_args()

    print("=" * 70)
    print(" Semester Project installer "
          f"({'EDGE / Raspberry Pi' if args.edge else 'FULL / PC'} mode)")
    print("=" * 70)

    check_python()
    make_venv(args.recreate, args.edge)
    install_packages(args.edge)
    verify(args.edge)

    activate = (".venv\\Scripts\\activate" if IS_WINDOWS else "source .venv/bin/activate")
    print("\n" + "=" * 70)
    print(" Install complete. Every session, activate the environment first:")
    print(f"    {activate}")
    print(" Then:")
    if args.edge:
        print("    bash run_on_pi.sh        # detect on a bundled demo video (no camera needed)")
    else:
        print("    python download.py")
        print("    python 01_simple_model.py   (then 02 ... 06, then detect.py)")
    print("=" * 70)
    pause()


if __name__ == "__main__":
    main()
