"""Control panel GUI - run and watch every pipeline step from one window.

Pure standard library (tkinter), so it runs even BEFORE install.py has
created the environment - in fact it can run the installer for you.

    python gui.py          (or double-click gui.bat on Windows)

What you get:
  - one button per pipeline step, with live done/running status markers
  - live console streaming of whatever step is running (pip, training, ...)
  - a Stop button that kills the running step
  - "Run remaining steps" to execute everything that isn't done yet, in order
  - a detection launcher (model variant + source) - video opens in its own
    OpenCV window, per-second stats stream into the log
  - a Results tab showing results/benchmark.csv as a table + chart opener

Implementation notes: steps run as subprocesses of the .venv interpreter
(install runs under the system interpreter that launched the GUI). Output is
read char-wise on a worker thread and handed to Tk through a queue; '\r'
progress lines (pip downloads etc.) update in place instead of spamming.
"""
from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

BASE = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"
VENV_PY = BASE / (".venv/Scripts/python.exe" if IS_WINDOWS else ".venv/bin/python")
MODELS = BASE / "models"
RESULTS = BASE / "results"


def _is_edge_device() -> bool:
    """True on a Raspberry Pi / ARM-Linux SBC - an inference-only device with no
    PyTorch. On edge we hide the Train/Optimize tabs and install --edge."""
    import platform
    return os.name != "nt" and platform.machine().lower() in ("aarch64", "armv7l", "armv6l")


IS_EDGE = _is_edge_device()

VARIANTS = ["simple", "pruned", "int8", "fp16", "pruned_quantized", "nas"]


def _any(*paths: Path) -> bool:
    return any(p.exists() for p in paths)


# Pipeline column = the three top-level stages. Models themselves are built in
# the Train / Optimize tabs (PC only); the benchmark compares every .onnx present.
STEPS: list[dict] = [
    dict(key="install", label="0a  Install environment", script="install.py",
         extra=(["--edge", "--no-pause"] if IS_EDGE else ["--no-pause"]), system_python=True,
         done=lambda: VENV_PY.exists()),
    dict(key="download", label="0b  Download data + base model", script="download.py",
         done=lambda: (MODELS / "base" / "yolov8n.pt").exists()
         and (BASE / "dataset" / "coco128" / "data_local.yaml").exists()),
    dict(key="bench", label="Benchmark all variants",
         script="06_benchmark.py",   # --show is added at run time from the checkbox
         done=lambda: (RESULTS / "benchmark.csv").exists()),
]


def default_source() -> str:
    """Best-effort read of detect.source from config.yaml without pyyaml."""
    try:
        text = (BASE / "config.yaml").read_text(encoding="utf-8")
        m = re.search(r'^\s*source:\s*"?([^"#\r\n]+)', text, re.MULTILINE)
        if m:
            return m.group(1).strip()
    except OSError:
        pass
    return "0"


def open_path(p: Path) -> None:
    try:
        if IS_WINDOWS:
            os.startfile(str(p))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception as e:  # noqa: BLE001
        messagebox.showerror("Open failed", f"{p}\n\n{e}")


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Semester Project - Control Panel")
        root.geometry("1200x720")
        root.minsize(1000, 600)

        style = ttk.Style()
        try:
            style.theme_use("vista" if IS_WINDOWS else "clam")
        except tk.TclError:
            pass

        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue = queue.Queue()
        self.current: dict | None = None
        self.chain: list[dict] = []
        self.t_start = 0.0
        self._open_line = False  # last log line is a '\r' progress line

        self._build_layout()
        self.refresh_statuses()
        self.log_line("Welcome. Run steps top to bottom; watch their output here.", "hdr")
        if not VENV_PY.exists():
            self.log_line("No .venv yet - start with '0a Install environment'.", "warn")
        self.root.after(80, self._poll)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------- layout --
    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        # ---------- left: pipeline ----------
        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 8))

        ttk.Label(left, text="Pipeline", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.step_marks: dict[str, tk.Label] = {}
        self.step_btns: dict[str, ttk.Button] = {}
        for step in STEPS:
            row = ttk.Frame(left)
            row.pack(fill="x", pady=1)
            mark = tk.Label(row, text="O", width=2, font=("Segoe UI", 10, "bold"))
            mark.pack(side="left")
            btn = ttk.Button(row, text=step["label"], width=30,
                             command=lambda s=step: self.run_step(s))
            btn.pack(side="left", fill="x", expand=True)
            self.step_marks[step["key"]] = mark
            self.step_btns[step["key"]] = btn

        bar = ttk.Frame(left)
        bar.pack(fill="x", pady=(8, 2))
        self.btn_all = ttk.Button(bar, text="Run remaining steps", command=self.run_remaining)
        self.btn_all.pack(side="left", fill="x", expand=True)
        self.btn_stop = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(4, 0))

        # ---------- left: detection ----------
        det = ttk.LabelFrame(left, text="Live detection", padding=6)
        det.pack(fill="x", pady=(12, 4))
        ttk.Label(det, text="Model (.onnx)").grid(row=0, column=0, sticky="w")
        models = self._scan_models()
        self.var_variant = tk.StringVar(value="simple.onnx" if "simple.onnx" in models
                                        else (models[0] if models else ""))
        # Lists EVERY .onnx in models/. postcommand re-scans so models you build
        # later (pruned, quantized_*, nas, ...) appear without restarting.
        self.cmb_variant = ttk.Combobox(det, textvariable=self.var_variant,
                                        values=models, state="readonly", width=24,
                                        postcommand=self._refresh_models)
        self.cmb_variant.grid(row=0, column=1, sticky="ew", pady=1)
        ttk.Label(det, text="Source").grid(row=1, column=0, sticky="w")
        self.var_source = tk.StringVar(value=default_source())
        # Editable dropdown: lists "0" (webcam) + every video in dataset/.
        # postcommand re-scans the folder each time the list is opened, so
        # newly downloaded clips show up without restarting the GUI. Still
        # editable, so you can type an rtsp:// URL or any custom path.
        self.cmb_source = ttk.Combobox(det, textvariable=self.var_source, width=22,
                                       values=self._scan_sources(),
                                       postcommand=self._refresh_sources)
        self.cmb_source.grid(row=1, column=1, sticky="ew", pady=1)
        ttk.Label(det, text='pick a video, "0" = webcam, or type rtsp://...',
                  foreground="#777").grid(row=2, column=0, columnspan=2, sticky="w")
        self.var_display = tk.BooleanVar(value=True)
        ttk.Checkbutton(det, text="show video window (detection + benchmark)",
                        variable=self.var_display).grid(row=3, column=0, columnspan=2,
                                                        sticky="w")
        self.btn_detect = ttk.Button(det, text="Run detection",
                                     command=self.run_detection)
        self.btn_detect.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        det.columnconfigure(1, weight=1)

        # ---------- left: build models (Train + Optimize tabs; PC only) ----------
        # Built either way (so the buttons exist), but only SHOWN on a PC. On a
        # Raspberry Pi there is no PyTorch, so training/optimization can't run -
        # the Pi only does detection + benchmark.
        build_nb = ttk.Notebook(left)
        if IS_EDGE:
            ttk.Label(left, text="Edge device (Raspberry Pi): training & optimization "
                      "run on the PC. This device does detection + benchmark.",
                      foreground="#777", wraplength=240, justify="left").pack(fill="x", pady=(10, 4))
        else:
            build_nb.pack(fill="x", pady=(10, 4))

        # --- Train tab: pick dataset + epochs -> trains a new model ---
        tr = ttk.Frame(build_nb, padding=6)
        build_nb.add(tr, text="  Train  ")
        ttk.Label(tr, text="Start from").grid(row=0, column=0, sticky="w")
        self.var_tr_src = tk.StringVar()
        srcs = self._scan_train_sources()
        self.cmb_tr_src = ttk.Combobox(
            tr, textvariable=self.var_tr_src, width=20, state="readonly",
            values=list(srcs),
            postcommand=lambda: self.cmb_tr_src.configure(values=list(self._scan_train_sources())))
        self.var_tr_src.set(next(iter(srcs), ""))   # base yolov8n first
        self.cmb_tr_src.grid(row=0, column=1, sticky="ew", pady=1)
        ttk.Label(tr, text="Dataset").grid(row=1, column=0, sticky="w")
        self.var_tr_data = tk.StringVar()
        dsets = self._scan_datasets()
        self.cmb_tr_data = ttk.Combobox(
            tr, textvariable=self.var_tr_data, width=20, state="readonly",
            values=list(dsets),
            postcommand=lambda: self.cmb_tr_data.configure(values=list(self._scan_datasets())))
        self.var_tr_data.set("vehicles_coco" if "vehicles_coco" in dsets else next(iter(dsets), ""))
        self.cmb_tr_data.grid(row=1, column=1, sticky="ew", pady=1)
        ttk.Label(tr, text="Epochs").grid(row=2, column=0, sticky="w")
        self.var_tr_epochs = tk.StringVar(value="20")
        ttk.Spinbox(tr, from_=0, to=300, textvariable=self.var_tr_epochs,
                    width=8).grid(row=2, column=1, sticky="w", pady=1)
        ttk.Label(tr, text="Output name").grid(row=3, column=0, sticky="w")
        self.var_tr_out = tk.StringVar(value="simple")
        ttk.Entry(tr, textvariable=self.var_tr_out, width=18).grid(row=3, column=1, sticky="ew", pady=1)
        self.btn_train = ttk.Button(tr, text="Train -> .pt + .onnx", command=self.train_model)
        self.btn_train.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(tr, text="trains a .pt source (ONNX can't be trained). 0 epochs = export only.",
                  foreground="#777", wraplength=240).grid(row=5, column=0, columnspan=2, sticky="w")
        tr.columnconfigure(1, weight=1)

        # --- Optimize tab: prune / quantize any model ---
        comb = ttk.Frame(build_nb, padding=6)
        build_nb.add(comb, text="  Optimize  ")
        ttk.Label(comb, text="Quantize .onnx").grid(row=0, column=0, sticky="w")
        self.var_q_src = tk.StringVar()
        self.cmb_q_src = ttk.Combobox(
            comb, textvariable=self.var_q_src, width=20, state="readonly",
            values=self._scan_models(),
            postcommand=lambda: self.cmb_q_src.configure(values=self._scan_models()))
        self.cmb_q_src.grid(row=0, column=1, sticky="ew", pady=1)
        self.btn_quant = ttk.Button(comb, text="Quantize -> INT8 + FP16",
                                    command=self.combine_quantize)
        self.btn_quant.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(1, 6))
        ttk.Label(comb, text="Prune .pt").grid(row=2, column=0, sticky="w")
        self.var_p_src = tk.StringVar()
        self.cmb_p_src = ttk.Combobox(
            comb, textvariable=self.var_p_src, width=20, state="readonly",
            values=self._scan_pt(),
            postcommand=lambda: self.cmb_p_src.configure(values=self._scan_pt()))
        self.cmb_p_src.grid(row=2, column=1, sticky="ew", pady=1)
        ttk.Label(comb, text="Prune %").grid(row=3, column=0, sticky="w")
        self.var_p_pct = tk.StringVar(value="30")
        ttk.Spinbox(comb, from_=1, to=90, textvariable=self.var_p_pct,
                    width=8).grid(row=3, column=1, sticky="w", pady=1)
        self.btn_prune = ttk.Button(comb, text="Prune this %",
                                    command=self.combine_prune)
        self.btn_prune.grid(row=4, column=0, columnspan=2, sticky="ew", pady=1)
        comb.columnconfigure(1, weight=1)

        # ---------- left: shortcuts ----------
        short = ttk.LabelFrame(left, text="Open", padding=6)
        short.pack(fill="x", pady=(8, 0))
        for text, target in (("config.yaml", BASE / "config.yaml"),
                             ("models folder", MODELS),
                             ("results folder", RESULTS),
                             ("dataset folder", BASE / "dataset")):
            ttk.Button(short, text=text,
                       command=lambda t=target: open_path(t)).pack(fill="x", pady=1)

        # ---------- right: log + results tabs ----------
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)

        log_tab = ttk.Frame(self.nb)
        self.nb.add(log_tab, text="  Log  ")
        font = ("Consolas", 9) if IS_WINDOWS else ("DejaVu Sans Mono", 9)
        self.txt = ScrolledText(log_tab, state="disabled", bg="#111418",
                                fg="#d6d6d6", insertbackground="#d6d6d6",
                                font=font, wrap="none")
        self.txt.pack(fill="both", expand=True)
        self.txt.tag_configure("hdr", foreground="#6cb6ff")
        self.txt.tag_configure("ok", foreground="#7ce38b")
        self.txt.tag_configure("err", foreground="#ff7b72")
        self.txt.tag_configure("warn", foreground="#e3b341")

        res_tab = ttk.Frame(self.nb)
        self.nb.add(res_tab, text="  Results  ")
        bar2 = ttk.Frame(res_tab)
        bar2.pack(fill="x", pady=2)
        ttk.Button(bar2, text="Refresh table", command=self.load_results).pack(side="left")
        ttk.Button(bar2, text="Open comparison chart",
                   command=self.show_chart).pack(side="left", padx=4)
        self.tree = ttk.Treeview(res_tab, show="headings")
        self.tree.pack(fill="both", expand=True)
        self.load_results()

        # ---------- bottom status bar ----------
        status = ttk.Frame(self.root, padding=(8, 2))
        status.pack(fill="x", side="bottom")
        self.lbl_state = ttk.Label(status, text="idle")
        self.lbl_state.pack(side="left")
        self.lbl_elapsed = ttk.Label(status, text="")
        self.lbl_elapsed.pack(side="right")

    # ----------------------------------------------------- source dropdown --
    def _scan_sources(self) -> list[str]:
        """'0' (webcam) followed by every video file under dataset/."""
        exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
        roots = [BASE / "dataset" / "demo", BASE / "dataset"]
        seen: set[str] = set()
        vids: list[str] = []
        for root in roots:
            if root.exists():
                for p in sorted(root.glob("*")):
                    if p.suffix.lower() in exts and p.name not in seen:
                        seen.add(p.name)
                        vids.append(p.relative_to(BASE).as_posix())
        return ["0"] + vids

    def _refresh_sources(self) -> None:
        self.cmb_source.configure(values=self._scan_sources())

    # ------------------------------------------------------ model dropdown --
    def _scan_models(self) -> list[str]:
        """Every .onnx file currently in models/ (filenames)."""
        if not MODELS.exists():
            return []
        return [p.name for p in sorted(MODELS.glob("*.onnx"))]

    def _refresh_models(self) -> None:
        self.cmb_variant.configure(values=self._scan_models())

    def _scan_pt(self) -> list[str]:
        """Every .pt checkpoint in models/ (prunable PyTorch models)."""
        if not MODELS.exists():
            return []
        return [p.name for p in sorted(MODELS.glob("*.pt"))]

    def _scan_train_sources(self) -> dict:
        """Trainable starting points -> their .pt path. ONNX is excluded because
        it's frozen and can't be trained. Always offers the pretrained base."""
        out: dict = {}
        base = MODELS / "base" / "yolov8n.pt"
        if base.exists():
            out["yolov8n (pretrained base)"] = base
        for p in sorted(MODELS.glob("*.pt")):
            out[p.name] = p
        return out

    def _scan_datasets(self) -> dict:
        """folder name -> its data.yaml path, for each dataset/* that has one."""
        out: dict = {}
        ddir = BASE / "dataset"
        if ddir.exists():
            for sub in sorted(p for p in ddir.iterdir() if p.is_dir()):
                yml = None
                for cand in ("data.yaml", "data_local.yaml"):
                    if (sub / cand).exists():
                        yml = sub / cand
                        break
                if yml is None:
                    found = list(sub.glob("*.yaml"))
                    yml = found[0] if found else None
                if yml is not None:
                    out[sub.name] = yml
        return out

    def train_model(self) -> None:
        if not self._can_launch():
            return
        dsets = self._scan_datasets()
        ds = self.var_tr_data.get().strip()
        if ds not in dsets:
            messagebox.showinfo("Pick a dataset", "Choose a dataset folder to train on.")
            return
        try:
            epochs = int(self.var_tr_epochs.get())
        except ValueError:
            messagebox.showwarning("Epochs", "Epochs must be a whole number.")
            return
        out = self.var_tr_out.get().strip() or "simple"
        srcs = self._scan_train_sources()
        src = self.var_tr_src.get().strip()
        if src not in srcs:
            messagebox.showinfo("Pick a source", "Choose a .pt model to start training from.")
            return
        cmd = [str(VENV_PY), "-u", str(BASE / "01_simple_model.py"),
               "--weights", str(srcs[src]),
               "--data", str(dsets[ds]), "--epochs", str(epochs), "--out", out]
        self.log_line(f"Training '{out}' from {src} on '{ds}' for {epochs} epochs - "
                      "CPU training is slow, watch the epoch lines below.", "warn")
        self._launch({"key": "train", "label": f"Train {out} from {src} on {ds} ({epochs} ep)"}, cmd)

    def combine_quantize(self) -> None:
        if not self._can_launch():
            return
        src = self.var_q_src.get().strip()
        if not src:
            messagebox.showinfo("Pick a model", "Choose an .onnx model to quantize.")
            return
        prefix = Path(src).stem  # quantize pruned.onnx -> pruned_static.onnx etc.
        cmd = [str(VENV_PY), "-u", str(BASE / "03_quantized_model.py"),
               "--onnx", str(MODELS / src), "--prefix", prefix]
        self._launch({"key": "quant", "label": f"Quantize {src} -> {prefix}_static/_fp16"}, cmd)

    def combine_prune(self) -> None:
        if not self._can_launch():
            return
        src = self.var_p_src.get().strip()
        if not src:
            messagebox.showinfo("Pick a model", "Choose a .pt model to prune.")
            return
        stem = Path(src).stem
        out = "pruned" if stem == "simple" else f"{stem}_pruned"
        try:
            pct = max(1, min(90, int(float(self.var_p_pct.get()))))
        except ValueError:
            messagebox.showwarning("Prune %", "Prune % must be a number (1-90).")
            return
        amount = pct / 100.0
        cmd = [str(VENV_PY), "-u", str(BASE / "02_pruned_model.py"),
               "--weights", str(MODELS / src), "--out", out, "--amount", str(amount)]
        self.log_line(f"Pruning {src} by {pct}% then fine-tuning to recover "
                      "(this trains - be patient).", "warn")
        self._launch({"key": "prune", "label": f"Prune {src} {pct}% -> {out}"}, cmd)

    def _can_launch(self) -> bool:
        if self.proc is not None:
            messagebox.showinfo("Busy", "A step is already running - stop it first.")
            return False
        if not VENV_PY.exists():
            messagebox.showwarning("No environment", "Run '0a Install environment' first.")
            return False
        return True

    # ------------------------------------------------------------ logging --
    def log_line(self, s: str, tag: str | None = None) -> None:
        self._log_insert(s, newline=True, tag=tag)

    def _log_insert(self, s: str, newline: bool, tag: str | None = None) -> None:
        if tag is None:
            low = s.lower()
            if "traceback" in low or "[error]" in low or "error:" in low:
                tag = "err"
            elif low.startswith(("[warn", "warning")):
                tag = "warn"
        self.txt.configure(state="normal")
        if self._open_line:  # replace the pending '\r' progress line
            self.txt.delete("end-1c linestart", "end-1c")
        self.txt.insert("end", s + ("\n" if newline else ""), tag or ())
        self._open_line = not newline
        # trim old lines so multi-hour runs don't bloat the widget
        if int(self.txt.index("end-1c").split(".")[0]) > 6000:
            self.txt.delete("1.0", "1500.0")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    # ------------------------------------------------------------ running --
    def run_step(self, step: dict) -> None:
        if self.proc is not None:
            messagebox.showinfo("Busy", "A step is already running - stop it first.")
            return
        if not step.get("system_python") and not VENV_PY.exists():
            messagebox.showwarning(
                "No environment",
                "Run '0a Install environment' first - it creates the .venv "
                "these steps execute in.")
            return
        py = sys.executable if step.get("system_python") else str(VENV_PY)
        cmd = [py, "-u", str(BASE / step["script"])] + list(step.get("extra", []))
        # Benchmark shows each model on video only when "show video window" is ticked;
        # unticked => it still measures size/accuracy/CPU/RAM but opens no windows.
        if step.get("key") == "bench" and self.var_display.get():
            cmd.append("--show")
            self.log_line("Each model shows in a video window for a few seconds, "
                          "one after another.", "warn")
        self._launch(step, cmd)

    def run_detection(self) -> None:
        if self.proc is not None:
            messagebox.showinfo("Busy", "A step is already running - stop it first.")
            return
        if not VENV_PY.exists():
            messagebox.showwarning("No environment", "Run '0a Install environment' first.")
            return
        model_name = self.var_variant.get().strip()
        if not model_name:
            messagebox.showwarning("No model", "No .onnx model found - run the pipeline steps first.")
            return
        cmd = [str(VENV_PY), "-u", str(BASE / "detect.py"),
               "--model", str(MODELS / model_name),
               "--source", self.var_source.get().strip()]
        if not self.var_display.get():
            cmd.append("--no-display")
        step = dict(key="detect", label=f"Live detection ({model_name})")
        if self.var_display.get():
            self.log_line("Video opens in a separate window - press q there to quit.",
                          "warn")
        self._launch(step, cmd)

    def run_remaining(self) -> None:
        todo = [s for s in STEPS if not s["done"]()]
        if not todo:
            messagebox.showinfo("Nothing to do", "Every step is already done.")
            return
        names = "\n".join("  " + s["label"] for s in todo)
        if not messagebox.askyesno(
                "Run remaining steps",
                f"Will run, in order:\n{names}\n\nThis can take a while on CPU. Start?"):
            return
        self.chain = todo[1:]
        self.run_step(todo[0])

    def _launch(self, step: dict, cmd: list[str]) -> None:
        self.current = step
        self.t_start = time.time()
        self.log_line("")
        self.log_line("=" * 72, "hdr")
        self.log_line(f">> {step['label']}    ({' '.join(Path(c).name if os.sep in c else c for c in cmd)})", "hdr")
        self.log_line("=" * 72, "hdr")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(BASE), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=flags,
            )
        except Exception as e:  # noqa: BLE001
            self.log_line(f"[ERROR] could not start: {e}", "err")
            self.proc, self.current = None, None
            return
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self._set_running_ui(True)
        self.refresh_statuses()

    def _reader(self, proc: subprocess.Popen) -> None:
        """Char-wise reader so '\\r' progress bars update in place."""
        buf = ""
        stream = proc.stdout
        assert stream is not None
        while True:
            ch = stream.read(1)
            if ch == "":
                break
            if ch == "\n":
                self.q.put(("line", buf))
                buf = ""
            elif ch == "\r":
                if buf:
                    self.q.put(("rline", buf))
                    buf = ""
            else:
                buf += ch
        if buf:
            self.q.put(("line", buf))
        self.q.put(("exit", proc.wait()))

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "line":
                    self._log_insert(str(payload), newline=True)
                elif kind == "rline":
                    self._log_insert(str(payload), newline=False)
                elif kind == "exit":
                    self._finished(int(payload))
        except queue.Empty:
            pass
        if self.proc is not None:
            mins, secs = divmod(int(time.time() - self.t_start), 60)
            label = self.current["label"] if self.current else "?"
            self.lbl_state.configure(text=f"running: {label}")
            self.lbl_elapsed.configure(text=f"elapsed {mins:02d}:{secs:02d}")
        self.root.after(80, self._poll)

    def _finished(self, rc: int) -> None:
        step = self.current
        self.proc, self.current = None, None
        if self._open_line:
            self._log_insert("", newline=True)
        if rc == 0:
            self.log_line(f"<< done: {step['label'] if step else '?'}", "ok")
        else:
            self.log_line(f"<< FAILED (exit {rc}): {step['label'] if step else '?'} "
                          "- scroll up for the error", "err")
            self.chain = []
        self._set_running_ui(False)
        self.refresh_statuses()
        self.lbl_state.configure(text="idle")
        if step and step.get("key") == "bench" and rc == 0:
            self.load_results()
            self.nb.select(1)
        if self.chain and rc == 0:
            nxt = self.chain.pop(0)
            self.run_step(nxt)

    def stop(self) -> None:
        if self.proc is None:
            return
        self.chain = []
        self.log_line("[stop] terminating...", "warn")
        proc = self.proc
        proc.terminate()

        def force() -> None:
            time.sleep(3)
            if proc.poll() is None:
                proc.kill()

        threading.Thread(target=force, daemon=True).start()

    # ------------------------------------------------------------- status --
    def _set_running_ui(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for btn in self.step_btns.values():
            btn.configure(state=state)
        self.btn_all.configure(state=state)
        self.btn_detect.configure(state=state)
        self.btn_train.configure(state=state)
        self.btn_quant.configure(state=state)
        self.btn_prune.configure(state=state)
        self.btn_stop.configure(state="normal" if running else "disabled")

    def refresh_statuses(self) -> None:
        for step in STEPS:
            mark = self.step_marks[step["key"]]
            if self.current is not None and self.current.get("key") == step["key"]:
                mark.configure(text=">", fg="#b58900")
            elif step["done"]():
                mark.configure(text="V", fg="#1a7f37")
            else:
                mark.configure(text="O", fg="#999999")

    # ------------------------------------------------------------ results --
    def load_results(self) -> None:
        import csv as _csv

        for item in self.tree.get_children():
            self.tree.delete(item)
        path = RESULTS / "benchmark.csv"
        if not path.exists():
            self.tree.configure(columns=("info",))
            self.tree.heading("info", text="no results yet - run step 6")
            return
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(_csv.reader(f))
        if not rows:
            return
        header, data = rows[0], rows[1:]
        self.tree.configure(columns=header)
        for col in header:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=110, anchor="center")
        self.tree.column(header[0], width=230, anchor="w")
        for r in data:
            self.tree.insert("", "end", values=r)

    def show_chart(self) -> None:
        """Display results/comparison.png inside the app. The Pi's default image
        viewer (eom) is unreliable, so we render the PNG in our own Tk window."""
        png = RESULTS / "comparison.png"
        if not png.exists():
            messagebox.showinfo("No chart yet",
                                "Run 'Benchmark all variants' first - it writes "
                                "results/comparison.png.")
            return
        try:
            img = tk.PhotoImage(file=str(png))     # Tk 8.6+ reads PNG with no extra libs
        except Exception:
            open_path(png)                         # ancient Tk: fall back to the OS viewer
            return
        # Integer-subsample a large chart down so it fits the screen (no Pillow needed).
        tw = int(self.root.winfo_screenwidth() * 0.9)
        th = int(self.root.winfo_screenheight() * 0.8)
        factor = 1
        while factor < 12 and (img.width() // factor > tw or img.height() // factor > th):
            factor += 1
        if factor > 1:
            img = img.subsample(factor)

        win = tk.Toplevel(self.root)
        win.title("Model comparison chart  (results/comparison.png)")
        cv = tk.Canvas(win, width=min(img.width(), tw), height=min(img.height(), th),
                       background="#202020", highlightthickness=0)
        hbar = ttk.Scrollbar(win, orient="horizontal", command=cv.xview)
        vbar = ttk.Scrollbar(win, orient="vertical", command=cv.yview)
        cv.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        cv.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)
        cv.create_image(0, 0, anchor="nw", image=img)
        cv.configure(scrollregion=(0, 0, img.width(), img.height()))
        cv.image = img    # keep a reference so Tk doesn't garbage-collect the image
        ttk.Button(win, text="Open full-size in external viewer",
                   command=lambda: open_path(png)).grid(row=2, column=0, columnspan=2,
                                                         sticky="ew")

    # -------------------------------------------------------------- close --
    def _on_close(self) -> None:
        if self.proc is not None:
            if not messagebox.askyesno("Quit", "A step is still running. Stop it and quit?"):
                return
            self.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    if "--selftest" in sys.argv:
        root.update_idletasks()
        root.update()
        print(f"GUI selftest OK ({root.winfo_width()}x{root.winfo_height()})")
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
