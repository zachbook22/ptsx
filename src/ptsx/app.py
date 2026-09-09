"""Desktop app: pick User WAV, Assistant WAV, output folder, then gate."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ptsx.pipeline import default_config, gate_files
from ptsx.transcribe import whisper_available

os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")


def main() -> int:
    App().run()
    return 0


class App:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("PTSX")
        self.root.minsize(560, 420)
        self.user_wav: Path | None = None
        self.asst_wav: Path | None = None
        self.outdir: Path | None = None
        self.busy = False
        self.log_q: queue.Queue[str] = queue.Queue()

        pad = {"padx": 12, "pady": 6}
        frm = ttk.Frame(self.root, padding=16)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Put in two aligned raws. Output goes to one folder.").pack(
            anchor=tk.W, **pad
        )

        self.user_var = tk.StringVar(value="No file selected")
        self.asst_var = tk.StringVar(value="No file selected")
        self.out_var = tk.StringVar(value="No folder selected")

        self._path_row(frm, "User WAV", self.user_var, self._pick_user, pad)
        self._path_row(frm, "Assistant WAV", self.asst_var, self._pick_asst, pad)
        self._path_row(frm, "Output folder", self.out_var, self._pick_out, pad)

        self.transcribe = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frm,
            text="Write conversation script (Whisper — slower on long files)",
            variable=self.transcribe,
        ).pack(anchor=tk.W, **pad)

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, **pad)
        self.run_btn = ttk.Button(btns, text="Run", command=self._run, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT)
        self.reveal_btn = ttk.Button(
            btns, text="Reveal in Finder", command=self._reveal, state=tk.DISABLED
        )
        self.reveal_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.log = tk.Text(frm, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True, **pad)

        self.root.after(120, self._drain_log)

    def _path_row(self, parent, label, var, cmd, pad) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, **pad)
        ttk.Button(row, text=f"{label}…", command=cmd, width=18).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=var).pack(side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True)

    def _pick_user(self) -> None:
        path = filedialog.askopenfilename(
            title="User raw WAV",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self.user_wav = Path(path)
            self.user_var.set(path)
            self._refresh()

    def _pick_asst(self) -> None:
        path = filedialog.askopenfilename(
            title="Assistant raw WAV",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self.asst_wav = Path(path)
            self.asst_var.set(path)
            self._refresh()

    def _pick_out(self) -> None:
        initial = None
        if self.user_wav:
            initial = str(self.user_wav.parent)
        path = filedialog.askdirectory(title="Output folder", initialdir=initial)
        if path:
            self.outdir = Path(path)
            self.out_var.set(path)
            self._refresh()

    def _refresh(self) -> None:
        ready = bool(self.user_wav and self.asst_wav and self.outdir) and not self.busy
        self.run_btn.configure(state=tk.NORMAL if ready else tk.DISABLED)
        self.reveal_btn.configure(
            state=tk.NORMAL if self.outdir and self.outdir.is_dir() else tk.DISABLED
        )

    def _log(self, msg: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, msg.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _drain_log(self) -> None:
        try:
            while True:
                self._log(self.log_q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(120, self._drain_log)

    def _run(self) -> None:
        if self.busy or not (self.user_wav and self.asst_wav and self.outdir):
            return
        if self.user_wav.resolve() == self.asst_wav.resolve():
            messagebox.showerror("PTSX", "User and assistant files must be different.")
            return
        if self.transcribe.get() and not whisper_available():
            messagebox.showerror(
                "PTSX",
                "Whisper is not installed. Run install.sh in the ptsx folder.",
            )
            return
        self.busy = True
        self._refresh()
        self._log("Starting…")
        user, asst, outdir = self.user_wav, self.asst_wav, self.outdir
        transcribe = self.transcribe.get()
        threading.Thread(
            target=self._worker, args=(user, asst, outdir, transcribe), daemon=True
        ).start()

    def _worker(self, user: Path, asst: Path, outdir: Path, transcribe: bool) -> None:
        class Writer:
            def write(self, s: str, q=self.log_q) -> int:
                if s:
                    q.put(s)
                return len(s)

            def flush(self) -> None:
                return None

        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = Writer()  # type: ignore[assignment]
        try:
            cfg = default_config()
            cfg.transcribe = transcribe
            paths = gate_files(user, asst, outdir, cfg)
            self.log_q.put("Done.")
            for key, path in paths.items():
                self.log_q.put(f"  {key}: {path}")
            self.root.after(0, lambda: self._done(True))
        except SystemExit as exc:
            self.log_q.put(str(exc) or "Stopped.")
            self.root.after(0, lambda: self._done(False))
        except Exception as exc:
            self.log_q.put(f"Error: {exc}")
            self.root.after(0, lambda: self._done(False, str(exc)))
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def _done(self, ok: bool, err: str | None = None) -> None:
        self.busy = False
        self._refresh()
        if ok:
            messagebox.showinfo("PTSX", f"Wrote files to:\n{self.outdir}")
        elif err:
            messagebox.showerror("PTSX", err)

    def _reveal(self) -> None:
        if self.outdir and self.outdir.is_dir():
            subprocess.run(["open", str(self.outdir)], check=False)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    raise SystemExit(main())
