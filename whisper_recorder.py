import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
import datetime
import queue
import time

# ── lazy-load whisper so UI opens instantly ──────────────────────────────────
whisper_model = None
MODEL_SIZE = "base"          # change to "tiny", "small", "medium" if you like
SAMPLE_RATE = 16000

# ── colours & fonts ──────────────────────────────────────────────────────────
BG        = "#0f0f0f"
SURFACE   = "#1a1a1a"
ACCENT    = "#e8ff47"        # electric lime
ACCENT2   = "#ff4747"        # record red
TEXT      = "#f0f0f0"
MUTED     = "#555555"
FONT_MONO = ("Courier New", 11)
FONT_UI   = ("Courier New", 10)
FONT_BIG  = ("Courier New", 28, "bold")

DEFAULT_RAW_FOLDER     = os.path.expanduser("~/Documents/Transcriptions/Raw")
DEFAULT_SUMMARY_FOLDER = os.path.expanduser("~/Documents/Transcriptions/Summaries")


# ── helpers ──────────────────────────────────────────────────────────────────
def load_whisper():
    global whisper_model
    if whisper_model is None:
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe(audio_path):
    load_whisper()
    segments, _ = whisper_model.transcribe(audio_path, beam_size=5)
    return " ".join(s.text.strip() for s in segments)


def ensure_folder(path):
    os.makedirs(path, exist_ok=True)


# ── main app ─────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WHISPER // RECORDER")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.geometry("680x560")
        self.minsize(500, 420)

        self.recording    = False
        self.audio_frames = []
        self.stream       = None
        self.status_var   = tk.StringVar(value="READY")
        self.timer_var    = tk.StringVar(value="00:00")
        self._timer_start = None
        self._timer_job   = None
        self._blink_state = True
        self._blink_job   = None
        self.raw_folder     = DEFAULT_RAW_FOLDER
        self.summary_folder = DEFAULT_SUMMARY_FOLDER

        self._build_ui()
        self._preload_whisper()   # background load so first transcription is fast

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── header ──
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(20, 0))

        tk.Label(hdr, text="WHISPER", font=FONT_BIG,
                 bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(hdr, text=" // RECORDER", font=("Courier New", 14),
                 bg=BG, fg=MUTED).pack(side="left", anchor="s", pady=6)

        # model badge
        tk.Label(hdr, text=f"[ {MODEL_SIZE.upper()} ]", font=FONT_UI,
                 bg=BG, fg=MUTED).pack(side="right", anchor="s", pady=6)

        # ── divider ──
        tk.Frame(self, bg=ACCENT, height=1).pack(fill="x", padx=24, pady=(8, 0))

        # ── status bar ──
        status_row = tk.Frame(self, bg=BG)
        status_row.pack(fill="x", padx=24, pady=(10, 0))

        self.dot = tk.Label(status_row, text="●", font=("Courier New", 14),
                            bg=BG, fg=MUTED)
        self.dot.pack(side="left")

        tk.Label(status_row, textvariable=self.status_var,
                 font=FONT_UI, bg=BG, fg=TEXT).pack(side="left", padx=(6, 0))

        tk.Label(status_row, textvariable=self.timer_var,
                 font=("Courier New", 13, "bold"),
                 bg=BG, fg=ACCENT).pack(side="right")

        # ── big toggle button ──
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(pady=18)

        self.rec_btn = tk.Button(
            btn_frame,
            text="▶  START RECORDING",
            font=("Courier New", 13, "bold"),
            bg=SURFACE, fg=ACCENT,
            activebackground=ACCENT, activeforeground=BG,
            relief="flat", cursor="hand2",
            padx=28, pady=12,
            command=self.toggle_recording,
        )
        self.rec_btn.pack()

        # ── transcription box ──
        tk.Label(self, text="TRANSCRIPTION", font=("Courier New", 9),
                 bg=BG, fg=MUTED).pack(anchor="w", padx=24)

        txt_frame = tk.Frame(self, bg=SURFACE, bd=0)
        txt_frame.pack(fill="both", expand=True, padx=24, pady=(4, 0))

        self.txt = tk.Text(
            txt_frame,
            font=FONT_MONO, bg=SURFACE, fg=TEXT,
            insertbackground=ACCENT,
            relief="flat", wrap="word",
            padx=12, pady=10,
            selectbackground=ACCENT, selectforeground=BG,
        )
        self.txt.pack(fill="both", expand=True, side="left")

        sb = tk.Scrollbar(txt_frame, command=self.txt.yview, bg=SURFACE,
                          troughcolor=SURFACE, activebackground=ACCENT)
        sb.pack(side="right", fill="y")
        self.txt.config(yscrollcommand=sb.set)

        self.txt.insert("1.0", "Your transcription will appear here.\nEdit freely before saving.")
        self.txt.config(state="disabled")

        # ── save section ──
        save_section = tk.Frame(self, bg=BG)
        save_section.pack(fill="x", padx=24, pady=(8, 0))

        tk.Frame(self, bg=MUTED, height=1).pack(fill="x", padx=24, pady=(0, 8))

        def _path_row(parent, label_text, folder_attr, choose_cmd):
            row = tk.Frame(parent, bg=BG)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label_text, font=("Courier New", 8), bg=BG,
                     fg=MUTED, width=12, anchor="w").pack(side="left")
            entry_var = tk.StringVar(value=getattr(self, folder_attr))
            entry = tk.Entry(
                row, textvariable=entry_var,
                font=("Courier New", 9), bg=SURFACE, fg=TEXT,
                insertbackground=ACCENT, relief="flat",
                selectbackground=ACCENT, selectforeground=BG,
            )
            entry.pack(side="left", fill="x", expand=True, padx=(4, 6))

            def _sync_from_entry(*_):
                setattr(self, folder_attr, entry_var.get())

            entry_var.trace_add("write", _sync_from_entry)

            tk.Button(
                row, text="BROWSE",
                font=("Courier New", 8), bg=SURFACE, fg=MUTED,
                activebackground=MUTED, activeforeground=BG,
                relief="flat", cursor="hand2", padx=8, pady=2,
                command=choose_cmd,
            ).pack(side="left")
            return entry_var

        self._raw_var     = _path_row(save_section, "RAW TRANSCRIPT",
                                      "raw_folder",     self._choose_raw_folder)
        self._summary_var = _path_row(save_section, "SUMMARY",
                                      "summary_folder", self._choose_summary_folder)

        # ── bottom bar ──
        bot = tk.Frame(self, bg=BG)
        bot.pack(fill="x", padx=24, pady=(4, 12))

        self.save_btn = tk.Button(
            bot, text="SAVE  ↓",
            font=("Courier New", 11, "bold"),
            bg=ACCENT, fg=BG,
            activebackground=TEXT, activeforeground=BG,
            relief="flat", cursor="hand2", padx=16, pady=6,
            command=self._save,
        )
        self.save_btn.pack(side="right")

        tk.Button(
            bot, text="CLEAR",
            font=FONT_UI, bg=SURFACE, fg=MUTED,
            activebackground=MUTED, activeforeground=BG,
            relief="flat", cursor="hand2", padx=10, pady=4,
            command=self._clear,
        ).pack(side="right", padx=(0, 8))

    # ── recording ─────────────────────────────────────────────────────────────
    def toggle_recording(self):
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self.recording    = True
        self.audio_frames = []
        self._set_status("RECORDING", ACCENT2)
        self.rec_btn.config(text="■  STOP RECORDING", bg=ACCENT2, fg=TEXT,
                            activebackground=TEXT)
        self._timer_start = time.time()
        self._tick_timer()
        self._blink()

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            callback=self._audio_cb,
        )
        self.stream.start()

    def _audio_cb(self, indata, frames, time_info, status):
        self.audio_frames.append(indata.copy())

    def _stop_recording(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self._cancel_timer()
        self._cancel_blink()
        self.dot.config(fg=MUTED)
        self._set_status("TRANSCRIBING…", ACCENT)
        self.rec_btn.config(state="disabled", text="TRANSCRIBING…",
                            bg=SURFACE, fg=MUTED)

        threading.Thread(target=self._do_transcribe, daemon=True).start()

    def _do_transcribe(self):
        try:
            audio = np.concatenate(self.audio_frames, axis=0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())

            result = transcribe(tmp)
            os.unlink(tmp)
            self.after(0, self._show_result, result)
        except Exception as e:
            self.after(0, self._show_result, f"[ERROR: {e}]")

    def _show_result(self, text):
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", text)
        self._set_status("DONE — edit and save", ACCENT)
        self.rec_btn.config(state="normal", text="▶  START RECORDING",
                            bg=SURFACE, fg=ACCENT, activebackground=ACCENT,
                            activeforeground=BG)
        self.timer_var.set("00:00")

    # ── save / clear ──────────────────────────────────────────────────────────
    def _save(self):
        content = self.txt.get("1.0", "end").strip()
        if not content or content.startswith("Your transcription"):
            messagebox.showwarning("Nothing to save", "Transcribe something first.")
            return
        raw_path = self.raw_folder.strip()
        if not raw_path:
            messagebox.showwarning("No folder", "Set a raw transcript folder first.")
            return
        ensure_folder(raw_path)
        stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(raw_path, f"transcript_{stamp}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        self._set_status(f"SAVED → {os.path.basename(filename)}", ACCENT)
        messagebox.showinfo("Saved", f"Transcript saved to:\n{filename}")

    def _clear(self):
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", "Your transcription will appear here.\nEdit freely before saving.")
        self.txt.config(state="disabled")
        self._set_status("READY", MUTED)
        self.dot.config(fg=MUTED)

    def _choose_raw_folder(self):
        chosen = filedialog.askdirectory(title="Choose raw transcript folder",
                                         initialdir=self.raw_folder or os.path.expanduser("~"))
        if chosen:
            self.raw_folder = chosen
            self._raw_var.set(chosen)

    def _choose_summary_folder(self):
        chosen = filedialog.askdirectory(title="Choose summary folder",
                                         initialdir=self.summary_folder or os.path.expanduser("~"))
        if chosen:
            self.summary_folder = chosen
            self._summary_var.set(chosen)

    # ── ui helpers ────────────────────────────────────────────────────────────
    def _set_status(self, msg, color=TEXT):
        self.status_var.set(msg)
        self.dot.config(fg=color)

    def _short_path(self, p):
        home = os.path.expanduser("~")
        return p.replace(home, "~") if p.startswith(home) else p

    def _tick_timer(self):
        if not self.recording:
            return
        elapsed = int(time.time() - self._timer_start)
        m, s    = divmod(elapsed, 60)
        self.timer_var.set(f"{m:02d}:{s:02d}")
        self._timer_job = self.after(1000, self._tick_timer)

    def _cancel_timer(self):
        if self._timer_job:
            self.after_cancel(self._timer_job)
            self._timer_job = None

    def _blink(self):
        if not self.recording:
            return
        self._blink_state = not self._blink_state
        self.dot.config(fg=ACCENT2 if self._blink_state else BG)
        self._blink_job = self.after(500, self._blink)

    def _cancel_blink(self):
        if self._blink_job:
            self.after_cancel(self._blink_job)
            self._blink_job = None

    def _preload_whisper(self):
        threading.Thread(target=load_whisper, daemon=True).start()


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()