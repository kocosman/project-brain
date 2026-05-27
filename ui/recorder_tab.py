import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
import time
import datetime
from pathlib import Path

from core.transcriber import transcribe, preload
from core.summarizer import summarize
from core.storage import list_projects, create_project, save_meeting, load_people, save_people

MEETING_TYPES = ["General", "Standup", "Design Review", "Client Call", "Retrospective"]
SAMPLE_RATE = 16000
LOW_CONF = 0.80

TXT_STYLE = dict(
    font=("Courier New", 11), wrap="word", relief="flat", padx=10, pady=8,
    bg="#1c1c1c", fg="#f0f0f0", insertbackground="#e8ff47",
    selectbackground="#e8ff47", selectforeground="#000000",
)


class RecorderTab(ctk.CTkFrame):
    def __init__(self, parent, settings):
        super().__init__(parent, fg_color="transparent")
        self.settings = settings
        self.recording = False
        self.audio_frames = []
        self.stream = None
        self._timer_start = None
        self._timer_job = None
        self._words = []
        self._speaker_runs = None
        self._speaker_entries = {}
        self._last_audio = None   # numpy int16 array kept for playback + save
        self._playing = False

        self._build_ui()
        self._refresh_projects()
        threading.Thread(target=preload,
                         args=(settings.get("whisper_model_size", "base"),),
                         daemon=True).start()

    def update_settings(self, settings):
        self.settings = settings
        self._refresh_projects()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # top controls
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=6, pady=(6, 2))

        r1 = ctk.CTkFrame(top, fg_color="transparent")
        r1.pack(fill="x", padx=8, pady=(6, 2))
        ctk.CTkLabel(r1, text="Project:", width=75, anchor="w").pack(side="left")
        self._project_var = tk.StringVar()
        self._project_combo = ctk.CTkComboBox(r1, variable=self._project_var,
                                               values=[], width=210, state="readonly")
        self._project_combo.pack(side="left", padx=(0, 8))
        ctk.CTkButton(r1, text="+ New Project", width=110,
                      command=self._new_project).pack(side="left")

        r2 = ctk.CTkFrame(top, fg_color="transparent")
        r2.pack(fill="x", padx=8, pady=(2, 6))
        ctk.CTkLabel(r2, text="Type:", width=75, anchor="w").pack(side="left")
        self._type_var = tk.StringVar(value=self.settings.get("default_meeting_type", "General"))
        ctk.CTkComboBox(r2, variable=self._type_var, values=MEETING_TYPES,
                        width=150, state="readonly").pack(side="left", padx=(0, 12))
        ctk.CTkLabel(r2, text="Name:", width=48, anchor="w").pack(side="left")
        self._name_var = tk.StringVar(value=datetime.date.today().strftime("%Y-%m-%d"))
        ctk.CTkEntry(r2, textvariable=self._name_var, width=180).pack(side="left", padx=(0, 12))

        # record row
        rec = ctk.CTkFrame(self)
        rec.pack(fill="x", padx=6, pady=2)
        self._rec_btn = ctk.CTkButton(rec, text="▶  START RECORDING",
                                       width=200, height=40,
                                       fg_color="#2d5a1b", hover_color="#3d7a25",
                                       command=self._toggle_recording)
        self._rec_btn.pack(side="left", padx=8, pady=6)
        self._timer_lbl = ctk.CTkLabel(rec, text="00:00",
                                        font=ctk.CTkFont(size=20, weight="bold"))
        self._timer_lbl.pack(side="left", padx=(0, 10))
        self._status_lbl = ctk.CTkLabel(rec, text="READY", text_color="gray")
        self._status_lbl.pack(side="left", padx=(0, 10))
        self._level_bar = ctk.CTkProgressBar(rec, width=140, mode="determinate")
        self._level_bar.set(0)
        self._level_bar.pack(side="left")

        # speaker naming panel — always in layout, empty = 0 height
        self._speaker_frame = ctk.CTkFrame(self)
        self._speaker_frame.pack(fill="x", padx=6, pady=0)

        # transcript header with confidence legend
        trans_hdr = ctk.CTkFrame(self, fg_color="transparent")
        trans_hdr.pack(fill="x", padx=8, pady=(6, 0))
        ctk.CTkLabel(trans_hdr, text="TRANSCRIPT", font=ctk.CTkFont(size=10),
                     text_color="gray").pack(side="left")
        ctk.CTkLabel(trans_hdr, text="  ● normal",
                     font=ctk.CTkFont(size=10), text_color="#f0f0f0").pack(side="left", padx=(12, 0))
        ctk.CTkLabel(trans_hdr, text="  ● low confidence",
                     font=ctk.CTkFont(size=10), text_color="#ff9900").pack(side="left")

        self._transcript = tk.Text(self, height=7, **TXT_STYLE)
        self._transcript.pack(fill="both", expand=True, padx=6, pady=(2, 2))
        self._transcript.insert("1.0", "Record a meeting — transcript appears here.")
        self._transcript.config(state="disabled")
        self._transcript.tag_config("low_conf", foreground="#ff9900")

        # summarize row
        sr = ctk.CTkFrame(self, fg_color="transparent")
        sr.pack(fill="x", padx=6, pady=2)
        self._sum_btn = ctk.CTkButton(sr, text="✦  Summarize with Ollama",
                                       command=self._do_summarize, state="disabled")
        self._sum_btn.pack(side="left", padx=(2, 8))
        self._sum_status = ctk.CTkLabel(sr, text="", text_color="gray")
        self._sum_status.pack(side="left")

        # summary
        ctk.CTkLabel(self, text="SUMMARY", font=ctk.CTkFont(size=10),
                     text_color="gray").pack(anchor="w", padx=8, pady=(4, 0))
        self._summary = tk.Text(self, height=7, **TXT_STYLE)
        self._summary.pack(fill="both", expand=True, padx=6, pady=(2, 2))
        self._summary.insert("1.0", "Click 'Summarize' after transcribing.")
        self._summary.config(state="disabled")

        # bottom bar
        bot = ctk.CTkFrame(self, fg_color="transparent")
        bot.pack(fill="x", padx=6, pady=(2, 8))
        self._save_btn = ctk.CTkButton(bot, text="SAVE  ↓", width=100,
                                        command=self._save, state="disabled")
        self._save_btn.pack(side="right", padx=(4, 2))
        self._export_btn = ctk.CTkButton(bot, text="Export PDF", width=100,
                                          fg_color="gray30", hover_color="gray40",
                                          command=self._export_pdf, state="disabled")
        self._export_btn.pack(side="right", padx=(4, 2))
        self._play_btn = ctk.CTkButton(bot, text="▶ Play Recording", width=130,
                                        fg_color="gray30", hover_color="gray40",
                                        command=self._toggle_playback, state="disabled")
        self._play_btn.pack(side="right", padx=(4, 2))
        ctk.CTkButton(bot, text="Clear", width=80,
                      fg_color="gray30", hover_color="gray40",
                      command=self._clear).pack(side="right", padx=(4, 2))

    # ── projects ──────────────────────────────────────────────────────────────
    def _refresh_projects(self):
        projects = list_projects(self.settings.get("projects_folder", ""))
        self._project_combo.configure(values=projects or ["(no projects yet)"])
        if projects and not self._project_var.get():
            self._project_var.set(projects[0])

    def _new_project(self):
        dialog = ctk.CTkInputDialog(text="Project name:", title="New Project")
        name = dialog.get_input()
        if name and name.strip():
            create_project(self.settings["projects_folder"], name.strip())
            self._refresh_projects()
            self._project_var.set(name.strip())

    # ── recording ─────────────────────────────────────────────────────────────
    def _toggle_recording(self):
        if not self.recording:
            self._start()
        else:
            self._stop()

    def _start(self):
        self.recording = True
        self.audio_frames = []
        self._clear_speaker_panel()
        self._set_status("RECORDING", "#ff4747")
        self._rec_btn.configure(text="■  STOP RECORDING",
                                fg_color="#7a1b1b", hover_color="#9a2525")
        self._timer_start = time.time()
        self._tick()
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                     dtype="int16", callback=self._audio_cb,
                                     blocksize=1024)
        self.stream.start()

    def _audio_cb(self, indata, frames, time_info, status):
        self.audio_frames.append(indata.copy())
        rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
        self.after(0, self._level_bar.set, min(1.0, rms / 3000.0))

    def _stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self._cancel_tick()
        self.after(0, self._level_bar.set, 0)

        if not self.audio_frames:
            self._set_status("No audio captured", "gray")
            self._rec_btn.configure(text="▶  START RECORDING",
                                    fg_color="#2d5a1b", hover_color="#3d7a25")
            return

        self._set_status("TRANSCRIBING…", "#e8ff47")
        self._rec_btn.configure(state="disabled", text="TRANSCRIBING…",
                                fg_color="gray30", hover_color="gray30")
        threading.Thread(target=self._run_transcribe, daemon=True).start()

    def _run_transcribe(self):
        try:
            audio = np.concatenate(self.audio_frames, axis=0)
            self._last_audio = audio  # keep for playback and save
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())

            text, words, speaker_runs = transcribe(
                tmp,
                model_size=self.settings.get("whisper_model_size", "base"),
                diarize=self.settings.get("diarize", True),
                hf_token=self.settings.get("hf_token", ""),
            )
            os.unlink(tmp)
            self.after(0, self._show_transcript, text, words, speaker_runs)
        except Exception as e:
            self.after(0, self._show_transcript, f"[ERROR: {e}]", [], None)

    def _show_transcript(self, text, words, speaker_runs):
        self._words = words
        self._speaker_runs = speaker_runs

        self._transcript.config(state="normal")
        self._transcript.delete("1.0", "end")
        self._transcript.insert("1.0", text)
        self._highlight_low_conf(text, words)

        if speaker_runs:
            self._populate_speaker_panel(speaker_runs)
        else:
            self._clear_speaker_panel()

        self._set_status("DONE — edit or summarize", "#e8ff47")
        self._rec_btn.configure(state="normal", text="▶  START RECORDING",
                                fg_color="#2d5a1b", hover_color="#3d7a25")
        self._timer_lbl.configure(text="00:00")
        self._sum_btn.configure(state="normal")
        self._save_btn.configure(state="normal")
        self._play_btn.configure(state="normal")

    def _highlight_low_conf(self, text, words):
        self._transcript.tag_remove("low_conf", "1.0", "end")
        content = self._transcript.get("1.0", "end-1c")
        pos = 0
        for w in words:
            if w["probability"] < LOW_CONF:
                clean = w["word"].strip()
                if not clean:
                    continue
                idx = content.lower().find(clean.lower(), pos)
                if idx == -1:
                    continue
                self._transcript.tag_add("low_conf", f"1.0+{idx}c", f"1.0+{idx+len(clean)}c")
                pos = idx + len(clean)

    # ── speaker naming panel ──────────────────────────────────────────────────
    def _populate_speaker_panel(self, speaker_runs):
        self._clear_speaker_panel()

        unique_speakers = []
        for r in speaker_runs:
            if r["speaker"] not in unique_speakers:
                unique_speakers.append(r["speaker"])

        # load known people for autocomplete
        known = []
        project = self._project_var.get()
        if project and not project.startswith("(no"):
            path = Path(self.settings["projects_folder"]) / project
            known = list(load_people(str(path)).keys())

        # colored panel so it's unmissable
        inner = ctk.CTkFrame(self._speaker_frame, fg_color="#2a2a1a", corner_radius=8)
        inner.pack(fill="x", padx=6, pady=(4, 6))

        ctk.CTkLabel(inner, text="👤  NAME THE SPEAKERS",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#e8ff47").pack(anchor="w", padx=12, pady=(8, 4))

        ctk.CTkLabel(inner, text="Type or select a name for each detected speaker, then click Apply.",
                     font=ctk.CTkFont(size=10), text_color="gray").pack(anchor="w", padx=12, pady=(0, 6))

        entries_row = ctk.CTkFrame(inner, fg_color="transparent")
        entries_row.pack(fill="x", padx=12, pady=(0, 10))

        self._speaker_entries = {}
        for label in unique_speakers:
            ctk.CTkLabel(entries_row, text=label, width=110,
                         font=ctk.CTkFont(size=11), anchor="w",
                         text_color="gray").pack(side="left", padx=(0, 4))
            combo = ctk.CTkComboBox(entries_row, values=known, width=150)
            combo.set("")
            combo.pack(side="left", padx=(0, 16))
            self._speaker_entries[label] = combo

        ctk.CTkButton(entries_row, text="Apply Names",
                      fg_color="#4a7a2a", hover_color="#5a9a32",
                      width=120, command=self._apply_names).pack(side="left")

    def _apply_names(self):
        mapping = {label: combo.get().strip()
                   for label, combo in self._speaker_entries.items()
                   if combo.get().strip()}
        if not mapping:
            return

        content = self._transcript.get("1.0", "end-1c")
        for label, name in mapping.items():
            content = content.replace(label, name)

        self._transcript.config(state="normal")
        self._transcript.delete("1.0", "end")
        self._transcript.insert("1.0", content)
        self._highlight_low_conf(content, self._words)

        # persist new names to people.json
        project = self._project_var.get()
        if project and not project.startswith("(no"):
            path = Path(self.settings["projects_folder"]) / project
            existing = load_people(str(path))
            for name in mapping.values():
                existing.setdefault(name, {})
            save_people(str(path), existing)

        self._clear_speaker_panel()

    def _clear_speaker_panel(self):
        for w in self._speaker_frame.winfo_children():
            w.destroy()
        self._speaker_entries = {}

    # ── playback ──────────────────────────────────────────────────────────────
    def _toggle_playback(self):
        if self._playing:
            sd.stop()
            self._playing = False
            self._play_btn.configure(text="▶ Play Recording")
        else:
            if self._last_audio is None:
                return
            self._playing = True
            self._play_btn.configure(text="■ Stop")
            def _play():
                sd.play(self._last_audio, samplerate=SAMPLE_RATE)
                sd.wait()
                self.after(0, self._on_playback_done)
            threading.Thread(target=_play, daemon=True).start()

    def _on_playback_done(self):
        self._playing = False
        self._play_btn.configure(text="▶ Play Recording")

    # ── summarize ─────────────────────────────────────────────────────────────
    def _do_summarize(self):
        transcript = self._transcript.get("1.0", "end").strip()
        if not transcript or transcript.startswith("Record"):
            return
        self._sum_btn.configure(state="disabled")
        self._sum_status.configure(text="Asking Ollama…")
        threading.Thread(target=self._run_summarize, args=(transcript,), daemon=True).start()

    def _run_summarize(self, transcript):
        result = summarize(transcript, self._type_var.get(),
                           self.settings.get("ollama_model", "llama3.1:8b"))
        self.after(0, self._show_summary, result)

    def _show_summary(self, text):
        self._summary.config(state="normal")
        self._summary.delete("1.0", "end")
        self._summary.insert("1.0", text)
        self._sum_btn.configure(state="normal")
        self._sum_status.configure(text="Done")
        self._export_btn.configure(state="normal")

    # ── save / export ─────────────────────────────────────────────────────────
    def _save(self):
        project = self._project_var.get().strip()
        if not project or project.startswith("(no"):
            messagebox.showwarning("No project", "Select or create a project first.")
            return
        transcript = self._transcript.get("1.0", "end").strip()
        if not transcript or transcript.startswith("Record"):
            messagebox.showwarning("Nothing to save", "Transcribe a recording first.")
            return
        summary = self._summary.get("1.0", "end").strip()
        if summary.startswith("Click"):
            summary = ""

        raw_name = self._name_var.get().strip() or datetime.date.today().isoformat()
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in raw_name)
        folder_name = f"{datetime.date.today().isoformat()}_{safe}"

        meta = {
            "project": project,
            "meeting_name": raw_name,
            "meeting_type": self._type_var.get(),
            "diarized": self._diarize_var.get(),
            "date": datetime.datetime.now().isoformat(),
        }
        saved = save_meeting(
            Path(self.settings["projects_folder"]) / project,
            folder_name, transcript, summary, meta,
        )
        # save audio alongside transcript if available
        if self._last_audio is not None:
            audio_path = saved / "recording.wav"
            with wave.open(str(audio_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(self._last_audio.tobytes())

        self._set_status(f"SAVED → {saved.name}", "#e8ff47")
        messagebox.showinfo("Saved", f"Meeting saved to:\n{saved}")

    def _export_pdf(self):
        summary = self._summary.get("1.0", "end").strip()
        if not summary or summary.startswith("Click"):
            messagebox.showwarning("No summary", "Generate a summary first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                             filetypes=[("PDF", "*.pdf")],
                                             title="Export summary PDF")
        if not path:
            return
        try:
            import markdown2
            from weasyprint import HTML
            html = markdown2.markdown(summary)
            styled = (
                "<html><body style='font-family:sans-serif;padding:48px;"
                f"max-width:780px;margin:auto'>{html}</body></html>"
            )
            HTML(string=styled).write_pdf(path)
            messagebox.showinfo("Exported", f"PDF saved to:\n{path}")
        except ImportError:
            messagebox.showerror("Missing packages", "Run: pip install markdown2 weasyprint")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _clear(self):
        for box, ph in [
            (self._transcript, "Record a meeting — transcript appears here."),
            (self._summary, "Click 'Summarize' after transcribing."),
        ]:
            box.config(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", ph)
            box.config(state="disabled")
        self._clear_speaker_panel()
        self._set_status("READY", "gray")
        self._sum_btn.configure(state="disabled")
        self._save_btn.configure(state="disabled")
        self._export_btn.configure(state="disabled")
        self._play_btn.configure(state="disabled", text="▶ Play Recording")
        self._sum_status.configure(text="")
        self._words = []
        self._speaker_runs = None
        self._last_audio = None
        self._playing = False

    # ── helpers ───────────────────────────────────────────────────────────────
    def _set_status(self, msg, color="gray"):
        self._status_lbl.configure(text=msg, text_color=color)

    def _tick(self):
        if not self.recording:
            return
        elapsed = int(time.time() - self._timer_start)
        m, s = divmod(elapsed, 60)
        self._timer_lbl.configure(text=f"{m:02d}:{s:02d}")
        self._timer_job = self.after(1000, self._tick)

    def _cancel_tick(self):
        if self._timer_job:
            self.after_cancel(self._timer_job)
            self._timer_job = None
