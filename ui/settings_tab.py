import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

from core.llm import check_ollama, list_models

MEETING_TYPES = ["General", "Standup", "Design Review", "Client Call", "Retrospective"]
WHISPER_SIZES = ["tiny", "base", "small", "medium"]


class SettingsTab(ctk.CTkFrame):
    def __init__(self, parent, settings, on_save):
        super().__init__(parent, fg_color="transparent")
        self.settings = dict(settings)
        self.on_save = on_save
        self._saved_lbl = None
        self._build_ui()
        threading.Thread(target=self._check_ollama_bg, daemon=True).start()

    def _section(self, title):
        f = ctk.CTkFrame(self)
        f.pack(fill="x", padx=8, pady=(8, 2))
        ctk.CTkLabel(f, text=title,
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=12, pady=(8, 4))
        return f

    def _row(self, parent):
        r = ctk.CTkFrame(parent, fg_color="transparent")
        r.pack(fill="x", padx=12, pady=(0, 8))
        return r

    def _build_ui(self):
        # ── Ollama ──
        ollama = self._section("Ollama (LLM)")

        status_row = self._row(ollama)
        self._dot = ctk.CTkLabel(status_row, text="●", text_color="gray", width=20,
                                  font=ctk.CTkFont(size=16))
        self._dot.pack(side="left")
        self._ollama_lbl = ctk.CTkLabel(status_row, text="Checking…")
        self._ollama_lbl.pack(side="left", padx=(4, 12))
        ctk.CTkButton(status_row, text="Check Status", width=120,
                      command=self._check_ollama).pack(side="left")

        model_row = self._row(ollama)
        ctk.CTkLabel(model_row, text="Model:", width=100, anchor="w").pack(side="left")
        self._model_var = tk.StringVar(value=self.settings.get("ollama_model", "llama3.2"))
        self._model_combo = ctk.CTkComboBox(model_row, variable=self._model_var,
                                             values=[self._model_var.get()], width=220)
        self._model_combo.pack(side="left", padx=(0, 8))
        ctk.CTkButton(model_row, text="Refresh list", width=110,
                      command=self._refresh_models).pack(side="left")

        # ── HuggingFace ──
        hf = self._section("HuggingFace (Speaker Diarization)")

        hf_row = self._row(hf)
        ctk.CTkLabel(hf_row, text="Token:", width=100, anchor="w").pack(side="left")
        self._hf_var = tk.StringVar(value=self.settings.get("hf_token", ""))
        ctk.CTkEntry(hf_row, textvariable=self._hf_var, width=300, show="*").pack(side="left", padx=(0, 8))
        ctk.CTkButton(hf_row, text="Show/Hide", width=90,
                      command=self._toggle_token_visibility).pack(side="left")
        self._hf_entry_ref = hf_row.winfo_children()[1]

        diarize_row = self._row(hf)
        self._diarize_var = tk.BooleanVar(value=self.settings.get("diarize", True))
        ctk.CTkCheckBox(diarize_row, text="Identify speakers by default",
                        variable=self._diarize_var).pack(side="left")

        ctk.CTkLabel(hf, text="Requires HF token above. Accept terms at hf.co/pyannote/speaker-diarization-3.1",
                     text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(0, 8))

        # ── Whisper ──
        whisper = self._section("Whisper (Speech-to-Text)")

        ws_row = self._row(whisper)
        ctk.CTkLabel(ws_row, text="Model size:", width=100, anchor="w").pack(side="left")
        self._whisper_var = tk.StringVar(value=self.settings.get("whisper_model_size", "base"))
        ctk.CTkOptionMenu(ws_row, values=WHISPER_SIZES, variable=self._whisper_var,
                          width=110).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(ws_row, text="tiny = fastest · medium = most accurate",
                     text_color="gray").pack(side="left")

        # ── Storage ──
        storage = self._section("Storage")

        pf_row = self._row(storage)
        ctk.CTkLabel(pf_row, text="Projects folder:", width=110, anchor="w").pack(side="left")
        self._projects_var = tk.StringVar(value=self.settings.get("projects_folder", ""))
        ctk.CTkEntry(pf_row, textvariable=self._projects_var, width=300).pack(side="left", padx=(0, 8))
        ctk.CTkButton(pf_row, text="Browse", width=80,
                      command=self._browse_folder).pack(side="left")

        # ── Defaults ──
        defaults = self._section("Defaults")

        mt_row = self._row(defaults)
        ctk.CTkLabel(mt_row, text="Meeting type:", width=110, anchor="w").pack(side="left")
        self._mt_var = tk.StringVar(value=self.settings.get("default_meeting_type", "General"))
        ctk.CTkOptionMenu(mt_row, values=MEETING_TYPES, variable=self._mt_var,
                          width=170).pack(side="left")

        # ── Save ──
        ctk.CTkButton(self, text="Save Settings", width=150,
                      command=self._save).pack(pady=16)

    # ── Ollama check ──────────────────────────────────────────────────────────
    def _check_ollama(self):
        self._ollama_lbl.configure(text="Checking…")
        self._dot.configure(text_color="gray")
        threading.Thread(target=self._check_ollama_bg, daemon=True).start()

    def _check_ollama_bg(self):
        ok = check_ollama()
        color = "#44bb44" if ok else "#bb4444"
        text = "Running" if ok else "Not running — start Ollama with:  ollama serve"
        self.after(0, lambda: self._dot.configure(text_color=color))
        self.after(0, lambda: self._ollama_lbl.configure(text=text))
        if ok:
            self.after(0, self._refresh_models)

    def _refresh_models(self):
        def _fetch():
            models = list_models()
            if models:
                self.after(0, lambda: self._model_combo.configure(values=models))
                if self._model_var.get() not in models:
                    self.after(0, lambda: self._model_var.set(models[0]))
        threading.Thread(target=_fetch, daemon=True).start()

    def _toggle_token_visibility(self):
        entry = self._hf_entry_ref
        entry.configure(show="" if entry.cget("show") == "*" else "*")

    def _browse_folder(self):
        chosen = filedialog.askdirectory(title="Choose projects folder",
                                          initialdir=self._projects_var.get())
        if chosen:
            self._projects_var.set(chosen)

    def _save(self):
        new = {
            "ollama_model": self._model_var.get().strip(),
            "whisper_model_size": self._whisper_var.get(),
            "projects_folder": self._projects_var.get().strip(),
            "default_meeting_type": self._mt_var.get(),
            "hf_token": self._hf_var.get().strip(),
            "diarize": self._diarize_var.get(),
        }
        self.on_save(new)
        if self._saved_lbl:
            self._saved_lbl.destroy()
        self._saved_lbl = ctk.CTkLabel(self, text="✓  Settings saved", text_color="#44bb44")
        self._saved_lbl.pack(pady=(0, 8))
        self.after(3000, lambda: self._saved_lbl.destroy() if self._saved_lbl else None)
