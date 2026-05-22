import customtkinter as ctk
import threading

from core.config import load_settings, save_settings
from core.llm import check_ollama
from ui.recorder_tab import RecorderTab
from ui.memory_tab import MemoryTab
from ui.settings_tab import SettingsTab

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PROJECT BRAIN")
        self.geometry("920x740")
        self.minsize(720, 580)

        self.settings = load_settings()
        self._build_ui()
        threading.Thread(target=self._startup_check, daemon=True).start()

    def _build_ui(self):
        self._tabs = ctk.CTkTabview(self)
        self._tabs.pack(fill="both", expand=True, padx=10, pady=10)

        for name in ("Recorder", "Project Memory", "Settings"):
            self._tabs.add(name)

        self._recorder = RecorderTab(self._tabs.tab("Recorder"), self.settings)
        self._recorder.pack(fill="both", expand=True)

        self._memory = MemoryTab(self._tabs.tab("Project Memory"), self.settings)
        self._memory.pack(fill="both", expand=True)

        self._settings = SettingsTab(self._tabs.tab("Settings"), self.settings,
                                      self._on_settings_saved)
        self._settings.pack(fill="both", expand=True)

    def _startup_check(self):
        if not check_ollama():
            self.after(0, self._warn_ollama)

    def _warn_ollama(self):
        win = ctk.CTkToplevel(self)
        win.title("Ollama Not Running")
        win.geometry("440x190")
        win.resizable(False, False)
        win.grab_set()
        ctk.CTkLabel(
            win,
            text=(
                "Ollama is not running.\n\n"
                "Start it in a terminal with:\n"
                "    ollama serve\n\n"
                "Summarize and Q&A will be unavailable until it's running."
            ),
            justify="left",
        ).pack(padx=24, pady=20)
        ctk.CTkButton(win, text="OK", command=win.destroy).pack(pady=(0, 16))

    def _on_settings_saved(self, new_settings):
        self.settings.update(new_settings)
        save_settings(self.settings)
        self._recorder.update_settings(self.settings)
        self._memory.update_settings(self.settings)


if __name__ == "__main__":
    app = App()
    app.mainloop()
