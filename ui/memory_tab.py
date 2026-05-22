import customtkinter as ctk
import tkinter as tk
import threading
from pathlib import Path

from core.storage import list_projects
from core.memory import recap_project, search, qa

TXT_STYLE = dict(
    font=("Courier New", 11), wrap="word", relief="flat", padx=10, pady=8,
    bg="#1c1c1c", fg="#f0f0f0", insertbackground="#e8ff47",
    selectbackground="#e8ff47", selectforeground="#000000",
)


class MemoryTab(ctk.CTkFrame):
    def __init__(self, parent, settings):
        super().__init__(parent, fg_color="transparent")
        self.settings = settings
        self._build_ui()
        self._refresh_projects()

    def update_settings(self, settings):
        self.settings = settings
        self._refresh_projects()

    def _build_ui(self):
        # ── project + recap ──
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=6, pady=(6, 2))

        r1 = ctk.CTkFrame(top, fg_color="transparent")
        r1.pack(fill="x", padx=8, pady=6)

        ctk.CTkLabel(r1, text="Project:", width=70, anchor="w").pack(side="left")
        self._project_var = tk.StringVar()
        self._project_combo = ctk.CTkComboBox(r1, variable=self._project_var,
                                               values=[], width=210, state="readonly")
        self._project_combo.pack(side="left", padx=(0, 10))
        ctk.CTkButton(r1, text="Recap Project", command=self._recap_project).pack(side="left", padx=(0, 8))
        self._recap_status = ctk.CTkLabel(r1, text="", text_color="gray")
        self._recap_status.pack(side="left")

        # ── search ──
        sr = ctk.CTkFrame(self, fg_color="transparent")
        sr.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(sr, text="Search:", width=60, anchor="w").pack(side="left", padx=(2, 0))
        self._search_var = tk.StringVar()
        search_entry = ctk.CTkEntry(sr, textvariable=self._search_var,
                                     placeholder_text="Search all projects and meetings…", width=340)
        search_entry.pack(side="left", padx=(4, 8))
        search_entry.bind("<Return>", lambda _: self._do_search())
        ctk.CTkButton(sr, text="Search", width=80, command=self._do_search).pack(side="left")

        # ── results box ──
        ctk.CTkLabel(self, text="RESULTS / RECAP", font=ctk.CTkFont(size=10),
                     text_color="gray").pack(anchor="w", padx=8, pady=(6, 0))
        self._output = tk.Text(self, height=9, **TXT_STYLE)
        self._output.pack(fill="both", expand=True, padx=6, pady=(2, 2))
        self._output.insert("1.0", "Recap a project or search across all meetings here.")
        self._output.config(state="disabled")

        # ── Q&A ──
        ctk.CTkLabel(self, text="ASK QUESTIONS", font=ctk.CTkFont(size=10),
                     text_color="gray").pack(anchor="w", padx=8, pady=(6, 0))
        self._chat = tk.Text(self, height=8, **TXT_STYLE)
        self._chat.pack(fill="both", expand=True, padx=6, pady=(2, 2))
        self._chat.config(state="disabled")
        self._chat.tag_config("q", foreground="#e8ff47", font=("Courier New", 11, "bold"))
        self._chat.tag_config("a", foreground="#f0f0f0")
        self._chat.tag_config("meta", foreground="#555555", font=("Courier New", 9))

        qa_row = ctk.CTkFrame(self, fg_color="transparent")
        qa_row.pack(fill="x", padx=6, pady=(2, 8))

        self._qa_var = tk.StringVar()
        qa_entry = ctk.CTkEntry(qa_row, textvariable=self._qa_var,
                                 placeholder_text="Ask a question about your meeting history…")
        qa_entry.pack(side="left", fill="x", expand=True, padx=(2, 8))
        qa_entry.bind("<Return>", lambda _: self._do_qa())

        self._ask_btn = ctk.CTkButton(qa_row, text="Ask", width=70, command=self._do_qa)
        self._ask_btn.pack(side="left", padx=(0, 8))

        self._scope_var = tk.StringVar(value="All projects")
        ctk.CTkOptionMenu(qa_row, values=["All projects", "Selected project"],
                          variable=self._scope_var, width=150).pack(side="left")

    # ── data ──────────────────────────────────────────────────────────────────
    def _refresh_projects(self):
        projects = list_projects(self.settings.get("projects_folder", ""))
        values = projects or ["(no projects yet)"]
        self._project_combo.configure(values=values)
        if projects and not self._project_var.get():
            self._project_var.set(projects[0])

    # ── recap ─────────────────────────────────────────────────────────────────
    def _recap_project(self):
        project = self._project_var.get().strip()
        if not project or project.startswith("(no"):
            return
        self._recap_status.configure(text="Asking Ollama…")
        path = str(Path(self.settings["projects_folder"]) / project)
        threading.Thread(target=self._run_recap, args=(path,), daemon=True).start()

    def _run_recap(self, path):
        result = recap_project(path, model=self.settings.get("ollama_model", "llama3.2"))
        self.after(0, lambda: self._show_output(result))
        self.after(0, lambda: self._recap_status.configure(text="Done"))

    # ── search ────────────────────────────────────────────────────────────────
    def _do_search(self):
        query = self._search_var.get().strip()
        if not query:
            return
        results = search(self.settings.get("projects_folder", ""), query)
        if not results:
            self._show_output(f'No results for "{query}".')
            return
        lines = [f'Found {len(results)} result(s) for "{query}":\n']
        for r in results:
            lines.append(f"▸ [{r['project']}]  {r['meeting']}")
            lines.append(f"  {r.get('snippet', '')}\n")
        self._show_output("\n".join(lines))

    # ── Q&A ───────────────────────────────────────────────────────────────────
    def _do_qa(self):
        question = self._qa_var.get().strip()
        if not question:
            return
        self._qa_var.set("")
        self._ask_btn.configure(state="disabled")
        project_filter = None
        if self._scope_var.get() == "Selected project":
            p = self._project_var.get().strip()
            if p and not p.startswith("(no"):
                project_filter = p
        self._append_chat(f"Q: {question}\n", "q")
        threading.Thread(target=self._run_qa,
                         args=(question, project_filter), daemon=True).start()

    def _run_qa(self, question, project_filter):
        answer = qa(self.settings.get("projects_folder", ""), question,
                    model=self.settings.get("ollama_model", "llama3.2"),
                    project_filter=project_filter)
        self.after(0, lambda: self._append_chat(f"A: {answer}\n\n", "a"))
        self.after(0, lambda: self._ask_btn.configure(state="normal"))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _show_output(self, text):
        self._output.config(state="normal")
        self._output.delete("1.0", "end")
        self._output.insert("1.0", text)
        self._output.config(state="disabled")

    def _append_chat(self, text, tag="a"):
        self._chat.config(state="normal")
        self._chat.insert("end", text, tag)
        self._chat.see("end")
        self._chat.config(state="disabled")
