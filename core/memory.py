from pathlib import Path
from core.llm import ask_ollama
from core.storage import list_projects, list_meetings

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load(name):
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


SYSTEM_MEMORY = _load("system_memory.md")
PROMPT_RECAP  = _load("recap.md")
PROMPT_QA     = _load("qa.md")


def _iter_summaries(projects_folder):
    for project_name in list_projects(projects_folder):
        project_path = Path(projects_folder) / project_name
        for meeting_folder in list_meetings(project_path):
            sf = meeting_folder / "summary.md"
            if sf.exists():
                yield {
                    "project": project_name,
                    "meeting": meeting_folder.name,
                    "summary": sf.read_text(encoding="utf-8"),
                    "path": str(meeting_folder),
                }


def search(projects_folder, query):
    q = query.lower()
    results = []
    seen = set()

    for entry in _iter_summaries(projects_folder):
        if q in entry["summary"].lower():
            idx = entry["summary"].lower().find(q)
            start = max(0, idx - 80)
            snippet = "…" + entry["summary"][start:idx + 120].replace("\n", " ") + "…"
            results.append({**entry, "snippet": snippet, "source": "summary"})
            seen.add(entry["meeting"])

    for project_name in list_projects(projects_folder):
        for meeting_folder in list_meetings(Path(projects_folder) / project_name):
            if meeting_folder.name in seen:
                continue
            tf = meeting_folder / "transcript.txt"
            if tf.exists():
                text = tf.read_text(encoding="utf-8")
                if q in text.lower():
                    idx = text.lower().find(q)
                    start = max(0, idx - 80)
                    snippet = "…" + text[start:idx + 120].replace("\n", " ") + "…"
                    results.append({
                        "project": project_name,
                        "meeting": meeting_folder.name,
                        "snippet": f"[transcript] {snippet}",
                        "path": str(meeting_folder),
                        "source": "transcript",
                    })

    return results


def recap_project(project_path, model="llama3.1:8b"):
    summaries = []
    for meeting_folder in list_meetings(Path(project_path)):
        sf = meeting_folder / "summary.md"
        if sf.exists():
            summaries.append(f"### {meeting_folder.name}\n{sf.read_text(encoding='utf-8')}")

    if not summaries:
        return "No summaries found for this project."

    prompt = PROMPT_RECAP.format(all_summaries="\n\n---\n\n".join(summaries))
    return ask_ollama(prompt, system=SYSTEM_MEMORY, model=model)


def qa(projects_folder, question, model="llama3.1:8b", project_filter=None):
    entries = list(_iter_summaries(projects_folder))
    if project_filter:
        entries = [e for e in entries if e["project"] == project_filter]

    if not entries:
        return "No meeting data found to answer from."

    all_summaries = "\n\n---\n\n".join(
        f"[{e['project']} / {e['meeting']}]\n{e['summary']}" for e in entries
    )
    prompt = PROMPT_QA.format(all_summaries=all_summaries, question=question)
    return ask_ollama(prompt, system=SYSTEM_MEMORY, model=model)
