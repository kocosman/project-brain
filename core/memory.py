from pathlib import Path
from core.llm import ask_ollama
from core.storage import list_projects, list_meetings


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


def recap_project(project_path, model="llama3.2"):
    summaries = []
    for meeting_folder in list_meetings(Path(project_path)):
        sf = meeting_folder / "summary.md"
        if sf.exists():
            summaries.append(f"### {meeting_folder.name}\n{sf.read_text(encoding='utf-8')}")

    if not summaries:
        return "No summaries found for this project."

    context = "\n\n---\n\n".join(summaries)
    prompt = (
        "Given these meeting summaries, write a cohesive project recap covering:\n"
        "- What has been accomplished\n"
        "- What is in progress\n"
        "- Open questions and blockers\n"
        "- Overall trajectory\n\n"
        f"Meeting summaries:\n{context}"
    )
    return ask_ollama(prompt, system="You are a project manager summarizing project history.", model=model)


def qa(projects_folder, question, model="llama3.2", project_filter=None):
    entries = list(_iter_summaries(projects_folder))
    if project_filter:
        entries = [e for e in entries if e["project"] == project_filter]

    if not entries:
        return "No meeting data found to answer from."

    context = "\n\n---\n\n".join(
        f"[{e['project']} / {e['meeting']}]\n{e['summary']}" for e in entries
    )
    system = (
        "You answer questions based only on the provided meeting notes. "
        "If the answer is not in the notes, say so clearly."
    )
    return ask_ollama(f"Context:\n{context}\n\nQuestion: {question}", system=system, model=model)
