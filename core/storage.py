import json
from pathlib import Path


def list_projects(projects_folder):
    folder = Path(projects_folder)
    if not folder.exists():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_dir())


def create_project(projects_folder, name):
    folder = Path(projects_folder) / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def list_meetings(project_path):
    folder = Path(project_path)
    if not folder.exists():
        return []
    return sorted((m for m in folder.iterdir() if m.is_dir()), reverse=True)


def save_meeting(project_path, meeting_name, transcript, summary, meta):
    folder = Path(project_path) / meeting_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "transcript.txt").write_text(transcript, encoding="utf-8")
    (folder / "summary.md").write_text(summary, encoding="utf-8")
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def load_people(project_path):
    f = Path(project_path) / "people.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_people(project_path, people):
    f = Path(project_path) / "people.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(people, indent=2, sort_keys=True), encoding="utf-8")


def load_meeting(meeting_path):
    folder = Path(meeting_path)
    result = {}
    for name, key in [("transcript.txt", "transcript"), ("summary.md", "summary")]:
        f = folder / name
        if f.exists():
            result[key] = f.read_text(encoding="utf-8")
    meta_f = folder / "meta.json"
    if meta_f.exists():
        try:
            result["meta"] = json.loads(meta_f.read_text(encoding="utf-8"))
        except Exception:
            result["meta"] = {}
    return result
