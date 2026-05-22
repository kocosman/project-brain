from pathlib import Path
from core.llm import ask_ollama

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load(name):
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


SYSTEM = _load("system_summarizer.md")

PROMPTS = {
    "General":       _load("general.md"),
    "Standup":       _load("standup.md"),
    "Design Review": _load("design_review.md"),
    "Client Call":   _load("client_call.md"),
    "Retrospective": _load("retrospective.md"),
}


def summarize(transcript, meeting_type="General", model="llama3.1:8b"):
    template = PROMPTS.get(meeting_type, PROMPTS["General"])
    prompt = template.format(transcript=transcript)
    return ask_ollama(prompt=prompt, system=SYSTEM, model=model)
