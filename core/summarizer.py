from core.llm import ask_ollama

_PROMPTS = {
    "General": """\
Structure this meeting transcript into the following format:

## Summary
[2-3 sentence overview]

## Key Points
- [bullet list]

## Action Items
- [owner: task]

## Decisions Made
- [list]

Transcript:
{transcript}""",

    "Standup": """\
Structure this standup transcript:

## Standup — {date}

### Done
- [per person]

### In Progress
- [per person]

### Blockers
- [per person, or "None"]

Transcript:
{transcript}""",

    "Design Review": """\
Structure this design review transcript:

## Design Review Summary

### Problem Statement
[what was reviewed]

### Proposed Solution
[key design decisions]

### Concerns Raised
- [list]

### Agreed Next Steps
- [list]

Transcript:
{transcript}""",

    "Client Call": """\
Structure this client call transcript:

## Client Call Summary

**Client:** [name if mentioned]

### Topics Discussed
- [list]

### Client Requests / Feedback
- [list]

### Commitments Made
- [what we promised]

### Follow-up Actions
- [owner: task]

Transcript:
{transcript}""",

    "Retrospective": """\
Structure this retrospective transcript:

## Retrospective Summary

### What Went Well
- [list]

### What Could Improve
- [list]

### Action Items
- [owner: action]

Transcript:
{transcript}""",
}

SYSTEM = "You are a concise, professional meeting summarizer. Output only the structured notes, no preamble."


def summarize(transcript, meeting_type="General", model="llama3.2"):
    import datetime
    template = _PROMPTS.get(meeting_type, _PROMPTS["General"])
    prompt = template.format(transcript=transcript, date=datetime.date.today().isoformat())
    return ask_ollama(prompt, system=SYSTEM, model=model)
