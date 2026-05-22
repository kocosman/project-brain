from core.llm import ask_ollama

SYSTEM = """
You are a precise meeting summarizer. You extract structured information
from raw transcripts. Return only valid markdown with the exact section
headers requested. No preamble, no commentary, no extra text.
If information for a section is not present in the transcript, write
'Nothing noted.' under that section. Never invent information.
"""

PROMPT_GENERAL = """
Summarize this meeting transcript into the following sections.
Use markdown headers exactly as shown.

## Decisions
List each decision made. One bullet per decision.
Include enough context to understand it without reading the transcript.

## Action items
Format: - [Owner or 'TBD'] — task description
One bullet per action item.

## Open threads
Unresolved questions, debates, or topics that need a follow-up.
One bullet per thread.

## Key context
Background information, constraints, or facts mentioned that will
matter in future meetings about this project.

Transcript:
{transcript}
"""

PROMPT_STANDUP = """
Summarize this standup transcript into exactly three sections.

## Done
What was completed since the last standup. One bullet per item.

## Doing
What is actively being worked on right now. One bullet per item.

## Blocked
Anything that is blocked, waiting on someone, or needs a decision.
Format: - [Person if known] — what they're blocked on

Transcript:
{transcript}
"""

PROMPT_DESIGN_REVIEW = """
Summarize this design review transcript.

## Decisions
Design decisions made or ratified. Include rationale if mentioned.

## Feedback
Specific feedback given on the design. Group by theme if possible.

## Action items
- [Owner or 'TBD'] — task description

## Open threads
Design questions left unresolved or deferred.

## Key context
Constraints, references, or principles mentioned that should
inform future design decisions.

Transcript:
{transcript}
"""

PROMPT_CLIENT_CALL = """
Summarize this client call transcript.

## Client requests
What the client asked for, requested changes, or expressed needs.

## Decisions
Anything agreed upon between client and team.

## Action items
- [Owner or 'TBD'] — task description
Flag client-facing commitments with [CLIENT COMMITMENT].

## Open threads
Unresolved questions or things to follow up with the client.

## Key context
Client preferences, constraints, or background that matters
for future interactions.

Transcript:
{transcript}
"""

PROMPT_RETRO = """
Summarize this retrospective transcript.

## What went well
Positive observations. One bullet per point.

## What didn't go well
Problems, friction, or failures identified. One bullet per point.

## Action items
Concrete improvements the team committed to.
- [Owner or 'TBD'] — what they will do differently

## Key context
Patterns or root causes identified that should be remembered.

Transcript:
{transcript}
"""

PROMPTS = {
    "General":       PROMPT_GENERAL,
    "Standup":       PROMPT_STANDUP,
    "Design Review": PROMPT_DESIGN_REVIEW,
    "Client Call":   PROMPT_CLIENT_CALL,
    "Retrospective": PROMPT_RETRO,
}


def summarize(transcript, meeting_type="General", model="llama3.2"):
    template = PROMPTS.get(meeting_type, PROMPT_GENERAL)
    prompt = template.format(transcript=transcript)
    return ask_ollama(prompt=prompt, system=SYSTEM, model=model)
