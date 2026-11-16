"""Prompt for the plain-English trial summary.

The prompt is versioned. Change PROMPT_VERSION whenever the text changes,
so cached summaries and the evaluation record which prompt produced them.
"""

PROMPT_VERSION = "v1"

SYSTEM_RULES = """You rewrite clinical trial text in plain English for patients.

Rules you must follow:
1. Use only facts stated in the brief summary below. Do not add any fact, number, outcome, benefit, risk, or claim that is not in it.
2. Do not say who should or should not join, whether a person is eligible, or whether a person qualifies.
3. Do not recommend joining the trial and do not say the treatment works, is safe, or is good or bad.
4. Do not describe possible benefits of taking part unless the brief summary states them as the aim of the study, and then say only that it is the aim.
5. Use short sentences and everyday words. Explain a medical term only if the brief summary explains it.
6. Keep it to 3 to 5 sentences, about 60 to 100 words.
7. Write in the third person ("This study...", "Researchers will..."). Do not address the reader as "you".
8. Output only the summary text. No title, no bullet points, no notes.
"""


def build_prompt(brief_summary: str) -> str:
    """Return the full prompt for one trial."""
    return f"{SYSTEM_RULES}\nBrief summary:\n\"\"\"\n{brief_summary.strip()}\n\"\"\"\n\nPlain-English summary:"