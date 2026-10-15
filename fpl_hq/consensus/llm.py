"""OpenAI model settings shared by every consensus stage.

gpt-4.1-mini shuts down on 2027-01-14; OpenAI names gpt-6-luna as its replacement.
GPT-6 models accept temperature=0 only with reasoning switched off, so stages that
want determinism pass DETERMINISTIC.
"""

MODEL = "gpt-6-luna"
DETERMINISTIC = {"reasoning_effort": "none", "temperature": 0}
