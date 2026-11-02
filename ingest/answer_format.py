"""The answer format of S7-S8: the prompt, the strict JSON schema and the output parser.

A model answers a question from numbered sources and returns one JSON object: whether the sources
are enough (`sufficiency`), the atomic claims it makes, each with the citations that support it
(`chunk_id` and a literal `quote`), and what is missing. This module only defines and reads that
format. It runs no model and reads no file. Whether a citation is genuine is decided by
ingest/answer_verifier.py, which checks provenance and citation integrity, never legal meaning.

Two kinds of defect are told apart. A *structural* defect (no JSON, invalid JSON, a field of the
wrong type or an extra field) makes the output unusable and every abstention rule abstains. A
*soft* defect breaks an instruction of the prompt that the schema cannot enforce without making
the rules of ingest/answer_verifier.py identical (sufficiency false with claims, sufficiency true
without claims, sufficiency false without a statement of what is missing); it is recorded and
counted, and the output is still read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROMPT_VERSION = "answer-prompt.v2"
MAX_NEW_TOKENS = 900

SYSTEM_PROMPT = """\
You answer questions about EU financial regulation (DORA, MiCA, PSD2) using ONLY the sources \
supplied in the user message. For this task you have no other knowledge.

Reply with exactly one JSON object and nothing else:
{"sufficiency": true or false, "claims": [...], "missing": "..."}

Rules
1. First decide whether the sources are enough to answer the question fully and correctly. If \
they are not, reply {"sufficiency": false, "claims": [], "missing": "<what the sources do not \
establish>"}. Do not attempt a partial answer.
2. If they are enough, set "sufficiency" to true, set "missing" to "" and list the claims that \
answer the question.
3. Each claim is atomic: one obligation, condition, exception, definition or figure. Do not join \
two statements that could stand alone.
4. Each claim has one or more citations. A citation is {"chunk_id": "...", "quote": "..."}. \
"chunk_id" is copied exactly from the id attribute of a source. "quote" is an exact, contiguous \
passage copied character for character from that source: the same words in the same order, with \
the same capital letters and punctuation. If the passage starts a sentence, keep the capital \
letter of its first word. Choose one sentence or clause of 10 to 40 words that contains the \
words which support the claim; never more than 80 words.
5. State a number, date, amount or percentage in a claim only if it appears in the sources you \
cite for that claim. Add no article numbers and no information that is not in the sources.
6. Never cite a source that was not supplied. Never paraphrase inside a quote.

Example 1. Sources: <source id="EXAMPLE:1">Financial entities shall keep records of all \
incidents for at least five years.</source> Question: How long must incident records be kept?
{"sufficiency": true, "claims": [{"text": "Financial entities must keep incident records for \
at least five years.", "citations": [{"chunk_id": "EXAMPLE:1", "quote": "Financial entities shall \
keep records of all incidents for at least five years"}]}], "missing": ""}

Example 2. Sources: <source id="EXAMPLE:2">Financial entities shall keep records of all \
incidents for at least five years.</source> Question: Which authority receives the incident \
reports?
{"sufficiency": false, "claims": [], "missing": "The sources do not say which authority \
receives incident reports."}"""

EXAMPLE_SOURCES = {
    "EXAMPLE:1": "Financial entities shall keep records of all incidents for at least five years."
}


def build_messages(question: str, sources: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    """The chat messages for one case: the fixed system prompt and the sources and question."""
    blocks = "\n".join(f'<source id="{cid}">\n{text}\n</source>' for cid, text in sources)
    user = f"Sources:\n{blocks}\n\nQuestion: {question}\n\nReply with the JSON object."
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def prompt_sha256() -> str:
    """The hash that identifies the prompt version: the system prompt, the user template and the
    token limit, so that none of them can change without the hash changing."""
    template = build_messages("{question}", [("{chunk_id}", "{text}")])[1]["content"]
    payload = json.dumps(
        {
            "version": PROMPT_VERSION,
            "system": SYSTEM_PROMPT,
            "user": template,
            "max": MAX_NEW_TOKENS,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_id: str
    quote: str


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(min_length=1)
    citations: list[Citation]


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sufficiency: bool
    claims: list[Claim]
    missing: str


@dataclass(frozen=True)
class ParsedOutput:
    """What came back. `answer` is None when the output is structurally unusable, and then
    `error` says why. `soft` lists the instructions the output broke without being unusable."""

    answer: Answer | None
    error: str | None
    extra_text: bool
    soft: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.answer is not None


def parse_output(raw: str) -> ParsedOutput:
    """Read one JSON object from the text of a model. A code fence or words around the object are
    tolerated and noted (`extra_text`); everything else follows the schema strictly."""
    text = raw.strip()
    if not text:
        return ParsedOutput(None, "empty_output", False, ())
    start = text.find("{")
    if start < 0:
        return ParsedOutput(None, "no_json_object", bool(text), ())
    try:
        data, end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return ParsedOutput(None, "invalid_json", start > 0, ())
    around = (text[:start] + text[start + end :]).replace("```json", "").replace("```", "")
    extra = bool(around.strip())
    try:
        answer = Answer.model_validate(data)
    except ValidationError:
        return ParsedOutput(None, "schema_invalid", extra, ())
    soft: list[str] = []
    if not answer.sufficiency and answer.claims:
        soft.append("sufficiency_false_with_claims")
    if answer.sufficiency and not answer.claims:
        soft.append("sufficiency_true_without_claims")
    if not answer.sufficiency and not answer.missing.strip():
        soft.append("insufficient_without_missing")
    return ParsedOutput(answer, None, extra, tuple(soft))


def render_sources(chunk_ids: Sequence[str], texts: Mapping[str, str]) -> list[tuple[str, str]]:
    """The (id, text) pairs of a context, in the order of the context."""
    return [(cid, texts[cid]) for cid in chunk_ids]
