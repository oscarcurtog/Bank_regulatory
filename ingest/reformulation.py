"""Round 4 of the retrieval work: closed-catalog legal reformulation for first-hop failures
(config/retrieval/necessity-selection.md).

A question can describe facts without naming the legal concept that governs them ("she learned
at work before the announcement and bought the token" is insider dealing, a word the question
never uses). A language model rewrites the question, and E5's ranking of the rewrite is fused
with its ranking of the original question by Reciprocal Rank Fusion (k = 60, equal weights):

- R4-A: the legally material facts only, in English, no legal concepts;
- R4-B: the same, plus at most 3 concepts copied from a closed catalog built once from the
  whole corpus (every article title and every defined term), each with the words of the
  question that motivate it. A concept that is not literally in the catalog is dropped.

This is not HyDE: no hypothetical legal text is written, and every concept is an existing
heading or defined term of the corpus. Nothing here reads a label.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ingest import dependencies as dp

MAX_CONCEPTS = 3
RRF_K = 60


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def catalog(corpus: Mapping[str, Any], definitions: Sequence[dp.Definition]) -> list[str]:
    """Every article title (act and article order), then every defined term (glossary order),
    each once, compared without case."""
    out: list[str] = []
    seen: set[str] = set()
    entries = [a.title for act in sorted(corpus) for a in corpus[act].articles]
    entries += [d.term for d in definitions]
    for entry in entries:
        key = normalize(entry).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(normalize(entry))
    return out


FACTS_TASK = (
    "List the legally material facts of the question in English, as short declarative "
    "sentences, one per line. Keep who acts, in what capacity, what they did or plan to do, and "
    "the assets, services, events, amounts and dates involved. Do not answer the question, do "
    "not name laws, articles or legal concepts, and do not add facts that are not in the "
    "question."
)
CONCEPTS_TASK = (
    f"Then choose at most {MAX_CONCEPTS} entries of the CATALOG that the question turns on. Copy "
    "each entry exactly as it is written in the catalog, and give the words of the question that "
    "point to it. Do not answer the question."
)


def facts_prompt(question: str) -> list[dict[str, str]]:
    system = "You extract the legally material facts of a question about EU financial regulation."
    user = (
        f"QUESTION: {question}\n\n{FACTS_TASK}\n\n"
        "Answer in this form and nothing else:\nFACTS:\n- <fact>\n- <fact>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def concepts_prompt(question: str, entries: Sequence[str]) -> list[dict[str, str]]:
    """The catalog opens the prompt (the system turn), so every question shares it."""
    system = (
        "You extract the legally material facts of a question about EU financial regulation, and "
        "choose the legal concepts it turns on from a closed catalog.\n\n"
        "CATALOG (article titles and defined terms of DORA, MiCA and PSD2):\n" + "\n".join(entries)
    )
    user = (
        f"QUESTION: {question}\n\n{FACTS_TASK}\n\n{CONCEPTS_TASK}\n\n"
        "Answer in this form and nothing else:\nFACTS:\n- <fact>\n- <fact>\nCONCEPTS:\n"
        "- <catalog entry> | <words of the question>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class Concept:
    entry: str  # the catalog's own spelling
    fragment: str
    fragment_in_question: bool


@dataclass(frozen=True)
class Reformulation:
    facts: tuple[str, ...]
    concepts: tuple[Concept, ...]
    rejected: tuple[str, ...]  # concepts not in the catalog, or beyond the limit


def _items(block: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"^\s*[-*]\s*(.+?)\s*$", block, re.MULTILINE)]


def parse(output: str, question: str, entries: Sequence[str] | None = None) -> Reformulation:
    """The facts, and (with a catalog) the concepts found literally in it, at most MAX_CONCEPTS,
    with whether their fragment is in the question; everything else is recorded as rejected."""
    m = re.search(r"FACTS:\s*(.*?)(?:\n\s*CONCEPTS:|\Z)", output, re.DOTALL | re.IGNORECASE)
    facts = tuple(f for f in _items(m.group(1)) if f) if m else ()
    if entries is None:
        return Reformulation(facts, (), ())
    by_key = {normalize(e).lower(): e for e in entries}
    c = re.search(r"CONCEPTS:\s*(.*)\Z", output, re.DOTALL | re.IGNORECASE)
    concepts: list[Concept] = []
    rejected: list[str] = []
    q = normalize(question).lower()
    for item in _items(c.group(1)) if c else []:
        entry, _, fragment = item.partition("|")
        key = normalize(entry).strip(" '\"`.").lower()
        canonical = by_key.get(key)
        if (
            canonical is None
            or len(concepts) >= MAX_CONCEPTS
            or any(x.entry == canonical for x in concepts)
        ):
            rejected.append(normalize(entry))
            continue
        frag = normalize(fragment).strip(" '\"`")
        concepts.append(Concept(canonical, frag, bool(frag) and frag.lower() in q))
    return Reformulation(facts, tuple(concepts), tuple(rejected))


def retrieval_text(r: Reformulation) -> str:
    """The text E5 ranks for a rewrite: the facts, then the concepts; empty if neither."""
    text = " ".join(r.facts)
    if r.concepts:
        text = f"{text} Legal concepts: {'; '.join(c.entry for c in r.concepts)}.".strip()
    return text
