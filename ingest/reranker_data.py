"""Synthetic training data for the requirement reranker (config/retrieval/reranker-finetuning.md).

E showed what to train: for each requirement, the passage that answers it should come first among
E5's 50. No DEV need, query, label, requirement or result is used here. The source is the corpus:
a local model writes 2 questions from one chunk, in the form of D's requirements; that chunk is a
provisional positive, not a perfect label, so every question goes through fixed filters:

- form: readable, a question, no article, paragraph, point or chunk reference;
- copying: no run of 8 or more words copied from the chunk;
- inherited text: a question whose words lean more on the chunk's inherited context (the article
  heading and lead-in its siblings share) than on its own body is ambiguous, and dropped;
- round trip: E5 must rank the source within its first 50 for the question.

Chunks whose bodies share at least half of their 5-word shingles with the source (twins, repeated
wording) are equivalent to it: never negatives, and counted as correct when measuring. Negatives
are E5's own candidates for the question, the hardest first. The split goes by article: every
question of an article's chunks falls in one split, by a hash of the article, never by pair.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ingest import decomposition as dc

MIN_BODY_WORDS = 20
QUESTIONS = 2
COPY_RUN = 8
SHINGLE = 5
EQUIVALENT_JACCARD = 0.5
DEPTH = 50
TRAIN_NEGATIVES = 3  # the two hardest, and one drawn from the rest of the 50
SELECT_CANDIDATES = 10  # validation for checkpoints: the positive and the 9 hardest
VALIDATION_PERCENT = 15
SELECT_SIZE = 100
REPORT_SIZE = 150
SEED = 20261015

SYSTEM = "You write the questions that a search of EU financial regulation must answer."
TASK = (
    f"Write {QUESTIONS} questions that this passage answers. Each asks about one point of law, "
    "in English, in one self-contained sentence that names who it concerns and what, and can be "
    "answered mainly from this passage. Do not answer, do not copy a long phrase from the "
    "passage, write no article, paragraph or point numbers, and add no fact that is not in the "
    "passage."
)
_LEAK = re.compile(
    r"\bArt(?:icles?|\.)\s*\d|\bparagraphs?\s+\d|\bpoints?\s+\(|\(\d+[a-z]?\)|\b[A-Z]{3,5}:\d",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+")
# frequent words that say nothing about which text a question leans on
_STOP = frozenset(
    "the and for that this with from shall which what when where must under their there these "
    "those have been does such into other than within about whose whom will would should could "
    "they them its any all each more most only also upon".split()
)


def eligible(body: str) -> bool:
    return len(body.split()) >= MIN_BODY_WORDS


def generation_prompt(text: str) -> list[dict[str, str]]:
    user = (
        f"PASSAGE:\n{text}\n\n{TASK}\n\n"
        "Answer in this form and nothing else:\nREQUIREMENTS:\n- <question>\n- <question>"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def questions(output: str) -> tuple[str, ...]:
    """At most QUESTIONS questions, read as D reads requirements."""
    return dc.parse(output).items[:QUESTIONS]


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def longest_copied_run(question: str, text: str) -> int:
    """The longest run of consecutive words of the question that appears in the text."""
    q, t = words(question), " " + " ".join(words(text)) + " "
    best = 0
    for i in range(len(q)):
        for j in range(i + best + 1, len(q) + 1):
            if " " + " ".join(q[i:j]) + " " in t:
                best = j - i
            else:
                break
    return best


def leans_on_inherited_text(question: str, context: str, body: str) -> bool:
    content = {w for w in words(question) if len(w) >= 4 and w not in _STOP}
    own, inherited = set(words(body)), set(words(context)) - set(words(body))
    return len(content & inherited) > len(content & own)


def rejection(question: str, context: str, body: str) -> str | None:
    """Why a question is dropped before retrieval, or None. The round trip comes after."""
    if not question.endswith("?"):
        return "not a question"
    if _LEAK.search(question):
        return "an article, paragraph, point or chunk reference"
    if longest_copied_run(question, context + " " + body) >= COPY_RUN:
        return f"a run of {COPY_RUN} or more words copied"
    if leans_on_inherited_text(question, context, body):
        return "leans on the inherited context"
    return None


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    w = words(text)
    return frozenset(tuple(w[i : i + SHINGLE]) for i in range(max(1, len(w) - SHINGLE + 1)))


def equivalents(bodies: Mapping[str, str]) -> dict[str, frozenset[str]]:
    """For each chunk, the other chunks whose bodies share at least half of their shingles."""
    sh = {c: shingles(b) for c, b in bodies.items()}
    index: dict[tuple[str, ...], set[str]] = {}
    for c, s in sh.items():
        for g in s:
            index.setdefault(g, set()).add(c)
    out: dict[str, frozenset[str]] = {}
    for c, s in sh.items():
        near = {o for g in s for o in index[g] if o != c}
        out[c] = frozenset(
            o for o in near if s and sh[o] and len(s & sh[o]) / len(s | sh[o]) >= EQUIVALENT_JACCARD
        )
    return out


def article_key(chunk_id: str, article_id: str) -> str:
    return f"{chunk_id.split(':')[0]}:{article_id}"


def in_validation(key: str) -> bool:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100 < VALIDATION_PERCENT


@dataclass(frozen=True)
class Item:
    item_id: str  # chunk id and question number
    question: str
    positive: str
    article: str
    candidates: tuple[str, ...]  # E5's first 50 for the question
    correct: frozenset[str]  # the positive and its equivalents


def train_negatives(item: Item, excluded: frozenset[str], rng: random.Random) -> list[str]:
    """The two hardest of E5's candidates that are neither correct nor excluded, then one drawn
    from the rest of the 50."""
    pool = [c for c in item.candidates if c not in item.correct and c not in excluded]
    hardest = pool[: TRAIN_NEGATIVES - 1]
    rest = pool[TRAIN_NEGATIVES - 1 :]
    return hardest + ([rng.choice(rest)] if rest else [])


def select_candidates(item: Item) -> list[str]:
    """The positive and the 9 hardest wrong candidates, for choosing the checkpoint."""
    wrong = [c for c in item.candidates if c not in item.correct]
    return [item.positive, *wrong[: SELECT_CANDIDATES - 1]]


def ranking_metrics(
    ranked: Mapping[str, Sequence[str]], correct: Mapping[str, frozenset[str]]
) -> dict[str, float]:
    """P@1, MRR and Recall@3, @5 and @10 of the first correct chunk, averaged over items."""
    ranks = []
    for item_id, order in ranked.items():
        hit = next((i for i, c in enumerate(order, 1) if c in correct[item_id]), None)
        ranks.append(hit)
    n = len(ranks)
    out = {"P@1": sum(r == 1 for r in ranks) / n, "MRR": sum(1 / r for r in ranks if r) / n}
    for k in (3, 5, 10):
        out[f"Recall@{k}"] = sum(r is not None and r <= k for r in ranks) / n
    return out


def fingerprint(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
