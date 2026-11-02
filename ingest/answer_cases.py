"""The cases of S7-S8: the split of DEV, the four contexts and the 15 negatives, all deterministic.

Nothing here generates an answer or reads a file. From the needs, the grades of the judged chunks
and the stored rankings of E5 and E, it builds the exact list of (question, context) cases that a
generator will later be run on. Every choice that is not forced is made by ordering the candidates
with a SHA-256 of a fixed seed string and the candidate id, so it does not depend on the version of
Python or on the order of the input.

Contexts
- GOLD_KNOWN: the known relevant chunks of DEV (grade 2 or 3), highest grade first, then chunk id.
  It is the ceiling of the generator with the evidence we know, not the complete legal evidence.
- E5@10 and E5@5: the first ten and the first five chunks of E5's stored ranking.
- E@10: the first ten chunks of E's stored ranking.

Negatives (15), cases where the right behaviour is to abstain
- natural: the DEV needs whose answerability is `none` (3), with the E5@10 and E@10 contexts;
- insufficient context (6): constructed known-insufficient contexts under the frozen DEV
  relevance definition. A valid question of the evaluation set with a context made only of chunks
  judged grade 0 or 1, none of grade 2 or 3 (grade 1 is "useful context that a correct and
  complete answer can leave out"). It is not legal ground truth: it only says that no chunk of
  known relevant evidence is in it. A chunk nobody judged is never used as if it were known to be
  a negative;
- mismatched context (6): a question with the E5@10 and E@10 contexts of another need, from a
  different act and sharing no chunk with the known relevant evidence of the question.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

SEED = "s7-s8-v1"
ACTS = ("DORA", "MiCA", "PSD2")
PROMPT_DEV_PER_ACT = 2
INSUFFICIENT_CASES = 6
MISMATCHED_CASES = 6
INSUFFICIENT_MIN_CHUNKS = 3
INSUFFICIENT_MAX_CHUNKS = 10

GOLD_KNOWN = "GOLD_KNOWN"
E5_10 = "E5@10"
E_10 = "E@10"
E5_5 = "E5@5"
INSUFFICIENT = "INSUFFICIENT_CONTEXT"
CONTEXTS = (GOLD_KNOWN, E5_10, E_10, E5_5)

POSITIVE = "positive"
NEG_NATURAL = "negative_natural"
NEG_INSUFFICIENT = "negative_insufficient_context"
NEG_MISMATCHED = "negative_mismatched_context"
PROMPT_DEV = "prompt_dev"


class CaseError(Exception):
    """The inputs cannot produce the cases the preregistration requires."""


@dataclass(frozen=True)
class NeedInfo:
    need_id: str
    answerability: str  # complete | partial | none
    acts: tuple[str, ...]


@dataclass(frozen=True)
class Split:
    prompt_dev: tuple[str, ...]  # 6 complete single-act needs, 2 per act
    evaluation: tuple[str, ...]  # the other complete needs
    partial: tuple[str, ...]  # outside the primary result
    none: tuple[str, ...]  # reserved for abstention


@dataclass(frozen=True)
class Case:
    case_id: str
    kind: str
    need_id: str  # the need of the question
    query_id: str
    question: str
    context_name: str
    chunk_ids: tuple[str, ...]  # in the order the model sees them
    source_need_id: str | None  # the need the context comes from, for mismatched cases


def _digest(seed: str, key: str, item: str) -> str:
    return hashlib.sha256(f"{seed}|{key}|{item}".encode()).hexdigest()


def hash_order(items: Sequence[str], key: str, seed: str = SEED) -> list[str]:
    """The items in an order fixed by SHA-256, independent of their input order."""
    return sorted(items, key=lambda item: (_digest(seed, key, item), item))


def make_split(needs: Sequence[NeedInfo], seed: str = SEED) -> Split:
    """Two complete single-act needs per act for prompt development, chosen by hash order; the
    remaining complete needs (cross-act ones included) are the evaluation set."""
    complete = [n for n in needs if n.answerability == "complete"]
    prompt_dev: list[str] = []
    for act in ACTS:
        candidates = [n.need_id for n in complete if n.acts == (act,)]
        if len(candidates) < PROMPT_DEV_PER_ACT:
            raise CaseError(f"{act}: {len(candidates)} complete single-act needs")
        prompt_dev += hash_order(candidates, f"prompt-dev|{act}", seed)[:PROMPT_DEV_PER_ACT]
    chosen = set(prompt_dev)
    return Split(
        prompt_dev=tuple(sorted(chosen)),
        evaluation=tuple(sorted(n.need_id for n in complete if n.need_id not in chosen)),
        partial=tuple(sorted(n.need_id for n in needs if n.answerability == "partial")),
        none=tuple(sorted(n.need_id for n in needs if n.answerability == "none")),
    )


# ---------------------------------------------------------------------------
# contexts
# ---------------------------------------------------------------------------


def known_relevant(grades: Mapping[str, int]) -> dict[str, int]:
    return {c: g for c, g in grades.items() if g >= 2}


def gold_known_context(grades: Mapping[str, int]) -> tuple[str, ...]:
    """The known relevant chunks, highest grade first, then chunk id."""
    return tuple(sorted(known_relevant(grades), key=lambda c: (-grades[c], c)))


def top_context(ranking: Sequence[str], k: int) -> tuple[str, ...]:
    return tuple(ranking[:k])


def insufficient_context(
    grades: Mapping[str, int],
    ranking: Sequence[str],
    maximum: int = INSUFFICIENT_MAX_CHUNKS,
    minimum: int = INSUFFICIENT_MIN_CHUNKS,
) -> tuple[str, ...] | None:
    """Only chunks judged grade 0 or 1 for the need, in the order of E5's ranking of the question,
    at most `maximum`. None if there are fewer than `minimum` of them."""
    known = {c for c, g in grades.items() if g in (0, 1)}
    chosen = tuple(c for c in ranking if c in known)[:maximum]
    return chosen if len(chosen) >= minimum else None


# ---------------------------------------------------------------------------
# negatives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NegativeSelection:
    insufficient: tuple[str, ...]  # need ids
    mismatched: tuple[tuple[str, str], ...]  # (question need, context need)


def select_negatives(
    split: Split,
    acts: Mapping[str, tuple[str, ...]],
    grades: Mapping[str, Mapping[str, int]],
    contexts_of: Callable[[str], set[str]],
    has_insufficient: Callable[[str], bool],
    seed: str = SEED,
) -> NegativeSelection:
    """Six needs for the insufficient-context negatives, then six (A, B) pairs for the mismatched
    ones, all from the evaluation set and with no need in both lists. `contexts_of(need)` is the
    set of chunks of the E5@10 and E@10 contexts of a need; `has_insufficient(need)` says whether
    a known-insufficient context can be built for it."""
    order = hash_order(split.evaluation, "negatives", seed)
    insufficient = [n for n in order if has_insufficient(n)][:INSUFFICIENT_CASES]
    if len(insufficient) < INSUFFICIENT_CASES:
        raise CaseError(f"only {len(insufficient)} needs allow an insufficient context")
    pairs: list[tuple[str, str]] = []
    for a in (n for n in order if n not in insufficient):
        known_a = set(known_relevant(grades[a]))
        partners = hash_order([b for b in split.evaluation if b != a], f"partner|{a}", seed)
        for b in partners:
            if set(acts[a]) & set(acts[b]) or contexts_of(b) & known_a:
                continue
            pairs.append((a, b))
            break
        if len(pairs) == MISMATCHED_CASES:
            break
    if len(pairs) < MISMATCHED_CASES:
        raise CaseError(f"only {len(pairs)} valid mismatched pairs")
    return NegativeSelection(tuple(insufficient), tuple(pairs))


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    query_id: str
    text: str


def _case(
    kind: str,
    need: str,
    question: Question,
    context: str,
    chunks: Sequence[str],
    source: str | None = None,
) -> Case:
    middle = f"{need}|from-{source}" if source else need
    return Case(
        f"{kind}|{middle}|{context}",
        kind,
        need,
        question.query_id,
        question.text,
        context,
        tuple(chunks),
        source,
    )


def positive_cases(
    need_ids: Sequence[str],
    questions: Mapping[str, Question],
    grades: Mapping[str, Mapping[str, int]],
    e5: Mapping[str, Sequence[str]],
    e: Mapping[str, Sequence[str]],
    kind: str = POSITIVE,
    contexts: Sequence[str] = CONTEXTS,
) -> list[Case]:
    """Each need with each requested context. `e5` and `e` map the query id to a full ranking."""
    out: list[Case] = []
    for need in sorted(need_ids):
        q = questions[need]
        built = {
            GOLD_KNOWN: gold_known_context(grades[need]),
            E5_10: top_context(e5[q.query_id], 10),
            E_10: top_context(e[q.query_id], 10),
            E5_5: top_context(e5[q.query_id], 5),
        }
        for context in contexts:
            out.append(_case(kind, need, q, context, built[context]))
    return out


def negative_cases(
    split: Split,
    selection: NegativeSelection,
    questions: Mapping[str, Question],
    grades: Mapping[str, Mapping[str, int]],
    e5: Mapping[str, Sequence[str]],
    e: Mapping[str, Sequence[str]],
) -> list[Case]:
    """The 15 negative cases as 24 generation cases: the natural ones and the mismatched ones with
    the E5@10 and E@10 contexts, the insufficient-context ones with their single context."""
    out: list[Case] = []
    for need in split.none:
        q = questions[need]
        out.append(_case(NEG_NATURAL, need, q, E5_10, top_context(e5[q.query_id], 10)))
        out.append(_case(NEG_NATURAL, need, q, E_10, top_context(e[q.query_id], 10)))
    for need in selection.insufficient:
        q = questions[need]
        chunks = insufficient_context(grades[need], e5[q.query_id])
        if chunks is None:
            raise CaseError(f"{need}: no insufficient context")
        out.append(_case(NEG_INSUFFICIENT, need, q, INSUFFICIENT, chunks))
    for a, b in selection.mismatched:
        qa, qb = questions[a], questions[b]
        out.append(_case(NEG_MISMATCHED, a, qa, E5_10, top_context(e5[qb.query_id], 10), b))
        out.append(_case(NEG_MISMATCHED, a, qa, E_10, top_context(e[qb.query_id], 10), b))
    return out
