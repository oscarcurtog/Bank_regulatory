"""The deterministic verifier of S7-S8 (layer 0) and the three abstention rules.

What it proves is **provenance and citation integrity**: the output is well formed, every claim has
a citation, every cited chunk exists and was in the context the model was given, every quote is
literal text of the chunk it cites, and no number in a claim is absent from the chunks it cites.
What it does **not** prove is legal entailment: a claim whose quote exists and contains its
numbers can still be unrelated to, or misread from, that quote. Nothing here tries to read
meaning. That gap is what the human anchor measures; the lexical overlap below is a diagnostic
that decides nothing.

The abstention rules are functions of one parsed output and its verdict. Two decide: A (SELF,
the model says its sources are sufficient) and C (COMBINED, the production candidate). B (layer 0
alone, ignoring the model's statement) is computed only as a reference and diagnostic: with the
schema rule that an insufficient answer has no claims, B and C can differ only when the model
breaks that instruction, and the generator is not altered to make them differ.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ingest.answer_format import Answer, Claim, ParsedOutput

MIN_QUOTE_WORDS = 5
MAX_QUOTE_WORDS = 80

# Failure codes of a citation and of a claim.
UNKNOWN_CHUNK = "unknown_chunk"
CHUNK_NOT_IN_CONTEXT = "chunk_not_in_context"
QUOTE_TOO_SHORT = "quote_too_short"
QUOTE_TOO_LONG = "quote_too_long"
QUOTE_NOT_LITERAL = "quote_not_literal"
NO_CITATION = "no_citation"
FIGURE_NOT_IN_CITED_CHUNKS = "figure_not_in_cited_chunks"

# A number: digit groups separated by a space (EUR 20 000), a decimal part with a dot or a comma,
# or plain digits.
_NUMBER = re.compile(r"\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it its of on or shall that the their "
    "them these this to under was were which with within without may must not no any all such "
    "other than also where who whose when if so can should would will been being those".split()
)


def normalize(text: str) -> str:
    """Unicode NFC and runs of any whitespace (including no-break spaces) collapsed to one space.
    Case, punctuation and every other character are kept: a quote is literal."""
    return " ".join(unicodedata.normalize("NFC", text).split())


_TYPOGRAPHIC = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
)


def fold(text: str) -> str:
    """A looser form of `normalize` for a diagnostic only: typographic apostrophes, quotation marks
    and dashes turned into plain ones, and case ignored."""
    return normalize(text).translate(_TYPOGRAPHIC).casefold()


def figures(text: str) -> set[str]:
    """The numbers in a text, in a canonical form: separators between digit groups removed and a
    decimal comma turned into a dot. Numbers spelled in words are not seen: not checkable here."""
    found = set()
    for token in _NUMBER.findall(text):
        found.add(re.sub(r"[   ]", "", token).replace(",", "."))
    return found


@dataclass(frozen=True)
class CitationVerdict:
    chunk_id: str
    failures: tuple[str, ...]
    near_literal: bool = (
        False  # diagnostic: not literal, but literal once case and glyphs are folded
    )

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class ClaimVerdict:
    citations: tuple[CitationVerdict, ...]
    failures: tuple[str, ...]  # claim-level failures; citation failures are in `citations`
    lexical_overlap: float  # diagnostic only: never used by a rule

    @property
    def passed(self) -> bool:
        return not self.failures and all(c.passed for c in self.citations)

    def all_failures(self) -> tuple[str, ...]:
        return (*self.failures, *(f for c in self.citations for f in c.failures))


@dataclass(frozen=True)
class AnswerVerdict:
    claims: tuple[ClaimVerdict, ...]

    @property
    def all_pass(self) -> bool:
        return bool(self.claims) and all(c.passed for c in self.claims)


def _verify_citation(
    chunk_id: str,
    quote: str,
    context_ids: frozenset[str],
    texts: Mapping[str, str],
) -> CitationVerdict:
    failures: list[str] = []
    if chunk_id not in texts:
        failures.append(UNKNOWN_CHUNK)
    if chunk_id not in context_ids:
        failures.append(CHUNK_NOT_IN_CONTEXT)
    words = len(normalize(quote).split())
    if words < MIN_QUOTE_WORDS:
        failures.append(QUOTE_TOO_SHORT)
    elif words > MAX_QUOTE_WORDS:
        failures.append(QUOTE_TOO_LONG)
    near = False
    if (
        chunk_id in texts
        and words >= MIN_QUOTE_WORDS
        and normalize(quote) not in normalize(texts[chunk_id])
    ):
        failures.append(QUOTE_NOT_LITERAL)
        near = fold(quote) in fold(texts[chunk_id])
    return CitationVerdict(chunk_id, tuple(failures), near)


def lexical_overlap(claim_text: str, quotes: Sequence[str]) -> float:
    """Diagnostic (layer 0.5): the share of the claim's content words that occur in its quotes.
    It is not evidence of support and no rule uses it."""
    words = {w.lower() for w in _WORD.findall(claim_text)} - _STOPWORDS
    words = {w for w in words if not w.isdigit()}
    if not words:
        return 1.0
    seen = {w.lower() for q in quotes for w in _WORD.findall(q)}
    return len(words & seen) / len(words)


def verify_claim(
    claim: Claim, context_ids: frozenset[str], texts: Mapping[str, str]
) -> ClaimVerdict:
    """Layer 0 for one claim. `context_ids` are the chunks the model was shown; `texts` holds the
    text of every chunk of the layer, so that a cited chunk outside the context is still read."""
    citations = tuple(
        _verify_citation(c.chunk_id, c.quote, context_ids, texts) for c in claim.citations
    )
    failures: list[str] = []
    if not claim.citations:
        failures.append(NO_CITATION)
    cited_text = " ".join(texts[c.chunk_id] for c in claim.citations if c.chunk_id in texts)
    if claim.citations and not figures(claim.text) <= figures(cited_text):
        failures.append(FIGURE_NOT_IN_CITED_CHUNKS)
    return ClaimVerdict(
        citations, tuple(failures), lexical_overlap(claim.text, [c.quote for c in claim.citations])
    )


def verify_answer(
    answer: Answer, context_ids: Sequence[str], texts: Mapping[str, str]
) -> AnswerVerdict:
    shown = frozenset(context_ids)
    return AnswerVerdict(tuple(verify_claim(c, shown, texts) for c in answer.claims))


# ---------------------------------------------------------------------------
# the three abstention rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decisions:
    """True means the rule lets the answer through; False means ABSTAIN."""

    a_self_only: bool
    b_verifier_only: bool
    c_combined: bool


def decide(parsed: ParsedOutput, verdict: AnswerVerdict | None) -> Decisions:
    """A answers if the model says its sources are sufficient. B ignores that statement and
    answers if there is at least one claim and every claim passes layer 0. C answers only if the
    model says sufficient, there is at least one claim and every claim passes: if any claim fails,
    ABSTAIN. A structurally unusable output makes all three abstain."""
    if parsed.answer is None:
        return Decisions(False, False, False)
    all_pass = verdict is not None and verdict.all_pass
    sufficient = parsed.answer.sufficiency
    return Decisions(sufficient, all_pass, sufficient and all_pass)
