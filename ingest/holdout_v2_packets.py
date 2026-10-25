"""TEST v2's packets and admissions (D-067): what each role receives, and what its answer must be
before it is admitted.

The packets reuse the frozen templates of the earlier construction wherever they fit: the
statement writer (statement_writer.v1), the query writer (query_variants.v1), the gatekeeper
(overlap_gate.v2), judge C (judge_c.v2) and the metadata adjudicator (metadata_adjudicator.v1).
Four templates are TEST v2's own:
- concept_author.v2: the concept author sees its anchor provisions, never their identifiers in
  the answer it gives;
- judge_b_pool.v3: judge_b_pool.v2 with two framing sentences made true for TEST v2 (the nodes
  come from a pool, in an order drawn per need, not in document order); the rubric, the task and
  the output are the same words;
- answerability.v1: the answerability of a need with frozen qrels, under the protocol's section 3;
- completeness_screen.v1: the chunk screen of the completeness diagnostic, high recall.

No packet says how a node or a chunk reached the pool, which system ranked it or where, or what
any other judge thought; the orchestrator's opaque ids name every item.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ingest import benchmark as bm
from ingest import holdout_admission as ha
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from ingest import holdout_v2 as v2

# A run of this many consecutive words shared with an anchor's article is near-copying, not
# terminology (hp.COPY_WORDS, the frozen anti-copy guard against DEV); bm.COPY_RUN (6) stays a
# diagnostic flag. Terms of art such as "critical ICT third-party service provider" are shorter.
ANCHOR_COPY_WORDS = hp.COPY_WORDS
VARIANTS = hpl.VARIANTS


@dataclass(frozen=True)
class WriterCell:
    """What the statement writer's frozen packet reads of a cell: its acts, no area (the writer
    never learns where the anchor is), its type and its answerability."""

    acts: list[str]
    area: None
    type: str
    answerability: str


@dataclass(frozen=True)
class NeedText:
    """What a judge's packet reads of a need: an opaque reference and the statement."""

    need_id: str
    statement: str


# ---------------------------------------------------------------------------
# the authors
# ---------------------------------------------------------------------------


def anchor_block(corpus: hp.Corpus, anchor: v2.Anchor) -> str:
    """One anchor as the concept author reads it: its act, its article's title (never its
    number), the own text of every ancestor inside the article, and the provision itself."""
    act, path = anchor.node_key
    parts = path.split(".")
    title = corpus.titles[(act, parts[0])]
    lines = [f"From {act}, the article titled: {title}"]
    for i in range(1, len(parts)):
        ancestor = (act, ".".join(parts[:i]))
        text = corpus.texts.get(ancestor, "").strip()
        if text:
            lines.append(f"Above it in the article: {text}")
    lines.append(f"ANCHOR: {corpus.texts[(act, path)].strip()}")
    return "\n".join(lines)


def _acts(slot: v2.Slot) -> str:
    if len(slot.acts) == 1:
        return slot.acts[0]
    return " and ".join(slot.acts) + ", both needed: no single act answers the question"


def concept_packet(corpus: hp.Corpus, candidate: v2.Candidate, ref: str) -> str:
    slot = candidate.slot
    return hp.template("concept_author.v2").substitute(
        opening=hp.HEADLESS_OPENING,
        cell_id=ref,
        acts=_acts(slot),
        type_name=slot.type,
        type_definition=hp.TYPE_DEFINITIONS[slot.type],
        answerability=slot.answerability,
        answerability_definition=hp.ANSWERABILITY_DEFINITIONS[slot.answerability],
        difficulty=slot.difficulty,
        difficulty_definition=hp.DIFFICULTY_DEFINITIONS[slot.difficulty],
        expected_answer=hp.EXPECTED_ANSWER[slot.number_or_date],
        anchors="\n\n".join(anchor_block(corpus, a) for a in candidate.anchors),
    )


def writer_packet(
    candidate: v2.Candidate, ref: str, concept: str, facts: Sequence[str], forbidden: Sequence[str]
) -> str:
    slot = candidate.slot
    cell = WriterCell(list(slot.acts), None, slot.type, slot.answerability)
    return hp.writer_packet(cell, ref, concept, facts, forbidden)  # type: ignore[arg-type]


def query_packet(statement: str) -> str:
    return bm.render_query_prompt(statement, VARIANTS)


def anchor_texts(corpus: hp.Corpus, candidate: v2.Candidate) -> list[str]:
    """The whole own text of every anchor's article: what copying is measured against."""
    out = []
    for anchor in candidate.anchors:
        act = anchor.act
        nodes = [k for k in corpus.order if k[0] == act and k[1].split(".")[0] == anchor.article]
        out.append(" ".join(corpus.texts[k] for k in nodes if corpus.texts.get(k)))
    return out


def longest_run(text: str, sources: Sequence[str]) -> int:
    """The longest run of consecutive words of text found verbatim in one source."""
    return bm.lexical_overlap(text, list(sources)).longest_run


def concept_problems(value: Any, anchors: Sequence[str], forbidden: Sequence[str]) -> list[str]:
    """The concept author's answer (concept_author.v2): the shape of concept_author.v1, and a
    concept the statement writer can receive without any anchor: no legal reference (the query
    renderer's frozen pattern), no legal wording or run of eight words of DEV or the fixtures
    (the frozen author guard), and no run of ANCHOR_COPY_WORDS words of an anchor's article."""
    problems = ha.concept_problems(value)
    if problems:
        return problems
    words = [str(value["concept"]), *(str(f) for f in value["facts"])]
    for i, text in enumerate(words):
        where = "concept" if i == 0 else f"fact {i}"
        if bm._LEGAL_REFERENCE.search(text):
            problems.append(f"{where}: a legal reference")
        if hp.author_leaks(text, forbidden):
            problems.append(f"{where}: legal wording or a run of DEV")
        if anchors and longest_run(text, anchors) >= ANCHOR_COPY_WORDS:
            problems.append(f"{where}: a run of {ANCHOR_COPY_WORDS} words of its anchor")
    return problems


def concept_fits(value: Mapping[str, Any]) -> bool:
    return str(value.get("fit", "")).strip().lower() == "fits"


def writer_problems(value: Any, forbidden: Sequence[str]) -> list[str]:
    """The statement writer's answer: a primary statement that cites no provision and holds no
    legal wording or run of DEV (the frozen statement checks)."""
    problems = ha.statement_problems(value)
    if problems:
        return problems
    found = hpl.statement_problems(str(value["primary"]).strip(), forbidden)
    return [name for name, hits in sorted(found.items()) if hits]


@dataclass(frozen=True)
class Leakage:
    longest_run: int  # the longest run of words of the statement or a query in its anchors
    share: float  # the mean share of content words of the statement and queries in its anchors
    rejected: bool


def leakage(statement: str, queries: Sequence[str], anchors: Sequence[str]) -> Leakage:
    """The anchor leakage diagnostic of one generated need, and its filter: a run of
    ANCHOR_COPY_WORDS consecutive words of an anchor's article in the statement or in a query is
    near-copying, and the candidate is not used. Shorter shared runs (terms of art) and shared
    vocabulary are measured and reported, never a filter."""
    texts = [statement, *queries]
    overlaps = [bm.lexical_overlap(t, list(anchors)) for t in texts]
    longest = max(o.longest_run for o in overlaps)
    share = sum(o.share for o in overlaps) / len(overlaps)
    return Leakage(longest, round(share, 4), longest >= ANCHOR_COPY_WORDS)


def gate_packet(ref: str, statement: str, existing: Sequence[hp.Existing]) -> str:
    return hp.gate_packet(ref, statement, existing)


# ---------------------------------------------------------------------------
# the judges of the pool
# ---------------------------------------------------------------------------


def pool_packet(
    corpus: hp.Corpus, need: NeedText, nodes: Sequence[str], ids: Mapping[str, str]
) -> str:
    """judge_b_pool.v3: the nodes in the order given (chunk groups in an order drawn per need),
    each under its opaque id, with the own text of its ancestors that are not in the packet."""
    judged = set(nodes)
    blocks = [f"CANDIDATE {ids[r]}\n" + corpus.node_block(r, judged, []) for r in nodes]
    return hp.template("judge_b_pool.v3").substitute(
        opening=hp.JUDGE_B_OPENING,
        rules=hp.judge_bc_rules(),
        need_id=need.need_id,
        statement=need.statement,
        candidates="\n\n".join(blocks),
    )


def pool_problems(answer: Any, by_id: Mapping[str, str]) -> list[str]:
    return ha.pool_problems(answer, by_id)


def c_packet(
    corpus: hp.Corpus,
    need: NeedText,
    nodes: Sequence[str],
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    ids: Mapping[str, str],
    seed: int = hp.JUDGE_C_SEED,
) -> tuple[str, list[hp.CItem]]:
    """judge_c.v2 on the nodes where the two blind judgments differ, no answerability item:
    each node under its opaque id, in document order, with the two positions in an order drawn
    for it from the need's opaque reference. TEST v2's own builder: the earlier one
    (holdout_packets.judge_c_packet) wrote its last disputed node where the need's reference
    goes, because its loop reuses the name of its `ref` parameter; it is kept, unchanged, to
    reproduce runs 1 to 7."""
    rng = random.Random(f"{seed}:{need.need_id}")
    judged = set(nodes)
    blocks, items = [], []
    for node in corpus.sort(nodes):
        a_first = rng.random() < 0.5
        pa, pb = (a[node], b[node]) if a_first else (b[node], a[node])
        items.append(hp.CItem(ids[node], node, "A" if a_first else "B"))
        blocks.append(
            f"ITEM {ids[node]}\n"
            + corpus.node_block(node, judged, [])
            + f"\nRationale A: {hp.show_judgment(pa)}. {pa['rationale']}"
            + f"\nRationale B: {hp.show_judgment(pb)}. {pb['rationale']}"
        )
    text = hp.template("judge_c.v2").substitute(
        rules=hp.judge_bc_rules(),
        need_id=need.need_id,
        statement=need.statement,
        items="\n\n".join(blocks),
    )
    return text, items


# ---------------------------------------------------------------------------
# answerability and metadata, after the qrels
# ---------------------------------------------------------------------------


def _evidence(corpus: hp.Corpus, finals: Mapping[str, Mapping[str, Any]]) -> list[str]:
    shown = [n for n, f in finals.items() if f.get("grade") is not None and f["grade"] >= 1]
    lines = []
    for ref in corpus.sort(shown):
        f = finals[ref]
        label = f"grade {f['grade']}" + (", ambiguous" if f.get("ambiguous") else "")
        lines.append(f"[{ref}] {label} | {corpus.heading(ref)}\n  {corpus.text(ref)}")
    return lines


def answerability_packet(
    corpus: hp.Corpus, need: NeedText, finals: Mapping[str, Mapping[str, Any]]
) -> str:
    lines = _evidence(corpus, finals)
    return hp.template("answerability.v1").substitute(
        opening=hp.HEADLESS_OPENING,
        rules=hp.judge_bc_rules(),
        need_id=need.need_id,
        statement=need.statement,
        evidence="\n".join(lines) or "none",
    )


def answerability_c_packet(
    corpus: hp.Corpus,
    need: NeedText,
    finals: Mapping[str, Mapping[str, Any]],
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    item_id: str,
    first_is_a: bool,
) -> tuple[str, list[hp.CItem]]:
    """judge_c.v2 on one item, the answerability, when the two blind answerability judgments
    differ: the final judged evidence (grade >= 1) with its text, and the two positions in the
    drawn order. The grades are final: the item asks for the answerability only."""
    lines = _evidence(corpus, finals)
    pa, pb = (first, second) if first_is_a else (second, first)
    block = (
        f"ITEM {item_id}: the answerability of the need (section 3). The grades below are final "
        "and are not to be decided again; decide only the answerability, from the words of the "
        "text shown.\n"
        + ("\n".join(lines) or "no judged evidence")
        + f"\nRationale A: answerability {pa['answerability']}. {pa['reason']}"
        + f"\nRationale B: answerability {pb['answerability']}. {pb['reason']}"
    )
    text = hp.template("judge_c.v2").substitute(
        rules=hp.judge_bc_rules(), need_id=need.need_id, statement=need.statement, items=block
    )
    return text, [hp.CItem(item_id, bm.ANSWERABILITY_TARGET, "A" if first_is_a else "B")]


def answerability_problems(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["the answer is not an object"]
    if value.get("answerability") not in ha.ANSWERABILITIES:
        return ["answerability: not complete, partial or none"]
    problems = []
    missing = value.get("missing")
    if value["answerability"] != "complete" and not (isinstance(missing, str) and missing.strip()):
        problems.append("missing: no text for a need that is not complete")
    if not (isinstance(value.get("reason"), str) and value["reason"].strip()):
        problems.append("reason: no text")
    return problems


@dataclass(frozen=True)
class QrelView:
    """What the metadata adjudicator's packet and its admission read of one final label."""

    act: str
    path: str
    grade: int
    ambiguous: bool


def qrel_views(finals: Mapping[str, Mapping[str, Any]]) -> list[QrelView]:
    """The final labels the metadata adjudicator reads: every resolved, non-ambiguous one with a
    grade of 1 or more. A pool holds hundreds of explicit 0s (the earlier construction judged a
    few dozen nodes), and none of them can support a driver; an ambiguous label is unjudged for
    the evaluation (D-019) and the packet would show it as a definite grade. Both are left out of
    the packet and of its check."""
    out = []
    for ref, f in finals.items():
        if f.get("grade") is None or f["grade"] < 1 or f.get("ambiguous"):
            continue
        act, _, path = ref.partition(":")
        out.append(QrelView(act, path, int(f["grade"]), bool(f.get("ambiguous"))))
    return out


def metadata_packet(
    corpus: hp.Corpus,
    need: NeedText,
    finals: Mapping[str, Mapping[str, Any]],
    answerability: str,
    gap_note: str | None,
) -> str:
    views: Any = qrel_views(finals)  # the attributes hp.metadata_packet reads of a bm.Qrel
    shown: Any = need  # and of a bm.Need: need_id and statement
    return hp.metadata_packet(corpus, shown, views, answerability, gap_note, ref=need.need_id)


def metadata_problems(
    value: Any, finals: Mapping[str, Mapping[str, Any]], answerability: str
) -> list[str]:
    verdict = ha.metadata_verdict(value, qrel_views(finals), answerability)  # type: ignore[arg-type]
    return [*verdict.problems, *verdict.violations]


# ---------------------------------------------------------------------------
# the completeness screen
# ---------------------------------------------------------------------------


def passage(corpus: hp.Corpus, text: str, first_node: bm.NodeKey) -> str:
    act, path = first_node
    return f"{corpus.heading(f'{act}:{path}')}\n{text.strip()}"


def screen_packet(need: NeedText, passages: Sequence[tuple[str, str]]) -> str:
    """completeness_screen.v1: (opaque id, passage text) in the order given."""
    blocks = [f"PASSAGE {pid}\n{text}" for pid, text in passages]
    return hp.template("completeness_screen.v1").substitute(
        opening=hp.HEADLESS_OPENING,
        rules=hp.judge_bc_rules(),
        need_id=need.need_id,
        statement=need.statement,
        passages="\n\n".join(blocks),
    )


_PASSAGE_ID = re.compile(r"^P-[0-9a-f]{12}$")


def screen_problems(value: Any, shown: Sequence[str]) -> list[str]:
    """Every passage shown decided once by its exact id, with an explicit true or false flag: a
    passage left out is a failure of the role's contract, never a `false`."""
    if not isinstance(value, Mapping) or not isinstance(value.get("passages"), list):
        return ["passages: not a list"]
    problems = []
    named = []
    for i, entry in enumerate(value["passages"], start=1):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("item_id"), str):
            problems.append(f"entry {i}: names no passage id")
            continue
        named.append(entry["item_id"])
        if not _PASSAGE_ID.match(entry["item_id"]):
            problems.append(f"entry {i}: not a passage id")
        if not isinstance(entry.get("flag"), bool):
            problems.append(f"entry {i}: flag not given as true or false")
    problems += ha._set_problems(named, list(shown), "passages")
    return problems
