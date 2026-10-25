"""TEST v2 (D-067): the pure core of the protocol, with no model and no state.

TEST v2 asks one question: on an independent private sample of information needs drawn from
the same task taxonomy as DEV, does the frozen system E improve over the frozen E5 at top-10
retrieval? Its qrels come from a pool of the compared systems' rankings, judged blind; the
incompleteness of that pool is measured, never fatal (config/holdout/test-v2-protocol.md).

Everything here is deterministic, from frozen inputs only:
- the population: DEV's answerable profiles (answerability complete or partial), each a slot;
  the `none` profiles never enter TEST v2;
- the candidate bank: UNIVERSE candidates, every slot BASE_COPIES or BASE_COPIES + 1 times,
  each with a priority drawn from the bank's private salt; the priority order is the order of
  the bank, and its unused tail is the single reserve;
- the anchors: a legal unit (an article) sampled per act, without replacement while the act has
  units left, and a provision inside it; never a raw chunk or node frequency;
- the completeness chains: per act, an order drawn before any judgment;
- the pool of a need and its decisive set, from frozen rankings;
- the judging order of a need's pooled chunks.

Nothing in this module reads a TEST need, a ranking's quality or a metric.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ingest import benchmark as bm
from ingest import evaluation as ev
from ingest import holdout as ho

SCHEMA = "holdout-v2.v1"
BANK_SCHEMA = "holdout-v2-bank.v1"
# the frozen design (D-067): the evaluable needs TEST v2 aims for, the candidates it may use
N_TARGET = 200
UNIVERSE = 260
# the seeds of the draws; each draw also takes the bank's private salt, so that nobody can
# compute the bank before the contract fixes the salt
BANK_SEED = 20261025
ANCHOR_SEED = 20261026
COMPLETENESS_SEED = 20261027
JUDGE_ORDER_SEED = 20261028
# a separate seed for every development bank: never the confirmatory one
DEVELOPMENT_SEED = 20261101
# the completeness diagnostic: per act, the first evaluable needs of its frozen chain
COMPLETENESS_STRATA = tuple(bm.ACTS)
COMPLETENESS_PER_STRATUM = 5
COMPLETENESS_SIZE = len(COMPLETENESS_STRATA) * COMPLETENESS_PER_STRATUM
# the pool's contributors and depths, frozen (D-067); E and E5 are the compared systems
POOL: tuple[tuple[str, int], ...] = (
    ("E", 20),
    ("E5", 20),
    ("BM25", 20),
    ("BGE-M3", 10),
    ("Qwen3-Embedding-0.6B", 20),
)
COMPARED = ("E", "E5")
DECISIVE_DEPTH = 10
FIRST_NEED_NUMBER = 401
ANCHOR_CONTRIBUTOR = "anchor"


class V2Error(Exception):
    """A rule of TEST v2's core is broken by its inputs: nothing is decided."""


def draw(seed: int, salt: bytes, purpose: str, *key: str) -> str:
    """The hex SHA-256 of one draw: reproducible from the seed, the salt and the key only."""
    text = ":".join((str(seed), salt.hex(), purpose, *key))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the population: DEV's answerable profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """One answerable DEV profile, in DEV's order. The DEV need itself never reaches TEST v2:
    only its profile (ho.Profile), which holds no topic."""

    slot_id: str
    acts: tuple[str, ...]
    type: str
    answerability: str
    difficulty: str
    number_or_date: bool


def slots(dev: bm.Benchmark) -> list[Slot]:
    """DEV's profiles with answerability complete or partial, one slot each (P01, P02, ...).
    A `none` profile is left out: it never enters a metric of S4, and a need anchored in the
    corpus cannot be one."""
    out: list[Slot] = []
    for need in sorted(dev.needs, key=lambda n: n.need_id):
        p = ho.profile_of(need)
        if p.answerability == "none":
            continue
        out.append(
            Slot(
                f"P{len(out) + 1:02d}",
                tuple(p.acts),
                p.type,
                p.answerability,
                p.difficulty,
                p.number_or_date,
            )
        )
    return out


# ---------------------------------------------------------------------------
# legal units: what an anchor is drawn from
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegalUnit:
    """An eligible article: in an area of the protocol's section 10 (so not a final or
    transitional provision), not a PSD2 article amended after the snapshot, with at least one
    provision of its own text. nodes are those provisions, in document order."""

    act: str
    article: str
    title: str
    area: str
    nodes: tuple[str, ...]


def legal_units(
    docs: Mapping[str, Any],
    areas: Sequence[ho.Area],
    excluded_psd2: frozenset[str],
    layer: ev.ChunkLayer,
) -> dict[str, list[LegalUnit]]:
    """Per act, its eligible articles in document order."""
    texts = bm.node_texts(docs)
    out: dict[str, list[LegalUnit]] = {}
    for act in bm.ACTS:
        units = []
        for article in docs[act].articles:
            area = ho.area_of(areas, act, article.path)
            if area is None or (act == "PSD2" and article.path in excluded_psd2):
                continue
            nodes = tuple(
                n.path
                for n in article.nodes
                if texts.get((act, n.path), "").strip() and (act, n.path) in layer.owner
            )
            if nodes:
                units.append(LegalUnit(act, article.path, article.title, area.name, nodes))
        out[act] = units
    return out


# ---------------------------------------------------------------------------
# the candidate bank
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Anchor:
    act: str
    article: str
    node: str  # the provision inside the article: "ACT:path" of its node
    area: str
    reuse: int  # how many earlier candidates (in priority order) drew this article

    @property
    def node_key(self) -> bm.NodeKey:
        return self.act, self.node.partition(":")[2]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str  # <slot>-<copy>: the slot's identity, never the priority
    slot: Slot
    priority: str  # the draw that orders the bank
    rank: int  # 1-based position in the priority order
    need_id: str  # the need id its queries take, fixed by its rank
    anchors: tuple[Anchor, ...] = field(default=())


@dataclass(frozen=True)
class Bank:
    seed: int
    candidates: tuple[Candidate, ...]  # in priority order
    reuse: dict[str, int]  # act -> anchors that drew an article already drawn
    capacity: dict[str, int]  # act -> eligible articles

    def record(self) -> dict[str, Any]:
        return {
            "schema": BANK_SCHEMA,
            "seed": self.seed,
            "universe": len(self.candidates),
            "capacity": dict(sorted(self.capacity.items())),
            "reuse": dict(sorted(self.reuse.items())),
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "rank": c.rank,
                    "need_id": c.need_id,
                    "priority": c.priority,
                    "slot": asdict(c.slot),
                    "anchors": [asdict(a) for a in c.anchors],
                }
                for c in self.candidates
            ],
        }

    def sha256(self) -> str:
        return sha256_value(self.record())


def copies(slot_ids: Sequence[str], salt: bytes, seed: int, universe: int) -> dict[str, int]:
    """How many candidates each slot gives: universe // slots for every slot, and one more for
    the universe % slots slots that come first in a draw. The proportions stay DEV's to within
    one candidate; no slot is chosen by anything but the draw."""
    if not slot_ids:
        raise V2Error("no answerable slot")
    base, extra = divmod(universe, len(slot_ids))
    first = sorted(slot_ids, key=lambda s: draw(seed, salt, "extra-copy", s))[:extra]
    return {s: base + (s in first) for s in slot_ids}


def build_bank(
    slot_list: Sequence[Slot],
    units: Mapping[str, Sequence[LegalUnit]],
    salt: bytes,
    seed: int = BANK_SEED,
    anchor_seed: int = ANCHOR_SEED,
    universe: int = UNIVERSE,
) -> Bank:
    """The frozen candidate bank.

    1. Each slot gives copies(...) candidates, named <slot>-<copy>.
    2. Every candidate's priority is a draw of its name; the bank is sorted by it, and the rank
       fixes its need id (N401 for the first).
    3. Anchors, per act, in priority order: a candidate needs one anchor in each of its
       profile's acts (a cross-act profile gets one per act). The act's eligible articles are
       put in an order drawn once, and the i-th demand of the act takes article i of that order,
       wrapping around only when every article has been drawn: uniform and without replacement
       while capacity permits, and a deterministic, counted reuse after. Inside the article,
       the provision starts at a draw over its eligible nodes made once for the article, and its
       k-th use takes the k-th provision after it: a reused article gives another provision
       while it has one left.
    """
    if any(not units.get(a) for a in bm.ACTS):
        raise V2Error("an act has no eligible legal unit")
    count = copies([s.slot_id for s in slot_list], salt, seed, universe)
    named = [(f"{s.slot_id}-{k:02d}", s) for s in slot_list for k in range(1, count[s.slot_id] + 1)]
    ordered = sorted(named, key=lambda item: draw(seed, salt, "priority", item[0]))
    order: dict[str, list[LegalUnit]] = {
        act: sorted(units[act], key=lambda u: draw(anchor_seed, salt, "article", act, u.article))
        for act in bm.ACTS
    }
    taken: Counter[str] = Counter()
    used: Counter[tuple[str, str]] = Counter()
    reuse: Counter[str] = Counter()
    candidates = []
    for rank, (name, slot) in enumerate(ordered, start=1):
        anchors = []
        for act in slot.acts:
            arts = order[act]
            unit = arts[taken[act] % len(arts)]
            taken[act] += 1
            earlier = used[(act, unit.article)]
            used[(act, unit.article)] += 1
            if earlier:
                reuse[act] += 1
            # one drawn start per article; its k-th use takes the k-th provision after it, so a
            # reused article gives another provision while it has one left
            start = int(draw(anchor_seed, salt, "provision", act, unit.article), 16)
            node = unit.nodes[(start + earlier) % len(unit.nodes)]
            anchors.append(Anchor(act, unit.article, f"{act}:{node}", unit.area, earlier))
        candidates.append(
            Candidate(
                name,
                slot,
                draw(seed, salt, "priority", name),
                rank,
                f"N{FIRST_NEED_NUMBER + rank - 1:03d}",
                tuple(anchors),
            )
        )
    return Bank(
        seed,
        tuple(candidates),
        {a: reuse[a] for a in bm.ACTS},
        {a: len(units[a]) for a in bm.ACTS},
    )


def bank_problems(bank: Bank, slot_list: Sequence[Slot], universe: int = UNIVERSE) -> list[str]:
    """The bank's rules, checked again on the bank itself."""
    problems = []
    if len(bank.candidates) != universe:
        problems.append(f"{len(bank.candidates)} candidates, not {universe}")
    if [c.rank for c in bank.candidates] != list(range(1, len(bank.candidates) + 1)):
        problems.append("the ranks are not the priority order")
    if [c.priority for c in bank.candidates] != sorted(c.priority for c in bank.candidates):
        problems.append("the candidates are not in priority order")
    if any(c.slot.answerability == "none" for c in bank.candidates):
        problems.append("a `none` profile entered the bank")
    per_slot = Counter(c.slot.slot_id for c in bank.candidates)
    if (
        set(per_slot) != {s.slot_id for s in slot_list}
        or max(per_slot.values()) - min(per_slot.values()) > 1
    ):
        problems.append("the slots are not replicated evenly")
    for c in bank.candidates:
        if tuple(a.act for a in c.anchors) != c.slot.acts:
            problems.append(f"{c.candidate_id}: its anchors are not one per act of its profile")
    if len({c.need_id for c in bank.candidates}) != len(bank.candidates):
        problems.append("a need id appears twice")
    return problems


# ---------------------------------------------------------------------------
# the completeness diagnostic: chains drawn before any judgment
# ---------------------------------------------------------------------------


def completeness_chains(
    bank: Bank, salt: bytes, seed: int = COMPLETENESS_SEED
) -> dict[str, list[str]]:
    """Per act, every candidate whose profile's first act it is, in an order drawn now. The
    diagnostic's slots of an act are the first COMPLETENESS_PER_STRATUM candidates of its chain
    that end up in the evaluable primary sample: the chain is fixed before any judgment, and
    which candidates are evaluable never depends on E or E5."""
    chains: dict[str, list[str]] = {act: [] for act in COMPLETENESS_STRATA}
    for c in bank.candidates:
        chains[c.slot.acts[0]].append(c.candidate_id)
    return {
        act: sorted(ids, key=lambda i: draw(seed, salt, "completeness", act, i))
        for act, ids in chains.items()
    }


def completeness_selection(
    chains: Mapping[str, Sequence[str]], primary: Iterable[str]
) -> dict[str, list[str]]:
    """The diagnostic's needs: per act, the first evaluable members of its chain."""
    sample = set(primary)
    return {
        act: [c for c in chain if c in sample][:COMPLETENESS_PER_STRATUM]
        for act, chain in chains.items()
    }


# ---------------------------------------------------------------------------
# the pool, the decisive set and the judging order
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pooled:
    chunk_id: str
    sources: tuple[tuple[str, str, int], ...]  # (contributor, query_id, rank); never shown


def pool(
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    query_ids: Sequence[str],
    anchor_chunks: Sequence[str],
    spec: Sequence[tuple[str, int]] = POOL,
) -> list[Pooled]:
    """The pool of one need: over its queries, every chunk each contributor ranks within its
    depth, and the chunks that own its anchors, each once. rankings: contributor -> query id ->
    chunk ids. A missing ranking contributes nothing (the evaluation scores it 0). Sorted by
    chunk id: an order that says nothing about a system."""
    found: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for contributor, depth in spec:
        for qid in query_ids:
            for rank, chunk in enumerate(rankings.get(contributor, {}).get(qid, ())[:depth], 1):
                found[chunk].append((contributor, qid, rank))
    for chunk in anchor_chunks:
        found[chunk].append((ANCHOR_CONTRIBUTOR, "", 0))
    return [Pooled(c, tuple(sorted(found[c]))) for c in sorted(found)]


def decisive_chunks(
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    query_ids: Sequence[str],
    depth: int = DECISIVE_DEPTH,
) -> list[str]:
    """The E-vs-E5 decisive set of one need: over its queries, the chunks in one compared
    system's top `depth` and not in the other's (E top-10 minus E5 top-10, plus E5 top-10
    minus E top-10)."""
    a, b = COMPARED
    out: set[str] = set()
    for qid in query_ids:
        top_a = set(rankings.get(a, {}).get(qid, ())[:depth])
        top_b = set(rankings.get(b, {}).get(qid, ())[:depth])
        out |= top_a ^ top_b
    return sorted(out)


def judged_nodes(
    chunk_id: str,
    layer: ev.ChunkLayer,
    texts: Mapping[bm.NodeKey, str],
    excluded_psd2: frozenset[str],
) -> list[str]:
    """The provisions of one chunk that are judged: every node it owns with its own text, in
    document order, outside the PSD2 articles amended after the snapshot. Its context
    (ancestors it does not own) is shown, never judged here."""
    out = []
    for act, path in layer.chunks[chunk_id].nodes:
        if not texts.get((act, path), "").strip():
            continue
        if act == "PSD2" and path.split(".")[0] in excluded_psd2:
            continue
        out.append(f"{act}:{path}")
    return out


def group_order(
    need_id: str, chunk_ids: Sequence[str], salt: bytes, stage: str, seed: int = JUDGE_ORDER_SEED
) -> list[str]:
    """The order a judge sees a need's chunk groups in: a draw per need and stage, so that no
    retriever rank and no document position orders the batches."""
    return sorted(chunk_ids, key=lambda c: draw(seed, salt, f"order:{stage}", need_id, c))


def batches(
    groups: Sequence[Sequence[str]],
    render: Any,
    count: Any,
    max_nodes: int,
    max_tokens: int,
) -> list[list[str]]:
    """Groups (each a chunk's nodes) packed in their order into batches, a group never split:
    a batch closes before the group that would make it hold more than max_nodes nodes or render
    to more than max_tokens tokens. A group that alone is over a ceiling is a batch of its own
    (it is never cut, and never left out). Deterministic."""
    out: list[list[str]] = []
    current: list[str] = []
    for group in groups:
        if not group:
            continue
        if current and (
            len(current) + len(group) > max_nodes or count(render([*current, *group])) > max_tokens
        ):
            out.append(current)
            current = []
        current = [*current, *group]
    if current:
        out.append(current)
    return out


# ---------------------------------------------------------------------------
# final labels, evaluability
# ---------------------------------------------------------------------------


def evaluable(finals: Mapping[str, Mapping[str, Any]]) -> bool:
    """Mechanical evaluability: at least one final, non-ambiguous label of grade >= 2. Nothing
    else (answerability, metadata, disagreement, difficulty) enters it."""
    return any(
        f.get("grade") is not None and f["grade"] >= bm.RELEVANT_GRADE and not f.get("ambiguous")
        for f in finals.values()
    )
