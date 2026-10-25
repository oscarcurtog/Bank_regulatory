"""TEST v2's adjudication (D-067), pure: who judges which node, and what the final label is.

- B1 judges every judged node of the need's pooled chunks: grouped by chunk, a chunk's nodes in
  document order, the groups in an order drawn per need; an explicit grade 0 is a label, kept.
- B2 judges, in packets of its own, every node B1 graded >= 1 or found ambiguous, and every
  node of the decisive set (the chunks in one compared system's top 10 and not in the other's,
  over the need's formulations), whatever B1 said.
- C decides exactly where B1 and B2 differ (grade or ambiguity), and nowhere else.
- The final label is the agreed one or C's; C may leave a node unresolved (no grade). A final
  ambiguity or an unresolved node stays in the qrels as it is: it never deletes the need.
- The qrels the evaluation reads are the final, non-ambiguous grades, the explicit 0s included
  (an ambiguous or unresolved node counts as unjudged, the evaluator's rule since D-019).

The ids the judges answer by are opaque (ho.item_id): the same id for a node in B1 and B2, so
that the two blind judgments name one candidate; C's ids are drawn apart.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ingest import benchmark as bm
from ingest import evaluation as ev
from ingest import holdout as ho
from ingest import holdout_judging as hj
from ingest import holdout_v2 as v2


def pool_ids(salt: bytes, need_id: str, nodes: Sequence[str]) -> dict[str, str]:
    """node -> opaque id, the same in B1 and B2."""
    return {n: ho.item_id(salt, f"v2-pool:{need_id}", n) for n in nodes}


def c_ids(salt: bytes, need_id: str, nodes: Sequence[str]) -> dict[str, str]:
    return {n: ho.item_id(salt, f"v2-c:{need_id}", n) for n in nodes}


def passage_id(salt: bytes, need_id: str, chunk_id: str) -> str:
    digest = hmac.new(salt, f"v2-screen:{need_id}:{chunk_id}".encode(), hashlib.sha256)
    return "P-" + digest.hexdigest()[:12]


@dataclass(frozen=True)
class NeedPool:
    """One need's frozen pool, as the adjudication reads it."""

    need_id: str
    chunks: tuple[str, ...]  # every pooled chunk, sorted by id
    judged: dict[str, tuple[str, ...]]  # chunk -> its judged nodes, in document order
    decisive: tuple[str, ...]  # the decisive chunks
    ineligible: tuple[str, ...]  # pooled chunks with no node to judge

    @property
    def nodes(self) -> list[str]:
        return [n for c in self.chunks for n in self.judged.get(c, ())]

    def decisive_nodes(self) -> list[str]:
        return [n for c in self.decisive for n in self.judged.get(c, ())]

    def record(self) -> dict[str, Any]:
        return {
            "need_id": self.need_id,
            "chunks": list(self.chunks),
            "judged": {c: list(n) for c, n in sorted(self.judged.items())},
            "decisive": list(self.decisive),
            "ineligible": list(self.ineligible),
        }


def need_pool(
    need_id: str,
    pooled: Sequence[v2.Pooled],
    decisive: Sequence[str],
    layer: ev.ChunkLayer,
    texts: Mapping[bm.NodeKey, str],
    excluded_psd2: frozenset[str],
) -> NeedPool:
    judged = {
        p.chunk_id: tuple(v2.judged_nodes(p.chunk_id, layer, texts, excluded_psd2)) for p in pooled
    }
    return NeedPool(
        need_id,
        tuple(p.chunk_id for p in pooled),
        {c: n for c, n in judged.items() if n},
        tuple(c for c in decisive if c in judged),
        tuple(c for c, n in judged.items() if not n),
    )


def groups(pool: NeedPool, chunks: Sequence[str], salt: bytes, stage: str) -> list[list[str]]:
    """The chunk groups of the given chunks in the order drawn for this need and stage."""
    ordered = v2.group_order(pool.need_id, [c for c in chunks if c in pool.judged], salt, stage)
    return [list(pool.judged[c]) for c in ordered]


def b2_nodes(pool: NeedPool, b1: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Every node B1 graded >= 1 or found ambiguous, and every node of the decisive set."""
    decisive = set(pool.decisive_nodes())
    return [n for n in pool.nodes if n in decisive or hj.positive_or_ambiguous(b1[n])]


def b2_chunks(pool: NeedPool, nodes: Sequence[str]) -> list[str]:
    wanted = set(nodes)
    return [c for c in pool.chunks if any(n in wanted for n in pool.judged.get(c, ()))]


def disputes(
    nodes: Sequence[str], b1: Mapping[str, Mapping[str, Any]], b2: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Exactly the nodes where the two blind judgments differ, in grade or in ambiguity."""
    return [n for n in nodes if hj.label(b1[n]) != hj.label(b2[n])]


def _label(j: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if j is None:
        return None
    grade, ambiguous = hj.label(j)
    return {"grade": grade, "ambiguous": bool(ambiguous)}


@dataclass(frozen=True)
class Final:
    grade: int | None
    ambiguous: bool
    outcome: str  # b1_negative | agreed | judge_c
    b1: dict[str, Any]
    b2: dict[str, Any] | None = None
    c: dict[str, Any] | None = None

    @property
    def unresolved(self) -> bool:
        return self.grade is None

    def record(self) -> dict[str, Any]:
        return {
            "grade": self.grade,
            "ambiguous": self.ambiguous,
            "outcome": self.outcome,
            "b1": self.b1,
            "b2": self.b2,
            "c": self.c,
        }


def finals(
    pool: NeedPool,
    b1: Mapping[str, Mapping[str, Any]],
    b2: Mapping[str, Mapping[str, Any]],
    c: Mapping[str, Mapping[str, Any]],
) -> dict[str, Final]:
    """The final label of every judged node of the pool."""
    out: dict[str, Final] = {}
    for node in pool.nodes:
        first = _label(b1[node])
        assert first is not None
        if node not in b2:
            if hj.positive_or_ambiguous(b1[node]) or node in set(pool.decisive_nodes()):
                raise ValueError(f"{node}: B2 is owed a judgment it did not give")
            out[node] = Final(first["grade"], first["ambiguous"], "b1_negative", first)
            continue
        second = _label(b2[node])
        assert second is not None
        if hj.label(b1[node]) == hj.label(b2[node]):
            out[node] = Final(first["grade"], first["ambiguous"], "agreed", first, second)
            continue
        if node not in c:
            raise ValueError(f"{node}: the judgments differ and C has not decided")
        third = _label(c[node])
        assert third is not None
        out[node] = Final(third["grade"], third["ambiguous"], "judge_c", first, second, third)
    return out


def qrel_record(
    finals_: Mapping[str, Final], texts: Mapping[bm.NodeKey, str]
) -> dict[str, dict[str, Any]]:
    """The need's qrel artifact: every judged node's final label, its judgments, its outcome,
    and the hash of the text it was judged on."""
    out = {}
    for node, f in sorted(finals_.items()):
        act, _, path = node.partition(":")
        out[node] = {**f.record(), "text_sha256": bm.sha256_text(texts[(act, path)])}
    return out


def evaluation_grades(record: Mapping[str, Mapping[str, Any]]) -> dict[bm.NodeKey, int]:
    """What the evaluation reads: node -> grade for every final, non-ambiguous, resolved label,
    explicit 0s included. An ambiguous or unresolved node counts as unjudged (D-019)."""
    out = {}
    for node, f in record.items():
        if f["grade"] is None or f["ambiguous"]:
            continue
        act, _, path = node.partition(":")
        out[(act, path)] = int(f["grade"])
    return out


@dataclass
class Summary:
    """Counts of one need's adjudication, never a label's text."""

    nodes: int = 0
    b2: int = 0
    disputes: int = 0
    grade_counts: dict[str, int] = field(default_factory=dict)
    ambiguous: int = 0
    unresolved: int = 0
    evaluable: bool = False


def summary(finals_: Mapping[str, Final], b2_count: int, dispute_count: int) -> Summary:
    grades: dict[str, int] = {}
    for f in finals_.values():
        key = "unresolved" if f.grade is None else str(f.grade)
        grades[key] = grades.get(key, 0) + 1
    labels = {n: {"grade": f.grade, "ambiguous": f.ambiguous} for n, f in finals_.items()}
    return Summary(
        len(finals_),
        b2_count,
        dispute_count,
        dict(sorted(grades.items())),
        sum(f.ambiguous for f in finals_.values()),
        sum(f.grade is None for f in finals_.values()),
        v2.evaluable(labels),
    )
