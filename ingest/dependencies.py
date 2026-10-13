"""Anchor-conditioned dependency retrieval (config/retrieval/dependency-retrieval.md, R1).

The anchors are the first chunks of a dense ranking. From the own text of the nodes each anchor
owns, three kinds of relation are read, deterministically:

- reference: the explicit legal references of ingest/references.py, to their exact targets;
- definition: a term defined in the anchor's act (a node whose text opens with "'term' means")
  that occurs in the anchor's own text, as a whole word, in any case, singular or plural;
- definition_reference: the explicit references inside a definition target's own text (a
  definition by reference), one step, definitions only.

Their targets, and the hits of short English questions about the defined terms (bridges), form
a candidate pool. The pool is never inserted at the top of a ranking: the candidate list keeps
the dense top n, then the rest of the pool by its best similarity to the query or a bridge.
Nothing here reads a label.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ingest import evaluation as ev
from ingest import references as xr
from ingest import retrieval as rt

NodeKey = tuple[str, str]
Kind = Literal["reference", "definition", "definition_reference", "bridge"]
_DEFINITION = re.compile(
    r"^\s*((?:'[^']{2,120}'(?:\s*,\s*|\s+or\s+|\s+and\s+)?)+)\s*(?:means|shall mean)\b"
)
BRIDGE_TEMPLATE = "What is {term}?"


@dataclass(frozen=True)
class Definition:
    act: str
    path: str
    term: str


def glossary(node_text: Mapping[NodeKey, str]) -> list[Definition]:
    """Every defined term of the corpus: the quoted terms that open a definition."""
    out = []
    for (act, path), text in sorted(node_text.items()):
        m = _DEFINITION.match(text)
        if m:
            out += [Definition(act, path, t) for t in re.findall(r"'([^']{2,120})'", m.group(1))]
    return out


def _term_pattern(term: str) -> re.Pattern[str]:
    stem = re.escape(term)
    if term.endswith("y") and term[-2:-1] not in ("a", "e", "o", "u"):
        forms = rf"(?:{stem}|{re.escape(term[:-1])}ies)"
    else:
        forms = rf"{stem}(?:s|es)?"
    return re.compile(rf"(?<![\w-]){forms}(?![\w-])", re.IGNORECASE)


class TermFinder:
    """The defined terms of one act that a text uses."""

    def __init__(self, definitions: Sequence[Definition]) -> None:
        self._by_act: dict[str, list[tuple[Definition, re.Pattern[str]]]] = {}
        for d in definitions:
            self._by_act.setdefault(d.act, []).append((d, _term_pattern(d.term)))

    def find(self, act: str, text: str) -> list[Definition]:
        hits = [(m.start(), d) for d, p in self._by_act.get(act, []) if (m := p.search(text))]
        return [d for _, d in sorted(hits, key=lambda h: (h[0], h[1].term))]


@dataclass(frozen=True)
class Relation:
    """Why a chunk is a candidate: which anchor, which relation, which target."""

    anchor: str
    anchor_rank: int
    kind: Kind
    detail: str  # the reference text, the term, or the bridge
    target_node: str  # act:path, or "" for a bridge hit
    candidate: str


@dataclass(frozen=True)
class DependencyIndex:
    """Each chunk's relations, independent of any query."""

    relations: Mapping[
        str, tuple[tuple[Kind, str, str, str], ...]
    ]  # chunk -> (kind, detail, node, target)
    terms: Mapping[str, tuple[Definition, ...]]  # chunk -> defined terms its own text uses


def build_index(
    layer: ev.ChunkLayer,
    node_text: Mapping[NodeKey, str],
    graph: xr.ReferenceGraph,
    definitions: Sequence[Definition],
) -> DependencyIndex:
    finder = TermFinder(definitions)
    definition_nodes = {(d.act, d.path) for d in definitions}
    relations: dict[str, tuple[tuple[Kind, str, str, str], ...]] = {}
    terms: dict[str, tuple[Definition, ...]] = {}
    for chunk_id, chunk in layer.chunks.items():
        out: list[tuple[Kind, str, str, str]] = []
        for edge in graph.edges.get(chunk_id, ()):
            node = f"{edge.reference.act}:{edge.resolved_path}"
            out += [("reference", edge.reference.text, node, t) for t in edge.targets]
        used: list[Definition] = []
        for act, path in chunk.nodes:
            for d in finder.find(act, node_text[(act, path)]):
                if (d.act, d.path) == (act, path) or d in used:
                    continue
                used.append(d)
                owner = layer.owner.get((d.act, d.path))
                if owner is None:
                    continue
                out.append(("definition", d.term, f"{d.act}:{d.path}", owner))
                if (d.act, d.path) in definition_nodes:
                    for edge in graph.edges.get(owner, ()):
                        if (edge.reference.source_act, edge.reference.source_path) != (
                            d.act,
                            d.path,
                        ):
                            continue
                        node = f"{edge.reference.act}:{edge.resolved_path}"
                        out += [("definition_reference", d.term, node, t) for t in edge.targets]
        relations[chunk_id] = tuple(out)
        terms[chunk_id] = tuple(used)
    return DependencyIndex(relations, terms)


def bridge_text(term: str) -> str:
    return BRIDGE_TEMPLATE.format(term=term)


@dataclass
class QueryCandidates:
    """One query's candidate list and why each candidate is in it."""

    ranked: rt.Ranked
    bridges: list[str]
    provenance: dict[str, list[Relation]] = field(default_factory=dict)
    pool_size: int = 0


def anchor_relations(
    base: rt.Ranked, index: DependencyIndex, anchors: int
) -> tuple[list[Relation], list[tuple[str, int]]]:
    """The relations of the first anchors, and the defined terms they use with the rank of
    the first anchor that uses each one."""
    relations: list[Relation] = []
    terms: dict[str, int] = {}
    for rank, anchor in enumerate(base.chunk_ids[:anchors], start=1):
        for kind, detail, node, target in index.relations.get(anchor, ()):
            if target != anchor:
                relations.append(Relation(anchor, rank, kind, detail, node, target))
        for d in index.terms.get(anchor, ()):
            terms.setdefault(d.term, rank)
    return relations, sorted(terms.items(), key=lambda t: (t[1], t[0]))


def choose_bridges(
    query_vector: np.ndarray,
    terms: Sequence[tuple[str, int]],
    bridge_vectors: Mapping[str, np.ndarray],
    limit: int,
) -> list[str]:
    """The `limit` bridges closest to the query (cosine of unit vectors), ties by the rank of
    the first anchor that uses the term, then the term."""
    scored = [(-float(bridge_vectors[bridge_text(t)] @ query_vector), rank, t) for t, rank in terms]
    return [bridge_text(t) for _, _, t in sorted(scored)[:limit]]


def candidate_list(
    base: rt.Ranked,
    relations: Sequence[Relation],
    bridge_hits: Mapping[str, Sequence[str]],
    probe_score: Callable[[str], float],
    keep: int,
) -> QueryCandidates:
    """The dense top `keep` in its order; then every other pool member by its probe score
    (best first, then chunk id); then every other chunk in the dense order."""
    provenance: dict[str, list[Relation]] = {}
    for r in relations:
        provenance.setdefault(r.candidate, []).append(r)
    for bridge, hits in bridge_hits.items():
        for rank, chunk in enumerate(hits, start=1):
            provenance.setdefault(chunk, []).append(Relation("", rank, "bridge", bridge, "", chunk))
    head = list(base.chunk_ids[:keep])
    placed = set(head)
    pool = sorted((c for c in provenance if c not in placed), key=lambda c: (-probe_score(c), c))
    order = head + pool
    in_pool = set(pool)
    order += [c for c in base.chunk_ids if c not in placed and c not in in_pool]
    n = len(order)
    ranked = rt.Ranked(base.query_id, tuple(order), tuple(float(n - i) for i in range(n)))
    return QueryCandidates(ranked, list(bridge_hits), provenance, len(head) + len(pool))


# ---------------------------------------------------------------------------
# E2: dense-seeded Personalized PageRank over the legal structure
# ---------------------------------------------------------------------------


def structure_matrix(index: DependencyIndex, chunk_ids: Sequence[str]) -> np.ndarray:
    """Row-stochastic transition matrix over chunks: from a chunk, one step to each distinct
    target of its relations, equally likely; a row of zeros for a chunk without targets."""
    position = {c: i for i, c in enumerate(chunk_ids)}
    w = np.zeros((len(chunk_ids), len(chunk_ids)), dtype=np.float64)
    for chunk, relations in index.relations.items():
        targets = {t for _, _, _, t in relations if t != chunk}
        for t in targets:
            w[position[chunk], position[t]] = 1.0 / len(targets)
    return w


def personalized_pagerank(
    seeds: np.ndarray, transition: np.ndarray, follow: float, tolerance: float = 1e-12
) -> np.ndarray:
    """p = (1 - follow) s + follow (W^T p + dangling mass s), iterated to a fixed point.
    `follow` is the probability of taking an edge (HippoRAG 2's damping of 0.5)."""
    if not 0 < follow < 1 or abs(seeds.sum() - 1) > 1e-9 or (seeds < 0).any():
        raise rt.RetrievalError("PageRank needs 0 < follow < 1 and a seed distribution")
    dangling = transition.sum(axis=1) == 0
    p = seeds.copy()
    for _ in range(10_000):
        nxt: np.ndarray = (1 - follow) * seeds + follow * (
            transition.T @ p + p[dangling].sum() * seeds
        )
        if np.abs(nxt - p).max() < tolerance:
            return nxt
        p = nxt
    raise rt.RetrievalError("PageRank did not converge")


def ppr_ranking(
    base: rt.Ranked,
    dense_scores: Mapping[str, float],
    transition: np.ndarray,
    chunk_ids: Sequence[str],
    seeds: int,
    follow: float,
) -> rt.Ranked:
    """E5's top `seeds` as the reset distribution (proportional to their cosine); chunks by
    PageRank score, ties by the base rank; never-reached chunks after, in the base order."""
    position = {c: i for i, c in enumerate(chunk_ids)}
    s = np.zeros(len(chunk_ids), dtype=np.float64)
    for c in base.chunk_ids[:seeds]:
        s[position[c]] = max(dense_scores[c], 0.0)
    if s.sum() <= 0:
        raise rt.RetrievalError(f"{base.query_id}: no positive seed")
    p = personalized_pagerank(s / s.sum(), transition, follow)
    base_rank = {c: i for i, c in enumerate(base.chunk_ids)}
    reached = [c for c in base.chunk_ids if p[position[c]] > 0]
    reached.sort(key=lambda c: (-p[position[c]], base_rank[c]))
    rest = [c for c in base.chunk_ids if p[position[c]] == 0]
    order = reached + rest
    return rt.Ranked(base.query_id, tuple(order), tuple(float(p[position[c]]) for c in order))
