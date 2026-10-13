"""Explicit legal cross-references, and the one-hop expansion that follows them
(config/retrieval/xref-expansion.json, config/retrieval/README.md experiment 3).

A deterministic parser reads the own text of a node and finds the references that name a
provision: "Article 46", "Article 36(11)", "Article 3(1), point (7)", "point (a) of Article
59(1)", "Articles 11 and 12", "Articles 16 to 22", "paragraph 3" (of the same article),
"point (b) of paragraph 2". The act is the node's own unless the reference names another:
"of this Regulation" keeps it, the acts of the corpus are recognized by their official
numbers, and any other act is outside the corpus and ignored. Not followed, by design: "that
Regulation" or "that Directive" (anaphora), a point or subparagraph without its article or
paragraph, Titles, Chapters, Sections, Annexes and recitals.

A reference names a node path. When that exact path does not exist, its nearest existing
ancestor is used and the trace says so. Its targets are the chunks that own the node or any
node below it, in layer order.

The expansion reads no label: it works on the legal text and on a base ranking only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ingest import evaluation as ev
from ingest import retrieval as rt


class ExpansionConfig(BaseModel):
    """config/retrieval/xref-expansion.json: what experiment 3 runs, fixed before it runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"]
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    anchor_k: tuple[int, ...] = Field(min_length=1)
    hops: Literal[1]
    reference_source: Literal["own_text_of_owned_nodes"]
    acts: dict[str, str]
    linearization: Literal["anchor_then_its_targets"]

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_NUM = r"\d+[a-z]?"
_PAREN = r"\((?:\d+[a-z]?|[a-z]{1,4})\)"
_ART = rf"{_NUM}(?:{_PAREN})*"
_SEP = r"(?:,\s*|\s+and\s+|\s+or\s+|\s+to\s+)"
_POINT_LIST = rf"\([a-z0-9]{{1,4}}\)(?:{_SEP}\([a-z0-9]{{1,4}}\))*"
_NUM_LIST = rf"{_NUM}(?:{_SEP}{_NUM})*"
_ACT = (
    r"(?:this|that)\s+(?:Regulation|Directive)"
    r"|(?:Treaty|Charter)\b[^,.;()]*"
    r"|(?:Commission\s+|Council\s+)?(?:Delegated\s+|Implementing\s+)?"
    r"(?:Regulation|Directive|Decision)\s+(?:\((?:EU|EC|EEC|Euratom)\)\s*)?(?:No\s+)?"
    r"\d{2,4}/\d+(?:/(?:EU|EC|EEC))?"
)
_QUALIFIER = rf"(?:,?\s+of\s+(?:the\s+)?(?P<act>{_ACT}))?"

Form = Literal[
    "points_of_article",
    "paragraphs_of_article",
    "articles",
    "article",
    "points_of_paragraph",
    "paragraphs",
]
# Most specific first: a span matched by one form is masked before the next form runs.
_PATTERNS: tuple[tuple[Form, re.Pattern[str]], ...] = (
    (
        "points_of_article",
        re.compile(
            rf"\bpoints?\s+(?P<points>{_POINT_LIST})\s+of\s+Article\s+(?P<article>{_ART}){_QUALIFIER}"
        ),
    ),
    (
        "paragraphs_of_article",
        re.compile(
            rf"\bparagraphs?\s+(?P<paras>{_NUM_LIST})\s+of\s+Article\s+(?P<article>{_NUM}){_QUALIFIER}"
        ),
    ),
    (
        "articles",
        re.compile(rf"\bArticles\s+(?P<articles>{_ART}(?:{_SEP}{_ART})+){_QUALIFIER}"),
    ),
    (
        "article",
        re.compile(
            rf"\bArticle\s+(?P<article>{_ART})(?P<more>(?:(?:,\s*|\s+and\s+|\s+or\s+){_PAREN})*)"
            rf"(?:,\s+points?\s+(?P<points>{_POINT_LIST}),?)?{_QUALIFIER}"
        ),
    ),
    (
        "points_of_paragraph",
        re.compile(rf"\bpoints?\s+(?P<points>{_POINT_LIST})\s+of\s+paragraph\s+(?P<para>{_NUM})\b"),
    ),
    ("paragraphs", re.compile(rf"\bparagraphs?\s+(?P<paras>{_NUM_LIST})\b")),
)


@dataclass(frozen=True)
class Reference:
    """One provision a node names: where, how, and the path it asks for."""

    source_act: str
    source_path: str
    start: int
    text: str
    form: Form
    act: str
    path: str


@dataclass(frozen=True)
class Skipped:
    source_act: str
    source_path: str
    text: str
    reason: Literal["anaphora", "outside_the_corpus"]


def _split_path(spec: str) -> list[str]:
    head = re.match(_NUM, spec)
    if head is None:
        raise ValueError(f"not an article or paragraph number: {spec!r}")
    return [head.group(0), *re.findall(r"\(([0-9a-z]+)\)", spec[head.end() :])]


def _expand(items: str, pattern: str) -> list[str]:
    """'(a), (b) and (d)' or '16 to 18': every named item, ranges filled when both ends are
    numbers or single letters."""
    found = [(m.group(1), m.start()) for m in re.finditer(pattern, items)]
    out: list[str] = []
    for i, (item, pos) in enumerate(found):
        between = items[found[i - 1][1] : pos] if i else ""
        prev = out[-1] if out else None
        if prev is not None and re.search(r"\bto\b", between):
            if prev.isdigit() and item.isdigit():
                out += [str(n) for n in range(int(prev) + 1, int(item) + 1)]
                continue
            if len(prev) == len(item) == 1 and prev.isalpha() and item.isalpha():
                out += [chr(c) for c in range(ord(prev) + 1, ord(item) + 1)]
                continue
        out.append(item)
    return out


def extract(
    source_act: str, source_path: str, text: str, acts: Mapping[str, str]
) -> tuple[list[Reference], list[Skipped]]:
    """The references in one node's own text, in text order, and the ones not followed."""
    article = source_path.split(".")[0]
    masked = text
    found: list[Reference] = []
    skipped: list[Skipped] = []
    for form, pattern in _PATTERNS:
        for m in pattern.finditer(masked):
            masked = masked[: m.start()] + " " * (m.end() - m.start()) + masked[m.end() :]
            g = m.groupdict()
            phrase = text[m.start() : m.end()]
            act = source_act
            if g.get("act"):
                named = re.sub(r"\s+", " ", g["act"])
                if named.startswith("that "):
                    skipped.append(Skipped(source_act, source_path, phrase, "anaphora"))
                    continue
                if not named.startswith("this "):
                    named = re.sub(r"^(?:Commission|Council) ", "", named)
                    if named not in acts:
                        skipped.append(
                            Skipped(source_act, source_path, phrase, "outside_the_corpus")
                        )
                        continue
                    act = acts[named]
            paths: list[list[str]] = []
            if form == "points_of_article":
                base = _split_path(g["article"])
                paths = [[*base, p] for p in _expand(g["points"], r"\(([a-z0-9]{1,4})\)")]
            elif form == "paragraphs_of_article":
                paths = [[g["article"], p] for p in _expand(g["paras"], rf"({_NUM})")]
            elif form == "articles":
                paths = [_split_path(s) for s in _expand(g["articles"], rf"({_ART})")]
            elif form == "article":
                base = _split_path(g["article"])
                more = re.findall(r"\(([0-9a-z]+)\)", g.get("more") or "")
                bases = [base] + [[*base[:-1], x] for x in more if len(base) > 1]
                if g.get("points"):
                    points = _expand(g["points"], r"\(([a-z0-9]{1,4})\)")
                    paths = [[*b, p] for b in bases for p in points]
                else:
                    paths = bases
            elif form == "points_of_paragraph":
                paths = [
                    [article, g["para"], p] for p in _expand(g["points"], r"\(([a-z0-9]{1,4})\)")
                ]
            else:  # paragraphs of the same article
                paths = [[article, p] for p in _expand(g["paras"], rf"({_NUM})")]
            for parts in paths:
                found.append(
                    Reference(
                        source_act, source_path, m.start(), phrase, form, act, ".".join(parts)
                    )
                )
    found.sort(key=lambda r: r.start)
    return found, skipped


# ---------------------------------------------------------------------------
# the graph over the chunk layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """A reference of a chunk's own node, resolved to a node and to the chunks that own it."""

    reference: Reference
    resolved_path: str  # the node used: the path asked for, or its nearest existing ancestor
    targets: tuple[str, ...]  # chunk ids, in layer order


@dataclass(frozen=True)
class ReferenceGraph:
    edges: Mapping[str, tuple[Edge, ...]]  # chunk_id -> its references, in text order
    skipped: tuple[Skipped, ...]
    unresolved: tuple[Reference, ...]

    def targets_of(self, chunk_id: str) -> list[tuple[Edge, str]]:
        return [(e, t) for e in self.edges.get(chunk_id, ()) for t in e.targets]


def build_graph(
    layer: ev.ChunkLayer,
    node_text: Mapping[tuple[str, str], str],
    chunk_order: Sequence[str],
    acts: Mapping[str, str],
) -> ReferenceGraph:
    """Every chunk's references, read from the own text of the nodes it owns (never from
    its inherited context), resolved against the nodes of the corpus."""
    nodes = set(node_text)
    position = {c: i for i, c in enumerate(chunk_order)}
    owned_below: dict[tuple[str, str], set[str]] = {}
    for (act, path), chunk_id in layer.owner.items():
        parts = path.split(".")
        for i in range(1, len(parts) + 1):
            owned_below.setdefault((act, ".".join(parts[:i])), set()).add(chunk_id)
    edges: dict[str, tuple[Edge, ...]] = {}
    skipped: list[Skipped] = []
    unresolved: list[Reference] = []
    for chunk_id in chunk_order:
        out: list[Edge] = []
        for act, path in layer.chunks[chunk_id].nodes:
            refs, skip = extract(act, path, node_text[(act, path)], acts)
            skipped += skip
            for ref in refs:
                resolved = ref.path
                while (ref.act, resolved) not in nodes and "." in resolved:
                    resolved = resolved.rsplit(".", 1)[0]
                if (ref.act, resolved) not in nodes:
                    unresolved.append(ref)
                    continue
                targets = sorted(
                    owned_below.get((ref.act, resolved), set()), key=position.__getitem__
                )
                out.append(Edge(ref, resolved, tuple(targets)))
        edges[chunk_id] = tuple(out)
    return ReferenceGraph(edges, tuple(skipped), tuple(unresolved))


# ---------------------------------------------------------------------------
# one-hop expansion of a ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """Why a chunk sits where it does in an expanded ranking."""

    position: int  # 1-based, in the expanded ranking
    chunk_id: str
    anchor: str
    anchor_rank: int
    reference: str
    target_node: str  # act:path actually used


def expand(base: rt.Ranked, graph: ReferenceGraph, k: int) -> tuple[rt.Ranked, list[Placement]]:
    """Anchor 1, then the chunks its references reach (reference order, then layer order),
    anchor 2 and its targets, up to anchor k; a chunk keeps its first position; then the rest
    of the base ranking in its order. One hop: a target's own references are not followed.

    Scores: the expanded ranking keeps no meaningful score, so each chunk gets one derived
    from its position (n down to 1); the trace says why every added chunk is there."""
    if k < 1:
        raise rt.RetrievalError("the expansion needs at least one anchor")
    order: list[str] = []
    placed: set[str] = set()
    trace: list[Placement] = []
    for rank, anchor in enumerate(base.chunk_ids[:k], start=1):
        if anchor not in placed:
            order.append(anchor)
            placed.add(anchor)
        for edge, target in graph.targets_of(anchor):
            if target in placed:
                continue
            order.append(target)
            placed.add(target)
            trace.append(
                Placement(
                    len(order),
                    target,
                    anchor,
                    rank,
                    edge.reference.text,
                    f"{edge.reference.act}:{edge.resolved_path}",
                )
            )
    order += [c for c in base.chunk_ids if c not in placed]
    n = len(order)
    return rt.Ranked(base.query_id, tuple(order), tuple(float(n - i) for i in range(n))), trace


def summarize_graph(graph: ReferenceGraph) -> dict[str, object]:
    """How much of the corpus the parser connects: for the record of every expanded run."""
    edges = [e for es in graph.edges.values() for e in es]
    per_chunk = [len({t for e in es for t in e.targets} - {c}) for c, es in graph.edges.items()]
    by_form: dict[str, int] = {}
    for e in edges:
        by_form[e.reference.form] = by_form.get(e.reference.form, 0) + 1
    cross_act = sum(e.reference.act != e.reference.source_act for e in edges)
    return {
        "references_followed": len(edges),
        "by_form": dict(sorted(by_form.items())),
        "cross_act": cross_act,
        "resolved_to_an_ancestor": sum(e.resolved_path != e.reference.path for e in edges),
        "unresolved": len(graph.unresolved),
        "skipped_anaphora": sum(s.reason == "anaphora" for s in graph.skipped),
        "skipped_outside_the_corpus": sum(s.reason == "outside_the_corpus" for s in graph.skipped),
        "chunks_with_a_target": sum(x > 0 for x in per_chunk),
        "targets_per_chunk": _quantiles(per_chunk),
        "targets_per_reference": _quantiles([len(e.targets) for e in edges]),
    }


def _quantiles(values: Iterable[int]) -> dict[str, float]:
    v = sorted(values)
    if not v:
        return {}
    pick = lambda p: v[min(len(v) - 1, round(p / 100 * (len(v) - 1)))]  # noqa: E731
    return {"min": v[0], "median": pick(50), "p90": pick(90), "max": v[-1]}
