"""C of retrieval iteration 3, redesigned before it ran: an expanded, deterministic candidate
population, and a set selector that judges each candidate's legal function
(config/retrieval/necessity-selection.md).

Population, per query, built without a label:
- hop 0: E5's first 20 chunks; the first 10 are the anchors;
- hop 1: R1's recorded pool (the anchors' explicit references and defined terms, and the hits of
  its bridges), and the definitions of the defined terms the question uses, in any act;
- hop 2: the anchors' definitions by reference (R1), and the explicit references and defined
  terms of every hop-1 member reached by a reference, a definition or a term of the question;
- the definitions of the terms used by every member of hop 0 or 1, one hop further, never past 2;
deduplicated by chunk, every route kept (source, relation, hop, clause), ordered by R1's list.

Selector: windows of at most 8 members in that order. For each member a language model names a
legal function and says whether removing it would leave a legally necessary part of the answer
unsupported. A fixed merge takes the necessary members function by function, in a fixed order of
functions and then the population's order, up to 10. The ranking is that set, then E5's order.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ingest import dependencies as dp
from ingest import necessity as nc
from ingest import retrieval as rt

HEAD = 20
ANCHORS = 10
WINDOW = 8
SET_SIZE = 10
MAX_HOP = 2


@dataclass(frozen=True)
class Route:
    """One way a member was reached."""

    source: str  # "question" or the chunk it was reached from
    kind: str  # e5, reference, definition, definition_reference, bridge, question_term
    hop: int
    relation: str
    clause: str


@dataclass(frozen=True)
class Member:
    chunk: str
    hop: int  # the shortest route
    position: int  # in R1's list: the population's order
    routes: tuple[Route, ...]
    visible_in: tuple[str, ...]  # members whose inherited context already holds its text


def _neutral(label: str) -> str:
    """A's labels name "the anchor"; here the source may be any provision of the route."""
    return label.replace("the anchor", "that provision")


def _links(structure: nc.Structure, source: str, kinds: Sequence[str]) -> list[tuple[str, nc.Link]]:
    """(candidate, link) for the references and defined terms of a chunk's own text."""
    out = []
    for kind, detail, node, target in structure.index.relations.get(source, ()):
        if target == source or kind not in kinds:
            continue
        if kind == "reference":
            out.append((target, nc.reference_link(structure, source, detail, node, target)))
        elif kind == "definition":
            out.append((target, nc.definition_link(structure, source, detail, target)))
        else:
            out.append(
                (target, nc.definition_reference_link(structure, source, detail, node, target))
            )
    return out


def question_terms(
    question: str,
    finder: dp.TermFinder,
    owner: Mapping[tuple[str, str], str],
    acts: Sequence[str],
) -> list[tuple[str, dp.Definition, str]]:
    """(defining chunk, definition, clause of the question) for every defined term of the given
    acts that the question uses."""
    out = []
    for act in acts:
        for d in finder.find(act, question):
            chunk = owner.get((d.act, d.path))
            if chunk is None:
                continue
            m = dp.term_pattern(d.term).search(question)
            clause = nc.segment_at(question, m.start(), m.end())[1] if m else ""
            out.append((chunk, d, clause))
    return out


def expanded_population(
    e5: rt.Ranked,
    r1_list: rt.Ranked,
    record: Mapping[str, Any],
    structure: nc.Structure,
    question: str,
    finder: dp.TermFinder,
    visible: Mapping[str, Sequence[str]],
    head: int = HEAD,
    anchors_n: int = ANCHORS,
) -> list[Member]:
    """One query's population, in R1's order (see the module docstring). `record` is R1's
    provenance for the query; `visible` maps a chunk to the chunks whose inherited context holds
    its own text."""
    if record["query_id"] != e5.query_id or r1_list.query_id != e5.query_id:
        raise nc.NecessityError(f"{e5.query_id}: the inputs are of different queries")
    routes: dict[str, list[Route]] = {}

    def add(chunk: str, route: Route) -> None:
        if route.hop <= MAX_HOP:
            routes.setdefault(chunk, []).append(route)

    for rank, c in enumerate(e5.chunk_ids[:head], start=1):
        add(c, Route("question", "e5", 0, f"retrieved for the question (E5 rank {rank})", ""))
    anchors = list(e5.chunk_ids[:anchors_n])
    rank_of = {a: i for i, a in enumerate(anchors, start=1)}
    terms_of = {a: {d.term for d in structure.index.terms.get(a, ())} for a in anchors}
    hop1_structural: set[str] = set()
    for candidate, rels in sorted(record["candidates"].items()):
        for rel in rels:
            kind, anchor = rel["kind"], rel["anchor"]
            if kind == "bridge":
                term = nc.term_of_bridge(rel["detail"])
                for a in anchors:
                    if term in terms_of[a] and a != candidate:
                        link = nc.bridge_link(structure, a, term, candidate)
                        add(candidate, Route(a, "bridge", 1, _neutral(link.label), link.context))
                continue
            if anchor not in rank_of or anchor == candidate:
                continue
            if kind == "reference":
                link = nc.reference_link(
                    structure, anchor, rel["detail"], rel["target_node"], candidate
                )
            elif kind == "definition":
                link = nc.definition_link(structure, anchor, rel["detail"], candidate)
            else:
                link = nc.definition_reference_link(
                    structure, anchor, rel["detail"], rel["target_node"], candidate
                )
            hop = 2 if kind == "definition_reference" else 1
            add(candidate, Route(anchor, kind, hop, _neutral(link.label), link.context))
            if hop == 1:
                hop1_structural.add(candidate)
    acts = sorted({d.act for d in structure.definitions})
    for chunk, d, clause in question_terms(question, finder, structure.layer.owner, acts):
        label = f"definition: the candidate defines '{d.term}', a term the question uses"
        add(chunk, Route("question", "question_term", 1, label, clause))
        hop1_structural.add(chunk)
    for source in sorted(hop1_structural):
        for target, link in _links(structure, source, ("reference", "definition")):
            add(target, Route(source, link.kind, 2, _neutral(link.label), link.context))
    for source in sorted(c for c, rs in routes.items() if min(r.hop for r in rs) <= 1):
        hop = min(r.hop for r in routes[source])
        for target, link in _links(structure, source, ("definition",)):
            add(target, Route(source, "definition", hop + 1, _neutral(link.label), link.context))
    position = {c: i for i, c in enumerate(r1_list.chunk_ids)}
    members = []
    for chunk, rs in routes.items():
        unique = list(dict.fromkeys(rs))
        unique.sort(key=lambda r: (r.hop, r.source != "question", r.source, r.kind, r.relation))
        seen = [v for v in visible.get(chunk, ()) if v in routes and v != chunk]
        members.append(
            Member(chunk, unique[0].hop, position[chunk], tuple(unique), tuple(sorted(seen)))
        )
    members.sort(key=lambda m: (m.position, m.chunk))
    return members


def visibility(
    contexts: Mapping[str, str],
    owned: Mapping[str, Sequence[dp.NodeKey]],
    node_text: Mapping[dp.NodeKey, str],
    article: Mapping[str, str],
) -> dict[str, list[str]]:
    """For each chunk, the other chunks of its article whose inherited context (the ancestors'
    text the chunker puts before the body) holds the own text of every node it owns."""

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    by_article: dict[str, list[str]] = {}
    for chunk in contexts:
        by_article.setdefault(article[chunk], []).append(chunk)
    out: dict[str, list[str]] = {}
    for chunk in contexts:
        own = [norm(node_text[n]) for n in owned[chunk]]
        if not own or not all(own):
            continue
        for other in by_article[article[chunk]]:
            if other != chunk and all(t in norm(contexts[other]) for t in own):
                out.setdefault(chunk, []).append(other)
    return out


def windows(members: Sequence[Member], size: int = WINDOW) -> list[list[Member]]:
    return [list(members[i : i + size]) for i in range(0, len(members), size)]


# ---------------------------------------------------------------------------
# the selector's prompt, its answer, and the merge, fixed before any output
# ---------------------------------------------------------------------------

FUNCTIONS = (
    "governing rule",
    "scope or applicability",
    "legal category or definition",
    "condition",
    "exception",
    "addressee or authority",
    "deadline or threshold",
    "resolving cross-reference",
    "merely related",
)
SYSTEM = (
    "You are a careful analyst of EU financial regulation. You judge which provisions are "
    "necessary evidence for answering a question."
)
INSTRUCTIONS = (
    "Below are candidate provisions, each numbered. Each says how it was found: from the "
    "question, or from another provision through a relation, with the number of hops and the "
    "clause that creates the relation.\n\n"
    "For each candidate, ask: if this candidate were removed, what legally necessary part of the "
    "answer would no longer be supported? Then give its legal function, copying exactly one of "
    "these names:\n"
    + "\n".join(FUNCTIONS)
    + "\n\nA candidate is necessary only if removing it would leave a legally necessary part of "
    "the answer unsupported. A related provision that repeats the general subject is not "
    "necessary.\n\n"
    "Answer with exactly one line per candidate, in order, and nothing else. Each line is the "
    "candidate's number in brackets, one function name from the list, a bar, and yes (necessary) "
    "or no. The form, shown on other numbers:\n"
    "[4] condition | yes\n"
    "[5] merely related | no"
)
_LINE = re.compile(
    r"^\s*\[(\d+)\]\s*(.+?)\s*\|\s*(?:necessary\s*:\s*)?(yes|no)\b", re.IGNORECASE | re.MULTILINE
)
CLAUSE_CHARS = 160


def _source_name(source: str) -> str:
    if source == "question":
        return "the question"
    act, path = source.split(":", 1)
    return f"{act} {path}"


def _short(clause: str, limit: int = CLAUSE_CHARS) -> str:
    if len(clause) <= limit:
        return clause
    cut = clause.rfind(" ", 0, limit)
    return clause[: cut if cut > 0 else limit] + "..."


def describe(member: Member, routes: int = 1) -> str:
    """How a member was found: its shortest route (at most `routes`), the clause shortened to
    about CLAUSE_CHARS characters."""
    parts = []
    for r in member.routes[:routes]:
        text = f"from {_source_name(r.source)} ({r.relation}; hop {r.hop})"
        if r.clause:
            text += f"; clause: {_short(r.clause)}"
        parts.append(text)
    line = "Found " + " / ".join(parts)
    if member.visible_in:
        line += "\nIts text already appears as heading context of: " + ", ".join(
            _source_name(v) for v in member.visible_in
        )
    return line


def window_prompt(
    question: str, window: Sequence[Member], text: Mapping[str, str]
) -> list[dict[str, str]]:
    """The chat messages for one window. The question and the instructions open the prompt, so
    every window of a query shares its first tokens."""
    blocks = [f"[{i}] {text[m.chunk]}\n{describe(m)}" for i, m in enumerate(window, start=1)]
    user = (
        f"QUESTION: {question}\n\n{INSTRUCTIONS}\n\nCANDIDATES:\n\n"
        + "\n\n".join(blocks)
        + f"\n\nAnswer now: {len(window)} lines, one per candidate."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class Judgment:
    function: str  # one of FUNCTIONS, or "unparsed"
    necessary: bool


def parse_judgments(output: str, n: int) -> dict[int, Judgment]:
    """The judgment of each candidate 1..n that the output names in the required form; the first
    line wins for a candidate named twice; an unknown function counts as unparsed."""
    out: dict[int, Judgment] = {}
    for m in _LINE.finditer(output):
        i = int(m.group(1))
        if not 1 <= i <= n or i in out:
            continue
        function = re.sub(r"\s+", " ", m.group(2).strip().strip("*").strip().lower())
        if function not in FUNCTIONS:
            function = "unparsed"
        out[i] = Judgment(function, m.group(3).lower() == "yes")
    return out


def merge(
    judged: Mapping[str, Judgment], members: Sequence[Member], limit: int = SET_SIZE
) -> list[str]:
    """The evidence set: the members judged necessary with a function other than merely related,
    taken one per function in turn (in the order of FUNCTIONS, then a last group for a necessary
    member whose function is not one of the names), each function in the population's order, up
    to `limit`."""
    order = [*FUNCTIONS[:-1], "unparsed"]
    groups: dict[str, list[str]] = {f: [] for f in order}
    for m in members:
        j = judged.get(m.chunk)
        if j is not None and j.necessary and j.function in groups:
            groups[j.function].append(m.chunk)
    chosen: list[str] = []
    while len(chosen) < limit and any(groups.values()):
        for function in order:
            if groups[function] and len(chosen) < limit:
                chosen.append(groups[function].pop(0))
    return chosen


def evidence_ranking(e5: rt.Ranked, chosen: Sequence[str]) -> rt.Ranked:
    """The evidence set in its order, then every other chunk in E5's order."""
    order = list(dict.fromkeys(chosen))
    placed = set(order)
    order += [c for c in e5.chunk_ids if c not in placed]
    n = len(order)
    return rt.Ranked(e5.query_id, tuple(order), tuple(float(n - i) for i in range(n)))
