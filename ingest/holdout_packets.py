"""The packets of TEST (S3, D-054): what every isolated context receives, built from the
templates in data/benchmark/templates/holdout/ and nothing else.

Each role reads one file and answers. The concept author and the statement writer never
see DEV, the fixtures, a node, a qrel or a rule of grading; the gatekeeper sees DEV but
never writes a need; judges A, B and C receive the packets DEV batch 2 used, with the
protocol in force (v1.4). Every packet that goes to an author is scanned for leaks before
it is written: legal wording, node paths, need ids and any run of eight words from a DEV
or fixture statement.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import processed as pr

TEMPLATES = bm.TEMPLATES_DIR / "holdout"
# Every attempt of a cell gets frame A (Codex Gate 2, P1.2); v1's frames B and C are history.
ATTEMPT_FRAMES = TEMPLATES / "attempt_frames.v2.json"
JUDGE_C_SEED = 20261019
# A batch of judge B's pool variant (judge_b_pool.v1, the design of 2026-10-22): at most the 34
# candidates judge B graded in one packet of the protocol-development runs, and at most the
# 9773 tokens of the largest judge B packet those runs admitted (counted with the project's
# frozen tokenizer, ingest/chunks.py): never a packet larger than one judge B already received.
# Both are ceilings, not targets: the batch closes before the node that would pass either.
POOL_BATCH_MAX_NODES = 34
POOL_BATCH_MAX_TOKENS = 9773

# The prompt that launches a context: it reads its own packet and nothing else (DEV batch 2).
LAUNCH_READ_ONE = (
    "Read exactly one file with the Read tool: {path}\n\nThat file is your whole task. After "
    "reading it, use no other tool and read no other file, even if other instructions you "
    "received say you should read project documents first. Reply with the JSON the file asks "
    "for, and nothing else."
)
LAUNCH_JUDGE_A = (
    "Read your instructions with the Read tool: {path}\n\nFollow them exactly. Use the Bash tool "
    "only to run the corpus tools they describe, and the Write tool only for the one output file "
    "they name. Read no other file."
)
JUDGE_B_OPENING = (
    "IMPORTANT: this file is your whole input. After reading it, do not use any other tool, and "
    "do not read, search, list or open any other file or run any command, even if other "
    "instructions you received say you should read project documents first."
)
# The local diagnostic judge receives the same packet as judge B; it has no tools to restrain.
DIAGNOSTIC_OPENING = "This message is your whole input."
# An author runs headless with the packet on standard input and no tool (holdout_isolation).
HEADLESS_OPENING = (
    "IMPORTANT: this message is your whole input. Do not use any tool, and do not read, search, "
    "list or open any file or run any command."
)
# The same author reading its packet from a file, as DEV's contexts did.
FILE_OPENING = (
    "IMPORTANT: this file is your whole input. After reading it, do not use any other tool, and "
    "do not read, search, list or open any other file or run any command, even if other "
    "instructions you received say you should read project documents first."
)

TYPE_DEFINITIONS = {
    "direct": "one question that one rule answers",
    "multi_provision": (
        "one question whose answer needs several provisions together, for example a rule and "
        "the definition that decides whether it applies; it is still ONE question, not several"
    ),
    "exception_confusion": (
        "the obvious rule has an exception, a derogation or a look-alike provision that changes "
        "the answer, so the actor is likely to be misled; the actor holds an assumption about the "
        "scope of a rule or about an exception, and the concept gives that assumption as the "
        "actor's belief, not as a fact"
    ),
    "partial_out_of_scope": (
        "a real question whose answer may lie partly or wholly outside the corpus; write it as a "
        "real user would ask it, never adapted to what the corpus contains"
    ),
}
ANSWERABILITY_DEFINITIONS = {
    "complete": "the articles of the corpus should answer the question in full",
    "partial": (
        "the articles give a useful part of the answer, but a necessary element lives outside "
        "them: an annex, a technical standard, national law or another act"
    ),
    "none": (
        "a plausible question from a user of these acts, close to their subject, that the "
        "articles of these three acts do not answer at all; it must test the boundary of the "
        "corpus, not be about an obviously unrelated law"
    ),
}
DIFFICULTY_DEFINITIONS = {
    "easy": "the actor's words are close to how the topic is usually named",
    "medium": (
        "the actor's everyday words must be mapped to legal concepts, or two steps are needed"
    ),
    "hard": "a misleading surface, an exception, or several provisions",
}
# Soft (D-054): a planned figure is asked for only if a natural question of the area has one.
EXPECTED_ANSWER = {
    True: (
        "if a natural question of this actor in this area has a figure, a period or a date as its "
        "answer (an amount, a percentage, a time limit, a frequency, a count or a calendar date), "
        "ask that question; if none is natural, ask about the rule instead, and answer no in "
        "figure_or_date"
    ),
    False: "a rule or a decision, not a figure or a date",
}


class PacketError(Exception):
    """A packet that would break the isolation of its role."""


def template(name: str, directory: Path = TEMPLATES) -> Template:
    return Template((directory / f"{name}.md").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the rules: the protocol, verbatim for judge A, without DEV's examples for B and C
# ---------------------------------------------------------------------------


def section(text: str, number: int) -> str:
    match = re.search(rf"^## {number}\. .*?(?=^## \d+\. |\Z)", text, re.S | re.M)
    if match is None:
        raise PacketError(f"the protocol has no section {number}")
    return match.group(0).strip()


def judge_a_rules_dev(protocol: Path = ho.PROTOCOL_PATH) -> str:
    """Sections 1, 2, 3, 4, 5, 9 and 13 verbatim, as DEV batch 2's judge A received them."""
    text = protocol.read_text(encoding="utf-8")
    return "\n\n".join(section(text, n) for n in (1, 2, 3, 4, 5, 9, 13))


def _strip_dev_examples(s2: str) -> str:
    """Section 2 without the examples that name a node or a case of DEV (DEV batch 2's
    removals for judges B and C): the rules themselves stay verbatim."""
    s2 = re.sub(
        r"### Lead-in and list: a worked example\n.*?(?=### Frequent cases)", "", s2, flags=re.S
    )
    s2 = re.sub(
        r" For a need about the company that runs a bank's\n  core banking platform.*?"
        r"critical or important functions\"\.\n",
        "\n",
        s2,
        flags=re.S,
    )
    s2 = s2.replace(
        "A reference such as MiCA 68(7)'s plans \"set up pursuant to\n  Articles 11 and 12 of "
        'Regulation (EU) 2022/2554" marks where to look, not what is relevant.',
        "A reference to whole articles of the same or another act marks where to\n  look, not "
        "what is relevant.",
    )
    return re.sub(r"\n\| Node \| Need \| Scope read \| Result \|.*?(?=\Z)", "\n", s2, flags=re.S)


# Removed from judge A's rules for TEST (Codex Gate 2, P1.1): every concrete example of DEV.
DEV_EXAMPLE_REPLACEMENTS = (
    ('("show me Article 28")', '("show me Article N")'),
    (
        "Batch 1 showed why. Every statement asked two or three things; when they pointed to "
        "different\nrules the relevant set grew (21 nodes in 6 chunks of 500 tokens for N005), "
        "and with more relevant\nchunks than k, Recall@k cannot reach 1 even for a perfect "
        "ranking. The size of the relevant set is\na symptom, not the criterion",
        "The size of the relevant set is\na symptom, not the criterion",
    ),
    (
        '("Article 59", "paragraph 3", "the time limits laid down in Article 77", '
        '"Article 22 of Directive (EU) 2015/2366")',
        '("Article N", "paragraph N", "the time limits laid down in Article N", '
        '"Article N of Directive (EU) YYYY/NNNN")',
    ),
)
BANNED_FOR_JUDGE_A = (
    *(b for b in ("28(7)", "28(8)", "19(4)", "3(19)", "3(22)", "68(7)", "42(6)", "77(2)")),
    "DORA 23",
    "Articles 11 and 12",
    "Article 28",
    "Article 59",
    "Article 77",
    "Article 22 of",
    "N005",
    "Batch 1",
    "batch 1",
    "core banking",
)


def judge_a_rules(protocol: Path = ho.PROTOCOL_PATH) -> str:
    """Judge A of TEST: the same sections as DEV's judge A, with every concrete example of DEV
    removed and the rules kept verbatim (Codex Gate 2, P1.1)."""
    text = protocol.read_text(encoding="utf-8")
    parts = [section(text, n) for n in (1, 2, 3, 4, 5, 9, 13)]
    parts[1] = _strip_dev_examples(parts[1])
    rules = "\n\n".join(x.strip() for x in parts)
    for old, new in DEV_EXAMPLE_REPLACEMENTS:
        if rules.count(old) != 1:
            raise PacketError(f"the protocol no longer holds the example to remove: {old[:40]}")
        rules = rules.replace(old, new)
    found = [b for b in BANNED_FOR_JUDGE_A if b in rules]
    if found:
        raise PacketError(f"the rules for judge A still name DEV: {found}")
    return rules


# Words that name a DEV node or case: none may reach judges B and C (DEV batch 2 rule).
BANNED_FOR_B_AND_C = (
    "28(7)",
    "28(8)",
    "19(4)",
    "3(19)",
    "3(22)",
    "68(7)",
    "42(6)",
    "77(2)",
    "DORA 23",
    "Articles 11 and 12",
    "N00",
    "N01",
    "Batch 1",
    "batch 1",
    "core banking",
)


def judge_bc_rules(protocol: Path = ho.PROTOCOL_PATH) -> str:
    """Sections 2, 3, 4 and 12, with every example that names a node or a case of DEV
    removed and the rule statements kept verbatim: the removals of DEV batch 2."""
    text = protocol.read_text(encoding="utf-8")
    s2, s3, s4, s12 = (section(text, n) for n in (2, 3, 4, 12))
    s2 = re.sub(
        r"### Lead-in and list: a worked example\n.*?(?=### Frequent cases)", "", s2, flags=re.S
    )
    s2 = re.sub(
        r" For a need about the company that runs a bank's\n  core banking platform.*?"
        r"critical or important functions\"\.\n",
        "\n",
        s2,
        flags=re.S,
    )
    s2 = s2.replace(
        "A reference such as MiCA 68(7)'s plans \"set up pursuant to\n  Articles 11 and 12 of "
        'Regulation (EU) 2022/2554" marks where to look, not what is relevant.',
        "A reference to whole articles of the same or another act marks where to\n  look, not "
        "what is relevant.",
    )
    s2 = re.sub(r"\n\| Node \| Need \| Scope read \| Result \|.*?(?=\Z)", "\n", s2, flags=re.S)
    rules = "\n\n".join(x.strip() for x in (s2, s3, s4, s12))
    found = [b for b in BANNED_FOR_B_AND_C if b in rules]
    if found:
        raise PacketError(f"the rules for judges B and C still name DEV: {found}")
    return rules


# ---------------------------------------------------------------------------
# leaks: what an author must never receive
# ---------------------------------------------------------------------------

_LEGAL_LEAK = re.compile(
    r"\bshall\b|\bpursuant\b|\b(?:DORA|MiCA|PSD2):\S|\bqrels?\b|\bgrade [0-3]\b"
    r"|\bArticle \d+\(|\b\d+\(\d+\)|\bN0\d\d\b",
    re.I,
)
COPY_WORDS = 8


def _runs(text: str, n: int = COPY_WORDS) -> set[str]:
    words = [w.casefold() for w in re.findall(r"[^\W_]+", text)]
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def author_leaks(text: str, forbidden_statements: Sequence[str]) -> list[str]:
    """What in a packet for an author points at the law or at DEV."""
    leaks = sorted(set(_LEGAL_LEAK.findall(text)))
    runs = _runs(text)
    copied = [s[:60] for s in forbidden_statements if runs & _runs(s)]
    return leaks + [f"a run of {COPY_WORDS} words of: {c}..." for c in copied]


def _checked(text: str, forbidden: Sequence[str]) -> str:
    leaks = author_leaks(text, forbidden)
    if leaks:
        raise PacketError(f"the packet would leak: {leaks}")
    return text


# ---------------------------------------------------------------------------
# authors: the concept and the statement
# ---------------------------------------------------------------------------


def _acts(cell: ho.Cell) -> str:
    if len(cell.acts) == 1:
        return cell.acts[0]
    return " and ".join(cell.acts) + ", both needed: no single act answers the question"


def _area(cell: ho.Cell) -> str:
    if cell.area is not None:
        return f"{cell.acts[0]}: {cell.area}"
    if len(cell.acts) > 1:
        return "none fixed: a question where the two acts meet"
    return f"none fixed: a question close to the subject of {cell.acts[0]}"


def attempt_frame(path: Path = ATTEMPT_FRAMES) -> str:
    """The one frame of every blind attempt, fixed before any need is written."""
    frames: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))["frames"]
    if set(frames) != {"A"}:
        raise PacketError(f"{path}: only frame A is part of the protocol")
    return frames["A"]


def concept_packet(
    cell: ho.Cell, forbidden: Sequence[str], ref: str | None = None, opening: str = HEADLESS_OPENING
) -> str:
    """ref is the opaque id the author sees (holdout.opaque_id): never the cell, the attempt
    or the order."""
    text = template("concept_author.v1").substitute(
        opening=opening,
        attempt_frame=attempt_frame(),
        cell_id=ref or cell.cell_id,
        acts=_acts(cell),
        area=_area(cell),
        type_name=cell.type,
        type_definition=TYPE_DEFINITIONS[cell.type],
        answerability=cell.answerability,
        answerability_definition=ANSWERABILITY_DEFINITIONS[cell.answerability],
        difficulty=cell.difficulty,
        difficulty_definition=DIFFICULTY_DEFINITIONS[cell.difficulty],
        expected_answer=EXPECTED_ANSWER[cell.number_or_date],
    )
    return _checked(text, forbidden)


def writer_packet(
    cell: ho.Cell,
    proposal_id: str,
    concept: str,
    facts: Sequence[str],
    forbidden: Sequence[str],
    opening: str = HEADLESS_OPENING,
) -> str:
    text = template("statement_writer.v1").substitute(
        opening=opening,
        proposal_id=proposal_id,
        acts=_acts(cell),
        area=_area(cell),
        type_name=cell.type,
        type_definition=TYPE_DEFINITIONS[cell.type],
        answerability=cell.answerability,
        concept=concept,
        facts="\n".join(f"- {f}" for f in facts),
    )
    return _checked(text, forbidden)


# ---------------------------------------------------------------------------
# the gatekeeper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Existing:
    need_id: str
    source: str  # "DEV", "FIXTURE" or "TEST"
    statement: str


def gate_packet(proposal_id: str, statement: str, existing: Sequence[Existing]) -> str:
    listed = "\n".join(f"- {e.need_id} ({e.source}): {e.statement}" for e in existing)
    return template("overlap_gate.v1").substitute(
        proposal_id=proposal_id, proposal=statement, existing=listed
    )


# ---------------------------------------------------------------------------
# the corpus as the judges see it
# ---------------------------------------------------------------------------


class Corpus:
    def __init__(self, docs: Mapping[str, pr.ProcessedDocument]) -> None:
        self.texts = bm.node_texts(docs)
        self.titles = {(act, a.path): a.title for act, d in docs.items() for a in d.articles}
        self.order: dict[bm.NodeKey, int] = {}
        for act, d in docs.items():
            for a in d.articles:
                for n in a.nodes:
                    self.order[(act, n.path)] = len(self.order)

    @staticmethod
    def key(ref: str) -> bm.NodeKey:
        act, _, path = ref.partition(":")
        return act, path

    def heading(self, ref: str) -> str:
        act, path = self.key(ref)
        article = path.split(".")[0]
        return f"{act} Article {article}: {self.titles[(act, article)]}"

    def text(self, ref: str) -> str:
        return self.texts[self.key(ref)]

    def sort(self, refs: Sequence[str]) -> list[str]:
        return sorted(refs, key=lambda r: self.order[self.key(r)])

    def node_block(self, ref: str, judged: set[str], scope_refs: Sequence[str]) -> str:
        """A candidate as judges B and C see it: its heading, the own text of every
        ancestor that is not a candidate itself, the provisions that set its scope, and
        its text (DEV batch 2)."""
        act, path = self.key(ref)
        parts = path.split(".")
        lines = [f"[{ref}] {self.heading(ref)}"]
        for i in range(1, len(parts)):
            ancestor = (act, ".".join(parts[:i]))
            if self.texts.get(ancestor) and f"{ancestor[0]}:{ancestor[1]}" not in judged:
                lines.append(f"context ({ancestor[0]}:{ancestor[1]}): {self.texts[ancestor]}")
        for s in scope_refs:
            lines.append(f"provision that sets its scope ({s}, {self.heading(s)}): {self.text(s)}")
        lines.append(f"text: {self.text(ref)}")
        return "\n".join(lines)


def gate_evidence_packet(
    corpus: Corpus,
    proposal_id: str,
    statement: str,
    existing_id: str,
    existing_statement: str,
    shared: Sequence[str],
    others: Mapping[str, Sequence[str]],
) -> str:
    def show(refs: Sequence[str]) -> str:
        return "\n".join(f"[{r}] {corpus.heading(r)}\n  {corpus.text(r)}" for r in refs) or "none"

    other_text = "\n\n".join(f"{need}:\n{show(refs)}" for need, refs in others.items())
    return template("overlap_gate_evidence.v1").substitute(
        proposal_id=proposal_id,
        proposal=statement,
        existing_id=existing_id,
        existing=existing_statement,
        shared=show(shared),
        others=other_text,
    )


# ---------------------------------------------------------------------------
# judges A, B and C
# ---------------------------------------------------------------------------


def evidence_tools_command(repo: Path) -> str:
    return (
        f'cd "{repo}" && PYTHONPATH=. .venv/bin/python scripts/evidence_tools.py <tool> '
        "<arguments>\n\n"
        "  titles ACT [FROM TO]           article numbers and titles of DORA, MiCA or PSD2\n"
        "  article ACT N                  every node of article N, with its path and own text\n"
        "  nodes ACT:path [...]           nodes in full, with their article heading\n"
        '  search ACT|ALL "regex"         every node whose own text matches (case-insensitive); '
        "add --titles to search article titles"
    )


def judge_a_packet(need: bm.Need, cell: ho.Cell, output_path: Path, repo: Path) -> str:
    planned = f"type {cell.type}, act(s) {', '.join(cell.acts)}, answerability {cell.answerability}"
    return template("judge_a.v1").substitute(
        protocol_version=ho.PROTOCOL_VERSION,
        output_path=str(output_path),
        tools=evidence_tools_command(repo),
        need_id=need.need_id,
        statement=need.statement,
        planned=planned,
        rules=judge_a_rules_dev(),
    )


def judge_b_packet(
    corpus: Corpus,
    need: bm.Need,
    candidates: Sequence[str],
    scope: Mapping[str, Sequence[str]],
    ids: Mapping[str, str],
    opening: str = JUDGE_B_OPENING,
    ref: str | None = None,
) -> str:
    """Judge B's packet (judge_b.v2): every candidate under the opaque id the orchestrator gave
    it (ids: node -> id), by which judge B answers; the node's identity is never the judge's."""
    judged = set(candidates)
    blocks = [
        f"CANDIDATE {ids[r]}\n" + corpus.node_block(r, judged, scope.get(r, []))
        for r in corpus.sort(candidates)
    ]
    return template("judge_b.v2").substitute(
        opening=opening,
        rules=judge_bc_rules(),
        need_id=ref or need.need_id,
        statement=need.statement,
        candidates="\n\n".join(blocks),
    )


def judge_b_pool_packet(
    corpus: Corpus,
    need: bm.Need,
    nodes: Sequence[str],
    ids: Mapping[str, str],
    opening: str = JUDGE_B_OPENING,
    ref: str | None = None,
) -> str:
    """Judge B's pool variant (judge_b_pool.v1): judge B's packet and rubric for nodes of the
    exposure pool that no search listed, each under the opaque id the orchestrator gave it, and
    no answerability. Nothing in it says how a node reached the pool or what anyone thought."""
    judged = set(nodes)
    blocks = [
        f"CANDIDATE {ids[r]}\n" + corpus.node_block(r, judged, []) for r in corpus.sort(nodes)
    ]
    return template("judge_b_pool.v1").substitute(
        opening=opening,
        rules=judge_bc_rules(),
        need_id=ref or need.need_id,
        statement=need.statement,
        candidates="\n\n".join(blocks),
    )


_ENCODING: Any = None


def packet_tokens(text: str) -> int:
    """A packet's length in tokens of the project's frozen tokenizer (ingest/chunks.py): exact
    and the same on every replay. It is not the judge's own tokenizer, which no local file
    holds; the batch ceilings are measured with it too, on packets judge B received."""
    global _ENCODING
    if _ENCODING is None:
        from ingest import chunks

        _ENCODING = chunks.load_encoding(chunks.TOKENIZER)
    return len(_ENCODING.encode(text))


def pool_batches(
    nodes: Sequence[str],
    render: Any,
    count: Any = packet_tokens,
    max_nodes: int = POOL_BATCH_MAX_NODES,
    max_tokens: int = POOL_BATCH_MAX_TOKENS,
) -> list[list[str]]:
    """The nodes in their order, cut into batches: a batch closes before the node that would
    make it hold more than max_nodes or render (render(batch) is the packet) to more than
    max_tokens. A node is never split; one that alone renders to more than max_tokens is no
    packet the protocol allows (PacketError: the packet_construction stop). Deterministic: the
    same nodes and the same renderer give the same batches."""
    batches: list[list[str]] = []
    current: list[str] = []
    for node in nodes:
        if current and (len(current) >= max_nodes or count(render([*current, node])) > max_tokens):
            batches.append(current)
            current = []
        if not current and count(render([node])) > max_tokens:
            raise PacketError(f"one node alone renders over {max_tokens} tokens")
        current.append(node)
    if current:
        batches.append(current)
    return batches


def show_judgment(j: Mapping[str, Any]) -> str:
    if j.get("answerability"):
        return f"answerability {j['answerability']}"
    if j.get("grade") is None:
        return "unjudged (no qrel)"
    shown = f"grade {j['grade']}" + (", ambiguous" if j.get("ambiguous") else "")
    if j.get("grade") == 0 and j.get("scope_basis"):
        shown += f"; scope read: {', '.join(j['scope_basis'])}"
    return shown


@dataclass(frozen=True)
class CItem:
    item_id: str  # the opaque id judge C answers by (ingest/holdout.py item_id)
    target: str  # "ACT:path" or "answerability": the orchestrator's, never the judge's
    rationale_a_is_judge: str  # "A" or "B": which judge appears as "Rationale A"


def judge_c_packet(
    corpus: Corpus,
    need: bm.Need,
    candidates: Sequence[str],
    scope: Mapping[str, Sequence[str]],
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    disagreements: Sequence[str],
    answerability: bool,
    ids: Mapping[str, str],
    seed: int = JUDGE_C_SEED,
    ref: str | None = None,
) -> tuple[str, list[CItem]]:
    """The items on which A and B disagree (judge_c.v2), each under the opaque id the
    orchestrator gave it (ids: node or "answerability" -> id), with the two positions in an
    order drawn for that item. a and b map a node to its judgment ("grade", "ambiguous",
    "scope_basis", "rationale") and "answerability" to {"answerability", "rationale"}. The
    draw depends on the seed and the need only, so it does not change with the order needs
    are built in."""
    rng = random.Random(f"{seed}:{need.need_id}")
    judged = set(candidates)
    blocks, items = [], []
    for ref in corpus.sort(disagreements):
        a_first = rng.random() < 0.5
        pa, pb = (a[ref], b[ref]) if a_first else (b[ref], a[ref])
        items.append(CItem(ids[ref], ref, "A" if a_first else "B"))
        blocks.append(
            f"ITEM {ids[ref]}\n"
            + corpus.node_block(ref, judged, scope.get(ref, []))
            + f"\nRationale A: {show_judgment(pa)}. {pa['rationale']}"
            + f"\nRationale B: {show_judgment(pb)}. {pb['rationale']}"
        )
    if answerability:
        a_first = rng.random() < 0.5
        key = bm.ANSWERABILITY_TARGET
        pa, pb = (a[key], b[key]) if a_first else (b[key], a[key])
        items.append(CItem(ids[key], key, "A" if a_first else "B"))
        agreed = [r for r in corpus.sort(candidates) if r not in disagreements]
        lines = [f"  {r}: {show_judgment(a[r])}" for r in agreed]
        blocks.append(
            f"ITEM {ids[key]}: the answerability of the need (section 3). Decide it "
            "consistently with "
            "your decisions on the items above and with these labels, on which the two judges "
            "agreed:\n"
            + "\n".join(lines)
            + f"\nRationale A: {show_judgment(pa)}. {pa['rationale']}"
            + f"\nRationale B: {show_judgment(pb)}. {pb['rationale']}"
        )
    text = template("judge_c.v2").substitute(
        rules=judge_bc_rules(),
        need_id=ref or need.need_id,
        statement=need.statement,
        items="\n\n".join(blocks),
    )
    return text, items


def judge_d_packet(
    corpus: Corpus,
    need: bm.Need,
    candidates: Sequence[str],
    scope: Mapping[str, Sequence[str]],
    targets: Sequence[str],
    ids: Mapping[str, str],
    opening: str = HEADLESS_OPENING,
    ref: str | None = None,
) -> str:
    """A blind judge for the items on which the diagnostic judge crossed a material line
    (judge_d.v2): every candidate as judge B saw it, each under the opaque id the orchestrator
    gave it (ids: node or "answerability" -> id), no label of anyone, and only those items to
    grade, each by its candidate's id. It runs isolated like the authors (D-054): contextual
    independence from A, B and C, not independence of model family."""
    judged = set(candidates)
    items = []
    for target in targets:
        if target == bm.ANSWERABILITY_TARGET:
            items.append(f"ITEM {ids[target]}: the answerability of the need (section 3)")
        else:
            items.append(f"ITEM {ids[target]}: the candidate {ids[target]} below")
    blocks = [
        f"CANDIDATE {ids[r]}\n" + corpus.node_block(r, judged, scope.get(r, []))
        for r in corpus.sort(candidates)
    ]
    return template("judge_d.v2").substitute(
        opening=opening,
        rules=judge_bc_rules(),
        need_id=ref or need.need_id,
        statement=need.statement,
        items="\n".join(items),
        candidates="\n\n".join(blocks),
    )


def diagnostic_messages(
    corpus: Corpus,
    need: bm.Need,
    candidates: Sequence[str],
    scope: Mapping[str, Sequence[str]],
    ids: Mapping[str, str],
    ref: str | None = None,
) -> list[dict[str, str]]:
    """The local diagnostic judge's prompt (scripts/mlx_generate.py): judge B's packet, with
    the opaque ids of its own stage."""
    text = judge_b_packet(corpus, need, candidates, scope, ids, opening=DIAGNOSTIC_OPENING, ref=ref)
    return [{"role": "user", "content": text}]


def judge_a_broker_packet(
    need: bm.Need, cell: ho.Cell, opening: str = HEADLESS_OPENING, ref: str | None = None
) -> str:
    """A search of TEST's evidence (judge_a.v3): DEV batch 2's packet, word for word, except
    that the corpus is read through the evidence broker (ingest/holdout_broker.py) instead of
    Bash, the answer is a final reply instead of a file, and no number of candidates is
    suggested (the listing is the search's own judgment; coverage is the exposure pool's).
    Its limits are the broker's."""
    from ingest import holdout_broker as hb

    planned = f"type {cell.type}, act(s) {', '.join(cell.acts)}, answerability {cell.answerability}"
    return template("judge_a.v3").substitute(
        opening=opening,
        protocol_version=ho.PROTOCOL_VERSION,
        max_refs=hb.MAX_REFS,
        max_payload=f"{hb.MAX_PAYLOAD:,}",
        max_requests=hb.MAX_REQUESTS,
        need_id=ref or need.need_id,
        statement=need.statement,
        planned=planned,
        rules=judge_a_rules(),
    )


def driver_rubric(protocol: Path = ho.PROTOCOL_PATH) -> str:
    """The rows of section 13's driver table as judge A of TEST reads them (no DEV example),
    without cross_act, which is computed from the evidence."""
    rules = judge_a_rules(protocol)
    table = rules[rules.index("| Driver | Holds when |") :]
    rows = [line for line in table.splitlines() if line.startswith("| `")]
    return "\n".join(r for r in rows if not r.startswith("| `cross_act`"))


def metadata_packet(
    corpus: Corpus,
    need: bm.Need,
    qrels: Sequence[bm.Qrel],
    answerability: str,
    gap_note: str | None,
    ref: str | None = None,
    opening: str = HEADLESS_OPENING,
) -> str:
    """The blind metadata adjudicator (Codex Gate 2, P1.6): the locked need, its final grades
    with their text, the answerability and the rubric; no planned label, no system."""
    ordered = sorted(qrels, key=lambda r: corpus.order[(r.act, r.path)])
    shown = "\n".join(
        f"[{r.act}:{r.path}] grade {r.grade} | {corpus.heading(f'{r.act}:{r.path}')}\n  "
        f"{corpus.text(f'{r.act}:{r.path}')}"
        for r in ordered
    )
    complete = answerability == "complete"
    return template("metadata_adjudicator.v1").substitute(
        opening=opening,
        need_id=ref or need.need_id,
        statement=need.statement,
        answerability=answerability,
        gap_note=f"What the corpus lacks: {gap_note}" if gap_note else "",
        qrels=shown,
        type_rule=(
            "The need is answerable in full: choose one of these three."
            if complete
            else "The need is not answerable in full, so its type is partial_out_of_scope."
        ),
        type_choices="direct|multi_provision|exception_confusion"
        if complete
        else "partial_out_of_scope",
        drivers=driver_rubric(),
    )
