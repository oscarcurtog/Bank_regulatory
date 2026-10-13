"""Retrieval benchmark (D-019): information needs, their query formulations and
node-level relevance judgments, validated against the processed layer and sealed
by a manifest.

One directory per benchmark (data/benchmark/dev/, data/benchmark/fixtures/), with
three JSON Lines files and a manifest:

    needs.jsonl     what the user wants to know; the unit of the DEV/TEST split
    queries.jsonl   how it is asked: several per need, in English and Spanish. A
                    query has no split of its own, it inherits its need's
    qrels.jsonl     which legal nodes answer the need, and how much (grade 0-3)
    manifest.json   the seal: schema, benchmark version, role, parser, corpus
                    snapshot, and the SHA-256 of every file it covers

Relevance lives on NODES (act + path), never on chunks, so the same judgments
evaluate any chunking: ingest/evaluation.py projects them onto a chunk layer.

Three boundaries, all fail-fast:

1. the manifest: a file that changed since the seal is refused;
2. the dataset: every record, then the rules that look at more than one record;
3. the corpus: every judged node exists, has text, and still has the text that
   was judged (text_sha256). A parser or corpus change never evaluates against
   stale labels: it fails, naming the need and the path to review.

seal() is the only writer of manifest.json, and it runs every check first. There
is no implicit reseal.

Unlike the other contracts of the repository, these models forbid unknown keys.
Their files are written by hand, and several fields are optional: a typo such as
"ambigous": true would otherwise be dropped in silence and change the metrics.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Self, get_args

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator

from ingest import download as dl
from ingest import hierarchy as h
from ingest import processed as pr
from ingest import psd2_exclusions as px

SCHEMA_VERSION: Literal["1"] = "1"
BENCHMARK_DIR = Path("data/benchmark")
DEV_DIR = BENCHMARK_DIR / "dev"
FIXTURES_DIR = BENCHMARK_DIR / "fixtures"
TEMPLATES_DIR = BENCHMARK_DIR / "templates"

NEEDS_FILE = "needs.jsonl"
QUERIES_FILE = "queries.jsonl"
QRELS_FILE = "qrels.jsonl"
MANIFEST_FILE = "manifest.json"

# Binary relevance for Recall, MRR and the evidence rules: grade >= 2 (D-019).
RELEVANT_GRADE = 2

Act = Literal["DORA", "MiCA", "PSD2"]
ACTS: tuple[Act, ...] = ("DORA", "MiCA", "PSD2")
Split = Literal["dev", "test"]
# What a benchmark directory is for. A fixture holds test data for the pipeline and
# never counts as DEV: its queries may have been written knowing the evidence.
Role = Literal["dev", "test", "fixture"]
NeedType = Literal["direct", "multi_provision", "exception_confusion", "partial_out_of_scope"]
# Properties of the EVIDENCE, defined in PROTOCOL.md section 13. Properties of a
# formulation (paraphrase, vocabulary, language) come from Query.lang, Query.style,
# Query.origin and the measured overlap.
Driver = Literal[
    "cross_act",
    "negation",
    "exception",
    "numeric",
    "date",
    "lexical_twin",
    "cross_reference",
    "level2_gap",
]
Difficulty = Literal["easy", "medium", "hard"]
Answerability = Literal["complete", "partial", "none"]
Status = Literal["draft", "reviewed", "ready"]
Lang = Literal["en", "es"]
Style = Literal["formal", "natural"]
# human:          written by a person from the need statement
# llm_from_need:  generated from the need statement alone, with a versioned template
# evidence_aware: written knowing the evidence; only allowed in a fixture
Origin = Literal["human", "llm_from_need", "evidence_aware"]
DataFile = Literal["needs.jsonl", "queries.jsonl", "qrels.jsonl"]
DATA_FILES: tuple[DataFile, ...] = ("needs.jsonl", "queries.jsonl", "qrels.jsonl")

NEED_ID_PATTERN = r"^N\d{3}$"
QUERY_ID_PATTERN = r"^N\d{3}-(en|es)-\d+$"
TEMPLATE_ID_PATTERN = r"^[a-z][a-z0-9_]*\.v\d+$"

NodeKey = tuple[str, str]  # (act, path)


def _trimmed(value: str) -> str:
    if value != value.strip():
        raise ValueError("leading or trailing whitespace")
    return value


Text = Annotated[str, Field(min_length=1), AfterValidator(_trimmed)]
Sha256 = Annotated[str, Field(pattern=dl.SHA256_PATTERN)]
TemplateId = Annotated[str, Field(pattern=TEMPLATE_ID_PATTERN)]
Grade = Annotated[int, Field(ge=0, le=3, strict=True)]
STRICT = ConfigDict(extra="forbid")


class BenchmarkError(Exception):
    """The benchmark files do not meet the contract."""


class StaleBenchmarkError(BenchmarkError):
    """The benchmark is valid but sealed against something that changed: its own
    files, the parser, or the corpus snapshot. Review the listed qrels, then seal
    again with a new benchmark_version."""


# ---------------------------------------------------------------------------
# contracts. Declaration order is serialization order.
# ---------------------------------------------------------------------------


class Need(BaseModel):
    """What a user wants to know. The unit of the split and of the qrels."""

    model_config = STRICT

    need_id: str = Field(pattern=NEED_ID_PATTERN)
    split: Split
    statement: Text
    type: NeedType
    drivers: list[Driver]
    difficulty: Difficulty
    acts: list[Act] = Field(min_length=1)
    answerability: Answerability
    gap_note: Text | None = None
    ambiguity_note: Text | None = None
    status: Status

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(set(self.drivers)) != len(self.drivers):
            raise ValueError(f"{self.need_id}: a driver appears twice")
        if len(set(self.acts)) != len(self.acts):
            raise ValueError(f"{self.need_id}: an act appears twice")
        complete = self.answerability == "complete"
        if complete != (self.gap_note is None):
            raise ValueError(
                f"{self.need_id}: gap_note is required exactly when answerability is not "
                f"'complete' (it says what the corpus lacks)"
            )
        if complete == (self.type == "partial_out_of_scope"):
            raise ValueError(
                f"{self.need_id}: type 'partial_out_of_scope' goes with answerability "
                f"'partial' or 'none', and only with them"
            )
        return self


class Query(BaseModel):
    """One formulation of a need."""

    model_config = STRICT

    query_id: str = Field(pattern=QUERY_ID_PATTERN)
    need_id: str = Field(pattern=NEED_ID_PATTERN)
    lang: Lang
    style: Style
    origin: Origin
    template: TemplateId | None = None
    text: Text

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not self.query_id.startswith(f"{self.need_id}-{self.lang}-"):
            raise ValueError(f"{self.query_id}: the id must be <need_id>-<lang>-<n>")
        # Spanish formulations are natural user language, not a translation shaped to
        # resemble the English legal text (D-019).
        if self.lang == "es" and self.style != "natural":
            raise ValueError(f"{self.query_id}: a Spanish query must have style 'natural'")
        if (self.origin == "llm_from_need") != (self.template is not None):
            raise ValueError(
                f"{self.query_id}: template is required exactly when origin is 'llm_from_need'"
            )
        return self


class Span(BaseModel):
    """A character range of a node's own text, for long nodes with several ideas."""

    model_config = STRICT

    start: int = Field(ge=0, strict=True)
    end: int = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.start >= self.end:
            raise ValueError(f"span start {self.start} is not before end {self.end}")
        return self


class Qrel(BaseModel):
    """One judgment: how relevant a legal node is to a need.

    grade 3: the direct evidence; 2: needed support (definition, exception, referred
    provision); 1: related but dispensable; 0: examined and NOT relevant, which is not
    the same as unjudged. ambiguous: the grade depends on an interpretation, so the
    judgment counts as unjudged and keeps its need out of any future TEST."""

    model_config = STRICT

    need_id: str = Field(pattern=NEED_ID_PATTERN)
    act: Act
    path: str = Field(pattern=h.LTREE_PATH_PATTERN)
    grade: Grade
    rationale: Text
    text_sha256: Sha256
    span: Span | None = None
    ambiguous: bool = Field(default=False, strict=True)

    @property
    def node(self) -> NodeKey:
        return (self.act, self.path)


class CorpusSource(BaseModel):
    model_config = STRICT

    source_celex: str = Field(pattern=dl.CELEX_PATTERN)
    source_sha256: Sha256


class BenchmarkManifest(BaseModel):
    """The seal of one benchmark directory, written only by seal()."""

    model_config = STRICT

    schema_version: Literal["1"]
    benchmark_version: str = Field(pattern=pr.VERSION_PATTERN)
    role: Role
    parser_version: str = Field(pattern=pr.VERSION_PATTERN)
    # the whole snapshot, not only the acts the qrels touch: hard negatives and
    # 'none' needs are judged against all of it
    corpus: dict[Act, CorpusSource]
    files: dict[DataFile, Sha256]
    templates: dict[TemplateId, Sha256]

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if set(self.corpus) != set(ACTS):
            raise ValueError(
                f"corpus must list every act {sorted(ACTS)}, got {sorted(self.corpus)}"
            )
        if set(self.files) != set(DATA_FILES):
            raise ValueError(f"files must list {sorted(DATA_FILES)}, got {sorted(self.files)}")
        return self


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    """One validated record per line. The first bad line aborts the whole file."""
    if not path.exists():
        raise BenchmarkError(f"{path} does not exist")
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()  # the trailing newline
    records: list[M] = []
    for number, line in enumerate(lines, start=1):
        try:
            records.append(model.model_validate_json(line))
        except ValidationError as e:
            raise BenchmarkError(
                f"{path}:{number} does not meet the {model.__name__} contract.\n{e}"
            ) from e
    return records


def write_jsonl(path: Path, records: Sequence[BaseModel]) -> None:
    """Every field of every record, in declaration order, one record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def template_path(template_id: str, templates_dir: Path = TEMPLATES_DIR) -> Path:
    return templates_dir / f"{template_id}.md"


@dataclass(frozen=True)
class Benchmark:
    """A loaded benchmark: its seal and its three files, already validated."""

    directory: Path
    manifest: BenchmarkManifest
    needs: list[Need]
    queries: list[Query]
    qrels: list[Qrel]
    # how each label was decided (section 12); empty where no record exists
    adjudications: list[Adjudication] = field(default_factory=list)

    def need(self, need_id: str) -> Need:
        return next(n for n in self.needs if n.need_id == need_id)

    def queries_of(self, need_id: str) -> list[Query]:
        return [q for q in self.queries if q.need_id == need_id]

    def qrels_of(self, need_id: str) -> list[Qrel]:
        return [r for r in self.qrels if r.need_id == need_id]


# ---------------------------------------------------------------------------
# the dataset boundary: rules across records
# ---------------------------------------------------------------------------


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def dataset_problems(
    needs: Sequence[Need],
    queries: Sequence[Query],
    qrels: Sequence[Qrel],
    role: Role,
    excluded_psd2: frozenset[str] = frozenset(),
) -> list[str]:
    """Every rule that looks at more than one record. Returns all the problems at
    once, so that one run lists everything to fix."""
    problems: list[str] = []
    need_ids = [n.need_id for n in needs]
    problems += [f"need {i} appears {c} times" for i, c in Counter(need_ids).items() if c > 1]
    query_ids = [q.query_id for q in queries]
    problems += [f"query {i} appears {c} times" for i, c in Counter(query_ids).items() if c > 1]
    keys = [(r.need_id, r.act, r.path) for r in qrels]
    problems += [f"qrel {k} appears {c} times" for k, c in Counter(keys).items() if c > 1]
    texts = Counter(_normalized(q.text) for q in queries)
    problems += [
        f"query {q.query_id}: the same text as another query"
        for q in queries
        if texts[_normalized(q.text)] > 1
    ]

    known = set(need_ids)
    problems += [
        f"query {q.query_id}: unknown need {q.need_id}" for q in queries if q.need_id not in known
    ]
    problems += [
        f"qrel {r.need_id} {r.act}:{r.path}: unknown need" for r in qrels if r.need_id not in known
    ]

    problems += [
        f"qrel {r.need_id} PSD2:{r.path}: Article {r.path.split('.')[0]} was amended after the "
        f"corpus snapshot ({px.PSD2_EXCLUSIONS_PATH.name}); no need may use it"
        for r in qrels
        if r.act == "PSD2" and r.path.split(".")[0] in excluded_psd2
    ]

    split = "dev" if role in ("dev", "fixture") else "test"
    problems += [
        f"need {n.need_id}: split {n.split!r} in a benchmark with role {role!r}"
        for n in needs
        if n.split != split
    ]
    if role != "fixture":
        problems += [
            f"query {q.query_id}: origin 'evidence_aware' is only allowed in a fixture"
            for q in queries
            if q.origin == "evidence_aware"
        ]

    for need in needs:
        mine = [q for q in queries if q.need_id == need.need_id]
        judged = [r for r in qrels if r.need_id == need.need_id]
        evidence = [r for r in judged if r.grade >= RELEVANT_GRADE and not r.ambiguous]
        if need.answerability == "none" and any(r.grade >= RELEVANT_GRADE for r in judged):
            problems.append(
                f"need {need.need_id}: answerability 'none' but it has qrels of grade >= "
                f"{RELEVANT_GRADE}"
            )
        if need.split == "test" and (
            need.ambiguity_note is not None or any(r.ambiguous for r in judged)
        ):
            problems.append(f"need {need.need_id}: an ambiguous need cannot be in TEST")
        if need.status == "draft":
            continue  # still being annotated: the evidence rules apply from 'reviewed' on
        langs = {q.lang for q in mine}
        problems += [
            f"need {need.need_id}: status {need.status!r} without a query in {lang!r}"
            for lang in get_args(Lang)
            if lang not in langs
        ]
        if need.answerability != "none":
            if not evidence:
                problems.append(
                    f"need {need.need_id}: answerability {need.answerability!r} without any "
                    f"non-ambiguous qrel of grade >= {RELEVANT_GRADE}"
                )
            # grade 3 answers the primary question (PROTOCOL.md v1.1, section 2): a need
            # resting on grade 2 alone has support but no answer, so it is not complete
            if need.answerability == "complete" and evidence and max(r.grade for r in evidence) < 3:
                problems.append(
                    f"need {need.need_id}: answerability 'complete' without a non-ambiguous "
                    f"qrel of grade 3 that answers its primary question"
                )
            evidence_acts = {r.act for r in evidence}
            if evidence and set(need.acts) != evidence_acts:
                problems.append(
                    f"need {need.need_id}: acts {sorted(need.acts)} but its evidence of grade "
                    f">= {RELEVANT_GRADE} is in {sorted(evidence_acts)}"
                )
    return problems


# ---------------------------------------------------------------------------
# adjudication records (PROTOCOL.md, section 12): how each label was decided
# ---------------------------------------------------------------------------

# Judge A annotates, judge B annotates without seeing A, and any disagreement goes to
# judge C, a fresh context that sees both rationales without knowing which judge was
# first or what the label used to be. The record lives next to the data files but
# outside the seal: sealing it would change the manifest contract, so instead it must
# agree with the sealed qrels, label by label.
ADJUDICATIONS_FILE = "adjudications.jsonl"
ANSWERABILITY_TARGET = "answerability"
NODE_REF_PATTERN = rf"^(?:{'|'.join(ACTS)}):{h.LTREE_PATH_PATTERN.removeprefix('^')}"
NodeRef = Annotated[str, Field(pattern=NODE_REF_PATTERN)]
# Articles whose nodes are definitions. A definition is grade 2 only when it is
# necessary to read the need or to decide whether a rule applies (section 2), so
# validate lists every definition graded 2 or 3 for the reviewer to check.
DEFINITIONS_ARTICLES: Mapping[str, str] = {"DORA": "3", "MiCA": "3", "PSD2": "4"}


class Judgment(BaseModel):
    """One judge's decision: on a candidate node (grade, or None to leave it
    unjudged, i.e. without a qrel) or on a need (answerability). A grade 0 names
    the provisions read to fix the node's scope: who it applies to, its conditions,
    exceptions and field, the node itself when it sets them."""

    model_config = STRICT

    grade: Grade | None = None
    ambiguous: bool = Field(default=False, strict=True)
    answerability: Answerability | None = None
    rationale: Text
    scope_basis: list[NodeRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.answerability is not None and (
            self.grade is not None or self.ambiguous or self.scope_basis
        ):
            raise ValueError("an answerability judgment carries no grade, flag or scope")
        if self.ambiguous and self.grade is None:
            raise ValueError("an ambiguous judgment still gives its best grade")
        if self.grade == 0 and not self.scope_basis:
            raise ValueError(
                "a grade 0 names the provisions read to fix the node's scope (scope_basis)"
            )
        return self

    @property
    def label(self) -> tuple[int | None, bool, str | None]:
        return (self.grade, self.ambiguous, self.answerability)

    @property
    def relevant(self) -> bool:
        return self.grade is not None and self.grade >= RELEVANT_GRADE and not self.ambiguous


class Adjudication(BaseModel):
    """How one label of one need was decided. There is no majority: when A and B
    disagree, C decides against the text, and may pick A's label, B's, another or
    ambiguous. An agreement goes to C only when it breaks an automatic check, when
    a later protocol version changes the rule it rests on, or when the methodological
    review finds that the protocol was applied differently from the rest of the
    benchmark, and then 'escalated' says which. The same field records that a fresh
    judge C decided again a label that an earlier C had decided, and the earlier
    decision is kept in its text; the record keeps A and B as they were and the
    latest C is the final label."""

    model_config = STRICT

    need_id: str = Field(pattern=NEED_ID_PATTERN)
    target: str = Field(pattern=rf"{NODE_REF_PATTERN}|^{ANSWERABILITY_TARGET}$")
    a: Judgment
    b: Judgment
    c: Judgment | None = None
    escalated: Text | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        where = f"{self.need_id} {self.target}"
        judges = [j for j in (self.a, self.b, self.c) if j is not None]
        on_need = self.target == ANSWERABILITY_TARGET
        if any((j.answerability is not None) != on_need for j in judges):
            raise ValueError(
                f"{where}: an answerability target takes answerability judgments, and a "
                f"node target takes grades"
            )
        if not self.agree and self.c is None:
            raise ValueError(
                f"{where}: judges A and B disagree, and a disagreement is never final "
                f"without judge C"
            )
        if self.c is not None and self.agree and self.escalated is None:
            raise ValueError(f"{where}: judge C on an agreement needs the reason ('escalated')")
        if self.escalated is not None and self.c is None:
            raise ValueError(f"{where}: an escalated agreement needs judge C")
        return self

    @property
    def agree(self) -> bool:
        return self.a.label == self.b.label

    @property
    def final(self) -> Judgment:
        return self.c if self.c is not None else self.a

    @property
    def crosses_threshold(self) -> bool:
        """A and B fall on different sides of the relevance line (grade >= 2)."""
        return self.a.relevant != self.b.relevant


def adjudication_problems(
    adjudications: Sequence[Adjudication],
    needs: Sequence[Need],
    qrels: Sequence[Qrel],
    require_all: bool,
) -> list[str]:
    """The record must say how every label came about, and agree with it: the final
    label of each target is the qrel (or no qrel when it is unjudged) and the need's
    answerability. With require_all, every reviewed or ready need is fully covered."""
    problems: list[str] = []
    keys = Counter((a.need_id, a.target) for a in adjudications)
    problems += [f"adjudication {k} appears {c} times" for k, c in keys.items() if c > 1]
    by_need = {n.need_id: n for n in needs}
    by_key = {(r.need_id, f"{r.act}:{r.path}"): r for r in qrels}
    for adj in adjudications:
        where = f"adjudication {adj.need_id} {adj.target}"
        final = adj.final
        if adj.need_id not in by_need:
            problems.append(f"{where}: unknown need")
        elif adj.target == ANSWERABILITY_TARGET:
            if final.answerability != by_need[adj.need_id].answerability:
                problems.append(
                    f"{where}: final {final.answerability!r}, the need says "
                    f"{by_need[adj.need_id].answerability!r}"
                )
        else:
            qrel = by_key.get((adj.need_id, adj.target))
            if final.grade is None and qrel is not None:
                problems.append(f"{where}: left unjudged, but a qrel exists")
            elif final.grade is not None and qrel is None:
                problems.append(f"{where}: final grade {final.grade}, but there is no qrel")
            elif qrel is not None and (qrel.grade, qrel.ambiguous) != (
                final.grade,
                final.ambiguous,
            ):
                problems.append(
                    f"{where}: final grade {final.grade} (ambiguous {final.ambiguous}), the "
                    f"qrel has {qrel.grade} (ambiguous {qrel.ambiguous})"
                )
    if require_all:
        for need in needs:
            if need.status == "draft":
                continue  # judged later: the record comes with the qrels
            if (need.need_id, ANSWERABILITY_TARGET) not in keys:
                problems.append(f"need {need.need_id}: no adjudication of its answerability")
            problems += [
                f"qrel {r.need_id} {r.act}:{r.path}: no adjudication record"
                for r in qrels
                if r.need_id == need.need_id and (r.need_id, f"{r.act}:{r.path}") not in keys
            ]
    return problems


def definition_qrels(qrels: Sequence[Qrel]) -> list[Qrel]:
    """Qrels of grade 2 or 3 on a definition: each must pass the necessity test."""
    return [
        r
        for r in qrels
        if r.grade >= RELEVANT_GRADE and r.path.split(".")[0] == DEFINITIONS_ARTICLES[r.act]
    ]


# ---------------------------------------------------------------------------
# the corpus boundary
# ---------------------------------------------------------------------------


def load_corpus(
    manifest_path: Path = dl.MANIFEST_PATH, processed_dir: Path = pr.PROCESSED_DIR
) -> dict[str, pr.ProcessedDocument]:
    """The processed layer of every act, through load_processed(): stale or broken
    layers abort before any judgment is compared against them."""
    manifest = dl.load_manifest(manifest_path)
    return {act: pr.load_processed(act, manifest, processed_dir) for act in ACTS}


def node_texts(docs: Mapping[str, pr.ProcessedDocument]) -> dict[NodeKey, str]:
    """(act, path) -> own text, for every node, empty ones included."""
    return {
        (act, node.path): node.own_text
        for act, doc in docs.items()
        for article in doc.articles
        for node in article.nodes
    }


def snapshot_problems(
    manifest: BenchmarkManifest, docs: Mapping[str, pr.ProcessedDocument]
) -> list[str]:
    """What separates the sealed snapshot from the processed layer in use."""
    problems: list[str] = []
    for act in ACTS:
        doc = docs.get(act)
        if doc is None:
            problems.append(f"{act}: no processed layer")
            continue
        if doc.parser_version != manifest.parser_version:
            problems.append(
                f"{act}: sealed with parser {manifest.parser_version}, the layer was written "
                f"by parser {doc.parser_version}"
            )
        sealed = manifest.corpus[act]
        if (doc.source_celex, doc.source_sha256) != (sealed.source_celex, sealed.source_sha256):
            problems.append(
                f"{act}: sealed against {sealed.source_celex} {sealed.source_sha256[:12]}, the "
                f"layer comes from {doc.source_celex} {doc.source_sha256[:12]}"
            )
    return problems


def qrel_problems(qrels: Sequence[Qrel], texts: Mapping[NodeKey, str]) -> list[str]:
    """Every judged node must exist, have text, and still have the text that was
    judged. Each problem names the need and the path to review."""
    problems: list[str] = []
    for r in qrels:
        where = f"need {r.need_id} {r.act}:{r.path}"
        text = texts.get(r.node)
        if text is None:
            problems.append(f"{where}: no such node in the processed layer")
        elif not text:
            problems.append(f"{where}: the node has no own text; judge one that has")
        elif sha256_text(text) != r.text_sha256:
            problems.append(f"{where}: the node text changed since it was judged")
        elif r.span is not None and r.span.end > len(text):
            problems.append(
                f"{where}: span ends at {r.span.end}, the text has {len(text)} characters"
            )
    return problems


def _fail(error: type[BenchmarkError], heading: str, problems: Sequence[str]) -> None:
    if problems:
        raise error(heading + "\n" + "\n".join(f"- {p}" for p in problems))


def adjudication_node_problems(
    adjudications: Sequence[Adjudication], texts: Mapping[NodeKey, str]
) -> list[str]:
    """Every node an adjudication names, as its target or in a scope check, exists."""
    problems: list[str] = []
    for adj in adjudications:
        refs = [adj.target] if adj.target != ANSWERABILITY_TARGET else []
        refs += [ref for j in (adj.a, adj.b, adj.c) if j is not None for ref in j.scope_basis]
        for ref in dict.fromkeys(refs):
            act, _, path = ref.partition(":")
            if (act, path) not in texts:
                problems.append(f"adjudication {adj.need_id} {adj.target}: no node {ref}")
    return problems


def check_corpus(benchmark: Benchmark, docs: Mapping[str, pr.ProcessedDocument]) -> None:
    """The third boundary, for a loaded benchmark. Needs the real corpus, so it runs
    locally (D-020); CI checks the first two boundaries only.

    A changed snapshot is reported with the qrels that the change actually broke, so
    that a reseal knows exactly what to review."""
    snapshot = snapshot_problems(benchmark.manifest, docs)
    texts = node_texts(docs)
    broken = qrel_problems(benchmark.qrels, texts)
    broken += adjudication_node_problems(benchmark.adjudications, texts)
    if snapshot:
        detail = broken or ["every qrel still matches: seal again with a new benchmark_version"]
        _fail(
            StaleBenchmarkError,
            f"{benchmark.directory} was sealed against another snapshot:",
            snapshot + detail,
        )
    _fail(
        StaleBenchmarkError,
        f"{benchmark.directory}: judged nodes no longer match the corpus:",
        broken,
    )


# ---------------------------------------------------------------------------
# the manifest boundary: load and seal
# ---------------------------------------------------------------------------


def _read_manifest(directory: Path) -> BenchmarkManifest:
    path = directory / MANIFEST_FILE
    if not path.exists():
        raise BenchmarkError(f"{path} does not exist: seal the benchmark first")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise BenchmarkError(f"{path} is not valid JSON: {e}") from e
    # the schema version first, as load_processed() does with the parser version
    version = raw.get("schema_version") if isinstance(raw, dict) else None
    if isinstance(version, str) and version != SCHEMA_VERSION:
        raise StaleBenchmarkError(
            f"{path} has schema_version {version} and this code reads {SCHEMA_VERSION}"
        )
    try:
        return BenchmarkManifest.model_validate(raw)
    except ValidationError as e:
        raise BenchmarkError(f"{path} does not meet the manifest contract.\n{e}") from e


def _read_records(directory: Path) -> tuple[list[Need], list[Query], list[Qrel]]:
    return (
        read_jsonl(directory / NEEDS_FILE, Need),
        read_jsonl(directory / QUERIES_FILE, Query),
        read_jsonl(directory / QRELS_FILE, Qrel),
    )


def _read_adjudications(directory: Path) -> list[Adjudication]:
    path = directory / ADJUDICATIONS_FILE
    return read_jsonl(path, Adjudication) if path.exists() else []


def _template_hashes(
    queries: Sequence[Query], templates_dir: Path
) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    problems: list[str] = []
    for template_id in sorted({q.template for q in queries if q.template is not None}):
        path = template_path(template_id, templates_dir)
        if path.exists():
            hashes[template_id] = sha256_file(path)
        else:
            problems.append(f"template {template_id}: {path} does not exist")
    return hashes, problems


def load_benchmark(
    directory: Path, templates_dir: Path = TEMPLATES_DIR, exclusions_path: Path | None = None
) -> Benchmark:
    """READ boundary for the files alone (no corpus needed): the seal, the hash of
    every file it covers, every record, and the rules across records. Use
    check_corpus() for the third boundary."""
    manifest = _read_manifest(directory)
    heading = f"{directory}: the files do not match the seal; validate them and seal again:"
    _fail(
        StaleBenchmarkError,
        heading,
        [
            f"{name}: changed since the seal"
            for name in DATA_FILES
            if not (directory / name).exists()
            or sha256_file(directory / name) != manifest.files[name]
        ],
    )
    needs, queries, qrels = _read_records(directory)
    hashes, missing = _template_hashes(queries, templates_dir)
    _fail(
        StaleBenchmarkError,
        heading,
        missing
        + [
            f"template {t}: changed since the seal"
            for t in sorted(set(hashes) | set(manifest.templates))
            if hashes.get(t) != manifest.templates.get(t)
        ],
    )
    excluded = _psd2_exclusions(exclusions_path, manifest.corpus["PSD2"])
    _fail(
        BenchmarkError,
        f"{directory} breaks the dataset rules:",
        dataset_problems(needs, queries, qrels, manifest.role, excluded),
    )
    adjudications = _read_adjudications(directory)
    _fail(
        BenchmarkError,
        f"{directory}: the adjudication record does not match the data:",
        adjudication_problems(
            adjudications, needs, qrels, require_all=(directory / ADJUDICATIONS_FILE).exists()
        ),
    )
    return Benchmark(directory, manifest, needs, queries, qrels, adjudications)


def _psd2_exclusions(path: Path | None, psd2: CorpusSource) -> frozenset[str]:
    """The excluded PSD2 articles, from a list built on the same PSD2 snapshot:
    a list derived from another version could miss the articles that matter."""
    exclusions = px.load_psd2_exclusions(path or px.PSD2_EXCLUSIONS_PATH)
    if (exclusions.base.celex, exclusions.base.sha256) != (psd2.source_celex, psd2.source_sha256):
        raise StaleBenchmarkError(
            f"the PSD2 exclusion list was built on {exclusions.base.celex} "
            f"{exclusions.base.sha256[:12]}, the benchmark uses {psd2.source_celex} "
            f"{psd2.source_sha256[:12]}: rebuild it with scripts/build_psd2_exclusions.py"
        )
    return exclusions.excluded()


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def seal(
    directory: Path,
    benchmark_version: str,
    role: Role,
    docs: Mapping[str, pr.ProcessedDocument],
    templates_dir: Path = TEMPLATES_DIR,
    exclusions_path: Path | None = None,
) -> BenchmarkManifest:
    """The only writer of manifest.json. It reads the files on disk, runs all three
    boundaries against the current processed layer, and writes nothing if any check
    fails. A stale qrel is never fixed here: it is reported, with its need and path.

    Returns the manifest in force. If neither the files nor the snapshot changed,
    nothing is written. If something changed, benchmark_version must grow."""
    if not re.fullmatch(pr.VERSION_PATTERN, benchmark_version):
        raise BenchmarkError(f"benchmark_version {benchmark_version!r} is not x.y.z")
    problems = [f"{act}: no processed layer" for act in ACTS if act not in docs]
    _fail(BenchmarkError, f"cannot seal {directory}:", problems)
    psd2 = CorpusSource(
        source_celex=docs["PSD2"].source_celex, source_sha256=docs["PSD2"].source_sha256
    )
    excluded = _psd2_exclusions(exclusions_path, psd2)
    needs, queries, qrels = _read_records(directory)
    hashes, missing = _template_hashes(queries, templates_dir)
    problems = dataset_problems(needs, queries, qrels, role, excluded) + missing
    adjudications = _read_adjudications(directory)
    problems += adjudication_problems(
        adjudications, needs, qrels, require_all=(directory / ADJUDICATIONS_FILE).exists()
    )
    parsers = {doc.parser_version for doc in docs.values()}
    if len(parsers) > 1:
        problems.append(f"the processed layer mixes parser versions {sorted(parsers)}")
    texts = node_texts(docs)
    problems += qrel_problems(qrels, texts) + adjudication_node_problems(adjudications, texts)
    _fail(BenchmarkError, f"cannot seal {directory}:", problems)

    manifest = BenchmarkManifest(
        schema_version=SCHEMA_VERSION,
        benchmark_version=benchmark_version,
        role=role,
        parser_version=parsers.pop(),
        corpus={
            act: CorpusSource(
                source_celex=docs[act].source_celex, source_sha256=docs[act].source_sha256
            )
            for act in ACTS
        },
        files={name: sha256_file(directory / name) for name in DATA_FILES},
        templates=hashes,
    )
    path = directory / MANIFEST_FILE
    if path.exists():
        current = _read_manifest(directory)
        if current.role != manifest.role:
            raise BenchmarkError(f"{path} has role {current.role!r}; a role never changes")
        unchanged = current.model_dump(exclude={"benchmark_version"}) == manifest.model_dump(
            exclude={"benchmark_version"}
        )
        if unchanged:
            return current
        if _version_tuple(benchmark_version) <= _version_tuple(current.benchmark_version):
            raise BenchmarkError(
                f"{path}: the files or the snapshot changed, so benchmark_version must be "
                f"greater than {current.benchmark_version}"
            )
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


# ---------------------------------------------------------------------------
# query variants: the prompt sees the need statement and nothing else
# ---------------------------------------------------------------------------

QUERY_TEMPLATE_ID = "query_variants.v1"
PROMPT_MARKER = "---PROMPT---"
# What a statement must not carry into the query prompt (PROTOCOL.md, step 1): an
# article, a paragraph or point, a structural unit, a node path or a CELEX number
# would tell the query writer where the answer is.
_LEGAL_REFERENCE = re.compile(
    r"\barts?\b\.?\s*\d|\barticles?\s*\d"  # Article 28, Art. 5, articles 3
    r"|\b\d+[a-z]?\s*\(\d+[a-z]?\)"  # 28(3), 35a(1)
    r"|\b(?:sub)?paragraphs?\s+\d|\bpoints?\s+\([a-z0-9]+\)"  # paragraph 2, point (a)
    r"|\b(?:title|chapter|section|annex|recital)s?\s+(?:[ivxlc]+|\d+)\b"  # Title III, Annex 1
    r"|\b(?:DORA|MiCA|PSD2):\S"  # a node path
    r"|\b[03]\d{4}[a-z]\d{4}\b",  # a CELEX number
    re.IGNORECASE,
)


def render_query_prompt(
    statement: str,
    variants: Sequence[tuple[str, str]],
    template_id: str = QUERY_TEMPLATE_ID,
    templates_dir: Path = TEMPLATES_DIR,
) -> str:
    """The prompt that asks an LLM for formulations of ONE need.

    It takes the statement as a plain string and the (lang, style) pairs wanted,
    so it cannot leak what it never receives: no Need fields such as the acts or
    the drivers, no node, no article text, no qrel, no answer. A statement that
    cites a legal reference is refused, since it would point at the answer.
    Spanish is natural only, as in the Query contract."""
    if not statement.strip():
        raise BenchmarkError("a query prompt needs a non-empty statement")
    reference = _LEGAL_REFERENCE.search(statement)
    if reference:
        raise BenchmarkError(
            f"the statement cites {reference.group(0)!r}: a need is written without legal "
            "references, so its query prompt cannot point at the answer"
        )
    for lang, style in variants:
        try:
            Query.model_validate(
                {
                    "query_id": f"N000-{lang}-1",
                    "need_id": "N000",
                    "lang": lang,
                    "style": style,
                    "origin": "human",
                    "text": "x",
                }
            )
        except ValidationError as e:
            raise BenchmarkError(f"formulation {lang} {style} is not allowed: {e}") from e
    if not variants or len(set(variants)) != len(variants):
        raise BenchmarkError("ask for at least one formulation, each once")
    text = template_path(template_id, templates_dir).read_text(encoding="utf-8")
    if text.count(PROMPT_MARKER) != 1:
        raise BenchmarkError(f"template {template_id} must contain {PROMPT_MARKER} exactly once")
    prompt = text.split(PROMPT_MARKER, 1)[1].strip()
    wanted = "\n".join(f"- {lang} {style}" for lang, style in variants)
    # variants first: a statement that happened to contain "{variants}" is left as it is
    return prompt.replace("{variants}", wanted).replace("{statement}", statement.strip()) + "\n"


# ---------------------------------------------------------------------------
# lexical overlap: a leakage diagnostic, never a filter
# ---------------------------------------------------------------------------

# How many consecutive query words shared with the evidence raise POSSIBLE COPY: a
# v1 heuristic (D-021), not a definition of leakage. The flag never rejects a query;
# a reviewer approves or rejects it with a reason. Terms of art such as "major
# ICT-related incident" are shorter.
COPY_RUN = 6
_WORD = re.compile(r"[^\W_]+(?:-[^\W_]+)*")
# English function words. Code stays in English (D-019), so Spanish queries keep
# their function words as content: against an English corpus their share is near
# zero anyway, and the diagnostic is read per language.
_STOPWORDS = frozenset(
    "a an the of to for in on at by and or is are be do does any its it from with what which "
    "how can under this that their our we has have will must as not no if when who whom".split()
)


@dataclass(frozen=True)
class Overlap:
    share: float  # content words of the query that appear in the evidence
    longest_run: int  # consecutive query words found verbatim in one evidence text

    @property
    def possible_copy(self) -> bool:
        return self.longest_run >= COPY_RUN


def _words(text: str) -> list[str]:
    return [w.casefold() for w in _WORD.findall(text)]


def lexical_overlap(query: str, evidence: Sequence[str]) -> Overlap:
    """How much a query shares with the text of its evidence (grade >= 2 nodes
    and their article headings). Reported per query to see whether retrieval
    depends on shared words; an expert may legitimately use official terms."""
    words = _words(query)
    content = {w for w in words if w not in _STOPWORDS}
    vocabulary = {w for text in evidence for w in _words(text)}
    joined = [" " + " ".join(_words(text)) + " " for text in evidence]
    longest = 0
    for start in range(len(words)):
        for end in range(len(words), start + longest, -1):
            run = " " + " ".join(words[start:end]) + " "
            if any(run in text for text in joined):
                longest = end - start
                break
    share = len(content & vocabulary) / len(content) if content else 0.0
    return Overlap(share, longest)


def evidence_texts(
    benchmark: Benchmark, need_id: str, docs: Mapping[str, pr.ProcessedDocument]
) -> list[str]:
    """The text a chunk would show for the evidence of a need: the own text of every
    non-ambiguous grade >= 2 node, and the heading of its article."""
    texts = node_texts(docs)
    titles = {(act, a.path): a.title for act, doc in docs.items() for a in doc.articles}
    out: list[str] = []
    for r in benchmark.qrels_of(need_id):
        if r.grade >= RELEVANT_GRADE and not r.ambiguous:
            article = r.path.split(".")[0]
            out.append(f"{r.act} Article {article}: {titles.get((r.act, article), '')}")
            out.append(texts.get(r.node, ""))
    return out
