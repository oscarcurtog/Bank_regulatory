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
from dataclasses import dataclass
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
# Properties of the EVIDENCE. Properties of a formulation (paraphrase, vocabulary,
# language) come from Query.lang, Query.style, Query.origin and the measured overlap.
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
            evidence_acts = {r.act for r in evidence}
            if evidence and set(need.acts) != evidence_acts:
                problems.append(
                    f"need {need.need_id}: acts {sorted(need.acts)} but its evidence of grade "
                    f">= {RELEVANT_GRADE} is in {sorted(evidence_acts)}"
                )
    return problems


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


def check_corpus(benchmark: Benchmark, docs: Mapping[str, pr.ProcessedDocument]) -> None:
    """The third boundary, for a loaded benchmark. Needs the real corpus, so it runs
    locally (D-020); CI checks the first two boundaries only.

    A changed snapshot is reported with the qrels that the change actually broke, so
    that a reseal knows exactly what to review."""
    snapshot = snapshot_problems(benchmark.manifest, docs)
    broken = qrel_problems(benchmark.qrels, node_texts(docs))
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
    return Benchmark(directory, manifest, needs, queries, qrels)


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
    parsers = {doc.parser_version for doc in docs.values()}
    if len(parsers) > 1:
        problems.append(f"the processed layer mixes parser versions {sorted(parsers)}")
    problems += qrel_problems(qrels, node_texts(docs))
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
