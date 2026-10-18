"""The external benchmark of T4 (D-053): the anti-money-laundering and counter-terrorism
questions of JuriFindIT (Dalla Noce et al., Findings of EACL 2026), adapted to our evaluator
without touching what a retriever does.

Adapt the interface, not the algorithm. The adapter only renames, converts and separates:
- corpus.jsonl holds doc_id and text, and it is the only thing a retriever reads. The 159
  synthetic root nodes of the published file (content "<< file root node >>", path "/") are
  left out, as the paper counts its 23,458 articles; nothing else is removed or rewritten.
- documents.jsonl keeps each article's file_name and path, for the failure analysis only.
- queries/<name>.jsonl holds query_id and text, nothing else: never the topics, the area or
  a relevant id.
- qrels/<name>.jsonl holds the relevant doc ids, in its own directory, read only by the
  evaluation once every ranking is written.

A question id is its split and its row in the pinned file ("validation-0012"): the published
questions have no id of their own. The area filter is the question's own field
(normative_area), never its relevant articles. Every input file is checked against its
SHA-256 before it is read and every output is hashed into manifest.json; loading refuses a
file whose bytes changed. pyarrow (requirements-dev.lock.txt) reads the parquet files and is
imported only when they are read.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DATASET = "jurifindit/JuriFindIT"
REVISION = "d765a61998a0dc962a9c06394ab546b5a5ee0aa5"
LICENSE = "cc-by-nc-sa-4.0"
# path -> (bytes, SHA-256 of the LFS object), from the Hub's file tree at REVISION
FILES: dict[str, tuple[int, str]] = {
    "corpus/corpus.parquet": (
        18_544_036,
        "33632b14e48cd75b106112c9771f30e1c471f364c57fe95a76bbb641f0f3c524",
    ),
    "questions/train.parquet": (
        115_021,
        "6c31d10b680b58ac95d91ef6bb5a7b4faf9a1927c32b2d579b6f6d8e9c8345d9",
    ),
    "questions/validation.parquet": (
        33_219,
        "4c64549f76143411e555ee30bb787cf7ac2899ee4f6479cd6b855441e4adf8c6",
    ),
}
SPLITS = ("train", "validation")
# The dataset's own label of the area, in Italian: "anti-money laundering and anti-terrorism".
AREA = "Antiriciclaggio e Antiterrorismo"
# The runtime estimate of E uses questions of another area, never the evaluated one: the first
# ESTIMATE_QUESTIONS of ESTIMATE_AREA ("civil code") in the validation file, by row.
ESTIMATE_AREA = "Codice civile"
ESTIMATE_QUESTIONS = 3
ROOT_PATH = "/"
# The published file's spelling; the paper typesets it as "«file root node»".
ROOT_CONTENT = "<< file root node >>"
EXPECTED = {
    "corpus_rows": 23_617,
    "root_nodes": 159,
    "articles": 23_458,
    "questions_train": 716,
    "questions_validation": 179,
    "aml_train": 81,
    "aml_validation": 27,
}
ADAPTER_VERSION = "1"
DATA_DIR = Path(f"data/external/jurifindit@{REVISION[:8]}")
RAW = "raw"
ADAPTED = "adapted"
QUERY_SETS = ("aml-validation", "aml-train", "estimate")
QREL_SETS = ("aml-validation", "aml-train")
MANIFEST = "manifest.json"

STRICT = ConfigDict(extra="forbid", frozen=True)


class ExternalError(Exception):
    """The external benchmark is not the pinned one, or its files do not meet the adapter."""


class FileRecord(BaseModel):
    model_config = STRICT

    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AdaptManifest(BaseModel):
    model_config = STRICT

    schema_version: Literal["1"]
    dataset: str
    revision: str
    license: str
    adapter_version: str
    inputs: dict[str, FileRecord]
    outputs: dict[str, FileRecord]
    counts: dict[str, int]
    area: str
    estimate_area: str
    transformations: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def url_of(path: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{path}"


def verify_inputs(raw_dir: Path) -> None:
    """Every input file at its pinned size and SHA-256, or ExternalError."""
    problems = []
    for path, (size, sha) in FILES.items():
        local = raw_dir / path
        if not local.exists():
            problems.append(f"{path}: missing")
        elif local.stat().st_size != size or sha256_file(local) != sha:
            problems.append(f"{path}: not the file pinned at {REVISION[:12]}")
    if problems:
        raise ExternalError("the raw files are not the pinned ones:\n" + "\n".join(problems))


def download(raw_dir: Path) -> list[str]:
    """Fetches each pinned file that is missing or wrong, checks it, and returns what it
    fetched. A file that arrives with another size or SHA-256 is deleted, never kept."""
    fetched = []
    for path, (size, sha) in FILES.items():
        local = raw_dir / path
        if local.exists() and local.stat().st_size == size and sha256_file(local) == sha:
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + ".part")
        with urllib.request.urlopen(url_of(path), timeout=120) as response, partial.open("wb") as f:
            while block := response.read(1 << 20):
                f.write(block)
        if partial.stat().st_size != size or sha256_file(partial) != sha:
            partial.unlink()
            raise ExternalError(f"{path}: the download is not the pinned file")
        partial.replace(local)
        fetched.append(path)
    verify_inputs(raw_dir)
    return fetched


def read_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
    return rows


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def _write(out_dir: Path, name: str, data: bytes, outputs: dict[str, FileRecord]) -> None:
    path = out_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    outputs[name] = FileRecord(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def _check(counts: Mapping[str, int], expected: Mapping[str, int]) -> None:
    wrong = {k: (counts.get(k), v) for k, v in expected.items() if counts.get(k) != v}
    if wrong:
        raise ExternalError(
            "the adapted counts are not the published ones: "
            + ", ".join(f"{k} {have} (expected {want})" for k, (have, want) in wrong.items())
        )


def adapt_rows(
    corpus: Sequence[Mapping[str, Any]],
    questions: Mapping[str, Sequence[Mapping[str, Any]]],
    out_dir: Path,
    inputs: Mapping[str, FileRecord],
    expected: Mapping[str, int] = EXPECTED,
) -> AdaptManifest:
    """The adapted files from rows already read, so that the rules are testable without a
    parquet file. Deterministic: the same rows give the same bytes."""
    roots = [r for r in corpus if r["path"] == ROOT_PATH]
    if any(r["content"] != ROOT_CONTENT for r in roots):
        raise ExternalError("a row with the root path is not a synthetic root node")
    if any(r["content"] == ROOT_CONTENT and r["path"] != ROOT_PATH for r in corpus):
        raise ExternalError("a root node's content outside the root path")
    articles = sorted((r for r in corpus if r["path"] != ROOT_PATH), key=lambda r: int(r["id"]))
    ids = [str(r["id"]) for r in articles]
    if len(set(ids)) != len(ids):
        raise ExternalError("an article id appears twice")
    outputs: dict[str, FileRecord] = {}
    _write(
        out_dir,
        "corpus.jsonl",
        _jsonl({"doc_id": str(r["id"]), "text": r["content"]} for r in articles),
        outputs,
    )
    _write(
        out_dir,
        "documents.jsonl",
        _jsonl(
            {"doc_id": str(r["id"]), "file_name": r["file_name"], "path": r["path"]}
            for r in articles
        ),
        outputs,
    )
    counts = {"corpus_rows": len(corpus), "root_nodes": len(roots), "articles": len(articles)}
    estimate: list[dict[str, str]] = []
    for split in SPLITS:
        rows = questions[split]
        counts[f"questions_{split}"] = len(rows)
        selected = [
            (f"{split}-{i:04d}", r) for i, r in enumerate(rows) if r["normative_area"] == AREA
        ]
        counts[f"aml_{split}"] = len(selected)
        _write(
            out_dir,
            f"queries/aml-{split}.jsonl",
            _jsonl({"query_id": qid, "text": r["question"]} for qid, r in selected),
            outputs,
        )
        _write(
            out_dir,
            f"qrels/aml-{split}.jsonl",
            _jsonl(
                {
                    "query_id": qid,
                    "doc_ids": sorted({str(d) for d in r["relevant_doc_ids"]}, key=int),
                }
                for qid, r in selected
            ),
            outputs,
        )
        if split == "validation":
            estimate = [
                {"query_id": f"{split}-{i:04d}", "text": r["question"]}
                for i, r in enumerate(rows)
                if r["normative_area"] == ESTIMATE_AREA
            ][:ESTIMATE_QUESTIONS]
    counts["estimate"] = len(estimate)
    _write(out_dir, "queries/estimate.jsonl", _jsonl(estimate), outputs)
    _check(counts, {**expected, "estimate": ESTIMATE_QUESTIONS})
    manifest = AdaptManifest(
        schema_version="1",
        dataset=DATASET,
        revision=REVISION,
        license=LICENSE,
        adapter_version=ADAPTER_VERSION,
        inputs=dict(inputs),
        outputs=outputs,
        counts=counts,
        area=AREA,
        estimate_area=ESTIMATE_AREA,
        transformations=[
            "parquet rows to JSON lines, fields renamed: id to doc_id, content to text, "
            "question to text",
            f"the {counts['root_nodes']} synthetic root nodes (path '/') left out of the corpus",
            "doc ids as decimal text; articles in ascending id order",
            "question ids from the split and the row in the pinned file",
            f"questions selected by their own field normative_area == {AREA!r}",
            "relevant_doc_ids moved to qrels/, as decimal text, sorted, each once",
            "topics and normative_area not written to the queries",
            f"the runtime-estimate questions: the first {ESTIMATE_QUESTIONS} of "
            f"{ESTIMATE_AREA!r} in validation, with no qrels",
        ],
    )
    (out_dir / MANIFEST).write_text(manifest.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return manifest


def adapt(raw_dir: Path, out_dir: Path) -> AdaptManifest:
    """The pinned raw files, checked, then adapted into out_dir."""
    verify_inputs(raw_dir)
    inputs = {p: FileRecord(bytes=size, sha256=sha) for p, (size, sha) in FILES.items()}
    corpus = read_rows(raw_dir / "corpus/corpus.parquet")
    questions = {s: read_rows(raw_dir / f"questions/{s}.parquet") for s in SPLITS}
    return adapt_rows(corpus, questions, out_dir, inputs)


def load_manifest(out_dir: Path) -> AdaptManifest:
    """The manifest, with every output file checked against it."""
    manifest = AdaptManifest.model_validate_json((out_dir / MANIFEST).read_text(encoding="utf-8"))
    for name, record in manifest.outputs.items():
        data = (out_dir / name).read_bytes()
        if len(data) != record.bytes or hashlib.sha256(data).hexdigest() != record.sha256:
            raise ExternalError(f"{out_dir / name} is not the file the adapter wrote")
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_corpus(out_dir: Path) -> list[tuple[str, str]]:
    """(doc_id, text) of every article, in id order: all a retriever reads of the corpus."""
    load_manifest(out_dir)
    return [(r["doc_id"], r["text"]) for r in _read_jsonl(out_dir / "corpus.jsonl")]


def load_documents(out_dir: Path) -> dict[str, tuple[str, str]]:
    """doc_id -> (file_name, path): metadata for the analysis, never for retrieval."""
    load_manifest(out_dir)
    return {
        r["doc_id"]: (r["file_name"], r["path"]) for r in _read_jsonl(out_dir / "documents.jsonl")
    }


def load_queries(out_dir: Path, name: str) -> list[tuple[str, str]]:
    """(query_id, text) of one query set: no field a retriever must not see exists here."""
    if name not in QUERY_SETS:
        raise ExternalError(f"unknown query set {name!r}")
    load_manifest(out_dir)
    rows = _read_jsonl(out_dir / f"queries/{name}.jsonl")
    if any(set(r) != {"query_id", "text"} for r in rows):
        raise ExternalError(f"queries/{name}.jsonl holds fields besides query_id and text")
    return [(r["query_id"], r["text"]) for r in rows]


def load_qrels(out_dir: Path, name: str) -> dict[str, frozenset[str]]:
    """query_id -> relevant doc ids. For the evaluation only, after every run is written."""
    if name not in QREL_SETS:
        raise ExternalError(f"unknown qrels set {name!r}")
    load_manifest(out_dir)
    return {
        r["query_id"]: frozenset(r["doc_ids"]) for r in _read_jsonl(out_dir / f"qrels/{name}.jsonl")
    }
