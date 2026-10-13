"""What every retriever of the retrieval iteration shares, whatever computes its scores
(config/retrieval/README.md):

- the ranking rule: every chunk of the layer, by score, best first, and by chunk id on a
  tie, as ingest/embeddings.rank_exact orders the dense models;
- the run file (data/runs/<system>-layer-<n>-dev-<version>/run.jsonl, not versioned): the
  whole ranking of every query, with its scores and whether the benchmark scores it. Its
  SHA-256 goes into the system's summary, and a run whose bytes no longer match is refused;
- weighted Reciprocal Rank Fusion, on ranks only: raw scores of different retrievers are
  never added;
- the diagnostics that compare runs with the labels: recall of a set of chunks, the oracle
  union of two top-k lists, and the dependent evidence of a need.

The labels enter here only through ingest/evaluation.py's projection (Target.chunk_grades),
and only to measure: nothing in this module decides a ranking from a label.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import download as dl
from ingest import evaluation as ev
from ingest.embeddings import Ranked, to_run

__all__ = [
    "DependentEvidence",
    "FrozenInputs",
    "Ranked",
    "RetrievalError",
    "dependent_evidence",
    "load_frozen_inputs",
    "load_system_run",
    "oracle_union_recall",
    "macro_by_need",
    "per_language",
    "rank_by_scores",
    "read_run",
    "recall_of",
    "relevant_chunks",
    "rrf",
    "to_run",
    "write_run",
]

RUNS_DIR = Path("data/runs")
RUN_FILE = "run.jsonl"


class RetrievalError(Exception):
    """A ranking, a run file or a fusion does not meet its contract."""


# ---------------------------------------------------------------------------
# rankings
# ---------------------------------------------------------------------------


def rank_by_scores(query_id: str, doc_ids: Sequence[str], scores: np.ndarray) -> Ranked:
    """Every document by score, best first, and by id on a tie."""
    if len(set(doc_ids)) != len(doc_ids):
        raise RetrievalError("a document id appears twice")
    if scores.shape != (len(doc_ids),):
        raise RetrievalError(f"{len(doc_ids)} documents and scores of shape {scores.shape}")
    id_order = np.empty(len(doc_ids), dtype=np.int64)
    id_order[sorted(range(len(doc_ids)), key=doc_ids.__getitem__)] = np.arange(len(doc_ids))
    order = np.lexsort((id_order, -scores))
    return Ranked(
        query_id, tuple(doc_ids[j] for j in order), tuple(float(scores[j]) for j in order)
    )


def rrf(
    query_id: str, rankings: Sequence[Sequence[str]], weights: Sequence[float], k: int
) -> Ranked:
    """Weighted Reciprocal Rank Fusion: score(c) = sum_i w_i / (k + rank_i(c)), ranks from 1.

    Every ranking must order the same documents, all of them: a document missing from one
    ranking would need a rank that does not exist, and inventing one is a silent choice."""
    if len(rankings) != len(weights) or not rankings:
        raise RetrievalError("one weight per ranking, and at least one ranking")
    if k <= 0 or any(w < 0 for w in weights):
        raise RetrievalError(f"k must be positive and weights non-negative: k={k}, {weights}")
    docs = sorted(rankings[0])
    for r in rankings:
        if len(r) != len(set(r)) or sorted(r) != docs:
            raise RetrievalError("RRF needs full rankings of the same documents, each once")
    fused = dict.fromkeys(docs, 0.0)
    for ranking, weight in zip(rankings, weights, strict=True):
        for position, doc in enumerate(ranking, start=1):
            fused[doc] += weight / (k + position)
    return rank_by_scores(query_id, docs, np.array([fused[d] for d in docs], dtype=np.float64))


# ---------------------------------------------------------------------------
# run files
# ---------------------------------------------------------------------------


def run_dir(system: str, max_tokens: int, benchmark_version: str, root: Path = RUNS_DIR) -> Path:
    return root / f"{system}-layer-{max_tokens}-dev-{benchmark_version}"


def write_run(path: Path, ranked: Sequence[Ranked], evaluable: Iterable[str]) -> str:
    """Writes every ranking, scored or not, and returns the file's SHA-256."""
    scored = set(evaluable)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in ranked:
            row = {
                "query_id": r.query_id,
                "evaluated": r.query_id in scored,
                "chunk_ids": list(r.chunk_ids),
                "scores": list(r.scores),
            }
            f.write(json.dumps(row) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(path: Path, sha256: str | None = None) -> tuple[list[Ranked], frozenset[str]]:
    """The rankings of a run file and the queries it marks as scored. With sha256, a file
    whose bytes differ from the recorded ones is refused: it is not the run that was measured."""
    if not path.exists():
        raise RetrievalError(f"{path}: no run file; run the system that writes it")
    data = path.read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise RetrievalError(f"{path} is not the run its summary recorded (SHA-256 differs)")
    ranked, evaluated = [], set()
    for line in data.decode("utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        ranked.append(Ranked(row["query_id"], tuple(row["chunk_ids"]), tuple(row["scores"])))
        if row["evaluated"]:
            evaluated.add(row["query_id"])
    return ranked, frozenset(evaluated)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def relevant_chunks(target: ev.Target) -> frozenset[str]:
    return frozenset(c for c, g in target.chunk_grades.items() if g >= bm.RELEVANT_GRADE)


def recall_of(found: Iterable[str], relevant: frozenset[str]) -> float:
    """Share of the relevant chunks among the found ones (binary relevance, grade >= 2)."""
    if not relevant:
        raise RetrievalError("recall needs at least one relevant chunk")
    return len(relevant & set(found)) / len(relevant)


@dataclass(frozen=True)
class DependentEvidence:
    """A need whose projected chunks hold an anchor (grade 3) and evidence the answer
    depends on without being the answer (grade 2): PROTOCOL.md section 2."""

    need_id: str
    anchors: frozenset[str]
    dependents: frozenset[str]


def dependent_evidence(targets: Iterable[ev.Target]) -> dict[str, DependentEvidence]:
    out = {}
    for t in targets:
        anchors = frozenset(c for c, g in t.chunk_grades.items() if g == 3)
        dependents = frozenset(c for c, g in t.chunk_grades.items() if g == 2)
        if anchors and dependents:
            out[t.need.need_id] = DependentEvidence(t.need.need_id, anchors, dependents)
    return out


def macro_by_need(values: Mapping[str, float], queries: Iterable[bm.Query]) -> float:
    """The mean of each need's queries, then the mean over needs: the 'global' value of
    the iteration (descriptive under D-020, since it mixes languages)."""
    by_need: dict[str, list[float]] = {}
    for q in queries:
        if q.query_id in values:
            by_need.setdefault(q.need_id, []).append(values[q.query_id])
    if not by_need:
        raise RetrievalError("no query to average")
    return statistics.fmean(statistics.fmean(v) for v in by_need.values())


def per_language(values: Mapping[str, float], queries: Iterable[bm.Query]) -> dict[str, float]:
    """D-020: the mean per (need, language), then per language over needs."""
    groups: dict[tuple[str, str], list[float]] = {}
    for q in queries:
        if q.query_id in values:
            groups.setdefault((q.need_id, q.lang), []).append(values[q.query_id])
    langs = sorted({lang for _, lang in groups})
    return {
        lang: statistics.fmean(statistics.fmean(v) for (_, la), v in groups.items() if la == lang)
        for lang in langs
    }


# ---------------------------------------------------------------------------
# the frozen inputs every experiment starts from
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenInputs:
    dev: bm.Benchmark
    docs: tuple[ch.ChunkedDocument, ...]
    layer: ev.ChunkLayer
    chunks: Mapping[str, ch.Chunk]  # chunk_id -> chunk, in layer order

    @property
    def targets(self) -> list[ev.Target]:
        return ev.targets(self.dev, self.layer)

    @property
    def evaluable(self) -> frozenset[str]:
        return frozenset(q.query_id for t in self.targets for q in t.queries)


def load_frozen_inputs(benchmark_dir: Path, expect_version: str, max_tokens: int) -> FrozenInputs:
    """The benchmark at the expected version and the chunk layer it was sealed against;
    anything else stops the experiment before a single score."""
    dev = bm.load_benchmark(benchmark_dir)
    if dev.manifest.benchmark_version != expect_version:
        raise RetrievalError(
            f"benchmark {dev.manifest.benchmark_version}, expected {expect_version}"
        )
    manifest = dl.load_manifest()
    docs = tuple(ch.load_chunks(act, manifest) for act in bm.ACTS)
    if {d.max_tokens for d in docs} != {max_tokens}:
        raise RetrievalError(f"the chunk layer was built at {sorted({d.max_tokens for d in docs})}")
    layer = ev.layer_from_documents(docs)
    ev.check_layer(dev.manifest, layer)
    chunks = {c.chunk_id: c for d in docs for c in d.chunks}
    return FrozenInputs(dev, docs, layer, chunks)


def oracle_union_recall(
    first: Sequence[str], second: Sequence[str], relevant: frozenset[str], k: int = 10
) -> float:
    """Share of the relevant chunks in the union of two top-k lists: a diagnostic of how
    much two retrievers complement each other, never a system (it sees up to 2k chunks)."""
    return recall_of(set(first[:k]) | set(second[:k]), relevant)


def load_system_run(
    system: str,
    max_tokens: int,
    benchmark_version: str,
    results: Path = Path("data/results"),
    runs: Path = RUNS_DIR,
) -> tuple[list[Ranked], dict[str, Any]]:
    """A system's run and summary; the run must be the one its summary recorded."""
    summary_path = (
        results / f"dev-{benchmark_version}" / f"{system}-layer-{max_tokens}" / "summary.json"
    )
    if not summary_path.exists():
        raise RetrievalError(f"{summary_path}: no summary; run the system first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sha = summary["identity"]["run_sha256"]
    ranked, _ = read_run(run_dir(system, max_tokens, benchmark_version, runs) / RUN_FILE, sha)
    return ranked, summary
