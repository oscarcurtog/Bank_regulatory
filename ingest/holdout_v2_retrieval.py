"""TEST v2's retrieval runs (D-067): the frozen systems pointed at a set of queries, nothing else.

The only change from DEV is the interface (S4 plan, section 2): the same library code that built
every DEV run, called on other queries, with the rankings written to a private folder instead of
data/runs/, and no evaluation at all. Nothing here reads a qrel, a benchmark's labels or a metric:
a ranking is produced, hashed and stored, and the pool is built from it later.

- E5, BGE-M3 and Qwen3-Embedding-0.6B: the cached vectors of the frozen layer (their identity
  checked as scripts/embed_baseline.py checks it), the queries encoded with the model's query
  prefix, exact ranking (em.rank_exact, ties by chunk id);
- BM25: the frozen index of scripts/bm25_baseline.py;
- E: D-4B's requirements (Qwen3-4B, greedy, the MLX worker of scripts/decomposition.py) from the
  question and E5's first three results; each requirement's first 50 E5 chunks reranked by
  Qwen3-Reranker-0.6B (scripts/requirement_reranking.py); D's rule (dc.assemble) unchanged.

Every system ranks the whole layer, as in DEV; the frozen file keeps each query's first
RUN_DEPTH chunks (ingest/evaluation.py MIN_RUN_DEPTH). The local models run one at a time, never
beside another heavy model: the runner calls the systems one at a time (the local resource
queue of D-067).

faithfulness() runs the same functions on DEV's queries and compares every ranking with the frozen
DEV run of the same system: the evidence that the interface changed and the systems did not.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import decomposition as dc
from ingest import download as dl
from ingest import embeddings as em
from ingest import evaluation as ev
from ingest import lexical as lx
from ingest import necessity as nc
from ingest import requirement_rerank as rq
from ingest import retrieval as rt

MAX_TOKENS = 500
RUN_DEPTH = ev.MIN_RUN_DEPTH  # the depth each frozen ranking keeps (100)
DENSE = {
    "E5": Path("config/embeddings/multilingual-e5-large.json"),
    "BGE-M3": Path("config/embeddings/bge-m3.json"),
    "Qwen3-Embedding-0.6B": Path("config/embeddings/qwen3-embedding-0.6b.json"),
}
BM25 = Path("config/retrieval/bm25.json")
DECOMPOSER = Path("config/retrieval/qwen3-4b-instruct-mlx.json")
RERANKER = Path("config/retrieval/qwen3-reranker-0.6b.json")
MLX_PYTHON = Path(".cache/mlx-venv/bin/python")
MAX_NEW_TOKENS = 256  # scripts/decomposition.py
SYSTEMS = ("E5", "BGE-M3", "Qwen3-Embedding-0.6B", "BM25", "E")
# the frozen DEV runs each system must reproduce on DEV's queries
DEV_RUNS = {
    "E5": "multilingual-e5-large",
    "BGE-M3": "bge-m3",
    "Qwen3-Embedding-0.6B": "qwen3-embedding-0.6b",
    "BM25": "bm25",
    "E": "e-d4b-qwen3-reranker-k5",
}
DEV_DECOMPOSITION = Path("data/runs/d-qwen3-4b-k5-layer-500-dev-0.10.0/outputs.jsonl")
DEV_RERANKER_SCORES = Path("data/runs/e-qwen3-reranker-0.6b-layer-500-dev-0.10.0/scores.jsonl")
CODE = (
    "ingest/holdout_v2_retrieval.py",
    "ingest/embeddings.py",
    "ingest/encoder.py",
    "ingest/lexical.py",
    "ingest/decomposition.py",
    "ingest/requirement_rerank.py",
    "ingest/necessity.py",
    "ingest/retrieval.py",
    "ingest/chunks.py",
    "scripts/mlx_generate.py",
)


class RetrievalV2Error(Exception):
    """A system cannot be run as frozen: nothing is ranked."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Layer:
    docs: tuple[ch.ChunkedDocument, ...]
    layer: ev.ChunkLayer
    chunks: dict[str, ch.Chunk]


def load_layer(max_tokens: int = MAX_TOKENS) -> Layer:
    docs = tuple(ch.load_chunks(act, dl.load_manifest()) for act in bm.ACTS)
    if {d.max_tokens for d in docs} != {max_tokens}:
        raise RetrievalV2Error(
            f"the chunk layer was built at {sorted({d.max_tokens for d in docs})}"
        )
    layer = ev.layer_from_documents(docs)
    return Layer(docs, layer, {c.chunk_id: c for d in docs for c in d.chunks})


@dataclass
class Run:
    system: str
    rankings: dict[str, list[str]]  # query id -> every chunk, best first
    refused: list[str] = field(default_factory=list)  # queries the system's contract refuses
    identity: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def lines(self) -> str:
        """The frozen file: each query's first RUN_DEPTH chunks, best first. Every metric, the
        pool and the diagnostics read 20 at most; the evaluator's floor is 100."""
        rows = [
            json.dumps(
                {"query_id": qid, "chunk_ids": self.rankings[qid][:RUN_DEPTH]}, ensure_ascii=False
            )
            for qid in sorted(self.rankings)
        ]
        return "".join(r + "\n" for r in rows)


# ---------------------------------------------------------------------------
# the dense systems
# ---------------------------------------------------------------------------


@dataclass
class Dense:
    config: em.EmbeddingConfig
    tokenizer: em.LoadedTokenizer
    doc_ids: tuple[str, ...]
    doc_vectors: np.ndarray
    identity: dict[str, Any]
    _encoder: list[Any] = field(default_factory=list)

    def encode(self, items: Sequence[tuple[str, str]]) -> tuple[list[str], np.ndarray, list[str]]:
        """Unit vectors of query texts, with the model's query prefix, exactly as the encoder
        computed the benchmark's queries. A query its contract refuses (over the model's input
        limit) is returned apart, never truncated."""
        kept, refused = [], []
        inputs = []
        for item_id, text in items:
            try:
                inputs += em.prepare_inputs(
                    [(item_id, text)], "query", self.tokenizer.tokenizer, self.config
                )
                kept.append(item_id)
            except em.TruncationRefusedError:
                refused.append(item_id)
        if not inputs:
            return kept, np.zeros((0, self.config.dimension), dtype=np.float32), refused
        if not self._encoder:
            from ingest.encoder import Encoder

            self._encoder.append(Encoder(self.config))
        vectors: np.ndarray = self._encoder[0].encode(inputs)
        return kept, vectors, refused

    def rank(self, items: Sequence[tuple[str, str]]) -> tuple[dict[str, list[str]], list[str]]:
        kept, vectors, refused = self.encode(items)
        ranked = em.rank_exact(kept, vectors, list(self.doc_ids), self.doc_vectors)
        return {r.query_id: list(r.chunk_ids) for r in ranked}, refused


def load_dense(config_path: Path, layer: Layer) -> Dense:
    """The cached vectors of the frozen layer, refused unless their identity is exactly the
    current one (the check of scripts/embed_baseline.py and ingest/dense.py)."""
    config = em.load_config(config_path)
    loaded = em.load_tokenizer(config)
    runtime = em.runtime_identity(config)
    identity = em.artifact_identity(config, loaded, runtime, em.layer_source(list(layer.docs)))
    meta, vectors = em.load_embedding_set(em.embedding_dir(config, f"layer-{MAX_TOKENS}"), identity)
    if list(meta.item_ids) != list(layer.chunks):
        raise RetrievalV2Error(f"{config.name}: the cached vectors are not in the layer's order")
    return Dense(
        config,
        loaded,
        tuple(meta.item_ids),
        vectors,
        {
            "config_file": str(config_path),
            "config_sha256": sha256_file(config_path),
            "config_hash": config.config_hash(),
            "artifact_identity": json.loads(identity.model_dump_json()),
            "vectors_sha256": em.matrix_sha256(vectors),
        },
    )


def dense_run(system: str, queries: Sequence[tuple[str, str]], layer: Layer) -> Run:
    start = time.perf_counter()
    dense = load_dense(DENSE[system], layer)
    rankings, refused = dense.rank(queries)
    return Run(system, rankings, refused, dense.identity, time.perf_counter() - start)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def bm25_run(queries: Sequence[tuple[str, str]], layer: Layer) -> Run:
    start = time.perf_counter()
    config = lx.load_config(BM25)
    versions = lx.check_versions(config)
    index = lx.build_index(config, [(cid, c.text) for cid, c in layer.chunks.items()])
    ranked = index.rank(list(queries))
    return Run(
        "BM25",
        {r.query_id: list(r.chunk_ids) for r in ranked},
        [],
        {
            "config_file": str(BM25),
            "config_sha256": sha256_file(BM25),
            "config_hash": config.config_hash(),
            "versions": versions,
        },
        time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# E
# ---------------------------------------------------------------------------


def decompose(
    rows: Sequence[Mapping[str, Any]],
    work: Path,
    worker: Callable[[Path, Path], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """D-4B's answers, by the MLX worker of scripts/decomposition.py: the prompts written as that
    script writes them, the worker run as it runs it, every prompt answered."""
    work.mkdir(parents=True, exist_ok=True)
    prompts, outputs = work / "prompts.jsonl", work / "outputs.jsonl"
    prompts.write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())
    if outputs.exists():
        outputs.unlink()
    (worker or _mlx_worker)(prompts, outputs)
    found = {r["id"]: r for r in map(json.loads, outputs.read_text(encoding="utf-8").splitlines())}
    missing = [r["id"] for r in rows if r["id"] not in found]
    if missing:
        raise RetrievalV2Error(f"{len(missing)} decomposition prompts have no output")
    return found


def _mlx_worker(prompts: Path, outputs: Path) -> None:
    if not MLX_PYTHON.exists():
        raise RetrievalV2Error(
            f"{MLX_PYTHON} is missing: the MLX environment of D-040 is not set up"
        )
    command = [str(MLX_PYTHON), "scripts/mlx_generate.py", "--config", str(DECOMPOSER)]
    command += ["--prompts", str(prompts), "--out", str(outputs)]
    command += ["--max-new-tokens", str(MAX_NEW_TOKENS)]
    subprocess.run(command, check=True, env=os.environ | {"HF_HOME": str(em.hf_home())})


def e_run(
    queries: Sequence[tuple[str, str]],
    layer: Layer,
    work: Path,
    e5: Run | None = None,
    answers: Mapping[str, str] | None = None,
    score_cache: Mapping[tuple[str, str], float] | None = None,
) -> Run:
    """E on the queries. answers (query id -> D-4B's output) and score_cache ((requirement,
    chunk) -> logit difference) let faithfulness() rebuild the DEV run from DEV's recorded
    answers and scores as scripts/requirement_reranking.py did, and check the regenerated ones
    against them; a TEST run gives neither and generates and scores everything."""
    start = time.perf_counter()
    dense = load_dense(DENSE["E5"], layer)
    if e5 is None:
        rankings, refused = dense.rank(queries)
        e5 = Run("E5", rankings, refused)
    question = {qid: text for qid, text in queries if qid in e5.rankings}
    if answers is None:
        rows = [
            {
                "id": f"d#{qid}",
                "messages": dc.requirements_prompt(
                    q, [layer.chunks[c].text for c in e5.rankings[qid][: dc.EXCERPTS]]
                ),
            }
            for qid, q in question.items()
        ]
        outputs = decompose(rows, work)
        answers = {qid: outputs[f"d#{qid}"]["output"] for qid in question}
    parsed = {qid: dc.parse(answers[qid]) for qid in question}
    texts = sorted({x for p in parsed.values() if p.readable for x in p.items})
    _, vectors, refused_reqs = dense.encode([(str(i), t) for i, t in enumerate(texts)])
    if refused_reqs:
        raise RetrievalV2Error("a requirement is over E5's input limit")
    by_text = dict(zip(texts, vectors, strict=True))
    doc_ids = list(dense.doc_ids)
    hits = {
        qid: [
            rt.rank_by_scores(qid, doc_ids, dense.doc_vectors @ by_text[x]).chunk_ids[: rq.DEPTH]
            for x in (p.items if p.readable else ())
        ]
        for qid, p in parsed.items()
    }
    jobs: dict[tuple[str, str], tuple[str, str, str]] = {}
    for qid, p in parsed.items():
        for x, ranking in zip(p.items if p.readable else (), hits[qid], strict=True):
            for c in ranking:
                jobs.setdefault((x, c), (rq.INSTRUCTION, x, layer.chunks[c].text))
    scores: dict[tuple[str, str], float] = dict(score_cache or {})
    pending = [k for k in jobs if k not in scores]
    if pending:
        config = nc.load_config(RERANKER)
        reranker = nc.NecessityReranker(config)
        groups: dict[str, list[tuple[str, str]]] = {}
        for k in pending:
            groups.setdefault(k[0], []).append(k)
        for part in groups.values():
            prompts = [reranker.prompt(*jobs[k]) for k in part]
            values = reranker.score_shared(prompts)
            scores |= {k: float(v) for k, v in zip(part, values, strict=True)}
    hits_rr = {
        qid: [
            tuple(rq.reorder(ranking, {c: scores[(x, c)] for c in ranking}))
            for x, ranking in zip(p.items if p.readable else (), hits[qid], strict=True)
        ]
        for qid, p in parsed.items()
    }
    rankings = {qid: list(dc.assemble(e5.rankings[qid], hits_rr[qid]).ranking) for qid in question}
    work.mkdir(parents=True, exist_ok=True)
    (work / "answers.json").write_text(json.dumps(answers, sort_keys=True, ensure_ascii=False))
    (work / "scores.jsonl").write_text(
        "".join(
            json.dumps(
                {"requirement": x, "chunk": c, "logit_diff": scores[(x, c)]}, ensure_ascii=False
            )
            + "\n"
            for x, c in sorted(jobs)
        )
    )
    return Run(
        "E",
        rankings,
        list(e5.refused),
        {
            "decomposer": {
                "config_file": str(DECOMPOSER),
                "config_sha256": sha256_file(DECOMPOSER),
            },
            "reranker": {"config_file": str(RERANKER), "config_sha256": sha256_file(RERANKER)},
            "requirement_retrieval": dense.identity,
            "answers_sha256": sha256_file(work / "answers.json"),
            "scores_sha256": sha256_file(work / "scores.jsonl"),
            "pairs_scored": len(pending),
        },
        time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# the run set, and its faithfulness on DEV
# ---------------------------------------------------------------------------


def identity() -> dict[str, Any]:
    """What the contract freezes of the retrieval: its code, its configs and the MLX worker."""
    files = [*CODE, *(str(p) for p in DENSE.values()), str(BM25), str(DECOMPOSER), str(RERANKER)]
    return {f: sha256_file(Path(f)) for f in sorted(files)}


def write_runs(runs: Mapping[str, Run], out: Path) -> dict[str, Any]:
    """Every run's file, written once, and the manifest that freezes them by hash."""
    out.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"schema": "holdout-v2-runs.v1", "code": identity(), "runs": {}}
    for system, run in sorted(runs.items()):
        path = out / f"{system}.jsonl"
        path.write_text(run.lines(), encoding="utf-8")
        manifest["runs"][system] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "queries": len(run.rankings),
            "refused": sorted(run.refused),
            "identity": run.identity,
            "seconds": round(run.seconds, 1),
        }
    text = json.dumps(manifest, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    (out / "manifest.json").write_text(text, encoding="utf-8")
    return manifest


def read_runs(out: Path, manifest: Mapping[str, Any]) -> dict[str, dict[str, list[str]]]:
    """The frozen rankings, each file checked against the manifest's hash first."""
    rankings: dict[str, dict[str, list[str]]] = {}
    for system, record in manifest["runs"].items():
        path = out / record["file"]
        if sha256_file(path) != record["sha256"]:
            raise RetrievalV2Error(f"{path} is not the frozen run")
        rankings[system] = {
            r["query_id"]: r["chunk_ids"] for r in map(json.loads, path.read_text().splitlines())
        }
    return rankings


def faithfulness(work: Path, systems: Sequence[str] = SYSTEMS) -> dict[str, Any]:
    """Each system run by this module on DEV's queries against its frozen DEV run: the same
    ranking for every query, or the differences counted. For E, the regenerated D-4B answers are
    compared with the recorded ones, the recomputed reranker scores with the recorded scores,
    and the ranking rebuilt from the recorded answers and scores with the frozen one."""
    layer = load_layer()
    dev = bm.load_benchmark(bm.DEV_DIR)
    queries = [(q.query_id, q.text) for q in dev.queries]
    frozen = {
        s: rt.read_run(
            rt.run_dir(DEV_RUNS[s], MAX_TOKENS, dev.manifest.benchmark_version) / rt.RUN_FILE
        )[0]
        for s in systems
    }
    report: dict[str, Any] = {}
    e5: Run | None = None
    for system in systems:
        expected = {r.query_id: list(r.chunk_ids) for r in frozen[system]}
        if system == "BM25":
            run = bm25_run(queries, layer)
        elif system == "E":
            recorded = {
                r["id"].split("#")[1]: r["output"]
                for r in map(json.loads, DEV_DECOMPOSITION.read_text().splitlines())
            }
            cached = {
                (r["requirement"], r["chunk"]): float(r["logit_diff"])
                for r in map(json.loads, DEV_RERANKER_SCORES.read_text().splitlines())
            }
            rebuilt = e_run(queries, layer, work / "E-recorded", e5, recorded, cached)
            regenerated = e_run(queries, layer, work / "E-regenerated", e5)
            answers = json.loads((work / "E-regenerated" / "answers.json").read_text())
            new_scores = {
                (r["requirement"], r["chunk"]): r["logit_diff"]
                for r in map(
                    json.loads, (work / "E-regenerated" / "scores.jsonl").read_text().splitlines()
                )
            }
            shared = [k for k in new_scores if k in cached]
            report["E-decomposition"] = {
                "queries": len(recorded),
                "identical_answers": sum(answers.get(q) == a for q, a in recorded.items()),
            }
            report["E-reranker"] = {
                "pairs_compared": len(shared),
                "max_abs_difference": max(
                    (abs(new_scores[k] - cached[k]) for k in shared), default=0.0
                ),
            }
            report["E-from-recorded"] = _compare(rebuilt.rankings, expected, rebuilt.seconds)
            run = regenerated
        else:
            run = dense_run(system, queries, layer)
            if system == "E5":
                e5 = run
        report[system] = _compare(run.rankings, expected, run.seconds)
        report[system]["refused"] = sorted(run.refused)
    return report


def _compare(
    got: Mapping[str, Sequence[str]], expected: Mapping[str, Sequence[str]], seconds: float
) -> dict[str, Any]:
    same = sum(list(got.get(q, [])) == list(r) for q, r in expected.items())
    top20 = sum(list(got.get(q, []))[:20] == list(r)[:20] for q, r in expected.items())
    return {
        "queries": len(expected),
        "identical_rankings": same,
        "identical_top20": top20,
        "seconds": round(seconds, 1),
    }
