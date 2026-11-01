"""Operational-cost harness of step S5.3: latency, memory, per-stage times, cold start and
reproducibility of retrieval pipelines, measured on one Apple M1, on the DEV queries.

The protocol is a private document whose SHA-256 is pinned in
config/retrieval/s5-s6-preregistration-a.json (stage A of the S5-S6 preregistration); this script
checks that pin before it measures anything, and every record it writes carries the hash.

No label is read here. It does not call load_benchmark, load_frozen_inputs or load_system_run,
which read qrels or metric summaries. It reads the queries' id, text and language, the chunk layer,
the stored vectors, and the stored rankings, whose SHA-256 it checks. Quality is never computed.

Subcommands:
    preflight   print the state of the machine against the protocol's preconditions
    smoke       run one system on the three warm-up queries only; nothing is recorded
    run         one system: pre-verification, its passes, validity, reproducibility
    cold        one cold start of one system (a fresh process, run by `drive`)
    drive       every system in the protocol's order, then the cold starts, then the report
    report      the summary, the Markdown report and the manifest of a results directory
"""

# ruff: noqa: E402
from __future__ import annotations

import time

START = time.perf_counter()  # the first thing that runs, for the cold start

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import cost_protocol as cp
from ingest import decomposition as dc
from ingest import download as dl
from ingest import embeddings as em
from ingest import lexical as lx
from ingest import necessity as nc
from ingest import requirement_rerank as rq
from ingest import retrieval as rt

PIN = Path("config/retrieval/s5-s6-preregistration-a.json")
# Plumbing tests only, never a measurement: S5_DRY_RUN_DIR (under .cache/) moves the results,
# skips the preconditions and the commit checks, and shortens the cooldown; S5_DRY_RUN_QUERIES
# keeps the first n queries of each pass. Every record made this way says `dry_run`.
DRY_RUN = os.environ.get("S5_DRY_RUN_DIR")
DRY_QUERIES = int(os.environ.get("S5_DRY_RUN_QUERIES", "0"))
if DRY_RUN and ".cache" not in Path(DRY_RUN).parts:
    raise SystemExit("S5_DRY_RUN_DIR must be under .cache/")
RESULTS = Path(DRY_RUN) if DRY_RUN else Path("data/results/dev-0.10.0/s5-cost-m1")
QUERIES = Path("data/benchmark/dev/queries.jsonl")
QUERIES_SHA256 = "8c1d859430139f3d40354fe0866c9e11d9c05e6458aba27969f1a9077d55f2c7"
RUNS = Path("data/runs")
E5 = Path("config/embeddings/multilingual-e5-large.json")
BM25 = Path("config/retrieval/bm25.json")
RERANKER = Path("config/retrieval/qwen3-reranker-0.6b.json")
DECOMPOSER = Path("config/retrieval/qwen3-4b-instruct-mlx.json")
MLX_PYTHON = Path(".cache/mlx-venv/bin/python")
MAX_TOKENS = 500
MAX_NEW_TOKENS = 256

STORED = {  # system -> (run directory name, SHA-256 of its run.jsonl)
    "S1": (
        "multilingual-e5-large",
        "b4880d4977fb242757849f4b8a20550f566fbad0fd5eda041505e28de83b9921",
    ),
    "S1p": (
        "pgvector-multilingual-e5-large",
        "2fa9a1f42cfe65e816e1aa0a1d049468f5c18ff5728c8943bd857feeab0ebc2e",
    ),
    "S2": ("bm25", "49b9d78fd41c12205e07a878ffeb2260d1110abd5ad99464b2d5f0e282c7aa76"),
    "S4": (
        "d-qwen3-4b-k5",
        "ccdf3e111731c78c19e2bcd616e6c46896527c0cdf8530547df88d9a3ee15317",
    ),
    "S5": (
        "e-d4b-qwen3-reranker-k5",
        "8ee79746f75e817320228e004344fb6eb09d5fe97fdcb815891834197b7ef521",
    ),
}
D_OUTPUTS_SHA256 = "0b5f139df9a9ca256a99a8baaa5748296da6dca034f36b1e3c388f937a9a3aef"
ORDER = ("S2", "S1", "S3", "S1p", "S6a", "S4", "S6b", "S5")
HEAVY = frozenset({"S4", "S5", "S6a", "S6b"})
FAST_PASSES, HEAVY_PASSES = 5, 2
COOLDOWN_SECONDS = 1 if DRY_RUN else 600
GRACE_SECONDS = 600
VERIFY_QUERIES = 5


# ---------------------------------------------------------------------------
# the machine
# ---------------------------------------------------------------------------


def out_of(command: Sequence[str]) -> str:
    done = subprocess.run(list(command), capture_output=True, text=True, check=False)
    return done.stdout + done.stderr


def snapshot() -> dict[str, Any]:
    return {
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "batt": out_of(["pmset", "-g", "batt"]),
        "pmset": out_of(["pmset", "-g"]),
        "therm": out_of(["pmset", "-g", "therm"]),
        "uptime": out_of(["uptime"]).strip(),
        "busiest": out_of(["ps", "-Ao", "pcpu,comm", "-r"]).splitlines()[:6],
        "docker_running": subprocess.run(
            ["pgrep", "-f", "Docker Desktop"], capture_output=True, check=False
        ).returncode
        == 0,
    }


def failures(snap: dict[str, Any], docker_allowed: bool) -> list[str]:
    found = []
    if not cp.on_ac_power(snap["batt"]):
        found.append("not on AC power")
    if not cp.low_power_mode_off(snap["pmset"]):
        found.append("low power mode is on")
    if not cp.thermal_ok(snap["therm"]):
        found.append("thermal warning or CPU limit")
    if not cp.load_ok(snap["uptime"]):
        found.append(f"load {cp.one_minute_load(snap['uptime'])} above {cp.LOAD_LIMIT}")
    if snap["docker_running"] and not docker_allowed:
        found.append("Docker Desktop is running")
    return found


def preconditions(
    docker_allowed: bool, grace: int = GRACE_SECONDS
) -> tuple[dict[str, Any], list[str]]:
    """The protocol's preconditions of a pass. Only the load average may wait (up to `grace`
    seconds, one reading every 30); power, low power mode, heat and Docker do not."""
    if DRY_RUN:
        return snapshot(), []
    deadline = time.monotonic() + grace
    while True:
        snap = snapshot()
        found = failures(snap, docker_allowed)
        only_load = found and all(f.startswith("load ") for f in found)
        if not only_load or time.monotonic() >= deadline:
            return snap, found
        time.sleep(30)


PINNED = {  # Appendix B of the protocol: configurations, measured code, other DEV files
    "config/embeddings/multilingual-e5-large.json": "2368eadd6b41b5874fe1be6c0a89bcd181e25ae07a1a8d155fd787af71d332b8",  # noqa: E501
    "config/retrieval/bm25.json": "65597fe35299a73b4eba0e3f160bd612cfec44ffb67faaddf775f56097f9088b",  # noqa: E501
    "config/retrieval/qwen3-reranker-0.6b.json": "8e1b4cd43eb106055166b3320e9550d81adcba4e078dc13bb4a09a7f08c20bc6",  # noqa: E501
    "config/retrieval/qwen3-4b-instruct-mlx.json": "9f8b638720508a169ae5d23edf35db9dc5e4005831ca4162a34c69cb02a747c5",  # noqa: E501
    "config/retrieval/decomposition.md": "46f8ad65ed222b644aa939893ef05fbb606afd30eb2562b2aaf5a1e8e1ca1729",  # noqa: E501
    "config/retrieval/requirement-reranking.md": "bc5371709842007a00127ca8d689a02e9c39f5bda149125566b3dd0968f2acc0",  # noqa: E501
    "ingest/decomposition.py": "5bb0a1069ad7e78f3ad7ee3bc438684b31e658fd3be5934264b1cb8cfad1296a",
    "ingest/requirement_rerank.py": "3237458b68c22c57ccb2890029b28715efba73f88d06034445ba7813609562e8",  # noqa: E501
    "ingest/necessity.py": "e520713d68ffe23b985fcc5d09a730a462bf39e232b8d839a1237dede5400814",
    "ingest/dense.py": "450b3aebf94c8e1e6f08ba98206748fbb4dbfb5382fcd9ba49563fa028a12ec8",
    "ingest/lexical.py": "e82813f8ebbc4794b9e22a3a8a002d56f7fbad181ad532980d5bc3044371f64d",
    "ingest/embeddings.py": "749d285b68a5d013ed3c6de58a2df37302a8f9600e8698173b249bfce42849e6",
    "ingest/encoder.py": "92f6127da8e2ff99d2514bc9852a5407c5dfe22b241533c2a42c7c31a3dec881",
    "ingest/retrieval.py": "d7b4479b070a5b19073e4825c072a47dd1f2564d4e0e4a402331444b519d1c27",
    "scripts/mlx_generate.py": "41a462498ecd1fe57cd1a5306a7bc9e5cc778c362e4383349f92738ae2066dbb",
    "scripts/decomposition.py": "fe59f92e81ab78bc808b4d4d03abadd9863bd46af245ff190da144f28e607e3b",
    "scripts/requirement_reranking.py": "3c29009726749ca9d558ecc5ed925a554dcb5f07918e52a91e4b5deeb74f5414",  # noqa: E501
    "compose.yaml": "a02f2afee2441f3a39c886ba7147e28c5f066c39e74d74cb0c1b27517179ecda",
    "data/runs/d-qwen3-4b-k5-layer-500-dev-0.10.0/prompts.jsonl": "db8276317c6feb85d7c3aa1124e8ce023dedb3b64e56ddc5ac1f40f80a014a4d",  # noqa: E501
    "data/benchmark/dev/manifest.json": "c845b684035dac91dec399604e6623a60c67a9564afba975f177b5d6482fb917",  # noqa: E501
}
HARNESS = ("scripts/s5_cost.py", "scripts/s5_mlx_worker.py", "ingest/cost_protocol.py")


def verify_pins() -> list[str]:
    """Every file the protocol pins, by SHA-256. Returns what is missing or different."""
    wrong = []
    for name, expected in PINNED.items():
        path = Path(name)
        if not path.exists():
            wrong.append(f"{name}: missing")
        elif cp.sha256_file(path) != expected:
            wrong.append(f"{name}: hash differs from the protocol")
    return wrong


def environment(layer_sha256: str) -> dict[str, Any]:
    """The stable part of the environment: what must be identical in every system's run."""
    import importlib.metadata as md

    def version(name: str) -> str:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return "not installed"

    def revision(path: str) -> dict[str, Any]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {k: data[k] for k in ("model_id", "revision") if k in data}

    return {
        "platform": platform.platform(),
        "machine": out_of(["sysctl", "-n", "hw.model"]).strip(),
        "chip": out_of(["sysctl", "-n", "machdep.cpu.brand_string"]).strip(),
        "memory_bytes": int(out_of(["sysctl", "-n", "hw.memsize"]).strip() or 0),
        "macos": out_of(["sw_vers"]).split(),
        "python": platform.python_version(),
        "packages": {
            n: version(n)
            for n in (
                "torch",
                "transformers",
                "numpy",
                "bm25s",
                "PyStemmer",
                "psycopg",
                "tokenizers",
            )
        },
        "mlx_environment": out_of([str(MLX_PYTHON), "-m", "pip", "list", "--format=freeze"])
        .strip()
        .splitlines()
        if MLX_PYTHON.exists()
        else [],
        "commit": out_of(["git", "rev-parse", "HEAD"]).strip(),
        "layer_sha256": layer_sha256,
        "queries_sha256": QUERIES_SHA256,
        "pinned_files": PINNED,
        "models": {
            "embedding": revision(str(E5)),
            "reranker": revision(str(RERANKER)),
            "decomposer": revision(str(DECOMPOSER)),
        },
        "seeds": {"base": cp.BASE_SEED, "repeat_offset": cp.REPEAT_OFFSET},
    }


def clean_tree() -> bool:
    """No tracked file changed since the commit: the measured code is the committed code."""
    return not out_of(["git", "status", "--porcelain", "--untracked-files=no"]).strip()


def guard(protocol_path: Path) -> tuple[str, str]:
    """What must hold before anything is measured: the harness is committed and pushed, the tree
    is clean, the protocol is the sealed one and the measured code and configurations are the
    pinned ones. Returns the protocol hash and the commit."""
    if DRY_RUN:
        return protocol_sha(protocol_path), "dry-run"
    for name in HARNESS:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", name], capture_output=True, check=False
        )
        if tracked.returncode != 0:
            raise SystemExit(f"{name} is not committed: commit the harness first")
    if not clean_tree():
        raise SystemExit("tracked files changed since the commit: commit the harness first")
    ahead = out_of(["git", "rev-list", "--count", "@{u}..HEAD"]).strip()
    if ahead != "0":
        raise SystemExit(f"the commit is not public yet ({ahead} commits ahead of origin)")
    protocol = protocol_sha(protocol_path)
    wrong = verify_pins()
    if wrong:
        raise SystemExit("pinned files differ from the protocol:\n  " + "\n  ".join(wrong))
    return protocol, out_of(["git", "rev-parse", "HEAD"]).strip()


def protocol_sha(protocol: Path) -> str:
    digest = cp.sha256_file(protocol)
    pinned = json.loads(PIN.read_text(encoding="utf-8"))["document_sha256"]
    if digest != pinned:
        raise SystemExit(f"{protocol} hashes to {digest}, the public pin says {pinned}")
    return digest


class Monitor(threading.Thread):
    """Watches the machine while a system is measured. Once a second it samples the resident
    memory of this process tree (a coarse, secondary observation, never a peak) and notes a jump
    between the wall clock and the monotonic clock, which only a suspension of the machine
    makes. Every 30 seconds it checks that the Mac is still on AC power. Events carry the
    monotonic time, so that a pass can ask whether anything happened while it ran."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.maximum_kb = 0
        self.events: list[tuple[float, str]] = []

    def run(self) -> None:
        last_wall, last_mono = time.time(), time.monotonic()
        tick = 0
        while not self._stop_event.is_set():
            listing = out_of(["ps", "-axo", "pid=,ppid=,rss="])
            self.maximum_kb = max(self.maximum_kb, cp.descendants_rss(listing, os.getpid()))
            wall, mono = time.time(), time.monotonic()
            if (wall - last_wall) - (mono - last_mono) > 5.0:
                self.events.append((mono, "the machine was suspended"))
            last_wall, last_mono = wall, mono
            tick += 1
            if tick % 30 == 0 and not cp.on_ac_power(out_of(["pmset", "-g", "batt"])):
                self.events.append((mono, "AC power was lost"))
            self._stop_event.wait(1.0)

    def since(self, mark: float) -> list[str]:
        return [what for at, what in self.events if at >= mark]

    def finish(self) -> int:
        self._stop_event.set()
        self.join(timeout=5)
        return self.maximum_kb


MONITOR: Monitor | None = None


# ---------------------------------------------------------------------------
# label-free inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Q:
    query_id: str
    text: str
    lang: str


def load_queries() -> dict[str, Q]:
    data = QUERIES.read_bytes()
    if hashlib.sha256(data).hexdigest() != QUERIES_SHA256:
        raise SystemExit(f"{QUERIES} is not the file the protocol pins")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    queries = {r["query_id"]: Q(r["query_id"], r["text"], r["lang"]) for r in rows}
    if len(queries) != 90:
        raise SystemExit(f"{len(queries)} queries, the protocol says 90")
    for text in cp.WARMUP_QUERIES:
        if text in {q.text for q in queries.values()}:
            raise SystemExit("a warm-up query is a DEV query")
    return queries


def read_stored(system: str) -> dict[str, tuple[str, ...]]:
    """query id -> the stored ranking, from a run file checked by its SHA-256. Only the query id
    and the order of the chunks are read; the run's `evaluated` flag is ignored."""
    name, sha = STORED[system]
    path = RUNS / f"{name}-layer-{MAX_TOKENS}-dev-0.10.0" / rt.RUN_FILE
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise SystemExit(f"{path} is not the run the protocol pins")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    return {r["query_id"]: tuple(r["chunk_ids"]) for r in rows}


def read_stored_generations() -> dict[str, str]:
    path = RUNS / f"d-qwen3-4b-k5-layer-{MAX_TOKENS}-dev-0.10.0" / "outputs.jsonl"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != D_OUTPUTS_SHA256:
        raise SystemExit(f"{path} is not the file the protocol pins")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    return {r["id"].split("#")[1]: r["output"] for r in rows}


class MlxWorker:
    """The persistent MLX process of scripts/s5_mlx_worker.py, started under /usr/bin/time -l so
    that its peak memory footprint is reported when it ends."""

    def __init__(self, time_file: Path) -> None:
        if not MLX_PYTHON.exists():
            raise SystemExit(f"{MLX_PYTHON} is missing: the MLX environment of D-040 is not set up")
        command = ["/usr/bin/time", "-l", "-o", str(time_file), str(MLX_PYTHON)]
        command += ["scripts/s5_mlx_worker.py", "--config", str(DECOMPOSER)]
        command += ["--max-new-tokens", str(MAX_NEW_TOKENS)]
        self.time_file = time_file
        self.process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=os.environ | {"HF_HOME": str(em.hf_home())},
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        first = json.loads(self.process.stdout.readline())
        if not first.get("ready"):
            raise SystemExit("the MLX worker did not start")
        self.load_seconds: float = first["load_seconds"]
        self.versions: dict[str, Any] = first["versions"]

    def generate(self, request_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps({"id": request_id, "messages": messages}) + "\n")
        self.process.stdin.flush()
        answer: dict[str, Any] = json.loads(self.process.stdout.readline())
        return answer

    def close(self) -> dict[str, int]:
        assert self.process.stdin is not None
        self.process.stdin.close()
        self.process.wait(timeout=60)
        return cp.parse_time_l(self.time_file.read_text(encoding="utf-8"))


class Context:
    """Everything a pipeline needs, loaded once and never timed."""

    def __init__(self, system: str, work: Path) -> None:
        self.system = system
        self.timings: dict[str, float] = {}
        t0 = time.perf_counter()
        self.queries = load_queries()
        manifest = dl.load_manifest()
        docs = tuple(ch.load_chunks(act, manifest) for act in bm.ACTS)
        if {d.max_tokens for d in docs} != {MAX_TOKENS}:
            raise SystemExit("the chunk layer is not the 500-token layer")
        self.chunks = {c.chunk_id: c.text for d in docs for c in d.chunks}
        self.doc_ids = tuple(self.chunks)
        self.layer_sha256 = hashlib.sha256(
            "\n".join(f"{c}\t{t}" for c, t in self.chunks.items()).encode()
        ).hexdigest()
        self.timings["layer"] = time.perf_counter() - t0
        self.encoder: Any = None
        self.doc_vectors: np.ndarray | None = None
        self.bm25: lx.BM25Index | None = None
        self.reranker: nc.NecessityReranker | None = None
        self.worker: MlxWorker | None = None
        self.pg: tuple[Any, int] | None = None
        self.pg_error = ""
        if system != "S2":
            self._load_dense(docs)
        if system in {"S2", "S3"}:
            t0 = time.perf_counter()
            config = lx.load_config(BM25)
            self.bm25 = lx.build_index(config, [(cid, text) for cid, text in self.chunks.items()])
            self.timings["bm25"] = time.perf_counter() - t0
        if system in {"S5", "S6a", "S6b"}:
            t0 = time.perf_counter()
            self.reranker = nc.NecessityReranker(nc.load_config(RERANKER))
            self.timings["reranker"] = time.perf_counter() - t0
        if system in {"S4", "S5"}:
            t0 = time.perf_counter()
            self.worker = MlxWorker(work / f"worker-time-{system}.txt")
            self.timings["mlx_worker"] = time.perf_counter() - t0
        if system == "S1p":
            self._connect_pg(docs)

    def _load_dense(self, docs: tuple[ch.ChunkedDocument, ...]) -> None:
        t0 = time.perf_counter()
        config = em.load_config(E5)
        self.config = config
        self.loaded = em.load_tokenizer(config)
        runtime = em.runtime_identity(config)
        identity = em.artifact_identity(config, self.loaded, runtime, em.layer_source(list(docs)))
        meta, vectors = em.load_embedding_set(
            em.embedding_dir(config, f"layer-{MAX_TOKENS}"), identity
        )
        if list(meta.item_ids) != list(self.chunks):
            raise SystemExit("the stored vectors are not in the layer's order")
        self.meta, self.doc_vectors = meta, vectors
        from ingest.encoder import Encoder

        self.encoder = Encoder(config)
        self.timings["e5"] = time.perf_counter() - t0

    def _connect_pg(self, docs: tuple[ch.ChunkedDocument, ...]) -> None:
        t0 = time.perf_counter()
        try:
            from ingest import vector_index as vi

            conn = vi.connect()
            vi.create_schema(conn)
            layer_id = vi.load_layer(conn, docs)
            assert self.doc_vectors is not None
            set_id = vi.load_embedding_set(conn, layer_id, self.meta, self.doc_vectors)
            self.pg = (conn, set_id)
        except Exception as error:  # noqa: BLE001 - recorded as `not measured`, never swapped
            self.pg_error = f"{type(error).__name__}: {error}"
        self.timings["pgvector"] = time.perf_counter() - t0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        inputs = em.prepare_inputs(
            [(str(i), t) for i, t in enumerate(texts)], "query", self.loaded.tokenizer, self.config
        )
        vectors: np.ndarray = self.encoder.encode(inputs)
        return vectors

    def sync(self) -> None:
        import torch

        torch.mps.synchronize()

    def close(self) -> dict[str, int]:
        memory: dict[str, int] = {}
        if self.worker is not None:
            memory = self.worker.close()
        if self.pg is not None:
            self.pg[0].close()
        return memory


# ---------------------------------------------------------------------------
# pipelines
# ---------------------------------------------------------------------------


class Stages:
    def __init__(self) -> None:
        self.ns: dict[str, int] = {}

    @contextmanager
    def timed(self, name: str):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.ns[name] = self.ns.get(name, 0) + time.perf_counter_ns() - t0


@dataclass
class Out:
    top: tuple[str, ...]
    ranking: tuple[str, ...] | None = None
    scores: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def e5_rank(
    ctx: Context, qid: str, text: str, st: Stages, name: str = "e5_rank"
) -> tuple[str, ...]:
    assert ctx.doc_vectors is not None
    with st.timed(name):
        vector = ctx.encode([text])[0]
        ranked = rt.rank_by_scores(qid, ctx.doc_ids, ctx.doc_vectors @ vector)
    return tuple(ranked.chunk_ids)


def s1(ctx: Context, qid: str, text: str, st: Stages) -> Out:
    assert ctx.doc_vectors is not None
    with st.timed("encode"):
        vector = ctx.encode([text])[0]
    with st.timed("search"):
        ranked = rt.rank_by_scores(qid, ctx.doc_ids, ctx.doc_vectors @ vector)
    return Out(tuple(ranked.chunk_ids[: cp.TOP]), tuple(ranked.chunk_ids))


def s1p(ctx: Context, qid: str, text: str, st: Stages) -> Out:
    from ingest import vector_index as vi

    assert ctx.pg is not None
    conn, set_id = ctx.pg
    with st.timed("encode"):
        vector = ctx.encode([text])[0]
    with st.timed("sql"):
        rows = vi.search(conn, set_id, vector, cp.TOP)
    return Out(tuple(c for c, _ in rows))


def s2(ctx: Context, qid: str, text: str, st: Stages) -> Out:
    assert ctx.bm25 is not None
    with st.timed("bm25"):
        ranked = rt.rank_by_scores(qid, ctx.bm25.doc_ids, ctx.bm25.scores(text))
    return Out(tuple(ranked.chunk_ids[: cp.TOP]), tuple(ranked.chunk_ids))


def s3(ctx: Context, qid: str, text: str, st: Stages) -> Out:
    assert ctx.bm25 is not None
    dense = e5_rank(ctx, qid, text, st, "e5")
    with st.timed("bm25"):
        lexical = rt.rank_by_scores(qid, ctx.bm25.doc_ids, ctx.bm25.scores(text)).chunk_ids
    with st.timed("fuse"):
        top = cp.rrf_fuse(dense, lexical)
    return Out(tuple(top))


def requirements(
    ctx: Context, qid: str, text: str, st: Stages, depth: int
) -> tuple[tuple[str, ...], dc.Requirements, list[tuple[str, ...]], dict[str, Any]]:
    """Steps shared by D and E: E5's ranking, the prompt, the generation, the parse and E5 over
    each readable requirement (its first `depth` chunks)."""
    assert ctx.worker is not None and ctx.doc_vectors is not None
    ranking = e5_rank(ctx, qid, text, st)
    with st.timed("prompt"):
        excerpts = [ctx.chunks[c] for c in ranking[: dc.EXCERPTS]]
        messages = dc.requirements_prompt(text, excerpts)
    with st.timed("llm"):
        answer = ctx.worker.generate(f"d#{qid}", messages)
    with st.timed("parse"):
        parsed = dc.parse(answer["output"])
    hits: list[tuple[str, ...]] = []
    if parsed.readable:
        with st.timed("req_encode"):
            vectors = ctx.encode(list(parsed.items))
        with st.timed("req_search"):
            hits = [
                tuple(rt.rank_by_scores(qid, ctx.doc_ids, ctx.doc_vectors @ v).chunk_ids[:depth])
                for v in vectors
            ]
    extra = {
        "generated": answer["output"],
        "llm_internal_ns": int(answer["seconds"] * 1e9),
        "readable": parsed.readable,
        "requirements": len(parsed.items),
    }
    return ranking, parsed, hits, extra


def s4(ctx: Context, qid: str, text: str, st: Stages) -> Out:
    ranking, _, hits, extra = requirements(ctx, qid, text, st, dc.POOL)
    with st.timed("assemble"):
        assembled = dc.assemble(ranking, hits)
    return Out(tuple(assembled.ranking[: cp.TOP]), tuple(assembled.ranking), {}, extra)


def s5(ctx: Context, qid: str, text: str, st: Stages) -> Out:
    assert ctx.reranker is not None
    ranking, parsed, hits, extra = requirements(ctx, qid, text, st, rq.DEPTH)
    scores: dict[str, float] = {}
    reranked: list[tuple[str, ...]] = []
    for index, (item, chunks) in enumerate(
        zip(parsed.items if parsed.readable else (), hits, strict=True)
    ):
        with st.timed("rerank_prompt"):
            prompts = [ctx.reranker.prompt(rq.INSTRUCTION, item, ctx.chunks[c]) for c in chunks]
        with st.timed("rerank_score"):
            values = ctx.reranker.score_shared(prompts)
            ctx.sync()
        with st.timed("reorder"):
            by_chunk = {c: float(v) for c, v in zip(chunks, values, strict=True)}
            reranked.append(tuple(rq.reorder(chunks, by_chunk)))
        scores.update({f"{qid}|{index}|{c}": v for c, v in by_chunk.items()})
    with st.timed("assemble"):
        assembled = dc.assemble(ranking, reranked)
    return Out(tuple(assembled.ranking[: cp.TOP]), tuple(assembled.ranking), scores, extra)


def s6(depth: int) -> Callable[[Context, str, str, Stages], Out]:
    def pipeline(ctx: Context, qid: str, text: str, st: Stages) -> Out:
        assert ctx.reranker is not None
        ranking = e5_rank(ctx, qid, text, st)
        candidates = ranking[:depth]
        with st.timed("rerank_prompt"):
            prompts = [ctx.reranker.prompt(rq.INSTRUCTION, text, ctx.chunks[c]) for c in candidates]
        with st.timed("rerank_score"):
            values = ctx.reranker.score_shared(prompts)
            ctx.sync()
        with st.timed("reorder"):
            by_chunk = {c: float(v) for c, v in zip(candidates, values, strict=True)}
            order = rq.reorder(candidates, by_chunk)
        scores = {f"{qid}|0|{c}": v for c, v in by_chunk.items()}
        return Out(tuple(order[: cp.TOP]), None, scores)

    return pipeline


PIPELINES: dict[str, Callable[[Context, str, str, Stages], Out]] = {
    "S1": s1,
    "S1p": s1p,
    "S2": s2,
    "S3": s3,
    "S4": s4,
    "S5": s5,
    "S6a": s6(20),
    "S6b": s6(50),
}


# ---------------------------------------------------------------------------
# passes
# ---------------------------------------------------------------------------


def one_query(ctx: Context, system: str, qid: str, text: str) -> tuple[Out, Stages, int]:
    stages = Stages()
    t0 = time.perf_counter_ns()
    out = PIPELINES[system](ctx, qid, text, stages)
    return out, stages, time.perf_counter_ns() - t0


def warm_up(ctx: Context, system: str) -> None:
    for i, text in enumerate(cp.WARMUP_QUERIES):
        one_query(ctx, system, f"warmup-{i}", text)


def verify_first(ctx: Context, system: str, order: Sequence[str]) -> list[str]:
    """The protocol's pre-verification: the first VERIFY_QUERIES queries of pass 1 against the
    stored rankings. Returns the ids that differ."""
    if system not in STORED:
        return []
    stored = read_stored(system)
    warm_up(ctx, system)
    bad = []
    for qid in list(order)[:VERIFY_QUERIES]:
        out, _, _ = one_query(ctx, system, qid, ctx.queries[qid].text)
        if not cp.same_top(out.top, stored[qid]):
            bad.append(qid)
    return bad


def run_pass(
    ctx: Context, system: str, label: str, seed: int, folder: Path, protocol: str, commit: str
) -> dict[str, Any]:
    docker_allowed = system == "S1p"
    snap_before, found = preconditions(docker_allowed)
    record: dict[str, Any] = {"pass": label, "seed": seed, "before": snap_before}
    if found:
        record |= {"status": "precondition_failed", "reasons": found}
        return record
    order = cp.pass_order(sorted(ctx.queries), seed)
    if DRY_QUERIES:  # the same few queries in every pass, so that passes can be compared
        keep = set(sorted(ctx.queries)[:DRY_QUERIES])
        order = [q for q in order if q in keep]
    stored = read_stored(system) if system in STORED else {}
    stored_text = read_stored_generations() if system in {"S4", "S5"} else {}
    first = label.startswith("1")  # pass 1 and its repetition are the full verification
    mark = time.monotonic()
    warm_up(ctx, system)
    totals: list[float] = []
    tops: dict[str, tuple[str, ...]] = {}
    scores: dict[str, float] = {}
    mismatches: list[dict[str, Any]] = []
    text_mismatches: list[str] = []
    started = time.perf_counter()
    lines = []
    for qid in order:
        out, stages, total = one_query(ctx, system, qid, ctx.queries[qid].text)
        totals.append(float(total))
        tops[qid] = out.top
        scores.update(out.scores)
        if first and qid in stored:
            reference = stored[qid]
            if not cp.same_top(out.top, reference):
                diverge = next(
                    (i for i, (a, b) in enumerate(zip(out.top, reference, strict=False)) if a != b),
                    0,
                )
                mismatches.append({"query_id": qid, "first_difference_rank": diverge + 1})
            elif out.ranking is not None and tuple(out.ranking) != tuple(reference):
                record.setdefault("full_ranking_differs", []).append(qid)
        if first and qid in stored_text and out.extra.get("generated") != stored_text[qid]:
            text_mismatches.append(qid)
        lines.append(
            json.dumps(
                {
                    "protocol_sha256": protocol,
                    "commit": commit,
                    "system": system,
                    "pass": label,
                    "seed": seed,
                    "query_id": qid,
                    "lang": ctx.queries[qid].lang,
                    "total_ns": total,
                    "stages_ns": stages.ns,
                    "llm_internal_ns": out.extra.get("llm_internal_ns"),
                    "requirements": out.extra.get("requirements"),
                    "readable": out.extra.get("readable"),
                }
            )
        )
    events = MONITOR.since(mark) if MONITOR is not None else []
    snap_after = snapshot()
    after = [f for f in failures(snap_after, docker_allowed) if not f.startswith("load ")]
    after = [] if DRY_RUN else after + events
    summary = cp.summarize(totals)
    half = min(30, len(totals) // 2)
    early, late = (
        cp.nearest_rank_percentile(totals[:half], 50),
        cp.nearest_rank_percentile(totals[-half:], 50),
    )
    with (folder / "timings.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    record |= {
        "status": "ok" if not after else "invalid_during_or_after",
        "reasons": after,
        "after": snap_after,
        "wall_seconds": time.perf_counter() - started,
        "n": summary.n,
        "p50_ns": summary.p50,
        "p95_ns": summary.p95,
        "max_ns": summary.maximum,
        "mean_ns": summary.mean,
        "first_vs_last_30_median": late / early if early else None,
        "rank_mismatches": mismatches,
        "generation_mismatches": text_mismatches,
        "unreadable_decompositions": sum(1 for ln in lines if json.loads(ln)["readable"] is False),
        "tops": {q: list(t) for q, t in tops.items()},
        "scores": scores,
    }
    return record


def read_rows(folder: Path, system: str, labels: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for line in (folder / "timings.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["system"] == system and row["pass"] in labels:
            rows.append(row)
    return rows


def stats(values: Sequence[float]) -> dict[str, float]:
    s = cp.summarize(values)
    return {"p50_ns": s.p50, "p95_ns": s.p95, "max_ns": s.maximum, "mean_ns": s.mean, "n": s.n}


def stage_table(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    per_stage: dict[str, list[float]] = {}
    for row in rows:
        for name, ns in row["stages_ns"].items():
            per_stage.setdefault(name, []).append(float(ns))
    return {name: stats(values) for name, values in per_stage.items()}


def run_system(system: str, protocol_path: Path) -> int:
    global MONITOR
    protocol, commit = guard(protocol_path)
    folder = RESULTS / system
    if (folder / "timings.jsonl").exists() or (folder / "summary.json").exists():
        raise SystemExit(f"{folder} already holds results: archive them, nothing is overwritten")
    folder.mkdir(parents=True, exist_ok=True)
    queries_order = cp.pass_order(sorted(load_queries()), cp.seed_for_pass(1))
    MONITOR = Monitor()
    MONITOR.start()
    t0 = time.perf_counter()
    ctx = Context(system, folder)
    load_seconds = time.perf_counter() - t0
    env = environment(ctx.layer_sha256)
    env_text = json.dumps(env, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    (folder / "environment.json").write_text(env_text, encoding="utf-8")
    summary: dict[str, Any] = {
        "protocol_sha256": protocol,
        "commit": commit,
        "dry_run": bool(DRY_RUN),
        "system": system,
        "environment_sha256": hashlib.sha256(env_text.encode()).hexdigest(),
        "load_seconds": ctx.timings | {"total": load_seconds},
    }
    if system == "S1p" and ctx.pg is None:
        summary |= {"status": "not_measured", "reason": ctx.pg_error or "no connection"}
        _finish(summary, {}, folder, ctx)
        return 0
    bad = verify_first(ctx, system, queries_order)
    summary["pre_verification"] = {"queries": VERIFY_QUERIES, "different": bad}
    if bad:
        summary["status"] = "verification_failed"
        _finish(summary, {}, folder, ctx)
        return 1
    heavy = system in HEAVY
    passes: dict[str, dict[str, Any]] = {}

    def execute(label: str, seed: int, cool: bool) -> dict[str, Any]:
        if cool:
            time.sleep(COOLDOWN_SECONDS)
        result = run_pass(ctx, system, label, seed, folder, protocol, commit)
        passes[label] = result
        return result

    def valid(result: dict[str, Any]) -> bool:
        return bool(result["status"] == "ok")

    unstable = "measurement_unstable"
    if heavy:
        # An invalid pass is repeated once; that repetition is the one the protocol allows.
        def attempt(n: int, cool: bool) -> tuple[str, dict[str, Any]]:
            result = execute(str(n), cp.seed_for_pass(n), cool)
            if valid(result):
                return str(n), result
            result = execute(f"{n}r", cp.repeat_seed(cp.seed_for_pass(n)), cool=True)
            return f"{n}r", result

        l1, r1 = attempt(1, cool=False)
        l2, r2 = attempt(2, cool=True)
        repeated = l1.endswith("r") or l2.endswith("r")
        usable = tuple(label for label, r in ((l1, r1), (l2, r2)) if valid(r))
        if len(usable) < 2:
            verdict = cp.Verdict(unstable, math.nan, usable)
        else:
            verdict = cp.judge_heavy(r1["p50_ns"], r2["p50_ns"])
            if verdict.status == "needs_repeat":
                if repeated:
                    verdict = cp.Verdict(unstable, verdict.drift, ("1",))
                else:
                    r2r = execute("2r", cp.repeat_seed(cp.seed_for_pass(2)), cool=True)
                    if valid(r2r):
                        verdict = cp.judge_heavy(r1["p50_ns"], r2["p50_ns"], r2r["p50_ns"])
                    else:
                        verdict = cp.Verdict(unstable, verdict.drift, ("1",))
            names = {"1": l1, "2": l2}
            verdict = cp.Verdict(
                verdict.status, verdict.drift, tuple(names.get(x, x) for x in verdict.reported)
            )
    else:

        def block(suffix: str) -> list[tuple[str, dict[str, Any]]]:
            out = []
            for n in range(1, FAST_PASSES + 1):
                seed = cp.seed_for_pass(n)
                seed = cp.repeat_seed(seed) if suffix else seed
                out.append((f"{n}{suffix}", execute(f"{n}{suffix}", seed, cool=False)))
            return out

        def medians(rows: list[tuple[str, dict[str, Any]]]) -> list[float]:
            return [r["p50_ns"] for _, r in rows]

        first_block = block("")
        first_ok = all(valid(r) for _, r in first_block)
        verdict = (
            cp.judge_fast(medians(first_block))
            if first_ok
            else cp.Verdict("needs_repeat", math.nan, ())
        )
        if verdict.status == "needs_repeat":
            second = block("r")
            labels_first = tuple(label for label, r in first_block if valid(r))
            if not all(valid(r) for _, r in second):
                verdict = cp.Verdict(unstable, math.nan, labels_first)
            elif first_ok:
                verdict = cp.judge_fast(medians(first_block), medians(second))
            else:
                again = cp.judge_fast(medians(second))
                labels_second = tuple(label for label, _ in second)
                verdict = (
                    cp.Verdict("valid_after_repeat", again.drift, labels_second)
                    if again.status == "valid"
                    else cp.Verdict(unstable, again.drift, labels_first)
                )
    summary |= analyse(system, passes, verdict, folder)
    _finish(summary, passes, folder, ctx)
    return 0


def analyse(
    system: str, passes: dict[str, dict[str, Any]], verdict: cp.Verdict, folder: Path
) -> dict[str, Any]:
    ok = {k: v for k, v in passes.items() if v["status"] == "ok"}
    reported = [label for label in verdict.reported if label in ok] or sorted(ok)
    analysis: dict[str, Any] = {
        "status": verdict.status,
        "drift": None if math.isnan(verdict.drift) else verdict.drift,
        "reported_passes": reported,
        "all_passes": {
            k: {
                f: v.get(f)
                for f in (
                    "status",
                    "reasons",
                    "seed",
                    "n",
                    "p50_ns",
                    "p95_ns",
                    "max_ns",
                    "mean_ns",
                    "wall_seconds",
                    "first_vs_last_30_median",
                    "rank_mismatches",
                    "generation_mismatches",
                    "full_ranking_differs",
                    "unreadable_decompositions",
                )
            }
            | {"before": v["before"], "after": v.get("after")}
            for k, v in passes.items()
        },
    }
    if not reported:
        analysis["no_valid_pass"] = True
        return analysis
    rows = read_rows(folder, system, reported)
    analysis |= stats([float(r["total_ns"]) for r in rows])
    analysis["stages"] = stage_table(rows)
    analysis["by_language"] = {
        lang: stats([float(r["total_ns"]) for r in rows if r["lang"] == lang])
        for lang in sorted({r["lang"] for r in rows})
    }
    first = ok[reported[0]]
    analysis["top10_identical_across_reported_passes"] = all(
        ok[label]["tops"] == first["tops"] for label in reported
    )
    analysis["rank_mismatch_count"] = len(first["rank_mismatches"])
    analysis["generation_mismatch_count"] = len(first["generation_mismatches"])
    if first["scores"] and len(reported) >= 2:
        second = ok[reported[1]]
        comparison = cp.compare_scores(first["scores"], second["scores"])
        analysis["score_reproducibility"] = {
            "passes": [reported[0], reported[1]],
            "outcome": comparison.outcome,
            "max_abs_diff": comparison.max_abs_diff,
            "compared": comparison.compared,
            "top10_identical": first["tops"] == second["tops"],
        }
    if system in {"S6a", "S6b"} and first["scores"]:
        path = folder / f"scores-{system}-pass{reported[0]}.jsonl"
        lines = []
        for key, value in sorted(first["scores"].items()):
            qid, _, chunk = key.split("|", 2)
            lines.append(
                json.dumps(
                    {"query_id": qid, "chunk_id": chunk, "logit_diff": value}, sort_keys=True
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.chmod(0o444)
        analysis["scores_file"] = {
            "path": str(path),
            "sha256": cp.sha256_file(path),
            "rows": len(lines),
            "pass": reported[0],
        }
    return analysis


def _finish(
    summary: dict[str, Any], passes: dict[str, dict[str, Any]], folder: Path, ctx: Context
) -> None:
    summary["sampled_tree_rss_kb_max"] = MONITOR.finish() if MONITOR is not None else None
    summary["worker_memory"] = ctx.close()
    (folder / "passes.json").write_text(
        json.dumps(summary.get("all_passes", {}), indent=1, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# cold start, smoke, drive, report
# ---------------------------------------------------------------------------


def heavy_imports(system: str) -> None:
    """The imports that the code of a system makes lazily, done up front so that the cold start
    can report them apart from loading the models."""
    if system != "S2":
        import torch  # noqa: F401
        import transformers  # noqa: F401
    if system in {"S2", "S3"}:
        import bm25s  # noqa: F401
    if system == "S1p":
        from ingest import vector_index  # noqa: F401


def cold(system: str, repetition: int, protocol_path: Path) -> int:
    module_imports = time.perf_counter() - START
    protocol, commit = guard(protocol_path)
    snap, found = preconditions(system == "S1p")
    if found:
        print("precondition_failed", found)
        return 2
    folder = RESULTS / system
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    heavy_imports(system)
    imports = module_imports + time.perf_counter() - t0
    t0 = time.perf_counter()
    ctx = Context(system, folder)
    loaded = time.perf_counter() - t0
    order = cp.pass_order(sorted(ctx.queries), cp.seed_for_pass(1))
    qid = order[0]
    if system == "S1p" and ctx.pg is None:
        row: dict[str, Any] = {"status": "not_measured", "reason": ctx.pg_error}
    else:
        out, stages, total = one_query(ctx, system, qid, ctx.queries[qid].text)
        row = {
            "status": "ok",
            "query_id": qid,
            "first_query_ns": total,
            "stages_ns": stages.ns,
            "top": list(out.top),
        }
    row |= {
        "protocol_sha256": protocol,
        "commit": commit,
        "system": system,
        "repetition": repetition,
        "import_seconds": imports,
        "load_seconds": ctx.timings | {"total": loaded},
        "worker_memory": ctx.close(),
        "before": snap,
    }
    with (folder / "cold-starts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return 0


def smoke(system: str, verify: bool) -> int:
    """One system on the three warm-up queries: plumbing only. Nothing is recorded. With
    `verify`, also the protocol's pre-verification on the first five queries of pass 1."""
    work = Path(".cache/s5-smoke")
    work.mkdir(parents=True, exist_ok=True)
    ctx = Context(system, work)
    print(f"{system}: loaded", {k: round(v, 2) for k, v in ctx.timings.items()})
    if system == "S1p" and ctx.pg is None:
        print("S1p: not measured:", ctx.pg_error)
        return 0
    for i, text in enumerate(cp.WARMUP_QUERIES):
        out, stages, total = one_query(ctx, system, f"warmup-{i}", text)
        print(
            f"  warm-up {i}: {total / 1e6:.1f} ms; stages",
            {k: round(v / 1e6, 1) for k, v in stages.ns.items()},
        )
        print("   top-3:", out.top[:3], "| requirements:", out.extra.get("requirements"))
    if verify and system in STORED:
        order = cp.pass_order(sorted(ctx.queries), cp.seed_for_pass(1))
        bad = verify_first(ctx, system, order)
        print("  pre-verification, ids that differ from the stored ranking:", bad)
    print("memory of the worker:", ctx.close())
    return 0


def preflight_command() -> int:
    snap = snapshot()
    found = failures(snap, docker_allowed=False)
    print(json.dumps({"snapshot": snap, "failures": found}, indent=1, ensure_ascii=False))
    return 1 if found else 0


FINISHED = {"valid", "valid_after_repeat", "measurement_unstable", "not_measured"}


def archive_partial(folder: Path) -> None:
    """A folder with results but no finished summary is moved aside, never overwritten."""
    if not folder.exists() or not any(folder.iterdir()):
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = folder.with_name(f"{folder.name}.partial-{stamp}")
    folder.rename(target)
    event = {"time_utc": stamp, "event": "partial results archived", "from": str(folder)}
    with (RESULTS / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event | {"to": str(target)}) + "\n")


def cold_rows(folder: Path) -> int:
    path = folder / "cold-starts.jsonl"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def drive(only: str | None, protocol_path: Path) -> int:
    guard(protocol_path)
    _, found = preconditions(only == "S1p")
    if found:
        print("preconditions fail, nothing started:", found)
        return 3
    systems = (only,) if only else tuple(s for s in ORDER if s != "S1p")
    python = sys.executable
    RESULTS.mkdir(parents=True, exist_ok=True)
    for system in systems:
        folder = RESULTS / system
        summary_path = folder / "summary.json"
        if summary_path.exists():
            status = json.loads(summary_path.read_text(encoding="utf-8")).get("status")
            if status in FINISHED:
                print(system, "already measured:", status, flush=True)
                continue
            print(system, "ended with status", status, "; a human decides:", summary_path)
            return 1
        archive_partial(folder)
        folder.mkdir(parents=True, exist_ok=True)
        time_file = folder / "main-time.txt"
        command = ["caffeinate", "-dims", "/usr/bin/time", "-l", "-o", str(time_file), python]
        command += [__file__, "run", "--system", system, "--protocol", str(protocol_path)]
        print("running", system, flush=True)
        code = subprocess.run(command, check=False).returncode
        memory = (
            cp.parse_time_l(time_file.read_text(encoding="utf-8")) if time_file.exists() else {}
        )
        (folder / "main-memory.json").write_text(
            json.dumps(memory, indent=1) + "\n", encoding="utf-8"
        )
        if code != 0:
            print(f"{system} ended with code {code}; stopping the drive", flush=True)
            return code
        if system in HEAVY:
            time.sleep(COOLDOWN_SECONDS)
    for system in systems:
        for repetition in range(cold_rows(RESULTS / system) + 1, 4):
            command = ["caffeinate", "-dims", python, __file__, "cold", "--system", system]
            command += ["--repetition", str(repetition), "--protocol", str(protocol_path)]
            print("cold start", system, repetition, flush=True)
            code = subprocess.run(command, check=False).returncode
            if code != 0:
                print(f"cold start of {system} ended with code {code}; stopping", flush=True)
                return code
    return report()


def ms(value: float) -> str:
    return f"{value / 1e6:,.1f}"


def gib(value: float) -> str:
    return f"{value / 2**30:.2f}"


def consistency_of_n() -> dict[str, Any] | None:
    """The protocol's last reproducibility item: S6a's scores against S6b's for the same pairs."""
    a, b = RESULTS / "S6a" / "summary.json", RESULTS / "S6b" / "summary.json"
    if not (a.exists() and b.exists()):
        return None
    files = []
    for path in (a, b):
        info = json.loads(path.read_text(encoding="utf-8")).get("scores_file")
        if not info:
            return None
        files.append(Path(info["path"]))

    def read(path: Path) -> dict[tuple[str, str], float]:
        rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        return {(r["query_id"], r["chunk_id"]): r["logit_diff"] for r in rows}

    small, large = read(files[0]), read(files[1])
    shared = {key: large[key] for key in small if key in large}
    comparison = cp.compare_scores(small, shared)
    return {
        "pairs_in_n20": len(small),
        "pairs_found_in_n50": len(shared),
        "outcome": comparison.outcome,
        "max_abs_diff": comparison.max_abs_diff,
    }


def report() -> int:
    systems: dict[str, Any] = {}
    for folder in sorted(p for p in RESULTS.iterdir() if p.is_dir() and ".partial-" not in p.name):
        path = folder / "summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        memory_file = folder / "main-memory.json"
        data["main_memory"] = (
            json.loads(memory_file.read_text(encoding="utf-8")) if memory_file.exists() else {}
        )
        cold_file = folder / "cold-starts.jsonl"
        data["cold_starts"] = (
            [json.loads(x) for x in cold_file.read_text(encoding="utf-8").splitlines()]
            if cold_file.exists()
            else []
        )
        main_peak = data["main_memory"].get("peak_memory_footprint", 0)
        worker_peak = (data.get("worker_memory") or {}).get("peak_memory_footprint", 0)
        data["sum_of_peaks_upper_bound_bytes"] = main_peak + worker_peak if main_peak else None
        systems[folder.name] = data
    environments = {d["environment_sha256"] for d in systems.values() if "environment_sha256" in d}
    summary = {
        "systems": systems,
        "environment_identical_across_systems": len(environments) <= 1,
        "s6a_vs_s6b_scores": consistency_of_n(),
    }
    (RESULTS / "cost-summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    reference = systems.get("S1", {}).get("p50_ns")
    lines = ["# S5.3 operational cost, measured on M1 (Apple M1, 16 GiB, fanless)", ""]
    lines += ["| System | Status | p50 (ms) | p95 (ms) | max (ms) | n | p50 / S1 |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for name in ORDER:
        d = systems.get(name)
        if d is None:
            continue
        if "p50_ns" in d:
            ratio = f"{d['p50_ns'] / reference:.1f}" if reference else ""
            lines.append(
                f"| {name} | {d['status']} | {ms(d['p50_ns'])} | {ms(d['p95_ns'])} | "
                f"{ms(d['max_ns'])} | {d['n']} | {ratio} |"
            )
        else:
            lines.append(f"| {name} | {d.get('status')} | | | | | |")
    lines += [
        "",
        "| System | Main peak footprint (GiB) | Worker peak (GiB) | Sum (upper bound, GiB) |",
    ]
    lines += ["|---|---|---|---|"]
    for name in ORDER:
        d = systems.get(name)
        if d is None:
            continue
        main_peak = d["main_memory"].get("peak_memory_footprint")
        worker_peak = (d.get("worker_memory") or {}).get("peak_memory_footprint")
        total = d.get("sum_of_peaks_upper_bound_bytes")
        lines.append(
            f"| {name} | {gib(main_peak) if main_peak else ''} | "
            f"{gib(worker_peak) if worker_peak else ''} | {gib(total) if total else ''} |"
        )
    lines += ["", "| System | Cold starts | Median load (s) | Median first query (ms) |"]
    lines += ["|---|---|---|---|"]
    for name in ORDER:
        d = systems.get(name)
        if d is None or not d["cold_starts"]:
            continue
        done = [c for c in d["cold_starts"] if c.get("status") == "ok"]
        if done:
            load = cp.nearest_rank_percentile([c["load_seconds"]["total"] for c in done], 50)
            first = cp.nearest_rank_percentile([c["first_query_ns"] for c in done], 50)
            lines.append(f"| {name} | {len(done)} | {load:.1f} | {ms(first)} |")
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        str(p.relative_to(RESULTS)): cp.sha256_file(p)
        for p in sorted(RESULTS.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"report written to {RESULTS}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("report")
    for name in ("smoke", "run", "cold"):
        p = sub.add_parser(name)
        p.add_argument("--system", choices=ORDER, required=True)
        if name == "smoke":
            p.add_argument("--verify", action="store_true")
        else:
            p.add_argument("--protocol", type=Path, required=True)
        if name == "cold":
            p.add_argument("--repetition", type=int, required=True)
    p = sub.add_parser("drive")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--only", choices=ORDER)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        return preflight_command()
    if args.command == "report":
        return report()
    if args.command == "smoke":
        return smoke(args.system, args.verify)
    if args.command == "run":
        return run_system(args.system, args.protocol)
    if args.command == "cold":
        return cold(args.system, args.repetition, args.protocol)
    return drive(args.only, args.protocol)


if __name__ == "__main__":
    sys.exit(main())
