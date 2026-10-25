"""TEST v2 (D-067): its development validation, its contract, and its construction.

    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py exclusions
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py faithfulness
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py pool-validation
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py pilot-generation <pilot id> --universe 27 \
        --workers 6 --certification <id> --certification-manifest <sha256>
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py pilot-judging <pilot id> --needs 2 \
        --workers 3 [--completeness <DEV need id>] --certification <id> \
        --certification-manifest <sha256>
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py development-references <pilot id> ...
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py forecast <pilot id> ...
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py prepare --certification <id> \
        --certification-manifest <sha256>
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py record-public-commit <sha>
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py validate --contract <sha256>
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py start --contract <sha256> --workers 6
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py resume --contract <sha256> --workers 6
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py status
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py evaluate --contract <sha256>
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py audit-export --contract <sha256>
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py audit-sa7 <labels.json> --contract <sha256>
    PYTHONPATH=. .venv/bin/python scripts/holdout_v2.py backup

Development only, never TEST material: faithfulness runs every frozen system on DEV's queries
and compares each ranking with its frozen DEV run; pool-validation builds the TEST v2 pool of
every DEV need from the frozen DEV runs; pilot-generation runs the anchored generation and the
global deduplication on a development bank (its own seed and salt); pilot-judging runs the
judging, the answerability, the metadata and, for one need, the completeness screen on DEV's
needs; forecast reads the pilots' logs and gives FAST, EXPECTED and SLOW wall times. Each pilot
is a namespace of its own under data/benchmark/test/v2-development/ with its raw folder under
~/.fundamento-private/holdout-v2-development-raw/: never TEST v2's.

prepare writes the public contract once (config/holdout/test-v2-contract.json), after the
opaque-id salt, the historical exclusions and the development references exist in TEST v2's
private provenance. start and resume are the only commands that run a model on TEST v2, and
each needs the contract's full SHA-256 typed again; start also needs the public commit that
holds the contract recorded. evaluate runs only on frozen qrels. status prints counts only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import secrets
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_certification as hcert
from ingest import holdout_exclusions as hxe
from ingest import holdout_executor as hx
from ingest import holdout_isolation as hi
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from ingest import holdout_roles as hr
from ingest import holdout_spool as hsp
from ingest import holdout_v2 as v2
from ingest import holdout_v2_audit as hau
from ingest import holdout_v2_contract as hc2
from ingest import holdout_v2_dedup as dd
from ingest import holdout_v2_engine as en
from ingest import holdout_v2_retrieval as rv
from ingest import holdout_v2_runner as hrun
from ingest import holdout_v2_state as vs

REPO = Path.cwd()


class V2RealExecutor(hx.RealExecutor):
    """The certified executor with one more classification (holdout_v2_engine CAPACITY): a
    failed call the provider refused for its capacity is never counted as a failure of its
    job. Every other outcome is the certified executor's own."""

    def _answer(self, role: str, run: hi.HeadlessRun) -> hpl.Session:
        failed = run.returncode not in (0, hsp.UNOBSERVED) or run.answer is None
        if failed and en.capacity_refusal(run.stream, run.stderr):
            detail = f"{role}: exit {run.returncode}, refused for the provider's capacity"
            raise hpl.TechnicalFailure(en.CAPACITY, detail)  # type: ignore[arg-type]
        return super()._answer(role, run)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False, default=str))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# the frozen inputs every world reads
# ---------------------------------------------------------------------------


def _common() -> dict[str, Any]:
    from ingest import psd2_exclusions as px

    dev = bm.load_benchmark(bm.DEV_DIR)
    fixtures = bm.load_benchmark(bm.FIXTURES_DIR)
    docs = bm.load_corpus()
    layer = rv.load_layer()
    excluded = px.load_psd2_exclusions(px.PSD2_EXCLUSIONS_PATH).excluded()
    return {
        "dev": dev,
        "fixtures": fixtures,
        "docs": docs,
        "layer": layer,
        "excluded": excluded,
        "corpus": hp.Corpus(docs),
    }


def _certification(certification: str, manifest_sha256: str) -> Path:
    namespace = hcert.selected(certification, manifest_sha256, REPO)
    pin = hcert.read_json(namespace / hcert.PROVENANCE / "runtime.json")
    live = hi.binary_identity(Path(pin["path"]))
    if (live["version"], live["sha256"]) != (pin["version"], pin["sha256"]):
        raise hcert.CertificationError(f"the binary at {pin['path']} is not the certified one")
    return namespace


def _executor_and_audit(
    certification: Path, raw: Path, frozen_templates: Mapping[str, str], forbidden: Sequence[str]
) -> tuple[Any, en.AuditContext]:
    """The real executor of the certified binary, and the audit context of every response."""
    from ingest import holdout_launcher as hla
    from ingest import holdout_runtime as hrt

    provenance = certification / hcert.PROVENANCE
    runtime = hcert.read_json(provenance / "runtime.json")
    gate1 = hcert.read_json(provenance / "gate1.json")
    claude = hrt.claude_runtime(provenance)
    model = claude["requested_model"]
    if runtime["model"] != model:
        raise hcert.CertificationError("the pinned model is not the certified requested model")
    inherited = list(gate1["environment_allowlist"]["inherited"])
    extra_env = [v for v in inherited if v not in hi.ENV_INHERITED]
    pin = {"version": runtime["version"], "sha256": runtime["sha256"]}
    spool = raw / "spool"
    ho.private_folder(spool, REPO)
    executor = V2RealExecutor(Path(runtime["path"]), model, pin, extra_env, spool, {})
    audit = en.AuditContext(
        terms=hla.project_terms(REPO),
        frozen_templates=dict(frozen_templates),
        forbidden=list(forbidden),
        pin={"version": runtime["version"], "organization_sha256": runtime["organization_sha256"]},
        tmp_root=str(Path(tempfile.gettempdir()).resolve()),
        claude_effective=claude["effective"],
    )
    return executor, audit


def _bge_vectors(layer: rv.Layer) -> Any:
    """BGE-M3 vectors of statements for the deduplication's prefilter: the model is loaded for
    the one call and released after it, so that no other local model ever runs beside it."""

    def vectors(texts: Sequence[str]) -> Any:
        import gc

        dense = rv.load_dense(rv.DENSE["BGE-M3"], layer)
        kept, found, refused = dense.encode([(str(i), t) for i, t in enumerate(texts)])
        del dense
        gc.collect()
        if refused or len(kept) != len(texts):
            raise rv.RetrievalV2Error(f"{len(refused)} statements over BGE-M3's input limit")
        return found

    return vectors


def _rank(layer: rv.Layer) -> Any:
    """The frozen systems, one call each: E5, BGE-M3 and Qwen3-Embedding-0.6B (dense), BM25,
    and E, which ranks E5 again on the same queries for its excerpts (deterministic), so that a
    resume never depends on what an earlier process held in memory."""

    def rank(system: str, queries: Sequence[tuple[str, str]], work: Path) -> rv.Run:
        if system in rv.DENSE:
            return rv.dense_run(system, queries, layer)
        if system == "BM25":
            return rv.bm25_run(queries, layer)
        if system == "E":
            return rv.e_run(queries, layer, work, rv.dense_run("E5", queries, layer))
        raise rv.RetrievalV2Error(f"{system} is no frozen system")

    return rank


def _store(namespace: Path, raw: Path, meta: Mapping[str, str]) -> vs.Store:
    if (namespace / vs.STATE_DIR / vs.DB_FILE).exists():
        store = vs.Store.open(namespace, raw)
        found = store.meta_all()
        differ = {k for k, v in meta.items() if found.get(k) != v}
        if differ:
            store.close()
            raise vs.StateError(f"the state was created with other {sorted(differ)}")
        return store
    ho.private_folder(raw, REPO)
    return vs.Store.create(namespace, raw, meta)


def _session(runner: hrun.Runner, workers: int, label: str) -> None:
    runner.engine.record(
        "session_started", None, {"workers": workers, "label": label, "at": _now()}
    )


# ---------------------------------------------------------------------------
# the historical exclusions
# ---------------------------------------------------------------------------


def exclusions() -> int:
    """The 99 historical statements of runs 1 to 7 (D-066, D-067): collected from their archived
    states, written once into TEST v2's private provenance. Counts and the hash only."""
    sources = []
    for n, record in enumerate(hc2.ABORTED_RUNS, start=1):
        directory = ho.HOLDOUT_DIR / "aborted" / f"run-{n}"
        raw = Path.home() / ".fundamento-private" / "holdout-aborted" / f"run-{n}" / "raw"
        sources.append(hxe.Source(f"run-{n}", directory, raw, str(record["contract_sha256"])))
    entries, states = hxe.collect(sources)
    provenance = hc2.NAMESPACE / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    identity = hxe.write(provenance / hc2.EXCLUSIONS_FILE, entries, states)
    _print({"identity": identity, "sources": {s.label: s.statements for s in states}})
    return 0


# ---------------------------------------------------------------------------
# development validation: retrieval faithfulness and the DEV pools
# ---------------------------------------------------------------------------


def faithfulness() -> int:
    work = hc2.DEVELOPMENT_ROOT / "faithfulness" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    work.mkdir(parents=True)
    report = rv.faithfulness(work)
    (work / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    _print(report)
    return 0


def dev_rankings(
    dev: bm.Benchmark, query_ids: Sequence[str] | None = None
) -> dict[str, dict[str, list[str]]]:
    from ingest import retrieval as rt

    out = {}
    for system, name in rv.DEV_RUNS.items():
        rows = rt.read_run(
            rt.run_dir(name, rv.MAX_TOKENS, dev.manifest.benchmark_version) / rt.RUN_FILE
        )[0]
        out[system] = {
            r.query_id: list(r.chunk_ids)
            for r in rows
            if query_ids is None or r.query_id in set(query_ids)
        }
    return out


def pool_validation() -> int:
    """Pilot B: the TEST v2 pool of every DEV need from the frozen DEV runs (no anchor: DEV has
    none), against DEV's own qrels: sizes, coverage, unique contributions, decisive sets."""
    from ingest import evaluation as ev
    from ingest import holdout_v2_judging as jv

    c = _common()
    dev, layer = c["dev"], c["layer"]
    rankings = dev_rankings(dev)
    sizes, nodes, decisive, covered, total = [], [], [], 0, 0
    unique: Counter[str] = Counter()
    for need in dev.needs:
        if need.status != "ready" or need.answerability == "none":
            continue
        qids = [q.query_id for q in dev.queries_of(need.need_id)]
        pooled = v2.pool(rankings, qids, [])
        pool = jv.need_pool(
            need.need_id,
            pooled,
            v2.decisive_chunks(rankings, qids),
            layer.layer,
            c["corpus"].texts,
            c["excluded"],
        )
        sizes.append(len(pool.chunks))
        nodes.append(len(pool.nodes))
        decisive.append(len(pool.decisive))
        grades = ev.project(ev.judged_nodes(dev.qrels_of(need.need_id)), layer.layer)
        relevant = {ch for ch, g in grades.items() if g >= bm.RELEVANT_GRADE}
        by = {p.chunk_id: {s[0] for s in p.sources} for p in pooled}
        covered += len(relevant & set(by))
        total += len(relevant)
        for ch in relevant & set(by):
            if len(by[ch]) == 1:
                unique[next(iter(by[ch]))] += 1
    report = {
        "needs": len(sizes),
        "median_chunks": statistics.median(sizes),
        "median_nodes": statistics.median(nodes),
        "median_decisive_chunks": statistics.median(decisive),
        "max_chunks": max(sizes),
        "max_nodes": max(nodes),
        "relevant_chunks_covered": covered,
        "relevant_chunks": total,
        "unique_relevant_by_contributor": dict(sorted(unique.items())),
    }
    _print(report)
    return 0


# ---------------------------------------------------------------------------
# the pilots: a world of their own each, never TEST v2's
# ---------------------------------------------------------------------------


def _pilot_paths(pilot: str) -> tuple[Path, Path]:
    if not pilot.replace("-", "").isalnum():
        raise ho.HoldoutError(f"{pilot!r} is not a pilot id")
    return hc2.DEVELOPMENT_ROOT / pilot, hc2.DEVELOPMENT_RAW / pilot


def _pilot_salt(namespace: Path) -> bytes:
    path = namespace / "provenance" / hc2.SALT_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32) + "\n", encoding="utf-8")
    return hc2.read_salt(namespace / "provenance")


def _historical(c: Mapping[str, Any]) -> list[dd.Reference]:
    found = hxe.read(hc2.NAMESPACE / "provenance" / hc2.EXCLUSIONS_FILE)
    refs = [dd.Reference(n.need_id, "DEV", n.statement) for n in c["dev"].needs]
    refs += [dd.Reference(n.need_id, "FIXTURE", n.statement) for n in c["fixtures"].needs]
    refs += [dd.Reference(f"historical:{e.key}", "DEVELOPMENT", e.statement) for e in found.entries]
    return refs


def _forbidden(c: Mapping[str, Any]) -> list[str]:
    return [n.statement for n in c["dev"].needs + c["fixtures"].needs]


def pilot_generation(
    pilot: str, universe: int, workers: int, certification: str, manifest: str
) -> int:
    """Pilot A: the anchored generation and the global deduplication on a development bank (the
    development seed and a salt of its own), frozen as topics; no retrieval, no judging."""
    namespace, raw = _pilot_paths(pilot)
    c = _common()
    salt = _pilot_salt(namespace)
    slots = v2.slots(c["dev"])
    units = v2.legal_units(c["docs"], ho.load_areas(), c["excluded"], c["layer"].layer)
    bank = v2.build_bank(slots, units, salt, v2.DEVELOPMENT_SEED, v2.DEVELOPMENT_SEED + 1, universe)
    cert = _certification(certification, manifest)
    executor, audit = _executor_and_audit(cert, raw, hr.template_hashes(), _forbidden(c))
    params = hrun.Params(workers=workers)

    def no_retrieval(*args: Any) -> rv.Run:
        raise ho.HoldoutError("a generation pilot retrieves nothing")

    world = hrun.World(
        namespace=namespace,
        corpus=c["corpus"],
        layer=c["layer"].layer,
        chunk_text={k: ch.text for k, ch in c["layer"].chunks.items()},
        excluded_psd2=c["excluded"],
        forbidden=_forbidden(c),
        references=_historical(c),
        bank=bank,
        chains={},
        audit_chain=[],
        salt=salt,
        params=params,
        audit=audit,
        rank=no_retrieval,
        write_runs=rv.write_runs,
        read_runs=rv.read_runs,
        vectors=_bge_vectors(c["layer"]),
    )
    meta = {"kind": "development-pilot-generation", "pilot": pilot, "bank_sha256": bank.sha256()}
    with _store(namespace, raw, meta) as store:
        runner = hrun.Runner(world, store, executor)
        _session(runner, workers, "pilot-generation")
        runner.freeze_bank()
        if "topics" not in runner.keyed:
            runner.generate()
            runner.deduplicate()
            runner.freeze_topics()
        topics = runner.topics()
        manifest_data = runner.read_frozen("topics", hrun.TOPICS_FILE)
        _print(
            {
                "pilot": pilot,
                "candidates": len(bank.candidates),
                "topics": len(topics),
                "generation": manifest_data["generation"],
                "deduplication": Counter(
                    "kept" if d["kept"] else f"duplicate:{d['closest_kind']}"
                    for d in manifest_data["deduplication"].values()
                ),
                "summary": runner.summary(),
            }
        )
    return 0


def _dev_topics(c: Mapping[str, Any], needs: Sequence[str]) -> list[dict[str, Any]]:
    """DEV needs as a pilot's topics: their statements and queries, no anchor."""
    dev = c["dev"]
    out = []
    for rank, need_id in enumerate(needs, start=1):
        need = next(n for n in dev.needs if n.need_id == need_id)
        out.append(
            {
                "rank": rank,
                "candidate_id": need_id,
                "need_id": need_id,
                "priority": "",
                "slot": {
                    "acts": list(need.acts),
                    "type": need.type,
                    "answerability": need.answerability,
                },
                "anchors": [],
                "statement": need.statement,
                "queries": [q.model_dump(mode="json") for q in dev.queries_of(need_id)],
                "leakage": None,
            }
        )
    return out


def pilot_judging(
    pilot: str,
    count: int,
    workers: int,
    completeness_need: str | None,
    certification: str,
    manifest: str,
    first: str | None = None,
) -> int:
    """Pilots C and D: the judging of DEV needs exactly as TEST v2 judges (pools from the frozen
    DEV runs, B1, B2, C, the qrel freeze, answerability, metadata) and, for one need, the
    completeness screen with its follow-up. Never TEST material."""
    namespace, raw = _pilot_paths(pilot)
    c = _common()
    dev = c["dev"]
    salt = _pilot_salt(namespace)
    eligible = sorted(
        (n.need_id for n in dev.needs if n.status == "ready" and n.answerability != "none"),
        key=lambda i: v2.draw(v2.DEVELOPMENT_SEED, salt, "pilot-need", i),
    )
    if first is not None:
        eligible = [first] + [n for n in eligible if n != first]
    if completeness_need is not None:
        eligible = [completeness_need] + [n for n in eligible if n != completeness_need]
    needs = eligible[:count]
    topics = _dev_topics(c, needs)
    query_ids = [q["query_id"] for t in topics for q in t["queries"]]
    rankings = dev_rankings(dev, query_ids)

    def rank(system: str, queries: Sequence[tuple[str, str]], work: Path) -> rv.Run:
        return rv.Run(system, dict(rankings[system]))  # DEV's frozen run, restricted

    chains: dict[str, list[str]] = {a: [] for a in v2.COMPLETENESS_STRATA}
    if completeness_need is not None:
        need = next(n for n in dev.needs if n.need_id == completeness_need)
        chains[need.acts[0]] = [completeness_need]
    cert = _certification(certification, manifest)
    executor, audit = _executor_and_audit(cert, raw, hr.template_hashes(), _forbidden(c))
    bank = v2.Bank(v2.DEVELOPMENT_SEED, (), {}, {})
    world = hrun.World(
        namespace=namespace,
        corpus=c["corpus"],
        layer=c["layer"].layer,
        chunk_text={k: ch.text for k, ch in c["layer"].chunks.items()},
        excluded_psd2=c["excluded"],
        forbidden=_forbidden(c),
        references=[],
        bank=bank,
        chains=chains,
        audit_chain=[],
        salt=salt,
        params=hrun.Params(workers=workers),
        audit=audit,
        rank=rank,
        write_runs=rv.write_runs,
        read_runs=rv.read_runs,
        vectors=_bge_vectors(c["layer"]),
        n_target=len(needs),
        topics_source=lambda: topics,
    )
    meta = {"kind": "development-pilot-judging", "pilot": pilot, "needs": ",".join(needs)}
    with _store(namespace, raw, meta) as store:
        runner = hrun.Runner(world, store, executor)
        _session(runner, workers, "pilot-judging")
        summary = runner.build()
    _print({"pilot": pilot, "needs": needs, "summary": summary})
    return 0


def development_references(pilots: Sequence[str]) -> int:
    """The generation pilots' statements, designated DEVELOPMENT topic-generation material
    (D-067): written once into TEST v2's private provenance as hidden references of its gate.
    Nothing else of a pilot becomes a reference."""
    refs: list[dd.Reference] = []
    for pilot in pilots:
        namespace, raw = _pilot_paths(pilot)
        with vs.Store.open(namespace, raw, readonly=True) as store:
            if store.meta("kind") != "development-pilot-generation":
                raise ho.HoldoutError(f"{pilot} is not a generation pilot")
            ledger = en.Ledger.fold(store.events())
        for key, p in sorted(ledger.keyed.items()):
            if key.startswith("generated/"):
                refs.append(
                    dd.Reference(
                        f"development:{pilot}:{p['candidate_id']}", "DEVELOPMENT", p["statement"]
                    )
                )
    path = hc2.NAMESPACE / "provenance" / hc2.DEVELOPMENT_REFERENCES_FILE
    identity = hc2.write_development_references(path, refs, pilots)
    _print(identity)
    return 0


# ---------------------------------------------------------------------------
# the forecast, from the pilots' logs
# ---------------------------------------------------------------------------


def _durations(
    events: Sequence[vs.Event],
) -> tuple[dict[str, list[float]], Counter[str], dict[str, Any]]:
    started: dict[str, tuple[str, datetime]] = {}
    out: dict[str, list[float]] = defaultdict(list)
    failures: Counter[str] = Counter()
    first = last = None
    workers = []
    for e in events:
        at = datetime.fromisoformat(e.at)
        if e.kind == "session_started":
            workers.append(e.payload["workers"])
        if e.kind == "run_started":
            started[e.payload["run_id"]] = (e.payload["role"], at)
            first = first or at
        elif e.kind == "run_responded" and e.payload["run_id"] in started:
            role, t0 = started[e.payload["run_id"]]
            out[role].append((at - t0).total_seconds())
            last = at
        elif e.kind == "run_failed":
            failures[e.payload["reason"]] += 1
    span = (last - first).total_seconds() if first and last else 0.0
    return out, failures, {"workers": workers, "wall_seconds": span}


def _quantile(xs: Sequence[float], q: float) -> float:
    ordered = sorted(xs)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else float("nan")


def forecast(pilots: Sequence[str]) -> int:
    """The pilots' measured latencies by role, their throughput and failures, and the FAST,
    EXPECTED and SLOW wall times of TEST v2 they imply (config/holdout/test-v2-protocol.md)."""
    per_role: dict[str, list[float]] = defaultdict(list)
    report: dict[str, Any] = {"pilots": {}}
    for pilot in pilots:
        namespace, raw = _pilot_paths(pilot)
        with vs.Store.open(namespace, raw, readonly=True) as store:
            events = store.events()
            kind = store.meta("kind")
        durations, failures, info = _durations(events)
        runs = sum(len(v) for v in durations.values())
        report["pilots"][pilot] = {
            "kind": kind,
            **info,
            "responses": runs,
            "responses_per_minute": runs / (info["wall_seconds"] / 60)
            if info["wall_seconds"]
            else None,
            "failures": dict(failures),
            "median_seconds": {r: statistics.median(v) for r, v in sorted(durations.items())},
        }
        for role, values in durations.items():
            per_role[role].extend(values)
    report["latency_seconds"] = {
        r: {
            "n": len(v),
            "p50": _quantile(v, 0.5),
            "p75": _quantile(v, 0.75),
            "p90": _quantile(v, 0.9),
        }
        for r, v in sorted(per_role.items())
    }
    _print(report)
    return 0


# ---------------------------------------------------------------------------
# the contract and TEST v2 itself
# ---------------------------------------------------------------------------


def prepare(certification: str, manifest: str) -> int:
    """The pre-registration contract (D-067), written once; its SHA-256 pinned in the private
    provenance. The salt is made here if absent: from it the bank, which nothing prints."""
    provenance = hc2.NAMESPACE / "provenance"
    if not (provenance / hc2.EXCLUSIONS_FILE).exists():
        print("STOP: the historical exclusions are missing (exclusions)")
        return 1
    if not (provenance / hc2.DEVELOPMENT_REFERENCES_FILE).exists():
        print("STOP: the development references are missing (development-references)")
        return 1
    salt = provenance / hc2.SALT_FILE
    if not salt.exists():
        salt.write_text(secrets.token_hex(32) + "\n", encoding="utf-8")
    cert = _certification(certification, manifest)
    private = hc2.read_private(provenance)
    contract = hc2.build(REPO, cert, private, hrun.Params())
    digest = hc2.write(contract)
    ho.write_json(
        provenance / hc2.PIN_FILE,
        {"path": hc2.CONTRACT_PATH.as_posix(), "sha256": digest, "pinned_at": _now()},
    )
    hc2.verify(digest)
    print(f"contract {hc2.CONTRACT_PATH}: {digest}")
    return 0


def record_public_commit(commit: str) -> int:
    provenance = hc2.NAMESPACE / "provenance"
    pin = json.loads((provenance / hc2.PIN_FILE).read_text(encoding="utf-8"))
    contract = hc2.verify(pin["sha256"])
    problems = hc2.committed_problems(contract, pin["sha256"], REPO, commit)
    if problems:
        print(f"STOP: {problems[:5]}")
        return 1
    ho.write_json(
        provenance / hc2.PUBLIC_COMMIT_FILE,
        {"commit": commit, "contract_sha256": pin["sha256"], "recorded_at": _now()},
    )
    print(f"public commit {commit} recorded for {pin['sha256'][:12]}")
    return 0


def _pinned(confirm: str) -> str:
    pin = json.loads((hc2.NAMESPACE / "provenance" / hc2.PIN_FILE).read_text(encoding="utf-8"))
    if confirm != pin["sha256"]:
        raise ho.HoldoutError("the contract's full SHA-256, typed again, is not the pinned one")
    return str(pin["sha256"])


def validate(confirm: str) -> int:
    pinned = _pinned(confirm)
    contract, _, problems = hc2.problems(pinned, REPO)
    public = (
        hc2.public_commit_problems(contract, pinned, REPO, hc2.NAMESPACE / "provenance")
        if contract
        else []
    )
    _print({"contract": pinned, "problems": problems, "public_commit_problems": public})
    return 1 if problems or public else 0


def _test_world(
    contract: Mapping[str, Any], private: hc2.Private, workers: int, pinned: str
) -> tuple[hrun.World, Any]:
    c = _common()
    slots = v2.slots(c["dev"])
    units = v2.legal_units(c["docs"], ho.load_areas(), c["excluded"], c["layer"].layer)
    bank = v2.build_bank(slots, units, private.salt)
    if bank.sha256() != contract["design"]["bank_sha256"]:
        raise ho.HoldoutError("the bank is not the contract's")
    reference = contract["certification"]
    cert = _certification(reference["id"], reference["manifest_sha256"])
    executor, audit = _executor_and_audit(
        cert, hc2.RAW_DIR, contract["roles"]["templates_sha256"], _forbidden(c)
    )
    params = hrun.Params(workers=workers, **contract["design"]["params"])

    def check(phase: str) -> None:
        _, _, problems = hc2.problems(pinned, REPO)
        if problems:
            raise en.V2Stop("contract", f"{phase}: {problems[:5]}")

    world = hrun.World(
        namespace=hc2.NAMESPACE,
        corpus=c["corpus"],
        layer=c["layer"].layer,
        chunk_text={k: ch.text for k, ch in c["layer"].chunks.items()},
        excluded_psd2=c["excluded"],
        forbidden=_forbidden(c),
        references=hc2.references(c["dev"], c["fixtures"], private),
        bank=bank,
        chains=v2.completeness_chains(bank, private.salt),
        audit_chain=hau.chain(bank, private.salt),
        salt=private.salt,
        params=params,
        audit=audit,
        rank=_rank(c["layer"]),
        write_runs=rv.write_runs,
        read_runs=rv.read_runs,
        vectors=_bge_vectors(c["layer"]),
        check=check,
    )
    return world, executor


def construct(action: str, confirm: str, workers: int) -> int:
    """START or RESUME of TEST v2: the only commands that run a model on TEST v2."""
    pinned = _pinned(confirm)
    contract, private, problems = hc2.problems(pinned, REPO)
    if contract is None or private is None or problems:
        print(f"STOP: {problems[:8]}")
        return 1
    exists = (hc2.NAMESPACE / vs.STATE_DIR / vs.DB_FILE).exists()
    if action == "start":
        if exists:
            print("STOP: a TEST v2 state exists: START never starts it again")
            return 1
        public = hc2.public_commit_problems(contract, pinned, REPO, hc2.NAMESPACE / "provenance")
        if public:
            print(f"STOP: {public[:5]}")
            return 1
    elif not exists:
        print("STOP: no TEST v2 state: RESUME never makes one up")
        return 1
    world, executor = _test_world(contract, private, workers, pinned)
    meta = {"kind": "test-v2", "contract_sha256": pinned, "bank_sha256": world.bank.sha256()}
    with _store(hc2.NAMESPACE, hc2.RAW_DIR, meta) as store:
        store.verify()
        runner = hrun.Runner(world, store, executor)
        _session(runner, workers, action)
        try:
            summary = runner.build()
        except en.V2Stop as stop:
            print(f"STOPPED (integrity): {stop.reason}")
            return 3
        except en.Paused as pause:
            print(f"PAUSED (provider): {pause}; resume later")
            return 4
    _print(summary)
    return 0


def status() -> int:
    """TEST v2's state, verified, read only, as counts."""
    with vs.Store.open(hc2.NAMESPACE, hc2.RAW_DIR, readonly=True) as store:
        summary = store.verify()
        ledger = en.Ledger.fold(store.events())
    keyed = ledger.keyed
    _print(
        {
            "events": summary["events"],
            "head": summary["head_sha256"],
            "freezes": [
                k
                for k in ("bank", "topics", "runs", "pools", "qrels", "completeness", "evaluation")
                if k in keyed
            ],
            "generated": sum(k.startswith("generated/") for k in keyed),
            "dropped": sum(k.startswith("drop/") for k in keyed),
            "judged": sum(k.startswith("judged/") for k in keyed),
            "evaluable": sum(
                bool(p.get("evaluable")) for k, p in keyed.items() if k.startswith("judged/")
            ),
            "stops": ledger.stops,
        }
    )
    return 0


def evaluate(confirm: str) -> int:
    """S4 on TEST v2: only on frozen qrels, once."""
    from ingest import holdout_v2_evaluation as hev

    pinned = _pinned(confirm)
    contract, private, problems = hc2.problems(pinned, REPO)
    if contract is None or private is None or problems:
        print(f"STOP: {problems[:8]}")
        return 1
    world, executor = _test_world(contract, private, 1, pinned)
    layer = world.layer
    world.evaluate = lambda inputs: hev.evaluate(hev.frozen_inputs(inputs, layer))
    with vs.Store.open(hc2.NAMESPACE, hc2.RAW_DIR) as store:
        store.verify()
        runner = hrun.Runner(world, store, executor)
        try:
            runner.evaluation()
        except hrun.NotReady as error:
            print(f"STOP: {error}")
            return 1
    print(f"evaluation written: {hrun.EVALUATION_FILE}")
    return 0


AUDIT_DIR = "human-audit"


def _frozen_file(keyed: Mapping[str, Any], key: str, relative: Path) -> Any:
    text = (hc2.NAMESPACE / relative).read_text(encoding="utf-8")
    if vs.sha_text(text) != keyed[key]["sha256"]:
        raise ho.HoldoutError(f"{relative} differs from its freeze")
    return json.loads(text)


def audit_export(confirm: str) -> int:
    """The optional human audit's blind packet (D-067): the decisive-set passages of its
    predetermined needs with decoys, under opaque ids, into the private namespace. Nothing of a
    system, a rank, a formulation or a judge's label is in it."""
    _pinned(confirm)
    with vs.Store.open(hc2.NAMESPACE, hc2.RAW_DIR, readonly=True) as store:
        keyed = en.Ledger.fold(store.events()).keyed
    if "human_audit/selection" not in keyed:
        print("STOP: the human audit's needs are not selected yet (after the qrel freeze)")
        return 1
    needs = keyed["human_audit/selection"]["needs"]
    salt = hc2.read_salt(hc2.NAMESPACE / "provenance")
    pools = _frozen_file(keyed, "pools", hrun.POOLS_FILE)["pools"]
    topics = {t["need_id"]: t for t in _frozen_file(keyed, "topics", hrun.TOPICS_FILE)["topics"]}
    layer = rv.load_layer()
    items = {n: hau.items(n, pools[n], salt) for n in needs}
    packet, template = hau.packet(
        [(n, topics[n]["statement"], items[n]) for n in needs],
        {c: ch.text for c, ch in layer.chunks.items()},
        salt,
    )
    out = hc2.NAMESPACE / AUDIT_DIR
    out.mkdir()
    (out / "packet.md").write_text(packet, encoding="utf-8")
    (out / "labels.template.json").write_text(json.dumps(template, indent=1) + "\n")
    key = [dataclasses.asdict(i) for n in needs for i in items[n]]
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"{sum(len(v) for v in items.values())} passages of {len(needs)} needs in {out}")
    return 0


def audit_sa7(labels_path: Path, confirm: str) -> int:
    """Sensitivity SA7 from the person's labels: written beside the audit, never into the qrels
    or the evaluation."""
    from ingest import holdout_v2_evaluation as hev

    pinned = _pinned(confirm)
    contract, private, problems = hc2.problems(pinned, REPO)
    if contract is None or private is None or problems:
        print(f"STOP: {problems[:8]}")
        return 1
    out = hc2.NAMESPACE / AUDIT_DIR
    key = [hau.Item(**i) for i in json.loads((out / "key.json").read_text(encoding="utf-8"))]
    labels = hau.read_labels(labels_path.read_text(encoding="utf-8"), key)
    world, _ = _test_world(contract, private, 1, pinned)
    with vs.Store.open(hc2.NAMESPACE, hc2.RAW_DIR) as store:
        store.verify()
        try:
            inputs = hrun.Runner(world, store, None).evaluation_inputs()
        except hrun.NotReady as error:
            print(f"STOP: {error}")
            return 1
    frozen = hev.frozen_inputs(inputs, world.layer)
    result = hev.sa7_human(
        dataclasses.replace(frozen, human={"labels": labels}), hev.primary_qrels(frozen)
    )
    target = out / "sa7.json"
    if target.exists():
        raise ho.HoldoutError(f"{target} exists: SA7 is computed once")
    target.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SA7 written: {target}")
    return 0


def backup() -> int:
    with vs.Store.open(hc2.NAMESPACE, hc2.RAW_DIR) as store:
        _print(vs.backup(store, ho.HOLDOUT_DIR, REPO, ho.BACKUP_DIR, "manual backup"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("exclusions")
    sub.add_parser("faithfulness")
    sub.add_parser("pool-validation")
    for name in ("pilot-generation", "pilot-judging"):
        p = sub.add_parser(name)
        p.add_argument("pilot")
        p.add_argument("--workers", type=int, required=True)
        p.add_argument("--certification", required=True)
        p.add_argument("--certification-manifest", required=True)
        if name == "pilot-generation":
            p.add_argument("--universe", type=int, required=True)
        else:
            p.add_argument("--needs", type=int, required=True)
            p.add_argument("--completeness")
            p.add_argument("--first")
    p = sub.add_parser("development-references")
    p.add_argument("pilots", nargs="+")
    p = sub.add_parser("forecast")
    p.add_argument("pilots", nargs="+")
    p = sub.add_parser("prepare")
    p.add_argument("--certification", required=True)
    p.add_argument("--certification-manifest", required=True)
    p = sub.add_parser("record-public-commit")
    p.add_argument("commit")
    for name in ("validate", "start", "resume", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--contract", required=True)
        if name in ("start", "resume"):
            p.add_argument("--workers", type=int, required=True)
    p = sub.add_parser("audit-export")
    p.add_argument("--contract", required=True)
    p = sub.add_parser("audit-sa7")
    p.add_argument("labels", type=Path)
    p.add_argument("--contract", required=True)
    sub.add_parser("status")
    sub.add_parser("backup")
    args = parser.parse_args(argv)
    try:
        if args.command == "exclusions":
            return exclusions()
        if args.command == "faithfulness":
            return faithfulness()
        if args.command == "pool-validation":
            return pool_validation()
        if args.command == "pilot-generation":
            return pilot_generation(
                args.pilot,
                args.universe,
                args.workers,
                args.certification,
                args.certification_manifest,
            )
        if args.command == "pilot-judging":
            return pilot_judging(
                args.pilot,
                args.needs,
                args.workers,
                args.completeness,
                args.certification,
                args.certification_manifest,
                args.first,
            )
        if args.command == "development-references":
            return development_references(args.pilots)
        if args.command == "forecast":
            return forecast(args.pilots)
        if args.command == "prepare":
            return prepare(args.certification, args.certification_manifest)
        if args.command == "record-public-commit":
            return record_public_commit(args.commit)
        if args.command == "validate":
            return validate(args.contract)
        if args.command in ("start", "resume"):
            return construct(args.command, args.contract, args.workers)
        if args.command == "status":
            return status()
        if args.command == "evaluate":
            return evaluate(args.contract)
        if args.command == "audit-export":
            return audit_export(args.contract)
        if args.command == "audit-sa7":
            return audit_sa7(args.labels, args.contract)
        if args.command == "backup":
            return backup()
    except (ho.HoldoutError, hcert.CertificationError, hc2.ContractV2Error, vs.StateError) as error:
        print(f"STOP: {error}")
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
