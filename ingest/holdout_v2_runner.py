"""TEST v2's construction, phase by phase (D-067, config/holdout/test-v2-protocol.md).

1. bank: the frozen candidate bank, its anchors, the completeness chains and the human audit's
   chain, written once and recorded by their hashes;
2. generation: every candidate's concept (with its anchors), statement and three queries, in
   parallel; the anchor-copy filter; a candidate the generation cannot use is dropped, never
   replaced by a new one;
3. deduplication: one global stage in priority order (ingest/holdout_v2_dedup.py);
4. topic freeze: the selected bank, written once; no topic text changes after it;
5. retrieval: every frozen system on every frozen query, one local model at a time, the runs
   frozen by hash (ingest/holdout_v2_retrieval.py);
6. pools: each need's pool and decisive set, frozen;
7. judging, in waves in priority order: B1, B2 and C (ingest/holdout_v2_judging.py), until the
   first N_TARGET evaluable needs exist or the bank is exhausted;
8. qrel freeze: the primary sample and its qrels, written once;
9. answerability and metadata of the primary sample, descriptive;
10. the completeness diagnostic on its frozen chains, and the human audit's selection;
11. the evaluation (ingest/holdout_v2_evaluation.py), which nothing before the qrel freeze can
    reach.

Every phase is idempotent: a restart replays what is admitted and goes on, and a phase whose
freeze is recorded verifies it instead of building it again. Only integrity failures stop it
(holdout_v2_engine.V2Stop); everything else (a dropped candidate, a duplicate, a need that is not
evaluable, an exhausted job, a disagreement, an ambiguity, a residual relevant chunk, a small N)
is a result, counted.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ingest import benchmark as bm
from ingest import evaluation as ev
from ingest import holdout as ho
from ingest import holdout_admission as ha
from ingest import holdout_judging as hj
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from ingest import holdout_v2 as v2
from ingest import holdout_v2_audit as hau
from ingest import holdout_v2_dedup as dd
from ingest import holdout_v2_engine as en
from ingest import holdout_v2_judging as jv
from ingest import holdout_v2_packets as v2p
from ingest import holdout_v2_retrieval as rv
from ingest import holdout_v2_state as vs

TOPICS_SCHEMA = "holdout-v2-topics.v1"
POOLS_SCHEMA = "holdout-v2-pools.v1"
QRELS_SCHEMA = "holdout-v2-qrels.v1"
COMPLETENESS_SCHEMA = "holdout-v2-completeness.v1"
BANK_FILE = Path("bank") / "bank.json"
TOPICS_FILE = Path("topics") / "topics.json"
RUNS_DIR = Path("runs")
RUNS_WORK = Path("runs-work")
POOLS_FILE = Path("pools") / "pools.json"
QRELS_FILE = Path("qrels") / "qrels.json"
COMPLETENESS_FILE = Path("completeness") / "completeness.json"
EVALUATION_FILE = Path("evaluation") / "evaluation.json"
SENSITIVITY_FILE = Path("evaluation") / "sensitivity.json"
STATUS_FILE = "status.json"
# S4: a system that fails technically runs again, unchanged, once; then it is not completed
RETRIEVAL_ATTEMPTS = 2


class NotReady(Exception):
    """A step was asked for before what it reads is frozen: nothing was done."""


@dataclass(frozen=True)
class Params:
    """The operational parameters the dry run may set before the contract (D-067), never the
    design. The batch ceilings shape the packets, so the contract freezes them and the bank's
    freeze records them; the workers and the backoff only pace the calls, and a resume may
    lower them (TEST then takes longer)."""

    workers: int = 6
    pool_max_nodes: int = hp.POOL_BATCH_MAX_NODES
    pool_max_tokens: int = hp.POOL_BATCH_MAX_TOKENS
    screen_max_passages: int = 50
    screen_max_tokens: int = 24_000
    backoff_seconds: float = 30.0

    def record(self) -> dict[str, Any]:
        """What shapes a packet: frozen."""
        return {
            "pool_max_nodes": self.pool_max_nodes,
            "pool_max_tokens": self.pool_max_tokens,
            "screen_max_passages": self.screen_max_passages,
            "screen_max_tokens": self.screen_max_tokens,
        }


@dataclass
class World:
    namespace: Path
    corpus: hp.Corpus
    layer: ev.ChunkLayer
    chunk_text: Mapping[str, str]  # chunk id -> its text: the screen's passages
    excluded_psd2: frozenset[str]
    forbidden: Sequence[str]  # DEV's and the fixtures' statements: the authors' guard
    references: Sequence[dd.Reference]  # the deduplication's hidden references
    bank: v2.Bank
    chains: Mapping[str, Sequence[str]]  # the completeness chains, per act
    audit_chain: Sequence[str]  # the human audit's chain
    salt: bytes
    params: Params
    audit: en.AuditContext
    # one frozen retriever on the queries ((query id, text) pairs) with a work folder of its
    # own: its run (holdout_v2_retrieval.Run); the systems in the order they run (E after E5)
    rank: Callable[[str, Sequence[tuple[str, str]], Path], rv.Run]
    write_runs: Callable[[Mapping[str, rv.Run], Path], dict[str, Any]]
    read_runs: Callable[[Path, Mapping[str, Any]], dict[str, dict[str, list[str]]]]
    # BGE-M3 vectors of statements: the deduplication's prefilter
    vectors: Callable[[Sequence[str]], np.ndarray]
    # the integrity checks at a phase boundary (contract, corpus, retrievers, runtime): raises
    # V2Stop; tests give a stand-in
    check: Callable[[str], None] = lambda phase: None
    evaluate: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None
    n_target: int = v2.N_TARGET
    count_tokens: Callable[[str], int] = field(default=hp.packet_tokens)
    # a development pilot's topics, instead of generation (DEV needs for pilot C): never TEST's
    topics_source: Callable[[], list[dict[str, Any]]] | None = None
    systems: Sequence[str] = rv.SYSTEMS

    @property
    def texts(self) -> Mapping[bm.NodeKey, str]:
        return self.corpus.texts


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _node(ref: str) -> bm.NodeKey:
    act, _, path = ref.partition(":")
    return act, path


class Runner:
    def __init__(self, world: World, store: vs.Store, executor: Any) -> None:
        self.w = world
        self.store = store
        self.engine = en.Engine(
            store, executor, world.audit, world.params.workers, world.params.backoff_seconds
        )
        self._by_id = {c.candidate_id: c for c in world.bank.candidates}

    # -----------------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------------

    @property
    def keyed(self) -> dict[str, dict[str, Any]]:
        return self.engine.ledger.keyed

    def record(self, kind: str, key: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return self.engine.record(kind, key, payload).payload
        except vs.Diverged as error:
            raise self.stop("state", str(error)) from error

    def stop(self, reason: str, detail: str) -> en.V2Stop:
        return self.engine.stop(en.V2Stop(reason, detail))

    def ref(self, kind: str, key: str) -> str:
        return ho.opaque_id(self.w.salt, f"v2-{kind}", key)

    def candidate(self, cid: str) -> v2.Candidate:
        return self._by_id[cid]

    def dropped(self, cid: str) -> bool:
        return f"drop/{cid}" in self.keyed

    def drop(self, cid: str, stage: str, reason: str) -> None:
        self.record(
            "candidate_dropped",
            f"drop/{cid}",
            {"candidate_id": cid, "stage": stage, "reason": reason},
        )

    def write_frozen(self, relative: Path, value: Any) -> str:
        """A frozen artifact: canonical JSON in the namespace, written once and kept in the
        state too. Its SHA-256 is what the freeze event records; a file already there must be
        exactly it."""
        text = vs.canonical(value) + "\n"
        digest, _ = self.store.put(text, "kept")
        path = self.w.namespace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if vs.sha_text(path.read_text(encoding="utf-8")) != digest:
                raise self.stop("topic_mutation", f"{relative} differs from what it freezes")
        else:
            path.write_text(text, encoding="utf-8")
        return digest

    def read_frozen(self, key: str, relative: Path) -> Any:
        """A frozen artifact, its file checked against the hash its freeze recorded."""
        digest = self.keyed[key]["sha256"]
        path = self.w.namespace / relative
        if not path.exists() or vs.sha_text(path.read_text(encoding="utf-8")) != digest:
            reason = "topic_mutation" if key in ("topics", "qrels") else "artifact"
            raise self.stop(reason, f"{relative} is missing or differs from its freeze")
        return json.loads(self.store.text(digest, "kept"))

    def status(self, phase: str, counts: Mapping[str, Any]) -> None:
        """Counts only: never a statement, a query, a label or a metric."""
        path = self.w.namespace / STATUS_FILE
        body = {"phase": phase, "at": _now(), **counts}
        path.write_text(json.dumps(body, sort_keys=True, indent=1) + "\n", encoding="utf-8")

    # -----------------------------------------------------------------------
    # the construction
    # -----------------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        try:
            self.w.check("start")
            self.freeze_bank()
            if "topics" not in self.keyed:
                if self.w.topics_source is not None:
                    self.import_topics()
                else:
                    self.generate()
                    self.deduplicate()
                    self.freeze_topics()
            self.topics()
            self.w.check("retrieval")
            self.retrieve()
            self.pools()
            self.w.check("judging")
            self.judge()
            self.freeze_qrels()
            self.answerability()
            self.metadata()
            self.completeness()
            self.audit_selection()
            if self.w.evaluate is not None:
                self.w.check("evaluation")
                self.evaluation()
        except vs.IntegrityError as error:
            raise self.stop("state", str(error)) from error
        return self.summary()

    # 1 -----------------------------------------------------------------------

    def freeze_bank(self) -> None:
        bank = self.w.bank
        record = {
            **bank.record(),
            "completeness_chains": {a: list(c) for a, c in sorted(self.w.chains.items())},
            "human_audit_chain": list(self.w.audit_chain),
        }
        digest = self.write_frozen(BANK_FILE, record)
        self.record(
            "bank_frozen",
            "bank",
            {
                "sha256": digest,
                "bank_sha256": bank.sha256(),
                "universe": len(bank.candidates),
                "params": self.w.params.record(),
            },
        )
        self.record(
            "anchors_assigned",
            "anchors",
            {
                "anchors_sha256": v2.sha256_value(
                    [[c.candidate_id, [asdict(a) for a in c.anchors]] for c in bank.candidates]
                ),
                "capacity": dict(sorted(bank.capacity.items())),
                "reuse": dict(sorted(bank.reuse.items())),
            },
        )

    # 2 -----------------------------------------------------------------------

    def _anchors(self, c: v2.Candidate) -> list[str]:
        return v2p.anchor_texts(self.w.corpus, c)

    def _concept_job(self, c: v2.Candidate) -> en.Job:
        anchors = self._anchors(c)
        forbidden = self.w.forbidden

        def admit(reply: str) -> Any:
            value = en.reply_json(reply)
            problems = v2p.concept_problems(value, anchors, forbidden)
            if problems:
                raise en.Inadmissible("concept_contract", problems)
            return value

        packet = v2p.concept_packet(self.w.corpus, c, self.ref("concept", c.candidate_id))
        return en.Job(f"gen/concept/{c.candidate_id}", "v2_concept_author", packet, admit)

    def _writer_job(self, c: v2.Candidate, concept: Mapping[str, Any]) -> en.Job:
        forbidden = self.w.forbidden

        def admit(reply: str) -> Any:
            value = en.reply_json(reply)
            problems = v2p.writer_problems(value, forbidden)
            if problems:
                raise en.Inadmissible("statement_contract", problems)
            return value

        facts = [str(x) for x in concept["facts"]]
        ref = self.ref("writer", c.candidate_id)
        packet = v2p.writer_packet(c, ref, str(concept["concept"]), facts, forbidden)
        return en.Job(
            f"gen/writer/{c.candidate_id}",
            "statement_writer",
            packet,
            admit,
            {"concept": v2.sha256_value(dict(concept))},
        )

    def _queries_job(self, c: v2.Candidate, statement: str) -> en.Job:
        def admit(reply: str) -> Any:
            try:
                queries = hpl.parse_queries(str(reply), c.need_id)
            except hpl.TechnicalFailure as error:
                raise en.Inadmissible("query_formulations", [error.detail]) from None
            return [q.model_dump(mode="json") for q in queries]

        return en.Job(
            f"gen/queries/{c.candidate_id}",
            "query_writer",
            v2p.query_packet(statement),
            admit,
            {"statement": vs.sha_text(statement)},
        )

    def generate(self) -> None:
        """The three hops of every candidate, each hop in parallel over the bank: no candidate
        waits for another, and the order calls finish in decides nothing."""
        todo = [
            c
            for c in self.w.bank.candidates
            if not self.dropped(c.candidate_id) and f"generated/{c.candidate_id}" not in self.keyed
        ]
        concepts = self.engine.run([self._concept_job(c) for c in todo])
        writing = []
        for c in todo:
            o = concepts[f"gen/concept/{c.candidate_id}"]
            if o.exhausted:
                self.drop(c.candidate_id, "concept", "technical_exhausted")
            elif not v2p.concept_fits(o.value):
                self.drop(c.candidate_id, "concept", "concept_unfit")
            else:
                writing.append((c, o.value))
        writer_jobs, packed = [], []
        for c, concept in writing:
            try:
                writer_jobs.append(self._writer_job(c, concept))
                packed.append((c, concept))
            except hp.PacketError:  # the frozen writer packet refuses what would leak
                self.drop(c.candidate_id, "statement", "packet_leak")
        written = self.engine.run(writer_jobs)
        formulating = []
        for c, _ in packed:
            o = written[f"gen/writer/{c.candidate_id}"]
            if o.exhausted:
                self.drop(c.candidate_id, "statement", "technical_exhausted")
            else:
                formulating.append((c, str(o.value["primary"]).strip()))
        formulated = self.engine.run([self._queries_job(c, s) for c, s in formulating])
        for c, statement in formulating:
            o = formulated[f"gen/queries/{c.candidate_id}"]
            if o.exhausted:
                self.drop(c.candidate_id, "queries", "technical_exhausted")
                continue
            leak = v2p.leakage(statement, [q["text"] for q in o.value], self._anchors(c))
            self.record(
                "candidate_generated",
                f"generated/{c.candidate_id}",
                {
                    "candidate_id": c.candidate_id,
                    "need_id": c.need_id,
                    "statement": statement,
                    "queries": o.value,
                    "leakage": {
                        "longest_run": leak.longest_run,
                        "share": leak.share,
                        "rejected": leak.rejected,
                    },
                },
            )
            if leak.rejected:
                self.drop(c.candidate_id, "leakage", "anchor_copy")
        # the filter's drop, from the recorded generation itself: a restart between the two
        # records still drops the candidate
        for c in self.w.bank.candidates:
            g = self.keyed.get(f"generated/{c.candidate_id}")
            if g is not None and g["leakage"]["rejected"] and not self.dropped(c.candidate_id):
                self.drop(c.candidate_id, "leakage", "anchor_copy")
        self.status("generation", self.generation_counts())

    def generation_counts(self) -> dict[str, Any]:
        drops = Counter(
            f"{p['stage']}:{p['reason']}" for k, p in self.keyed.items() if k.startswith("drop/")
        )
        generated = sum(k.startswith("generated/") for k in self.keyed)
        return {"generated": generated, "dropped": dict(sorted(drops.items()))}

    def generated(self) -> list[dict[str, Any]]:
        """The generated candidates the generation kept, in priority order: never one dropped,
        never one whose recorded generation copies its anchor."""
        out = []
        for c in self.w.bank.candidates:
            g = self.keyed.get(f"generated/{c.candidate_id}")
            if g is not None and not self.dropped(c.candidate_id) and not g["leakage"]["rejected"]:
                out.append(g)
        return out

    def generation_report(self) -> dict[str, Any]:
        """The anchor leakage diagnostic (D-067): its distribution over every generated
        candidate, the rejection rate, and the drops by stage, by profile and by acts. Counts
        and overlaps only."""
        rows = [p for k, p in self.keyed.items() if k.startswith("generated/")]
        runs = [int(p["leakage"]["longest_run"]) for p in rows]
        shares = [float(p["leakage"]["share"]) for p in rows]
        rejected = sum(bool(p["leakage"]["rejected"]) for p in rows)
        drops = [p for k, p in self.keyed.items() if k.startswith("drop/")]

        def by(attribute: Callable[[v2.Candidate], str]) -> dict[str, dict[str, int]]:
            out: dict[str, Counter[str]] = {}
            for c in self.w.bank.candidates:
                out.setdefault(attribute(c), Counter())["candidates"] += 1
            for p in drops:
                out[attribute(self.candidate(p["candidate_id"]))][
                    f"{p['stage']}:{p['reason']}"
                ] += 1
            return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}

        return {
            "generated": len(rows),
            "longest_run": {str(k): v for k, v in sorted(Counter(runs).items())},
            "share": {
                "mean": float(np.mean(shares)) if shares else None,
                "quartiles": [float(x) for x in np.quantile(shares, [0.25, 0.5, 0.75])]
                if shares
                else [],
            },
            "anchor_copy_rejections": rejected,
            "rejection_rate": rejected / len(rows) if rows else None,
            "dropped": dict(sorted(Counter(f"{p['stage']}:{p['reason']}" for p in drops).items())),
            "by_profile": by(lambda c: c.slot.slot_id),
            "by_acts": by(lambda c: "+".join(c.slot.acts)),
        }

    # 3 -----------------------------------------------------------------------

    def _gate_job(self, cid: str, n: int, statement: str, shown: dd.Shown) -> en.Job:
        ids = [s for s, _, _ in shown.existing]

        def admit(reply: str) -> Any:
            value = en.reply_json(reply)
            problems = ha.gate_problems(value, ids)
            if problems:
                raise en.Inadmissible("gate_contract", problems)
            return value

        ref = self.ref("gatekeeper", f"{cid}:{n}")
        existing = [hp.Existing(s, kind, text) for s, kind, text in shown.existing]
        packet = v2p.gate_packet(ref, statement, existing)
        return en.Job(f"dedup/gate/{cid}/{n}", "gatekeeper", packet, admit)

    def _items(self) -> list[dd.Item]:
        return [
            dd.Item(
                g["candidate_id"],
                self.candidate(g["candidate_id"]).rank,
                g["statement"],
                tuple((a.act, a.article) for a in self.candidate(g["candidate_id"]).anchors),
            )
            for g in self.generated()
        ]

    def deduplicate(self) -> None:
        """One global stage over everything generated, in priority order: the neighborhoods
        are nominated once and recorded before the first gate call; the gate runs in waves of
        candidates whose earlier neighbors are decided; resolve() must then give, with the
        recorded answers, exactly the decisions recorded."""
        items = self._items()
        refs = {r.key: r for r in self.w.references}
        if len(refs) != len(self.w.references):
            raise self.stop("contract", "two deduplication references share a key")
        if "dedup/neighborhoods" not in self.keyed:
            keys = [r.key for r in self.w.references] + [i.candidate_id for i in items]
            texts = [r.statement for r in self.w.references] + [i.statement for i in items]
            vectors = dict(zip(keys, self.w.vectors(texts), strict=True))
            hoods = dd.neighborhoods(items, self.w.references, vectors)
            self.record(
                "dedup_neighborhoods",
                "dedup/neighborhoods",
                {
                    "neighborhoods": {k: h.record() for k, h in sorted(hoods.items())},
                    "references": dict(sorted(Counter(r.kind for r in self.w.references).items())),
                    "references_sha256": v2.sha256_value(
                        [[r.key, r.kind, r.statement] for r in self.w.references]
                    ),
                },
            )
        recorded = self.keyed["dedup/neighborhoods"]["neighborhoods"]
        if set(recorded) != {i.candidate_id for i in items}:
            raise self.stop("state", "the recorded neighborhoods are not the generated candidates")
        hoods = {
            k: dd.Neighborhood(tuple(h["references"]), tuple(h["bank"]))
            for k, h in recorded.items()
        }
        resolved = {i.candidate_id for i in items if f"dedup/{i.candidate_id}" in self.keyed}
        kept = {
            i.candidate_id: i.statement
            for i in items
            if self.keyed.get(f"dedup/{i.candidate_id}", {}).get("kept")
        }
        pending = sorted((i for i in items if i.candidate_id not in resolved), key=lambda i: i.rank)
        while pending:
            wave = dd.ready(pending, hoods, resolved)
            if not wave:
                raise self.stop("state", "the deduplication cannot progress")
            jobs, plan = [], {}
            for item in wave:
                gates = dd.packets(
                    item.candidate_id, hoods[item.candidate_id], kept, refs, self.w.salt
                )
                plan[item.candidate_id] = gates
                jobs += [
                    self._gate_job(item.candidate_id, n, item.statement, s)
                    for n, s in enumerate(gates, start=1)
                ]
            outcomes = self.engine.run(jobs)
            for item in sorted(wave, key=lambda i: i.rank):
                cid = item.candidate_id
                answers = [outcomes[f"dedup/gate/{cid}/{n}"] for n in range(1, len(plan[cid]) + 1)]
                verdicts = [
                    "technical_exhausted" if o.exhausted else str(o.value["verdict"])
                    for o in answers
                ]
                keep = all(v == "independent" for v in verdicts)
                closest_kind = None
                for o, shown in zip(answers, plan[cid], strict=True):
                    if not o.exhausted and o.value["verdict"] != "independent":
                        kinds = {s: kind for s, kind, _ in shown.existing}
                        closest_kind = kinds.get(o.value.get("closest") or "")
                        break
                self.record(
                    "dedup_decided",
                    f"dedup/{cid}",
                    {
                        "candidate_id": cid,
                        "kept": keep,
                        "verdicts": verdicts,
                        "closest_kind": closest_kind,
                    },
                )
                resolved.add(cid)
                if keep:
                    kept[cid] = item.statement
            pending = [i for i in pending if i.candidate_id not in resolved]
        self._check_dedup(items, hoods, refs)
        self.status("deduplication", self._dedup_counts(items))

    def _check_dedup(
        self,
        items: Sequence[dd.Item],
        hoods: Mapping[str, dd.Neighborhood],
        refs: Mapping[str, dd.Reference],
    ) -> None:
        """The recorded decisions are exactly resolve()'s on the recorded answers: the parallel
        waves and the one-by-one order in priority give one selection."""

        def oracle(cid: str, n: int, shown: dd.Shown) -> Mapping[str, Any]:
            job = f"dedup/gate/{cid}/{n}"
            if job in self.engine.ledger.admitted:
                return dict(self.engine.ledger.admitted[job])
            return {"verdict": "technical_exhausted"}

        again = dd.resolve(items, hoods, refs, self.w.salt, oracle)
        for cid, d in again.items():
            if d.kept != self.keyed[f"dedup/{cid}"]["kept"]:
                raise self.stop("state", f"the deduplication of {cid} diverges from resolve()")

    def _dedup_counts(self, items: Sequence[dd.Item]) -> dict[str, Any]:
        decided = [self.keyed[f"dedup/{i.candidate_id}"] for i in items]
        return {
            "candidates": len(items),
            "kept": sum(d["kept"] for d in decided),
            "duplicates_by_kind": dict(
                sorted(
                    Counter(
                        "technical_exhausted"
                        if "technical_exhausted" in d["verdicts"]
                        else str(d["closest_kind"])
                        for d in decided
                        if not d["kept"]
                    ).items()
                )
            ),
        }

    # 4 -----------------------------------------------------------------------

    def selected(self) -> list[dict[str, Any]]:
        return [
            g
            for g in self.generated()
            if self.keyed.get(f"dedup/{g['candidate_id']}", {}).get("kept")
        ]

    @staticmethod
    def topic_hash(topic: Mapping[str, Any]) -> str:
        return v2.sha256_value(dict(topic))

    def freeze_topics(self) -> None:
        topics = []
        for g in self.selected():
            c = self.candidate(g["candidate_id"])
            topics.append(
                {
                    "rank": c.rank,
                    "candidate_id": c.candidate_id,
                    "need_id": c.need_id,
                    "priority": c.priority,
                    "slot": asdict(c.slot),
                    "anchors": [asdict(a) for a in c.anchors],
                    "statement": g["statement"],
                    "queries": g["queries"],
                    "leakage": g["leakage"],
                }
            )
        self._freeze_topic_list(topics, {"generation": self.generation_report()})

    def import_topics(self) -> None:
        """A development pilot's topics (pilot C: DEV's needs and queries), frozen exactly as
        TEST's would be, so that every later phase runs unchanged on them."""
        assert self.w.topics_source is not None
        self._freeze_topic_list(self.w.topics_source(), {"generation": "imported for a pilot"})

    def _freeze_topic_list(self, topics: list[dict[str, Any]], report: Mapping[str, Any]) -> None:
        manifest = {
            "schema": TOPICS_SCHEMA,
            "bank_sha256": self.w.bank.sha256(),
            "seeds": {
                "bank": self.w.bank.seed,
                "anchor": v2.ANCHOR_SEED,
                "completeness": v2.COMPLETENESS_SEED,
                "judge_order": v2.JUDGE_ORDER_SEED,
                "human_audit": hau.AUDIT_SEED,
            },
            "templates_sha256": dict(sorted(self.w.audit.frozen_templates.items())),
            "references": dict(sorted(Counter(r.kind for r in self.w.references).items())),
            "references_sha256": v2.sha256_value(
                [[r.key, r.kind, r.statement] for r in self.w.references]
            ),
            "topics": topics,
            "topic_sha256": {t["need_id"]: self.topic_hash(t) for t in topics},
            "dropped": [
                {"candidate_id": p["candidate_id"], "stage": p["stage"], "reason": p["reason"]}
                for k, p in sorted(self.keyed.items())
                if k.startswith("drop/")
            ],
            "deduplication": {
                p["candidate_id"]: {
                    "kept": p["kept"],
                    "verdicts": p["verdicts"],
                    "closest_kind": p["closest_kind"],
                }
                for k, p in sorted(self.keyed.items())
                if k.startswith("dedup/") and "candidate_id" in p
            },
            **report,
        }
        digest = self.write_frozen(TOPICS_FILE, manifest)
        self.record("topics_frozen", "topics", {"sha256": digest, "topics": len(topics)})
        self.status("topics", {"topics": len(topics), **self.generation_counts()})

    def topics(self) -> list[dict[str, Any]]:
        """The frozen topics, checked: the file against its freeze and every topic against its
        own hash. A topic that changed after the freeze is an integrity stop."""
        manifest = self.read_frozen("topics", TOPICS_FILE)
        topics: list[dict[str, Any]] = manifest["topics"]
        if {t["need_id"]: self.topic_hash(t) for t in topics} != manifest["topic_sha256"]:
            raise self.stop("topic_mutation", "a frozen topic differs from its hash")
        return topics

    # 5 -----------------------------------------------------------------------

    def _run_problems(self, run: rv.Run, expected: set[str]) -> list[str]:
        """A run of the topics: every query ranked or refused by its contract, no chunk twice,
        no chunk outside the layer (the evaluator's check of a run)."""
        problems = []
        if set(run.rankings) | set(run.refused) != expected or set(run.rankings) & set(run.refused):
            problems.append("its queries are not the topics' queries")
        for qid, ranking in run.rankings.items():
            if len(set(ranking)) != len(ranking):
                problems.append(f"{qid}: a chunk ranked twice")
            if any(c not in self.w.layer.chunks for c in ranking):
                problems.append(f"{qid}: a chunk outside the layer")
        return problems

    def _ranked(self, system: str) -> rv.Run:
        """A system's run as recorded when it completed."""
        p = self.keyed[f"retrieval/{system}/ranked"]
        rows = map(json.loads, self.store.text(p["sha256"], "kept").splitlines())
        rankings = {r["query_id"]: list(r["chunk_ids"]) for r in rows}
        return rv.Run(system, rankings, list(p["refused"]), dict(p["identity"]), p["seconds"])

    def retrieve(self) -> None:
        """Every frozen retriever on every frozen query, one at a time, each run recorded when
        it completes; a system that fails technically runs again, unchanged, once, and is then
        not completed (S4). The run set is written once and moved into place whole. Nothing is
        evaluated."""
        if "runs" in self.keyed:
            return
        topics = self.topics()
        queries = [(q["query_id"], q["text"]) for t in topics for q in t["queries"]]
        expected = {qid for qid, _ in queries}
        done: dict[str, rv.Run] = {}
        not_completed = []
        for system in self.w.systems:
            key = f"retrieval/{system}"
            if f"{key}/ranked" in self.keyed:
                done[system] = self._ranked(system)
                continue
            # only an attempt that failed counts: one a process interruption ended (a kill, a
            # sleep, a crash of the orchestrator) is recorded as interrupted and runs again
            started = sum(k.startswith(f"{key}/attempt/") for k in self.keyed)
            for n in range(1, started + 1):
                if f"{key}/failed/{n}" not in self.keyed:
                    self.record(
                        "retrieval_interrupted",
                        f"{key}/interrupted/{n}",
                        {"system": system, "attempt": n},
                    )
            failed = sum(k.startswith(f"{key}/failed/") for k in self.keyed)
            while failed < RETRIEVAL_ATTEMPTS and f"{key}/not_completed" not in self.keyed:
                started += 1
                self.record(
                    "retrieval_attempt",
                    f"{key}/attempt/{started}",
                    {"system": system, "attempt": started},
                )
                try:
                    run = self.w.rank(system, queries, self.w.namespace / RUNS_WORK / system)
                except Exception as error:  # a technical failure: once more, unchanged
                    self.record(
                        "retrieval_failed",
                        f"{key}/failed/{started}",
                        {"system": system, "attempt": started, "error": type(error).__name__},
                    )
                    failed += 1
                    continue
                problems = self._run_problems(run, expected)
                if problems:
                    raise self.stop("retrieval_identity", f"{system}: {problems[:3]}")
                digest, _ = self.store.put(run.lines(), "kept")
                self.record(
                    "system_ranked",
                    f"{key}/ranked",
                    {
                        "system": system,
                        "sha256": digest,
                        "refused": sorted(run.refused),
                        "identity": run.identity,
                        "seconds": round(run.seconds, 1),
                    },
                )
                done[system] = self._ranked(system)
                break
            if system not in done:
                self.record(
                    "system_not_completed",
                    f"{key}/not_completed",
                    {"system": system, "failed_attempts": failed},
                )
                not_completed.append(system)
        final = self.w.namespace / RUNS_DIR
        if not final.exists():
            staging = self.w.namespace / f"{RUNS_DIR}.staging-{_now()}"
            self.w.write_runs(done, staging)
            staging.rename(final)
        text = (final / "manifest.json").read_text(encoding="utf-8")
        manifest = json.loads(text)
        for system, record in manifest["runs"].items():
            if record["sha256"] != self.keyed[f"retrieval/{system}/ranked"]["sha256"]:
                raise self.stop("artifact", f"the run file of {system} is not the run recorded")
        if set(manifest["runs"]) != set(done):
            raise self.stop("artifact", "the run set is not the runs recorded")
        digest, _ = self.store.put(text, "kept")
        self.record(
            "runs_frozen",
            "runs",
            {
                "sha256": digest,
                "systems": sorted(done),
                "not_completed": sorted(not_completed),
                "queries": len(expected),
            },
        )

    def rankings(self) -> dict[str, dict[str, list[str]]]:
        final = self.w.namespace / RUNS_DIR
        text = (final / "manifest.json").read_text(encoding="utf-8")
        if vs.sha_text(text) != self.keyed["runs"]["sha256"]:
            raise self.stop("artifact", "the runs' manifest differs from its freeze")
        try:
            return self.w.read_runs(final, json.loads(text))
        except Exception as error:
            raise self.stop("artifact", f"the runs do not verify: {error}") from error

    # 6 -----------------------------------------------------------------------

    def pools(self) -> None:
        if "pools" in self.keyed:
            return
        rankings = self.rankings()
        out = {}
        for t in self.topics():
            qids = [q["query_id"] for q in t["queries"]]
            anchors = [self.w.layer.owner[_node(a["node"])] for a in t["anchors"]]
            pooled = v2.pool(rankings, qids, anchors)
            decisive = v2.decisive_chunks(rankings, qids)
            pool = jv.need_pool(
                t["need_id"], pooled, decisive, self.w.layer, self.w.texts, self.w.excluded_psd2
            )
            out[t["need_id"]] = {
                **pool.record(),
                "sources": {p.chunk_id: [list(s) for s in p.sources] for p in pooled},
            }
        digest = self.write_frozen(POOLS_FILE, {"schema": POOLS_SCHEMA, "pools": out})
        self.record("pools_frozen", "pools", {"sha256": digest, "needs": len(out)})
        sizes = sorted(len(p["chunks"]) for p in out.values())
        self.status(
            "pools", {"needs": len(out), "median_chunks": sizes[len(sizes) // 2] if sizes else 0}
        )

    def need_pools(self) -> dict[str, jv.NeedPool]:
        pools = self.read_frozen("pools", POOLS_FILE)["pools"]
        return {
            n: jv.NeedPool(
                n,
                tuple(p["chunks"]),
                {c: tuple(v) for c, v in p["judged"].items()},
                tuple(p["decisive"]),
                tuple(p["ineligible"]),
            )
            for n, p in pools.items()
        }

    # 7 -----------------------------------------------------------------------

    def _pool_jobs(
        self,
        scope: str,
        role: str,
        pool: jv.NeedPool,
        statement: str,
        nodes: Sequence[str],
        stage: str,
    ) -> list[en.Job]:
        """The batch jobs of one blind judgment of a pool's nodes: chunk groups in the order
        drawn for the need and the stage, a chunk's nodes in document order, never a group
        split; every node under its opaque id, the same in B1 and B2."""
        salt = self.w.salt
        ids = jv.pool_ids(salt, scope, pool.nodes)
        wanted = set(nodes)
        order = v2.group_order(scope, [c for c in pool.chunks if c in pool.judged], salt, stage)
        groups = [[n for n in pool.judged[c] if n in wanted] for c in order]
        need = v2p.NeedText(self.ref(role, scope), statement)

        def render(batch: Sequence[str]) -> str:
            return v2p.pool_packet(self.w.corpus, need, batch, ids)

        cuts = v2.batches(
            groups,
            render,
            self.w.count_tokens,
            self.w.params.pool_max_nodes,
            self.w.params.pool_max_tokens,
        )
        jobs = []
        for number, batch in enumerate(cuts, start=1):
            by_id = {ids[n]: n for n in batch}

            def admit(reply: str, by_id: dict[str, str] = by_id) -> Any:
                value = en.reply_json(reply)
                problems = ha.pool_problems(value, by_id)
                if problems:
                    raise en.Inadmissible("judge_answer_complete", problems)
                return hj.pool_judgments(value, by_id)

            jobs.append(en.Job(f"{stage}/{scope}/{number}", role, render(batch), admit))
        return jobs

    def _c_jobs(
        self,
        scope: str,
        pool: jv.NeedPool,
        statement: str,
        nodes: Sequence[str],
        b1: Mapping[str, Any],
        b2: Mapping[str, Any],
        stage: str,
    ) -> list[en.Job]:
        """Judge C on exactly the disputed nodes, in batches of the drawn chunk order."""
        if not nodes:
            return []
        ids = jv.c_ids(self.w.salt, scope, nodes)
        wanted = set(nodes)
        ordered = [
            n
            for c in v2.group_order(scope, list(pool.chunks), self.w.salt, stage)
            for n in pool.judged.get(c, ())
            if n in wanted
        ]
        sizing = v2p.NeedText(self.ref("judge_c", f"{scope}:sizing"), statement)

        def render(batch: Sequence[str]) -> str:
            return v2p.c_packet(self.w.corpus, sizing, batch, b1, b2, ids)[0]

        cuts = v2.batches(
            [[n] for n in ordered],
            render,
            self.w.count_tokens,
            self.w.params.pool_max_nodes,
            self.w.params.pool_max_tokens,
        )
        jobs = []
        for number, batch in enumerate(cuts, start=1):
            need = v2p.NeedText(self.ref("judge_c", f"{scope}:{number}"), statement)
            packet, items = v2p.c_packet(self.w.corpus, need, batch, b1, b2, ids)

            def admit(reply: str, items: list[hp.CItem] = items) -> Any:
                value = en.reply_json(reply)
                problems = ha.judge_c_problems(value, items)
                if problems:
                    raise en.Inadmissible("judge_answer_complete", problems)
                by_item = {i["item_id"]: i for i in value["items"]}
                return {item.target: by_item[item.item_id] for item in items}

            jobs.append(en.Job(f"{stage}/{scope}/{number}", "judge_c", packet, admit))
        return jobs

    def _stage_done(
        self, prefix: str, scope: str, stage: str, labels: Mapping[str, Any], batches: int
    ) -> None:
        self.record(
            "need_stage",
            f"{prefix}{stage}/{scope}",
            {
                "need_id": scope,
                "stage": stage,
                "nodes": len(labels),
                "batches": batches,
                "labels_sha256": v2.sha256_value(dict(labels)),
            },
        )

    def adjudicate(
        self, needs: Sequence[tuple[str, jv.NeedPool, str]], prefix: str, decisive: bool
    ) -> dict[str, tuple[dict[str, jv.Final] | None, str | None]]:
        """B1, B2 and C over many needs at once, each stage in parallel across them. Returns
        need -> (finals, None), or (None, the stage whose job was exhausted)."""
        out: dict[str, tuple[dict[str, jv.Final] | None, str | None]] = {}

        def owed(p: jv.NeedPool) -> jv.NeedPool:
            return p if decisive else jv.NeedPool(p.need_id, p.chunks, p.judged, (), p.ineligible)

        b1_jobs = {
            s: self._pool_jobs(s, "v2_pool_judge_1", p, st, p.nodes, f"{prefix}b1")
            for s, p, st in needs
        }
        done = self.engine.run([j for js in b1_jobs.values() for j in js])
        b1: dict[str, dict[str, Any]] = {}
        alive = []
        for s, p, st in needs:
            outcomes = [done[j.job_id] for j in b1_jobs[s]]
            if any(o.exhausted for o in outcomes):
                out[s] = (None, f"{prefix}b1")
                continue
            b1[s] = {n: label for o in outcomes for n, label in o.value.items()}
            if set(b1[s]) != set(p.nodes):
                raise self.stop("state", f"{s}: B1 does not cover its pool exactly")
            self._stage_done(prefix, s, "b1", b1[s], len(outcomes))
            alive.append((s, p, st))
        b2_sets, b2_jobs = {}, {}
        for s, p, st in alive:
            b2_sets[s] = jv.b2_nodes(owed(p), b1[s])
            b2_jobs[s] = self._pool_jobs(s, "v2_pool_judge_2", p, st, b2_sets[s], f"{prefix}b2")
        done = self.engine.run([j for js in b2_jobs.values() for j in js])
        b2: dict[str, dict[str, Any]] = {}
        still = []
        for s, p, st in alive:
            outcomes = [done[j.job_id] for j in b2_jobs[s]]
            if any(o.exhausted for o in outcomes):
                out[s] = (None, f"{prefix}b2")
                continue
            b2[s] = {n: label for o in outcomes for n, label in o.value.items()}
            if set(b2[s]) != set(b2_sets[s]):
                raise self.stop("state", f"{s}: B2 does not cover what it owes exactly")
            self._stage_done(prefix, s, "b2", b2[s], len(outcomes))
            still.append((s, p, st))
        c_jobs = {
            s: self._c_jobs(
                s, p, st, jv.disputes(b2_sets[s], b1[s], b2[s]), b1[s], b2[s], f"{prefix}c"
            )
            for s, p, st in still
        }
        done = self.engine.run([j for js in c_jobs.values() for j in js])
        for s, p, _ in still:
            outcomes = [done[j.job_id] for j in c_jobs[s]]
            if any(o.exhausted for o in outcomes):
                out[s] = (None, f"{prefix}c")
                continue
            c = {n: label for o in outcomes for n, label in o.value.items()}
            self._stage_done(prefix, s, "c", c, len(outcomes))
            try:
                out[s] = (jv.finals(owed(p), b1[s], b2[s], c), None)
            except ValueError as error:
                raise self.stop("state", f"{s}: {error}") from error
        return out

    def judge(self) -> None:
        """Waves in priority order, each as large as the evaluable needs still missing: the
        first N_TARGET evaluable needs end the judging; an exhausted judgment makes its need a
        technical failure and the next one in priority order comes in."""
        topics = self.topics()
        pools = self.need_pools()
        statement = {t["need_id"]: t["statement"] for t in topics}
        order = [t["need_id"] for t in topics]
        while True:
            judged = [n for n in order if f"judged/{n}" in self.keyed]
            evaluable = [n for n in judged if self.keyed[f"judged/{n}"]["evaluable"]]
            remaining = [n for n in order if n not in set(judged)]
            waves = sorted(k for k in self.keyed if k.startswith("wave/"))
            last = self.keyed[waves[-1]]["needs"] if waves else []
            if any(f"judged/{n}" not in self.keyed for n in last):
                wave = list(last)  # a wave a restart interrupted: the same wave, never another
            else:
                if len(evaluable) >= self.w.n_target or not remaining:
                    break
                wave = remaining[: self.w.n_target - len(evaluable)]
                self.record("wave_planned", f"wave/{len(waves) + 1:04d}", {"needs": wave})
            results = self.adjudicate([(n, pools[n], statement[n]) for n in wave], "judge/", True)
            for n in wave:
                finals, exhausted = results[n]
                if finals is None:
                    self.record(
                        "need_judged",
                        f"judged/{n}",
                        {"need_id": n, "evaluable": False, "technical": exhausted},
                    )
                    continue
                record = jv.qrel_record(finals, self.w.texts)
                digest, _ = self.store.put(vs.canonical(record), "kept")
                b1 = {k: f.b1 for k, f in finals.items()}
                s = jv.summary(
                    finals,
                    len(jv.b2_nodes(pools[n], b1)),
                    sum(f.outcome == "judge_c" for f in finals.values()),
                )
                self.record(
                    "need_judged",
                    f"judged/{n}",
                    {"need_id": n, "evaluable": s.evaluable, "qrels": digest, "summary": asdict(s)},
                )
            judged = [n for n in order if f"judged/{n}" in self.keyed]
            self.status(
                "judging",
                {
                    "judged": len(judged),
                    "evaluable": sum(self.keyed[f"judged/{n}"]["evaluable"] for n in judged),
                    "technical": sum(
                        bool(self.keyed[f"judged/{n}"].get("technical")) for n in judged
                    ),
                },
            )

    # 8 -----------------------------------------------------------------------

    def freeze_qrels(self) -> None:
        if "qrels" in self.keyed:
            return
        order = [t["need_id"] for t in self.topics()]
        judged = [n for n in order if f"judged/{n}" in self.keyed]
        evaluable = [n for n in judged if self.keyed[f"judged/{n}"]["evaluable"]]
        primary = evaluable[: self.w.n_target]
        qrels = {
            n: json.loads(self.store.text(self.keyed[f"judged/{n}"]["qrels"], "kept"))
            for n in primary
        }
        technical = [n for n in judged if self.keyed[f"judged/{n}"].get("technical")]
        manifest = {
            "schema": QRELS_SCHEMA,
            "n": len(primary),
            "n_target": self.w.n_target,
            "primary": primary,
            "qrels": qrels,
            "judged_unused": [n for n in evaluable if n not in set(primary)],
            "not_evaluable": [
                n for n in judged if n not in set(evaluable) and n not in set(technical)
            ],
            "technical": technical,
            "not_judged": [n for n in order if n not in set(judged)],
        }
        digest = self.write_frozen(QRELS_FILE, manifest)
        self.record("qrels_frozen", "qrels", {"sha256": digest, "n": len(primary)})
        self.status("qrels", {"n": len(primary), "n_target": self.w.n_target})

    def qrels(self) -> dict[str, Any]:
        found: dict[str, Any] = self.read_frozen("qrels", QRELS_FILE)
        return found

    def primary(self) -> list[str]:
        return list(self.qrels()["primary"])

    # 9 -----------------------------------------------------------------------

    def _answerability_job(
        self, role: str, need: str, statement: str, finals: Mapping[str, Any]
    ) -> en.Job:
        def admit(reply: str) -> Any:
            value = en.reply_json(reply)
            problems = v2p.answerability_problems(value)
            if problems:
                raise en.Inadmissible("answerability_contract", problems)
            return value

        text = v2p.NeedText(self.ref(role, need), statement)
        packet = v2p.answerability_packet(self.w.corpus, text, finals)
        return en.Job(f"answerability/{role}/{need}", role, packet, admit)

    def _answerability_c_job(
        self,
        need: str,
        statement: str,
        finals: Mapping[str, Any],
        a: Mapping[str, Any],
        b: Mapping[str, Any],
    ) -> en.Job:
        item_id = ho.item_id(self.w.salt, f"v2-answerability-c:{need}", bm.ANSWERABILITY_TARGET)
        draw = v2.draw(v2.JUDGE_ORDER_SEED, self.w.salt, "answerability-c", need)
        text = v2p.NeedText(self.ref("judge_c_answerability", need), statement)
        packet, items = v2p.answerability_c_packet(
            self.w.corpus, text, finals, a, b, item_id, int(draw, 16) % 2 == 0
        )

        def admit(reply: str) -> Any:
            value = en.reply_json(reply)
            problems = ha.judge_c_problems(value, items)
            if problems:
                raise en.Inadmissible("judge_answer_complete", problems)
            return value["items"][0]

        return en.Job(f"answerability/judge_c/{need}", "judge_c", packet, admit)

    def answerability(self) -> None:
        """Descriptive, after the qrels: the first blind judgment; a `none` against the need's
        grade >= 2 evidence (every primary need has some) is a contradiction, and a second
        blind judgment follows; where the two differ, judge C decides. An exhausted judgment
        leaves the answerability `unknown`. Nothing here filters the sample."""
        topics = {t["need_id"]: t for t in self.topics()}
        qrels = self.qrels()["qrels"]
        needs = [n for n in self.primary() if f"answerability/{n}" not in self.keyed]
        first = self.engine.run(
            [
                self._answerability_job("v2_answerability", n, topics[n]["statement"], qrels[n])
                for n in needs
            ]
        )

        def a1(n: str) -> en.Outcome:
            return first[f"answerability/v2_answerability/{n}"]

        contradicted = [
            n for n in needs if not a1(n).exhausted and a1(n).value["answerability"] == "none"
        ]
        second = self.engine.run(
            [
                self._answerability_job("v2_answerability_2", n, topics[n]["statement"], qrels[n])
                for n in contradicted
            ]
        )

        def a2(n: str) -> en.Outcome:
            return second[f"answerability/v2_answerability_2/{n}"]

        disputed = [
            n
            for n in contradicted
            if not a2(n).exhausted and a2(n).value["answerability"] != "none"
        ]
        c_jobs = {
            n: self._answerability_c_job(
                n, topics[n]["statement"], qrels[n], a1(n).value, a2(n).value
            )
            for n in disputed
        }
        decided = self.engine.run(list(c_jobs.values()))
        for n in needs:
            if a1(n).exhausted:
                payload = {"answerability": "unknown", "path": "first_exhausted", "missing": None}
            elif n not in contradicted:
                v = a1(n).value
                payload = {
                    "answerability": v["answerability"],
                    "path": "first",
                    "missing": v.get("missing"),
                }
            elif a2(n).exhausted:
                payload = {
                    "answerability": "unknown",
                    "path": "contradiction_second_exhausted",
                    "missing": a1(n).value.get("missing"),
                }
            elif n not in disputed:
                payload = {
                    "answerability": "none",
                    "path": "contradiction_confirmed",
                    "missing": a1(n).value.get("missing"),
                }
            else:
                c = decided[c_jobs[n].job_id]
                if c.exhausted:
                    payload = {
                        "answerability": "unknown",
                        "path": "judge_c_exhausted",
                        "missing": None,
                    }
                else:
                    chosen = c.value["answerability"]
                    source = a1(n).value if chosen == "none" else a2(n).value
                    same = chosen == source["answerability"]
                    payload = {
                        "answerability": chosen,
                        "path": "judge_c",
                        "missing": source.get("missing") if same else c.value.get("reason"),
                    }
            self.record("answerability_decided", f"answerability/{n}", {"need_id": n, **payload})

    def metadata(self) -> None:
        """The blind metadata adjudicator on every primary need whose answerability is known;
        an exhausted adjudication, or an unknown answerability, leaves the metadata `unknown`.
        It never removes a need."""
        topics = {t["need_id"]: t for t in self.topics()}
        qrels = self.qrels()["qrels"]
        needs = [n for n in self.primary() if f"metadata/{n}" not in self.keyed]
        jobs = {}
        for n in needs:
            answer = self.keyed[f"answerability/{n}"]
            kind = answer["answerability"]
            if kind == "unknown":
                continue
            gap = None if kind == "complete" else (answer.get("missing") or None)
            finals = qrels[n]

            def admit(reply: str, finals: Mapping[str, Any] = finals, kind: str = kind) -> Any:
                value = en.reply_json(reply)
                problems = v2p.metadata_problems(value, finals, kind)
                if problems:
                    raise en.Inadmissible("metadata_answer", problems)
                return value

            text = v2p.NeedText(self.ref("metadata", n), topics[n]["statement"])
            packet = v2p.metadata_packet(self.w.corpus, text, finals, kind, gap)
            jobs[n] = en.Job(f"metadata/job/{n}", "metadata_adjudicator", packet, admit)
        done = self.engine.run(list(jobs.values()))
        for n in needs:
            if n not in jobs or done[jobs[n].job_id].exhausted:
                payload: dict[str, Any] = {
                    "type": "unknown",
                    "difficulty": "unknown",
                    "drivers": {},
                }
            else:
                v = done[jobs[n].job_id].value
                payload = {
                    "type": v["type"],
                    "difficulty": v["difficulty"],
                    "drivers": v["drivers"],
                }
            self.record("metadata_decided", f"metadata/{n}", {"need_id": n, **payload})

    # 10 ----------------------------------------------------------------------

    def _screen_jobs(
        self, need: str, statement: str, chunks: Sequence[str]
    ) -> list[tuple[en.Job, dict[str, str]]]:
        """The chunk screen of one completeness need: every chunk outside its pool that owns a
        node to judge, in an order drawn for the need, about screen_max_passages a batch."""
        salt = self.w.salt
        ordered = v2.group_order(need, chunks, salt, "screen")
        pid = {c: jv.passage_id(salt, need, c) for c in ordered}
        text = {
            c: v2p.passage(self.w.corpus, self.w.chunk_text[c], self.w.layer.chunks[c].nodes[0])
            for c in ordered
        }
        ref = v2p.NeedText(self.ref("screen", need), statement)

        def render(batch: Sequence[str]) -> str:
            return v2p.screen_packet(ref, [(pid[c], text[c]) for c in batch])

        cuts = v2.batches(
            [[c] for c in ordered],
            render,
            self.w.count_tokens,
            self.w.params.screen_max_passages,
            self.w.params.screen_max_tokens,
        )
        out = []
        for number, batch in enumerate(cuts, start=1):
            by_pid = {pid[c]: c for c in batch}

            def admit(reply: str, by_pid: dict[str, str] = by_pid) -> Any:
                value = en.reply_json(reply)
                problems = v2p.screen_problems(value, list(by_pid))
                if problems:
                    raise en.Inadmissible("screen_contract", problems)
                return {e["item_id"]: bool(e["flag"]) for e in value["passages"]}

            job = en.Job(f"completeness/screen/{need}/{number}", "v2_screen", render(batch), admit)
            out.append((job, by_pid))
        return out

    def completeness(self) -> None:
        """The completeness diagnostic: per act, the first COMPLETENESS_PER_STRATUM primary
        needs of its chain frozen with the bank; for each, the chunk screen of the whole layer
        outside its pool, and the flagged chunks' nodes through B1, B2 and C. Only sensitivity
        SA2 reads what it finds: it never touches the primary qrels, the sample or the run."""
        if "completeness" in self.keyed:
            return
        topics = {t["need_id"]: t for t in self.topics()}
        primary = set(self.primary())
        need_of = {t["candidate_id"]: t["need_id"] for t in topics.values()}
        chosen = v2.completeness_selection(
            self.w.chains, [t["candidate_id"] for t in topics.values() if t["need_id"] in primary]
        )
        self.record(
            "completeness_selected",
            "completeness/selection",
            {"selection": {a: [need_of[c] for c in cs] for a, cs in sorted(chosen.items())}},
        )
        selection = self.keyed["completeness/selection"]["selection"]
        needs = [n for a in v2.COMPLETENESS_STRATA for n in selection.get(a, [])]
        pools = self.need_pools()
        todo = [n for n in needs if f"completeness/{n}" not in self.keyed]
        outside = {
            n: [
                c
                for c in self.w.layer.chunks
                if c not in set(pools[n].chunks)
                and v2.judged_nodes(c, self.w.layer, self.w.texts, self.w.excluded_psd2)
            ]
            for n in todo
        }
        screen_jobs = {n: self._screen_jobs(n, topics[n]["statement"], outside[n]) for n in todo}
        done = self.engine.run([j for js in screen_jobs.values() for j, _ in js])
        follow, flagged_of, incomplete = [], {}, {}
        for n in todo:
            flags: dict[str, bool] = {}
            for job, by_pid in screen_jobs[n]:
                o = done[job.job_id]
                if o.exhausted:
                    incomplete[n] = "screen_batch_exhausted"
                    continue
                flags |= {by_pid[p]: f for p, f in o.value.items()}
            flagged = sorted(c for c, f in flags.items() if f)
            flagged_of[n] = flagged
            self.record(
                "completeness_screened",
                f"completeness/screen/{n}",
                {
                    "need_id": n,
                    "screened": len(outside[n]),
                    "answered": len(flags),
                    "flagged": len(flagged),
                    "flags_sha256": v2.sha256_value(flags),
                },
            )
            judged = {
                c: tuple(v2.judged_nodes(c, self.w.layer, self.w.texts, self.w.excluded_psd2))
                for c in flagged
            }
            pool = jv.NeedPool(f"{n}:completeness", tuple(flagged), judged, (), ())
            follow.append((f"{n}:completeness", pool, topics[n]["statement"]))
        results = self.adjudicate(follow, "completeness/", False)
        for n in todo:
            finals, exhausted = results.get(f"{n}:completeness", (None, None))
            residual = sorted(
                {
                    self.w.layer.owner[_node(node)]
                    for node, f in (finals or {}).items()
                    if f.grade is not None and f.grade >= bm.RELEVANT_GRADE and not f.ambiguous
                }
            )
            record = jv.qrel_record(finals, self.w.texts) if finals else {}
            digest, _ = self.store.put(vs.canonical(record), "kept")
            self.record(
                "completeness_followed",
                f"completeness/{n}",
                {
                    "need_id": n,
                    "screened": len(outside[n]),
                    "flagged": len(flagged_of.get(n, [])),
                    "followed_nodes": len(finals or {}),
                    "residual_chunks": residual,
                    "finals": digest,
                    "incomplete": incomplete.get(n) or exhausted,
                },
            )
        manifest = {
            "schema": COMPLETENESS_SCHEMA,
            "selection": selection,
            "needs": {n: self.keyed[f"completeness/{n}"] for n in needs},
        }
        digest = self.write_frozen(COMPLETENESS_FILE, manifest)
        self.record("completeness_frozen", "completeness", {"sha256": digest, "needs": len(needs)})
        self.status("completeness", {"needs": len(needs)})

    def audit_selection(self) -> None:
        """The optional human audit's needs: the first AUDIT_SIZE primary needs of the chain
        frozen with the bank. Recorded only: the audit itself is a person's, and never blocks."""
        if "human_audit/selection" in self.keyed:
            return
        topics = self.topics()
        primary = set(self.primary())
        need_of = {t["candidate_id"]: t["need_id"] for t in topics}
        chosen = hau.selection(
            self.w.audit_chain, [t["candidate_id"] for t in topics if t["need_id"] in primary]
        )
        self.record(
            "human_audit_selected", "human_audit/selection", {"needs": [need_of[c] for c in chosen]}
        )

    # 11 ----------------------------------------------------------------------

    def evaluation_inputs(self) -> dict[str, Any]:
        """What the evaluation reads: every frozen artifact, checked against its freeze. Only a
        complete construction is evaluated: the qrels frozen (before that, no E-vs-E5 metric can
        be computed), every primary need's answerability and metadata decided, the
        completeness diagnostic frozen and the human audit's needs selected; the evaluation is
        written once, so it never runs on less."""
        if "qrels" not in self.keyed:
            raise NotReady("the qrels are not frozen: no metric exists before")
        primary = self.primary()
        missing = [
            n
            for n in primary
            if f"answerability/{n}" not in self.keyed or f"metadata/{n}" not in self.keyed
        ]
        if missing or "completeness" not in self.keyed or "human_audit/selection" not in self.keyed:
            raise NotReady("the construction is not complete: no evaluation before it is")
        completeness = self.read_frozen("completeness", COMPLETENESS_FILE)
        return {
            "not_completed": list(self.keyed["runs"].get("not_completed", [])),
            "topics": self.read_frozen("topics", TOPICS_FILE),
            "qrels": self.qrels(),
            "pools": self.read_frozen("pools", POOLS_FILE),
            "rankings": self.rankings(),
            "completeness": completeness,
            "completeness_finals": {
                n: json.loads(self.store.text(r["finals"], "kept"))
                for n, r in (completeness.get("needs") or {}).items()
            },
            "answerability": {
                n: self.keyed[f"answerability/{n}"]
                for n in primary
                if f"answerability/{n}" in self.keyed
            },
            "metadata": {
                n: self.keyed[f"metadata/{n}"] for n in primary if f"metadata/{n}" in self.keyed
            },
        }

    def evaluation(self) -> None:
        if "evaluation" in self.keyed or self.w.evaluate is None:
            return
        result = self.w.evaluate(self.evaluation_inputs())
        sensitivity = self.write_frozen(SENSITIVITY_FILE, result.get("sensitivity", {}))
        self.record("sensitivity_done", "sensitivity", {"sha256": sensitivity})
        digest = self.write_frozen(EVALUATION_FILE, result)
        self.record("evaluation_done", "evaluation", {"sha256": digest})
        self.status("evaluated", {"n": result.get("n")})

    # -----------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Counts only, never a statement, a query, a label or a metric."""
        k = self.keyed
        judged = [p for key, p in k.items() if key.startswith("judged/")]
        return {
            "events": self.store.head()[0],
            "candidates": len(self.w.bank.candidates),
            **self.generation_counts(),
            "topics": k.get("topics", {}).get("topics"),
            "judged": len(judged),
            "evaluable": sum(p["evaluable"] for p in judged),
            "technical_needs": sum(bool(p.get("technical")) for p in judged),
            "n": k.get("qrels", {}).get("n"),
            "completeness": k.get("completeness", {}).get("needs"),
            "evaluated": "evaluation" in k,
            "stops": len(self.engine.ledger.stops),
        }
