"""TEST v2's pre-registration contract (D-067): everything that decides TEST v2, frozen by content
before its first topic exists.

prepare() writes it once, public, beside the protocol and the analysis plan: hashes and rules
only. What would expose TEST stays in the private TEST v2 namespace (data/benchmark/test/v2/,
inside the private TEST history that refuses every push): the opaque-id salt, the candidate bank
it draws, the historical exclusions and the development references. The contract publishes
their identities, so that what the construction later reads can be checked against it.

problems() is the preflight: the contract as pinned, every file it freezes with its frozen
content, the corpus and the chunk layer, the systems recomputed from their sources, the
retrieval code, the role contracts and templates, the sealed Gate 1 certification and the
binary it pinned, the MLX runtime of E's decomposer, the private paths, the private inputs
against their identities, and the bank recomputed from the salt. The construction runs only
when it is empty, and START also needs the public commit recorded that holds the contract
(public_commit_problems).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_certification as hcert
from ingest import holdout_contract as hc
from ingest import holdout_exclusions as hxe
from ingest import holdout_isolation as hi
from ingest import holdout_packets as hp
from ingest import holdout_roles as hr
from ingest import holdout_runtime as hrt
from ingest import holdout_v2 as v2
from ingest import holdout_v2_audit as hau
from ingest import holdout_v2_dedup as dd
from ingest import holdout_v2_engine as en
from ingest import holdout_v2_evaluation as hev
from ingest import holdout_v2_packets as v2p
from ingest import holdout_v2_retrieval as rv
from ingest import holdout_v2_runner as hrun
from ingest import holdout_v2_state as vs

SCHEMA = "holdout-v2-contract.v1"
CONTRACT_PATH = Path("config/holdout/test-v2-contract.json")
PROTOCOL_PATH = Path("config/holdout/test-v2-protocol.md")
PLAN_PATH = Path("config/holdout/s4-analysis-plan.v2.md")
NAMESPACE = ho.HOLDOUT_DIR / "v2"
DEVELOPMENT_ROOT = ho.HOLDOUT_DIR / "v2-development"
RAW_DIR = vs.RAW_DIR
DEVELOPMENT_RAW = Path.home() / ".fundamento-private" / "holdout-v2-development-raw"
SALT_FILE = "opaque_salt.hex"
EXCLUSIONS_FILE = "historical_exclusions.json"
DEVELOPMENT_REFERENCES_FILE = "development_references.json"
PIN_FILE = "contract_pin.json"
PUBLIC_COMMIT_FILE = "public_commit.json"
DEVELOPMENT_REFERENCES_SCHEMA = "holdout-v2-development-references.v1"
AUDIT_MODULE = Path("ingest/holdout_isolation.py")

# run 7, the first confirmatory attempt (D-067): with runs 1 to 6 (holdout_contract), the
# sources of the 99 historical statements, and what TEST v2 replaces
RUN_7 = {
    "status": "ABORTED_CONFIRMATORY_MINIMUM_STOP",
    "contract_sha256": "6762368c07204a849413a50a0bb167a84056e82f5638a33bc83df457c73cbdf3",
    "public_commit": "a216b14eaa3c13cc05a925f42a16d3b115b73f85",
    "ready_needs": 0,
    "reason": (
        "The frozen minimum feasibility rule: the first five evaluable cells exhausted their "
        "attempts, so at most 49 of the required 50 evaluable needs stayed reachable. 240 "
        "events, 0 READY, no retrieval and no E-vs-E5 result. An external methodological review "
        "followed, before any TEST retrieval existed"
    ),
    "change": "TEST v2 (D-067): a new prospective protocol, not a repair of run 7",
}
ABORTED_RUNS = (*hc.ABORTED_RUNS, RUN_7)

# the code that builds and evaluates TEST v2: every module of the repository its entry points
# import, at any depth (code_files), so that the metrics, the projection, the relevance
# threshold, the query renderer and the leakage measures are frozen with the rest; a change
# after the contract is a change of method
ENTRY_POINTS = (
    "scripts/holdout_v2.py",
    "scripts/mlx_generate.py",
    "ingest/holdout_v2.py",
    "ingest/holdout_v2_audit.py",
    "ingest/holdout_v2_contract.py",
    "ingest/holdout_v2_dedup.py",
    "ingest/holdout_v2_engine.py",
    "ingest/holdout_v2_evaluation.py",
    "ingest/holdout_v2_judging.py",
    "ingest/holdout_v2_packets.py",
    "ingest/holdout_v2_retrieval.py",
    "ingest/holdout_v2_runner.py",
    "ingest/holdout_v2_state.py",
)
# the earlier list, kept as the floor code_files must cover
CODE_FILES = (
    "ingest/holdout_v2.py",
    "ingest/holdout_v2_audit.py",
    "ingest/holdout_v2_contract.py",
    "ingest/holdout_v2_dedup.py",
    "ingest/holdout_v2_engine.py",
    "ingest/holdout_v2_evaluation.py",
    "ingest/holdout_v2_judging.py",
    "ingest/holdout_v2_packets.py",
    "ingest/holdout_v2_retrieval.py",
    "ingest/holdout_v2_runner.py",
    "ingest/holdout_v2_state.py",
    "ingest/holdout.py",
    "ingest/holdout_admission.py",
    "ingest/holdout_broker.py",
    "ingest/holdout_certification.py",
    "ingest/holdout_checks.py",
    "ingest/holdout_contract.py",
    "ingest/holdout_controller.py",
    "ingest/holdout_exclusions.py",
    "ingest/holdout_executor.py",
    "ingest/holdout_isolation.py",
    "ingest/holdout_judging.py",
    "ingest/holdout_launcher.py",
    "ingest/holdout_packets.py",
    "ingest/holdout_pipeline.py",
    "ingest/holdout_process.py",
    "ingest/holdout_roles.py",
    "ingest/holdout_runtime.py",
    "ingest/holdout_spool.py",
    "ingest/holdout_state.py",
    "ingest/psd2_exclusions.py",
    "ingest/run_report.py",
    "scripts/holdout_v2.py",
)
INPUTS = (
    "data/benchmark/PROTOCOL.md",
    "data/benchmark/psd2_excluded_articles.json",
    *(
        f"data/benchmark/{split}/{name}"
        for split in ("dev", "fixtures")
        for name in ("manifest.json", "needs.jsonl", "queries.jsonl", "qrels.jsonl")
    ),
)
# the operational parameters the dry run may set (D-067): never the design
OPERATIONAL_KNOBS = (
    "workers (concurrent model calls; a resume may lower it)",
    "backoff_seconds (after a provider failure)",
    "the batch ceilings, frozen here: pool_max_nodes, pool_max_tokens, screen_max_passages, "
    "screen_max_tokens",
    "scheduling of independent jobs",
    "checkpoint (backup) frequency",
    "the local resource queue: one local model at a time",
)


class ContractV2Error(Exception):
    """The contract cannot be built or does not hold: nothing runs."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


# ---------------------------------------------------------------------------
# the private inputs, by identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Private:
    """The private inputs of TEST v2, read once from the namespace's provenance."""

    salt: bytes
    exclusions: tuple[hxe.Exclusion, ...]
    exclusions_identity: dict[str, Any]
    exclusions_sources: tuple[str, ...]  # the contracts they were collected under
    development: tuple[dd.Reference, ...]
    development_identity: dict[str, Any]


def read_salt(provenance: Path) -> bytes:
    text = (provenance / SALT_FILE).read_text(encoding="utf-8").strip()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ContractV2Error("the opaque-id salt is not 32 bytes in hex")
    return bytes.fromhex(text)


def development_identity(references: Sequence[dd.Reference]) -> dict[str, Any]:
    return {
        "schema": DEVELOPMENT_REFERENCES_SCHEMA,
        "sha256": v2.sha256_value([[r.key, r.kind, r.statement] for r in references]),
        "count": len(references),
    }


def write_development_references(
    path: Path, references: Sequence[dd.Reference], pilots: Sequence[str]
) -> dict[str, Any]:
    """The development topic material TEST v2's gate keeps as hidden references (D-067): the
    statements its pilots generated, and nothing else. Written once."""
    if path.exists():
        raise ContractV2Error(f"{path} exists: the development references are written once")
    if any(r.kind != "DEVELOPMENT" for r in references):
        raise ContractV2Error("a development reference must be of kind DEVELOPMENT")
    identity = development_identity(references)
    body = {
        **identity,
        "pilots": list(pilots),
        "references": [[r.key, r.statement] for r in references],
    }
    path.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return identity


def read_development_references(path: Path) -> tuple[tuple[dd.Reference, ...], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    refs = tuple(dd.Reference(k, "DEVELOPMENT", s) for k, s in data["references"])
    identity = development_identity(refs)
    if {k: data[k] for k in identity} != identity:
        raise ContractV2Error(f"{path} does not hold its own identity")
    return refs, identity


def read_private(provenance: Path) -> Private:
    found = hxe.read(provenance / EXCLUSIONS_FILE)
    refs, identity = read_development_references(provenance / DEVELOPMENT_REFERENCES_FILE)
    return Private(
        read_salt(provenance),
        tuple(found.entries),
        dict(found.identity),
        tuple(sorted(s.contract_sha256 for s in found.sources)),
        refs,
        identity,
    )


def references(dev: bm.Benchmark, fixtures: bm.Benchmark, private: Private) -> list[dd.Reference]:
    """Every hidden reference of the deduplication: DEV, the fixtures, the 99 historical
    statements and the development topics, each under its own key."""
    out = [dd.Reference(n.need_id, "DEV", n.statement) for n in dev.needs]
    out += [dd.Reference(n.need_id, "FIXTURE", n.statement) for n in fixtures.needs]
    out += [
        dd.Reference(f"historical:{e.key}", "DEVELOPMENT", e.statement) for e in private.exclusions
    ]
    out += list(private.development)
    return out


# ---------------------------------------------------------------------------
# the bank, from the salt
# ---------------------------------------------------------------------------


def bank_inputs(
    dev: bm.Benchmark, docs: Mapping[str, Any], layer: Any, excluded_psd2: frozenset[str]
) -> tuple[list[v2.Slot], dict[str, list[v2.LegalUnit]]]:
    return v2.slots(dev), v2.legal_units(docs, ho.load_areas(), excluded_psd2, layer)


def bank_record(bank: v2.Bank, salt: bytes) -> dict[str, Any]:
    """What the runner writes as bank/bank.json: the bank, its completeness chains and the
    human audit's chain."""
    return {
        **bank.record(),
        "completeness_chains": {
            a: list(c) for a, c in sorted(v2.completeness_chains(bank, salt).items())
        },
        "human_audit_chain": hau.chain(bank, salt),
    }


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


def design(
    slots: Sequence[v2.Slot], bank: v2.Bank, salt: bytes, params: hrun.Params
) -> dict[str, Any]:
    """The frozen design of D-067, as rules and identities."""
    return {
        "question": (
            "On an independent private sample of information needs drawn from DEV's task "
            "taxonomy, does the frozen E improve over the frozen E5 at top-10 retrieval?"
        ),
        "n_target": v2.N_TARGET,
        "candidate_universe": v2.UNIVERSE,
        "population": {
            "slots": [asdict(s) for s in slots],
            "slots_sha256": v2.sha256_value([asdict(s) for s in slots]),
            "rule": "DEV's answerable profiles (answerability complete or partial), one slot each "
            "in DEV's need order; the none profiles never enter TEST v2",
            "copies": "universe // slots for every slot, one more for the universe % slots slots "
            "first in a draw ('extra-copy')",
        },
        "priority": {
            "seed": v2.BANK_SEED,
            "rule": "candidate <slot>-<copy>, ordered by the hex SHA-256 of "
            "'<seed>:<salt hex>:priority:<candidate id>'; the rank fixes the need id (N401 for the "
            "first); independent of completion order, of any system's output and of judging",
        },
        "anchors": {
            "seed": v2.ANCHOR_SEED,
            "unit": "the article (legal unit): in an area of the protocol's section 10, outside "
            "the PSD2 articles amended after the snapshot, with at least one provision of its own "
            "text owned by a chunk; never a raw chunk or node frequency",
            "rule": "per act of the profile (a cross-act profile gets one anchor per act), in "
            "priority order: the act's eligible articles in an order drawn once ('article'), the "
            "i-th demand taking article i, wrapping only when all are drawn (uniform, without "
            "replacement while capacity permits, then a deterministic counted reuse); the "
            "provision a draw over the article's eligible nodes ('provision'), shifted by its "
            "earlier uses",
            "capacity": dict(sorted(bank.capacity.items())),
            "reuse": dict(sorted(bank.reuse.items())),
        },
        "bank_sha256": bank.sha256(),
        "bank_file_sha256": _text_sha(vs.canonical(bank_record(bank, salt)) + "\n"),
        "reserve": "the earliest candidates in priority order that survive deduplication and "
        "become evaluable form the primary sample; the unused tail is the single reserve; no "
        "regeneration, no second reserve; a bank exhausted before N freezes the N achieved. A "
        "candidate leaves only for an outcome-independent reason: a duplicate, an unusable or "
        "anchor-copying generation, an exhausted technical retry, mechanical non-evaluability",
        "params": params.record(),
    }


def generation() -> dict[str, Any]:
    return {
        "hops": {
            "A_concept": "v2_concept_author (concept_author.v2): the profile and its anchors "
            "with their article's title and ancestors; nothing of DEV, the fixtures, the "
            "exclusions, other candidates, systems, rankings, qrels or metrics",
            "B_need": "statement_writer (statement_writer.v1): the concept and its facts only, "
            "never an anchor, an identifier or a location",
            "C_queries": "query_writer (query_variants.v1): the statement only; three "
            "formulations (EN formal, EN natural, ES natural)",
        },
        "concept_checks": "the concept_author.v1 shape; no legal reference "
        "(benchmark._LEGAL_REFERENCE); no legal wording or 8-word run of DEV or the fixtures "
        f"(holdout_packets.author_leaks); no run of {v2p.ANCHOR_COPY_WORDS} words of its "
        "anchor's article",
        "leakage_filter": f"a run of {v2p.ANCHOR_COPY_WORDS} consecutive words of an anchor's "
        "article in the statement or a query drops the candidate (near-copying); shorter runs and "
        "shared vocabulary are measured and reported, never a filter",
        "parallel": "every hop in parallel over the bank; completion order decides nothing",
    }


def deduplication(private: Private, dev: bm.Benchmark, fixtures: bm.Benchmark) -> dict[str, Any]:
    return {
        "method": "one global stage after the whole bank is generated, in priority order",
        "prefilter": {
            "mandatory": "every reference or earlier bank candidate with the same normalized "
            f"statement or a content-word Jaccard >= {dd.MANDATORY_JACCARD}; every earlier bank "
            "candidate sharing an anchor article",
            "bounded": f"the {dd.TOP_REFERENCES} most similar references and {dd.TOP_BANK} most "
            f"similar earlier candidates by BGE-M3 cosine, the {dd.TOP_LEXICAL} most similar of "
            "either by Jaccard",
            "packet_size": dd.MAX_SHOWN,
            "vectors": "BGE-M3 (config/embeddings/bge-m3.json), the frozen query encoding",
        },
        "decision": "overlap_gate.v2 under opaque ids; a candidate is kept only if every packet "
        "finds it independent; only kept earlier candidates are shown; priority wins",
        "references": {
            "DEV": len(dev.needs),
            "FIXTURE": len(fixtures.needs),
            "historical_exclusions": private.exclusions_identity,
            "historical_sources": list(private.exclusions_sources),
            "development_topics": private.development_identity,
            "statement": "the 99 historical statements stay hidden references because they are "
            "part of D-066's documented history, keeping them avoids reinterpreting it, and "
            "corpus anchoring makes the cost low; their exclusion is NOT claimed to be strictly "
            "necessary for statistical validity. Authors never see them. No recursive "
            "blacklist: only designated DEVELOPMENT topic-generation material may become a "
            "development reference; technical failures never do",
        },
    }


def pooling() -> dict[str, Any]:
    return {
        "contributors": [{"system": s, "depth": d} for s, d in v2.POOL],
        "anchor": "the chunk that owns each anchor, whether or not any retriever returns it",
        "union": "over the need's three formulations, each chunk once",
        "judged": "every node a pooled chunk owns with its own text, outside the excluded PSD2 "
        "articles; its ancestors are shown as context",
        "provenance": "kept internally (contributor, formulation, rank); never shown to a judge",
        "decisive_set": f"the union over the formulations of E top-{v2.DECISIVE_DEPTH} minus E5 "
        f"top-{v2.DECISIVE_DEPTH}, plus E5 top-{v2.DECISIVE_DEPTH} minus E top-{v2.DECISIVE_DEPTH}",
        "pool_only": "BGE-M3 and Qwen3-Embedding-0.6B contribute to the pool, never to a "
        "confirmatory comparison",
        "excluded": "no one-hop expansion, no sixth retriever",
    }


def judging(params: hrun.Params) -> dict[str, Any]:
    return {
        "B1": "v2_pool_judge_1 (judge_b_pool.v3): every judged node of the pool, explicit 0s "
        "kept, opaque ids, exact coverage; nodes grouped by chunk in document order, the groups "
        f"in an order drawn per need and stage (seed {v2.JUDGE_ORDER_SEED})",
        "B2": "v2_pool_judge_2, independent and blind: every node B1 graded >= 1 or found "
        "ambiguous, and every node of the decisive set, whatever B1 said",
        "C": "judge_c (judge_c.v2) on exactly the nodes where B1 and B2 differ in grade or "
        "ambiguity",
        "final": "the agreed label or C's; a final ambiguity stays in the qrels and never "
        "deletes the need; an ambiguous or unresolved node counts as unjudged (D-019)",
        "retired": "judge D, the Qwen diagnostic, adjudication_uncertain and "
        "final_qrels_unambiguous rejections",
        "batches": params.record() | {"tokenizer": "cl100k_base (ingest/chunks.py)"},
        "waves": "in priority order, each as large as the evaluable needs still missing",
    }


def after_qrels() -> dict[str, Any]:
    return {
        "evaluability": "mechanical: at least one final non-ambiguous node of grade >= 2",
        "qrel_freeze": "the first N_TARGET evaluable needs in priority order; fewer if the bank "
        "is exhausted",
        "answerability": "descriptive, after the qrels (answerability.v1): a `none` against "
        "grade >= 2 evidence is a contradiction recorded and routed to a second blind judgment "
        "(v2_answerability_2), and to judge C where the two differ; exhausted: unknown. It never "
        "filters the sample",
        "metadata": "metadata_adjudicator on the final grades >= 1; exhausted or unknown "
        "answerability: unknown. It never removes a need",
    }


def completeness() -> dict[str, Any]:
    return {
        "needs": v2.COMPLETENESS_SIZE,
        "strata": list(v2.COMPLETENESS_STRATA),
        "per_stratum": v2.COMPLETENESS_PER_STRATUM,
        "seed": v2.COMPLETENESS_SEED,
        "chains": "per act, every candidate whose profile's first act it is, in an order drawn "
        "with the bank ('completeness'); the stratum's needs are the first of its chain in the "
        "primary sample",
        "screen": "v2_screen (completeness_screen.v1): every chunk of the layer outside the "
        "need's pool that owns a node to judge, one blind judge, high recall, about "
        "screen_max_passages per batch",
        "follow_up": "the flagged chunks' nodes through B1, B2 and C; only a final "
        "non-ambiguous grade >= 2 counts as a verified residual relevant chunk",
        "use": "sensitivity SA2 only: it never modifies the primary qrels, restarts or "
        "invalidates TEST",
        "omission_audit_stop": "retired, with no hidden equivalent",
    }


def evaluation_record() -> dict[str, Any]:
    return {
        "plan_file": PLAN_PATH.as_posix(),
        "plan_sha256": _sha(PLAN_PATH),
        "co_primary": list(hev.PRIMARY),
        "relevance": f"Recall@10 at grade >= {bm.RELEVANT_GRADE}; nDCG@10 the grades as linear "
        "gains (ingest/ir_metrics.py)",
        "unit": "the information need: the mean of its three formulations first",
        "bootstrap": {
            "function": "ingest/run_report.py paired_bootstrap",
            "seed": hev.SEED,
            "resamples": hev.REPS,
            "interval": "percentile, sorted means at int(0.025 * reps) and int(0.975 * reps) - 1",
        },
        "verdicts": "holds if the lower end > 0, reverses if the upper end < 0, inconclusive "
        "otherwise; S4's table of claims",
        "missing_query": "scored 0, never dropped",
        "evaluator": {
            "ir-measures": hc._lock_version("ir-measures"),
            "pytrec-eval-terrier": hc._lock_version("pytrec-eval-terrier"),
            "projection": "ingest/evaluation.py project: a chunk's grade is the highest final "
            "grade it owns; an ambiguous or unresolved label is unjudged (D-019)",
            "relevant_grade": bm.RELEVANT_GRADE,
        },
        "incomplete_systems": "a system that fails technically runs again, unchanged, once; then "
        "it is not completed; if E or E5 is not completed the primary comparison is not made",
        "sensitivity": [
            "SA1 ambiguity: final ambiguous labels at their grade",
            "SA2 augmented qrels on the 15 completeness needs",
            "SA3 fixed-scale G@10: DCG@10 linear / DCG@10 of ten grade-3 chunks",
            "SA4 residual relevance: unjudged top-10 bounds; residual relevance outside the pool "
            "in fixed scenarios (1 chunk of grade 2, 3 chunks of grade 3 per need)",
            "SA5 nDCG gains: grade 1 as 0; exponential 2^g - 1",
            "SA6 paired t interval and paired sign-flip permutation test",
            "SA7 optional human audit, never corrective",
            "SA8 leave-one-contributor-out and unique relevant",
        ],
        "pool_diagnostics": [
            "Judged@10 E",
            "Judged@10 E5",
            "Judged@20 E",
            "Judged@20 E5",
            "relevant chunks contributed uniquely by each pool system",
            "cumulative relevant discovery by depth",
            "pool size distribution",
            "leave-one-contributor-out / unique relevant (SA8)",
            "decisive-set coverage",
            "evaluable needs whose only grade >= 2 evidence one compared system alone brought",
        ],
        "human_audit": {
            "needs": hau.AUDIT_SIZE,
            "seed": hau.AUDIT_SEED,
            "decoys_per_need": hau.DECOYS_PER_NEED,
            "rule": "optional, blind, non-blocking, non-corrective: reported through SA7; TEST "
            "1.0.0 is never replaced",
        },
    }


def execution() -> dict[str, Any]:
    return {
        "technical_executions_per_job": en.MAX_RUNS,
        "retry": "the same packet in a new isolated context; a valid answer is never run again; "
        "after the last execution the job is exhausted, recorded once, and its candidate or need "
        "moves on by the frozen priority",
        "provider_capacity": "a call the provider refuses for its capacity (the CLI's final "
        "result or stderr naming a usage limit, rate_limit_error or overloaded_error) never "
        f"counts toward the limit: a wait of {en.CAPACITY_COOLDOWN_SECONDS:.0f} s doubling to an "
        f"hour, once per episode, and a pause after {en.OUTAGE_MAX_SECONDS:.0f} s without a "
        "response; every other failure counts",
        "isolation": "every response audited by the role's contract (holdout_roles "
        "audit_role_run); a single forbidden-content finding that disappears once metadata "
        "values are neutralized is retried; any other finding is an integrity stop",
        "hard_stops": list(en.INTEGRITY),
        "not_stops": "new relevance, low pool recall, ambiguity, disagreement, hard needs, N < "
        "200 after the bank, any result of E or E5, a small effect, an interval across 0",
        "state": {"schema": vs.SCHEMA, "single_writer": True, "events": sorted(EVENT_KINDS)},
        "operational_knobs": list(OPERATIONAL_KNOBS),
        "local_queue": "the local models (E5, BGE-M3, Qwen3-Embedding, BM25, Qwen3-4B, "
        "Qwen3-Reranker) one at a time, never beside another heavy model",
    }


EVENT_KINDS = (
    "bank_frozen",
    "anchors_assigned",
    "run_started",
    "run_responded",
    "run_failed",
    "job_admitted",
    "candidate_dropped",
    "candidate_generated",
    "dedup_neighborhoods",
    "dedup_decided",
    "topics_frozen",
    "runs_frozen",
    "pools_frozen",
    "need_stage",
    "need_judged",
    "qrels_frozen",
    "answerability_decided",
    "metadata_decided",
    "wave_planned",
    "job_exhausted",
    "retrieval_attempt",
    "retrieval_failed",
    "retrieval_interrupted",
    "system_ranked",
    "system_not_completed",
    "completeness_selected",
    "completeness_screened",
    "completeness_followed",
    "completeness_frozen",
    "human_audit_selected",
    "sensitivity_done",
    "evaluation_done",
    "session_started",
    "capacity_wait",
    "paused",
    "stopped",
)


def retrieval_identity() -> dict[str, Any]:
    """E, E5 and BM25 as the earlier contract recomputes them from their DEV runs, the pool-only
    systems by their configs and DEV runs, and the code TEST v2 runs them with."""
    pool_only = {}
    for system in ("BGE-M3", "Qwen3-Embedding-0.6B"):
        config = rv.DENSE[system]
        pool_only[system] = {
            "config_file": config.as_posix(),
            "config_sha256": _sha(config),
            "dev_run": rv.DEV_RUNS[system],
        }
    return {
        "systems": hc.systems(),
        "pool_only": pool_only,
        "code_sha256": rv.identity(),
        "run_depth": rv.RUN_DEPTH,
    }


def code_files(repo: Path = Path(".")) -> tuple[str, ...]:
    """Every file of the repository the entry points import, at any depth (lazy imports
    inside functions included), by a static reading of their import statements."""
    import ast

    seen: set[str] = set()
    todo = list(ENTRY_POINTS)
    while todo:
        path = todo.pop()
        if path in seen or not (repo / path).is_file():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse((repo / path).read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == "ingest":
                todo += [f"ingest/{a.name}.py" for a in node.names]
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ingest."):
                todo.append(str(node.module).replace(".", "/") + ".py")
            elif isinstance(node, ast.Import):
                todo += [
                    a.name.replace(".", "/") + ".py"
                    for a in node.names
                    if a.name.startswith("ingest.")
                ]
    return tuple(sorted(seen | set(CODE_FILES)))


def code_identity(repo: Path) -> dict[str, Any]:
    files = code_files(repo)
    dirty = _git(repo, "status", "--porcelain", "--", *files)
    if dirty:
        raise ContractV2Error(f"uncommitted changes in the contract's code:\n{dirty}")
    return {
        "built_on_commit": _git(repo, "rev-parse", "HEAD"),
        "files_sha256": {f: _sha(repo / f) for f in files},
        "inputs_sha256": {f: _sha(repo / f) for f in INPUTS},
        "protocol": {"file": PROTOCOL_PATH.as_posix(), "sha256": _sha(repo / PROTOCOL_PATH)},
    }


def runtime_identity(certification: Path) -> dict[str, Any]:
    claude = hrt.claude_runtime(certification / hcert.PROVENANCE)
    mlx = hrt.mlx_runtime(config_path=rv.DECOMPOSER, max_new_tokens=rv.MAX_NEW_TOKENS)
    text = json.dumps([claude, mlx]).lower()
    if "email" in text or "organization" in text:
        raise ContractV2Error("a runtime identity carries the account's identity: never public")
    return {"claude_runtime": claude, "mlx_runtime_decomposer": mlx}


def build(
    repo: Path,
    certification: Path,
    private: Private,
    params: hrun.Params,
) -> dict[str, Any]:
    dev = bm.load_benchmark(bm.DEV_DIR)
    fixtures = bm.load_benchmark(bm.FIXTURES_DIR)
    layer = rv.load_layer()
    from ingest import psd2_exclusions as px

    excluded = px.load_psd2_exclusions(px.PSD2_EXCLUSIONS_PATH).excluded()
    slots, units = bank_inputs(dev, bm.load_corpus(), layer.layer, excluded)
    bank = v2.build_bank(slots, units, private.salt)
    problems = v2.bank_problems(bank, slots)
    if problems:
        raise ContractV2Error(f"the bank breaks its rules: {problems}")
    roles = hr.contracts_record()
    return {
        "schema": SCHEMA,
        "status": "prepared for freeze; TEST v2 not started",
        "decision": "D-067",
        "protocol": PROTOCOL_PATH.as_posix(),
        "design": design(slots, bank, private.salt, params),
        "generation": generation(),
        "deduplication": deduplication(private, dev, fixtures),
        "pooling": pooling(),
        "judging": judging(params),
        "after_qrels": after_qrels(),
        "completeness": completeness(),
        "evaluation": evaluation_record(),
        "execution": execution(),
        "roles": {
            "role_contracts_sha256": _text_sha(json.dumps(roles, sort_keys=True)),
            "templates_sha256": hr.template_hashes(),
            "template_files": {n: hr.template_path(n).as_posix() for n in hr.template_hashes()},
            "rules_sha256": _text_sha(hp.judge_bc_rules()),
            "query_template": bm.QUERY_TEMPLATE_ID,
        },
        "corpus": hc.corpus(),
        "retrieval": retrieval_identity(),
        "runtime": runtime_identity(certification),
        "certification": hcert.reference(certification),
        "code": code_identity(repo),
        "aborted_runs": [dict(r) for r in ABORTED_RUNS],
    }


def write(contract: Mapping[str, Any], path: Path = CONTRACT_PATH) -> str:
    if path.exists():
        raise ContractV2Error(f"{path} exists: the contract is written once")
    path.write_text(json.dumps(contract, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return _sha(path)


def frozen_files(contract: Mapping[str, Any]) -> dict[str, str]:
    """Every file the contract freezes, by path, with its SHA-256."""
    files = dict(contract["code"]["files_sha256"]) | dict(contract["code"]["inputs_sha256"])
    files[contract["code"]["protocol"]["file"]] = contract["code"]["protocol"]["sha256"]
    files[contract["evaluation"]["plan_file"]] = contract["evaluation"]["plan_sha256"]
    files |= contract["retrieval"]["code_sha256"]
    for name, sha in contract["roles"]["templates_sha256"].items():
        files[contract["roles"]["template_files"][name]] = sha
    for system in contract["retrieval"]["pool_only"].values():
        files[system["config_file"]] = system["config_sha256"]
    for system in contract["retrieval"]["systems"].values():
        for part in (system, *(v for v in system.values() if isinstance(v, dict))):
            if "config_file" in part:
                files[part["config_file"]] = part["config_sha256"]
    return dict(sorted(files.items()))


def verify(
    pinned_sha256: str, path: Path = CONTRACT_PATH, repo: Path = Path(".")
) -> dict[str, Any]:
    """The contract as pinned, and every file it freezes with the content it froze."""
    if not path.exists() or _sha(path) != pinned_sha256:
        raise ContractV2Error(f"{path} is not the pinned contract")
    contract: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != SCHEMA:
        raise ContractV2Error(f"{path} is not a TEST v2 contract")
    changed = [
        f
        for f, h in frozen_files(contract).items()
        if not (repo / f).exists() or _sha(repo / f) != h
    ]
    if changed:
        raise ContractV2Error(f"files differ from the contract: {changed}")
    return contract


def committed_problems(
    contract: Mapping[str, Any], pinned: str, repo: Path, commit: str
) -> list[str]:
    """What the recorded public commit lacks to date the frozen machinery: the contract as
    pinned and every frozen file with its frozen content, read from the commit itself."""
    problems = []
    for f, h in {CONTRACT_PATH.as_posix(): pinned, **frozen_files(contract)}.items():
        done = subprocess.run(
            ["git", "-C", str(repo), "show", f"{commit}:{f}"], capture_output=True, check=False
        )
        if done.returncode != 0:
            problems.append(f"{f}: not in {commit[:12]}")
        elif hashlib.sha256(done.stdout).hexdigest() != h:
            problems.append(f"{f}: in {commit[:12]} with other content")
    return problems


def private_path_problems(repo: Path, namespace: Path, raw: Path) -> list[str]:
    problems = []
    ignored = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "-q", str(namespace.resolve() / "x")], check=False
    )
    if ignored.returncode != 0:
        problems.append(f"{namespace} is not ignored by the public repository")
    hook = ho.HOLDOUT_DIR / ".git" / "hooks" / "pre-push"
    if not hook.exists() or hook.read_text(encoding="utf-8") != ho.NO_PUSH_HOOK:
        problems.append("the private TEST history does not refuse every push")
    if raw.resolve().is_relative_to(repo.resolve()):
        problems.append(f"{raw} is inside the repository")
    return problems


def problems(
    pinned: str,
    repo: Path = Path("."),
    namespace: Path = NAMESPACE,
    raw: Path = RAW_DIR,
) -> tuple[dict[str, Any] | None, Private | None, list[str]]:
    """The preflight of TEST v2: everything the contract freezes, checked again now."""
    try:
        contract = verify(pinned, repo / CONTRACT_PATH, repo)
    except (ContractV2Error, OSError, json.JSONDecodeError) as error:
        return None, None, [f"the contract: {error}"]
    found: list[str] = []
    found += [f"corpus: {p}" for p in hc.corpus_problems(contract)]
    try:
        if hc.systems() != contract["retrieval"]["systems"]:
            found.append("E, E5 or BM25 recomputed from their sources differ from the contract")
    except hc.ContractError as error:
        found.append(f"the systems: {error}")
    if rv.identity() != contract["retrieval"]["code_sha256"]:
        found.append("the retrieval code or configs differ from the contract")
    import importlib.metadata

    for package in ("ir-measures", "pytrec-eval-terrier"):
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != contract["evaluation"]["evaluator"][package]:
            found.append(f"the evaluator {package} installed is not the contract's")
    roles = hr.contracts_record()
    if _text_sha(json.dumps(roles, sort_keys=True)) != contract["roles"]["role_contracts_sha256"]:
        found.append("the role contracts differ from the contract")
    if hr.template_hashes() != contract["roles"]["templates_sha256"]:
        found.append("the templates differ from the contract")
    reference = contract["certification"]
    try:
        cert = hcert.selected(reference["id"], reference["manifest_sha256"], repo)
        if hcert.reference(cert) != reference:
            found.append("the certification is not the one the contract records")
        gate1 = hcert.read_json(cert / hcert.PROVENANCE / "gate1.json")
        if _sha(repo / AUDIT_MODULE) != gate1["audit"]["module_sha256"]:
            found.append("the isolation audit is not the code Gate 1 passed with")
        pin = hcert.read_json(cert / hcert.PROVENANCE / "runtime.json")
        live = hi.binary_identity(Path(pin["path"]))
        if (live["version"], live["sha256"]) != (pin["version"], pin["sha256"]):
            found.append("the pinned Claude Code binary changed")
        claude = json.loads(json.dumps(hrt.claude_runtime(cert / hcert.PROVENANCE)))
        differ = hrt.runtime_differences(claude, contract["runtime"]["claude_runtime"])
        if differ:
            found.append(f"the Claude runtime differs: {differ[:5]}")
    except (hcert.CertificationError, OSError, KeyError, hrt.RuntimeIdentityError) as error:
        found.append(f"the certification: {error}")
    try:
        mlx = json.loads(
            json.dumps(hrt.mlx_runtime(config_path=rv.DECOMPOSER, max_new_tokens=rv.MAX_NEW_TOKENS))
        )
        differ = hrt.runtime_differences(mlx, contract["runtime"]["mlx_runtime_decomposer"])
        if differ:
            found.append(f"the MLX runtime of E's decomposer differs: {differ[:5]}")
    except hrt.RuntimeIdentityError as error:
        found.append(f"the MLX runtime: {error}")
    found += private_path_problems(repo, namespace, raw)
    private = None
    try:
        private = read_private(namespace / "provenance")
        dedup = contract["deduplication"]["references"]
        if private.exclusions_identity != dedup["historical_exclusions"]:
            found.append("the historical exclusions are not the contract's")
        if list(private.exclusions_sources) != sorted(
            str(r["contract_sha256"]) for r in ABORTED_RUNS
        ):
            found.append("the historical exclusions are not collected from the seven aborted runs")
        if private.development_identity != dedup["development_topics"]:
            found.append("the development references are not the contract's")
        dev = bm.load_benchmark(bm.DEV_DIR)
        from ingest import psd2_exclusions as px

        layer = rv.load_layer()
        excluded = px.load_psd2_exclusions(px.PSD2_EXCLUSIONS_PATH).excluded()
        slots, units = bank_inputs(dev, bm.load_corpus(), layer.layer, excluded)
        bank = v2.build_bank(slots, units, private.salt)
        if bank.sha256() != contract["design"]["bank_sha256"]:
            found.append("the bank drawn from the salt is not the contract's")
    except (ContractV2Error, OSError, KeyError, ho.HoldoutError, json.JSONDecodeError) as error:
        found.append(f"the private inputs: {error}")
    return contract, private, found


def public_commit_problems(
    contract: Mapping[str, Any], pinned: str, repo: Path, provenance: Path
) -> list[str]:
    path = provenance / PUBLIC_COMMIT_FILE
    if not path.exists():
        return ["the public commit holding the contract is not recorded"]
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("contract_sha256") != pinned:
        return ["the recorded public commit is of another contract"]
    return committed_problems(contract, pinned, repo, record["commit"])
