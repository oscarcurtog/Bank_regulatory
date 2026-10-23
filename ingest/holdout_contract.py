"""The pre-authoring evaluation contract of TEST (S3, D-054, Codex Gate 2 P1.5).

Before a single TEST need is written, this fixes what S4 will run and how it will be read:
the exact configurations of E, E5 and BM25 as they produced their DEV results, the code that
runs them, the corpus and chunk layer they read, the evaluation and the analysis. Every value
is read from the file or the module that defines it, never typed here, and each source is
recorded with its SHA-256. No system runs to build it.

After authoring begins, nothing in it may change. The freeze of TEST adds only derived
properties of TEST (dataset hashes, counts, realized distributions, rejection and overlap
statistics, file identities); S4 checks every file hash of this contract before it runs.

What the contract freezes is CONTENT: the SHA-256 of every file it names (frozen_files), and
the corpus and chunk layer on disk. verify() checks only that, never which commit is checked
out, so committing these exact files changes nothing it checks. Two commits are recorded as
provenance, never compared with HEAD: E_RESULT_COMMIT, the historical commit of E's DEV result
that the code is diffed against when the contract is built, and built_on_commit, the HEAD the
contract was built on. The public pre-authoring commit cannot be inside the contract (it
contains it); it is recorded after it is made, outside the contract, by
ingest/holdout_launcher.py record_public_commit, which checks that the commit holds exactly
the frozen content (committed_problems).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import decomposition as dc
from ingest import evaluation as ev
from ingest import ir_metrics as irm
from ingest import requirement_rerank as rq
from ingest import run_report as rr

CONTRACT_PATH = Path("config/holdout/pre-authoring-contract.json")
PLAN_PATH = Path("config/holdout/s4-analysis-plan.md")
SCHEMA = "holdout-pre-authoring-contract.v1"
DEV_RESULTS = Path("data/results/dev-0.10.0")
# The commit that recorded E's DEV result (M-049); the code is compared with it.
E_RESULT_COMMIT = "7636908"
CODE_FILES = (
    "ingest/benchmark.py",
    "ingest/chunks.py",
    "ingest/decomposition.py",
    "ingest/dense.py",
    "ingest/download.py",
    "ingest/embeddings.py",
    "ingest/encoder.py",
    "ingest/evaluation.py",
    "ingest/hierarchy.py",
    "ingest/ir_metrics.py",
    "ingest/lexical.py",
    "ingest/necessity.py",
    "ingest/processed.py",
    "ingest/reranker.py",
    "ingest/requirement_rerank.py",
    "ingest/retrieval.py",
    "ingest/run_report.py",
    "scripts/bm25_baseline.py",
    "scripts/compare_embedding_runs.py",
    "scripts/decomposition.py",
    "scripts/embed_baseline.py",
    "scripts/mlx_generate.py",
    "scripts/report_embedding_run.py",
    "scripts/requirement_reranking.py",
)
# The code that builds TEST: a change after authoring is a change of method.
CONSTRUCTION_FILES = (
    "ingest/corpus_tools.py",
    "ingest/holdout.py",
    "ingest/holdout_admission.py",
    "ingest/holdout_broker.py",
    "ingest/holdout_certification.py",
    "ingest/holdout_checks.py",
    "ingest/holdout_contract.py",
    "ingest/holdout_controller.py",
    "ingest/holdout_executor.py",
    "ingest/holdout_isolation.py",
    "ingest/holdout_judging.py",
    "ingest/holdout_launcher.py",
    "ingest/holdout_ledger.py",
    "ingest/holdout_packets.py",
    "ingest/holdout_pipeline.py",
    "ingest/holdout_process.py",
    "ingest/holdout_roles.py",
    "ingest/holdout_runtime.py",
    "ingest/holdout_spool.py",
    "ingest/holdout_state.py",
    "ingest/psd2_exclusions.py",
    "scripts/evidence_tools.py",
    "scripts/holdout.py",
    "scripts/mlx_prompt_tokens.py",
)
# What else decides how TEST is built, read by that code: the protocol (the judges' rules,
# the rubric, the areas), the evidence discovery of every need (three searches, the exposure
# pool, the omission audit's search), every reachable failure with its classification, the
# diagnostic judge's
# config, and DEV and the fixtures (the gatekeeper's comparisons, the authors' forbidden
# wording)
EVIDENCE_PATH = Path("config/holdout/evidence-discovery.md")
FAILURES_PATH = Path("config/holdout/failure-paths.md")
# The amendment of the second, aborted construction (CONSTRUCTION_AMENDMENT_1): public
# history, kept beside its run's record (ABORTED_RUNS); the clean restart amends nothing
AMENDMENT_PATH = Path("config/holdout/construction-amendment-1.json")
CONSTRUCTION_INPUTS = (
    "data/benchmark/PROTOCOL.md",
    EVIDENCE_PATH.as_posix(),
    FAILURES_PATH.as_posix(),
    "config/retrieval/qwen3-8b-mlx.json",
    *(
        f"data/benchmark/{split}/{name}"
        for split in ("dev", "fixtures")
        for name in ("manifest.json", "needs.jsonl", "queries.jsonl", "qrels.jsonl")
    ),
)
CONFIG_FILES = {
    "E5": "config/embeddings/multilingual-e5-large.json",
    "BM25": "config/retrieval/bm25.json",
    "decomposer": "config/retrieval/qwen3-4b-instruct-mlx.json",
    "reranker": "config/retrieval/qwen3-reranker-0.6b.json",
}
# Read from the diff of each file since E_RESULT_COMMIT on 2026-10-19, and kept as a
# check: the contract refuses to build if any other code file differs from that commit.
CHANGES_SINCE_E = {
    "ingest/decomposition.py": (
        "adds assemble_scored (G1, iteration 5); assemble, E's rule, unchanged"
    ),
    "ingest/ir_metrics.py": "adds binary_scores (T4); the graded measures unchanged",
    "ingest/lexical.py": (
        "adds the Italian analyzer of T4; the English and Porter branch is equivalent, and "
        "tests/unit/test_lexical.py keeps the DEV config hash e086bdb4"
    ),
}


class ContractError(Exception):
    """The contract cannot be built, or a file no longer matches it."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise ContractError(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout.strip()


def _summary(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((DEV_RESULTS / name / "summary.json").read_text("utf-8"))
    return data


def _lock_version(package: str) -> str:
    for line in Path("requirements-dev.lock.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{package}=="):
            return line.split("==", 1)[1].split()[0]
    raise ContractError(f"{package} is not pinned in requirements-dev.lock.txt")


def _constant(path: str, name: str) -> int:
    """An integer constant of a script, read from its source (scripts are not modules)."""
    import re

    match = re.search(rf"^{name} = (\d+)\b", Path(path).read_text(encoding="utf-8"), re.M)
    if match is None:
        raise ContractError(f"{path} defines no {name}")
    return int(match.group(1))


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_identity(repo: Path) -> dict[str, Any]:
    """The SHA-256 of every file that runs E, E5, BM25 and the evaluation, and what changed
    since E's DEV result. Refuses an uncommitted change or an unexplained one."""
    dirty = _git(repo, "status", "--porcelain", "--", *CODE_FILES, *CONFIG_FILES.values())
    if dirty:
        raise ContractError(f"uncommitted changes in the contract's files:\n{dirty}")
    changed = set(
        _git(repo, "diff", "--name-only", E_RESULT_COMMIT, "HEAD", "--", *CODE_FILES).split()
    )
    unexplained = sorted(changed - set(CHANGES_SINCE_E))
    if unexplained:
        raise ContractError(f"changed since E's result, with no recorded reason: {unexplained}")
    return {
        # provenance only: verify() never compares either with the commit checked out
        "built_on_commit": _git(repo, "rev-parse", "HEAD"),
        "e_result_commit": _git(repo, "rev-parse", E_RESULT_COMMIT),
        "files_sha256": {f: _sha(repo / f) for f in CODE_FILES},
        "changes_since_e_result": {f: CHANGES_SINCE_E[f] for f in sorted(changed)},
        "python": sys.version.split()[0],
    }


def systems() -> dict[str, Any]:
    e5 = _summary("multilingual-e5-large-layer-500")
    bm25 = _summary("bm25-layer-500")
    d4b = _summary("d-qwen3-4b-k5-layer-500")
    e = _summary("e-d4b-qwen3-reranker-k5-layer-500")
    config = {k: json.loads(Path(p).read_text(encoding="utf-8")) for k, p in CONFIG_FILES.items()}
    if (
        e["system"]["decomposer"] != config["decomposer"]
        or e["system"]["reranker"] != config["reranker"]
    ):
        raise ContractError("E's recorded models differ from their config files")
    if bm25["system"] != {"kind": "bm25", **config["BM25"]}:
        raise ContractError("BM25's recorded system differs from its config file")
    ranking = "score descending, then chunk id ascending (ingest/retrieval.py rank_by_scores)"
    return {
        "E5": {
            "config_file": CONFIG_FILES["E5"],
            "config_sha256": _sha(Path(CONFIG_FILES["E5"])),
            "config": config["E5"],
            "config_hash": e5["config_hash"],
            "tokenizer_sha256": e5["identity"]["tokenizer_sha256"],
            "runtime": e5["identity"]["runtime"],
            "scoring": "dot product of the normalized mean-pooled vectors (its similarity)",
            "ranking": ranking,
            "depth": "every chunk of the layer, at least ingest/evaluation.MIN_RUN_DEPTH",
            "dev_run_sha256": e5["identity"]["run_sha256"],
        },
        "BM25": {
            "config_file": CONFIG_FILES["BM25"],
            "config_sha256": _sha(Path(CONFIG_FILES["BM25"])),
            "config": config["BM25"],
            "config_hash": bm25["config_hash"],
            "versions": bm25["identity"]["versions"],
            "corpus_unit": "the chunk_text of every chunk of the layer",
            "ranking": ranking,
            "dev_run_sha256": bm25["identity"]["run_sha256"],
        },
        "E": {
            "kind": e["system"]["kind"],
            "rule": e["system"]["rule"],
            "d_rule": d4b["system"]["rule"],
            "decomposer": {
                "config_file": CONFIG_FILES["decomposer"],
                "config_sha256": _sha(Path(CONFIG_FILES["decomposer"])),
                "config": config["decomposer"],
                "prompt": "ingest/decomposition.py requirements_prompt",
                "system_sha256": _text_sha(dc.SYSTEM),
                "necessary_sha256": _text_sha(dc.NECESSARY),
                "task_sha256": _text_sha(dc.TASK),
                "excerpts": dc.EXCERPTS,
                "requirements": [dc.MIN_REQUIREMENTS, dc.MAX_REQUIREMENTS],
                "max_new_tokens": _constant("scripts/decomposition.py", "MAX_NEW_TOKENS"),
                "decoding": config["decomposer"]["decoding"],
                "unreadable": "fewer than 2 readable requirements keep E5's ranking",
            },
            "requirement_retrieval": {
                "model": "E5, the same contract",
                "depth": rq.DEPTH,
            },
            "reranker": {
                "config_file": CONFIG_FILES["reranker"],
                "config_sha256": _sha(Path(CONFIG_FILES["reranker"])),
                "config": config["reranker"],
                "instruction": rq.INSTRUCTION,
                "order": "descending score; equal scores keep E5's order",
            },
            "assembler": {
                "function": "ingest/decomposition.py assemble",
                "K": dc.K,
                "keep": dc.KEEP,
                "depth": dc.DEPTH,
                "after_the_ten": "the rest of E5's ranking, in E5's order",
            },
            "dev_identity": e["identity"],
        },
    }


def corpus() -> dict[str, Any]:
    dev = json.loads((bm.DEV_DIR / bm.MANIFEST_FILE).read_text(encoding="utf-8"))
    layer = _summary("multilingual-e5-large-layer-500")["layer"]
    return {
        "acts": dev["corpus"],
        "parser_version": dev["parser_version"],
        "psd2_exclusions_sha256": _sha(bm.BENCHMARK_DIR / "psd2_excluded_articles.json"),
        "layer": layer,
        "query_template": dev["templates"],
    }


def evaluation() -> dict[str, Any]:
    return {
        "relevance": (
            f"binary at grade >= {bm.RELEVANT_GRADE}; nDCG uses the grades as linear gains"
        ),
        "metrics": {
            "co_primary": ["Recall@10", f"nDCG@{irm.NDCG_CUTOFF}"],
            "secondary": ["Recall@5", "Recall@20", f"MRR@{irm.MRR_CUTOFF}"],
        },
        "min_run_depth": ev.MIN_RUN_DEPTH,
        "evaluator": {
            "ir-measures": _lock_version("ir-measures"),
            "pytrec-eval-terrier": _lock_version("pytrec-eval-terrier"),
        },
        "unit": "the information need",
        "evaluable": "needs of answerability complete or partial; none never enters a metric",
        "aggregation": (
            "each metric per query, the mean over a need's queries (its three formulations: "
            "EN formal, EN natural, ES natural), then the mean over the evaluable needs: "
            "ingest/retrieval.py macro_by_need and ingest/run_report.py paired_bootstrap, the "
            "computation of M-049's +0.061"
        ),
        "aggregation_note": (
            "D-020 calls this global value descriptive because it mixes languages; S4 makes it "
            "primary by the S3 decision that the statistical unit is the need, and reports the "
            "languages as secondary strata"
        ),
        "missing_rankings": "a query absent from a run is scored 0 and listed (evaluation.py)",
    }


def analysis(plan: Path = PLAN_PATH) -> dict[str, Any]:
    text = plan.read_text(encoding="utf-8")
    if "OPEN" in text or "DRAFT" in text:
        raise ContractError(f"{plan} still has an OPEN or DRAFT item")
    return {
        "plan_file": plan.as_posix(),
        "plan_sha256": _sha(plan),
        "primary_comparison": "E against E5",
        "bootstrap": {
            "function": "ingest/run_report.py paired_bootstrap",
            "seed": rr.SEED,
            "resamples": 10_000,
            "level": 0.95,
            "interval": "percentile: sorted means at int(0.025 * reps) and int(0.975 * reps) - 1",
            "resampling": "needs with replacement (random.Random(seed).choices), each need's "
            "delta the mean over its queries",
        },
        "effect_size": "d_z = mean of the per-need deltas / their standard deviation (n - 1)",
    }


def construction(
    grid_path: Path,
    plan: Mapping[str, Any],
    roles: Mapping[str, Any],
    claude_runtime: Mapping[str, Any],
    mlx_runtime: Mapping[str, Any],
    certification: Mapping[str, Any],
) -> dict[str, Any]:
    """What fixes how TEST is built, also frozen before authoring: the grid, the order of the
    cells, the omission audit, the role contracts, the templates, the broker's limits, the
    Claude and MLX runtimes (ingest/holdout_runtime.py), and the sealed Gate 1 certification
    Claude's runtime is read from (ingest/holdout_certification.py reference), which the
    launcher selects by it and nothing else. The lists themselves stay in the private TEST
    directory; the contract publishes their hashes. The account's email and organization never
    enter it."""
    text = json.dumps([claude_runtime, mlx_runtime, certification]).lower()
    if "email" in text or "organization" in text:
        raise ContractError("a runtime identity carries the account's identity: never public")
    from ingest import holdout_broker as hb
    from ingest import holdout_checks as hk
    from ingest import holdout_controller as hct
    from ingest import holdout_packets as hp
    from ingest import holdout_process as hproc
    from ingest import holdout_roles as hr
    from ingest import holdout_state as hs

    return {
        "grid_sha256": _sha(grid_path),
        "order": {k: plan["order"][k] for k in ("algorithm", "seed", "sha256")},
        "omission_audit": {k: plan["omission_audit"][k] for k in ("algorithm", "seed", "sha256")}
        | {"cells": len(plan["omission_audit"]["cells"])},
        "role_contracts_sha256": _text_sha(json.dumps(roles, sort_keys=True)),
        "templates_sha256": hr.template_hashes(),
        "broker": {
            "escalation_at": hb.ESCALATION_AT,
            "ceiling": hb.MAX_REQUESTS,
            "max_payload_characters": hb.MAX_PAYLOAD,
            "violations_forgiven": hb.MAX_VIOLATIONS,
        },
        "attempts_per_cell": 3,
        "min_evaluable": 50,
        # the person's rules of 2026-10-19 (ingest/holdout_pipeline.py, holdout_state.py)
        "technical_executions_per_stage": hs.MAX_TECHNICAL_RUNS,
        "resume": "same cell, need, stage and frozen inputs; admitted outputs never run again",
        "state_schema": hs.SCHEMA,
        "operational_checkpoints": list(hs.CHECKPOINTS),
        "statement_review": "code and blind roles only; no person edits or approves a statement",
        "evidence_discovery": {
            "spec": EVIDENCE_PATH.as_posix(),
            "searches": ["search_1", "search_2", "search_3"],
            "audit_search": {
                "stage": "audit_search",
                "cells": len(plan["omission_audit"]["cells"]),
            },
            "first_judgment": "the label of the first of search_1, search_2, search_3 that listed "
            "the node; the answerability, search_1's",
            "exposure_pool": "every node a normal search received in full (an article or nodes "
            "page) or listed, that can be judged; a search snippet is not a full reading",
            "unlisted_nodes": "judge B's pool variant (judge_b_pool.v2), blind, every node shown "
            "graded explicitly 0 to 3 with its ambiguity, exactly once (a node left out or "
            "ungraded is a failure of the role's contract, never a grade 0); an explicit grade 0 "
            "that is not ambiguous is final, a grade >= 1 or an ambiguity gets a second blind "
            "judgment by the same contract, and judge C decides where they differ",
            "pool_batches": {
                "max_nodes": hp.POOL_BATCH_MAX_NODES,
                "max_tokens": hp.POOL_BATCH_MAX_TOKENS,
                "tokenizer": "cl100k_base (ingest/chunks.py)",
            },
            "judge_d_and_diagnostic": "every listed node, every pool node with a final grade >= 1, "
            "every pool node any blind judgment graded >= 1 or found ambiguous",
            "audit_threshold": "a node the audit search listed or received in full, outside the "
            "normal exposure pool, adjudicated grade >= 1 stops",
        },
        "process_deadlines_seconds": {
            "headless": hproc.HEADLESS_SECONDS,
            "broker_turn": hproc.BROKER_TURN_SECONDS,
            "broker_run": hproc.BROKER_RUN_SECONDS,
            "diagnostic_generation": hproc.LOCAL_SECONDS,
            "diagnostic_token_count": hproc.TOKEN_COUNT_SECONDS,
            "termination_grace": hproc.GRACE_SECONDS,
        },
        "controller_policy": hct.POLICY,
        "controller_deadlines_seconds": {
            "invocation": hct.INVOCATION_SECONDS,
            "stall": hct.STALL_SECONDS,
        },
        "code_sha256": {f: _sha(Path(f)) for f in CONSTRUCTION_FILES},
        "inputs_sha256": {f: _sha(Path(f)) for f in CONSTRUCTION_INPUTS},
        "template_files": {n: hr.template_path(n).as_posix() for n in hr.template_hashes()},
        "checks": hk.record(),
        "claude_runtime": dict(claude_runtime),
        "mlx_runtime": dict(mlx_runtime),
        "certification": dict(certification),
    }


# The constructions started under an earlier contract and aborted: what this contract replaces,
# and why. Nothing of an aborted run is reused.
ABORTED_RUNS = (
    {
        "status": "ABORTED_PROTOCOL_DEFECT_QWEN_DIAGNOSTIC",
        "contract_sha256": "fe18917551276d676a6f3e0c5c20d14ee34f2f4a66c9de18407f264366a858f9",
        "public_commit": "3226afad8fdb8f424b6486f540312e31cb16278a",
        "ready_needs": 0,
        "reason": (
            "The frozen Qwen diagnostic recovery policy assumed that a repeated technical "
            "execution could produce a different semantic output. In the real S3 run Qwen3-8B "
            "was running greedy deterministic decoding, so a structurally malformed but "
            "successfully generated response was likely reproducible. Retrying the identical "
            "semantic execution was therefore not a meaningful recovery strategy."
        ),
        "change": "the diagnostic's failure semantics only (holdout_checks.py "
        "diagnostic_admissible); sampling, grid, order, prompts, judges and systems unchanged",
    },
    {
        "status": "ABORTED_PROTOCOL_DEVELOPMENT_RUN",
        "contract_sha256": "e45ccb1125f6a648759493876f7ebe0c32364a24c438f2ff4ea0961705ed9e26",
        "base_contract_sha256": (
            "d4f3a1f2826d5a18ff8a1e43ac8ae62347b99a20b07a24121591e5e63956b381"
        ),
        "public_commit": "73bfb18f2ca90a8b7d5c32d0e57ed6896f078f48",
        "base_public_commit": "57ec4c9f3370a76f1125304112e58a04a0c2566b",
        "amendments": ["config/holdout/construction-amendment-1.json"],
        "ready_needs": 8,
        "reason": (
            "The omission audit's hard stop: a second blind search found evidence judge A had "
            "missed, adjudicated grade >= 1, so that the frozen rule called for the evidence "
            "search to be corrected for every need alike. The review that followed also found "
            "control-flow defects (outputs admitted without their full contract, a second "
            "searcher's declared problem with no transition, audits bound to the wrong attempt, "
            "unbounded processes, a controller that could act on a stale failure). The run is "
            "kept whole as a protocol-development run, and TEST restarts prospectively under one "
            "final procedure for every need, rather than being repaired."
        ),
        "change": (
            "evidence discovery: two blind searches for every need, their union adjudicated "
            "once, and the omission audit as a third blind search (config/holdout/"
            "evidence-discovery.md); complete admission of every output; bounded processes with "
            "verified cleanup; the diagnostic's token preflight; every reachable failure "
            "classified (config/holdout/failure-paths.md); the unattended controller in the "
            "repository. Sampling, grid, order, omission cells, templates, judges, thresholds, "
            "systems and the S4 metrics unchanged"
        ),
    },
    {
        "status": "ABORTED_PROTOCOL_DEVELOPMENT_RUN",
        "contract_sha256": "c6b0c6912b67c077c86738e186180c5000d305965e585a74744e5d63b215dbd6",
        "public_commit": "1765ae38f5a5381f5346f095ec55ff710cfcf363",
        "ready_needs": 6,
        "reason": (
            "Three independent executions of the metadata adjudicator gave a well-formed answer "
            "violating the same frozen semantic rule of its stage, so that technical retries "
            "could not repair it and the stage was left pending_technical_unresolved. The "
            "transition for that class was defined prospectively, and TEST restarts under it."
        ),
        "change": (
            "a semantic admissibility failure, the same frozen invariant violated in the three "
            "executions of search A, search B or the metadata adjudicator, rejects the attempt "
            "(semantic_inadmissible); contract failures and search C keep their technical "
            "exhaustion. Everything else unchanged"
        ),
    },
    {
        "status": "ABORTED_PROTOCOL_DEVELOPMENT_RUN",
        "contract_sha256": "433edd5d633777c575e81302423c19b8646c566e20ceb0a9698b7250c1634225",
        "public_commit": "3740fc7911f1f2bcc24d109e6045aef5eda471b9",
        "ready_needs": 5,
        "reason": (
            "Three executions of judge C answered the item they were given for another legal "
            "node, a failure of the role's contract, so that the stage was left "
            "pending_technical_unresolved. A candidate's identity was carried by its act and "
            "path, which the model wrote back itself, rather than by an id the orchestrator "
            "owns."
        ),
        "change": (
            "the identity of what a judge grades: every candidate of judges B, C and D and of "
            "the diagnostic is given under an opaque item id, answered by that exact id and "
            "mapped to its node by the orchestrator alone (judge_b.v2, judge_c.v2, judge_d.v2); "
            "a missing, unknown, repeated or other id stays a technical contract failure. "
            "Rubrics, thresholds, retries and every methodological rule unchanged"
        ),
    },
    {
        "status": "ABORTED_PROTOCOL_DEVELOPMENT_RUN",
        "contract_sha256": "b824db37fb9e13202f4d3efdc38a706a03c9c56fe030607a39ae5a6a167846d7",
        "public_commit": "808762e078f1e96a3f71a02c0b560b447332fca7",
        "ready_needs": 3,
        "reason": (
            "The omission audit's hard stop: the audit search listed a node outside the union "
            "of searches A and B, adjudicated grade >= 1. A structural review, counts only, "
            "found that most omissions were nodes the missing search had read in full and not "
            "listed: coverage depended on each search's discretionary, softly capped listing"
        ),
        "change": (
            "three fixed blind searches for every need; a mechanical exposure pool (every node "
            "a search received in full, listed or not, is judged: judge B's pool variant, a "
            "second blind judgment for a grade >= 1 or an ambiguity, judge C on a difference); "
            "the audit search's coverage is its own exposure; judge D and the diagnostic on the "
            "listed nodes and the pool's contested or relevant ones; no number of candidates in "
            "the searches' template (judge_a.v3); the controller's outer deadline 72 hours. The "
            "rubrics, the omission threshold >= 1, the retries and every other rule unchanged"
        ),
    },
)


def amends(path: Path = AMENDMENT_PATH) -> dict[str, Any]:
    """What an amended contract carries of its amendment: the base it amends, the point of the
    log where it applies, the identities it may change, the first transition the amended code
    must make on replay, and the amendment file's hash."""
    record = json.loads(path.read_text(encoding="utf-8"))
    keys = ("base_contract_sha256", "base_public_commit", "after_event", "identity_changes")
    return {
        "amendment": record["id"],
        "amendment_file": path.as_posix(),
        "amendment_sha256": _sha(path),
        **{k: record[k] for k in keys},
        "first_transition": record["first_transition"],
    }


def build(
    repo: Path = Path("."), construction_record: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if construction_record is None:
        raise ContractError("the construction record is required: the grid, order and roles")
    return {
        "schema": SCHEMA,
        "status": "frozen before authoring; only derived TEST properties are added at the freeze",
        "construction": dict(construction_record),
        "code": code_identity(repo),
        "corpus": corpus(),
        "systems": systems(),
        "evaluation": evaluation(),
        "analysis": analysis(),
        "aborted_runs": [dict(r) for r in ABORTED_RUNS],
        "derived_at_freeze_only": [
            "dataset hashes",
            "final counts",
            "realized distributions",
            "rejection counts",
            "overlap statistics",
            "file identities",
        ],
    }


def write(contract: Mapping[str, Any], path: Path = CONTRACT_PATH) -> str:
    if path.exists():
        raise ContractError(f"{path} exists: the contract is written once")
    path.write_text(json.dumps(contract, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return _sha(path)


def committed_problems(
    contract: Mapping[str, Any], pinned_sha256: str, repo: Path, commit: str
) -> list[str]:
    """What a commit lacks to be the timestamp of the frozen machinery: the contract file as
    pinned, and every frozen file with exactly its frozen content, read from the commit itself
    (git show), never from the working tree."""
    problems = []
    for f, h in {CONTRACT_PATH.as_posix(): pinned_sha256, **frozen_files(contract)}.items():
        done = subprocess.run(
            ["git", "-C", str(repo), "show", f"{commit}:{f}"], capture_output=True, check=False
        )
        if done.returncode != 0:
            problems.append(f"{f}: not in {commit[:12]}")
        elif hashlib.sha256(done.stdout).hexdigest() != h:
            problems.append(f"{f}: in {commit[:12]} with other content")
    return problems


def corpus_problems(contract: Mapping[str, Any]) -> list[str]:
    """The corpus on disk against the contract: the acts (CELEX and source SHA-256), the
    parser, the PSD2 exclusions and the chunk layer (ingest/embeddings.py layer_source)."""
    from ingest import chunks as ch
    from ingest import download as dl
    from ingest import embeddings as em

    expected = contract["corpus"]
    problems = []
    processed = bm.load_corpus()
    for act, source in expected["acts"].items():
        doc = processed.get(act)
        if doc is None or (doc.source_celex, doc.source_sha256) != (
            source["source_celex"],
            source["source_sha256"],
        ):
            problems.append(f"{act}: not the contract's source")
    if {d.parser_version for d in processed.values()} != {expected["parser_version"]}:
        problems.append("the parser version differs")
    if _sha(bm.BENCHMARK_DIR / "psd2_excluded_articles.json") != expected["psd2_exclusions_sha256"]:
        problems.append("the PSD2 exclusion list differs")
    manifest = dl.load_manifest()
    layer = em.layer_source([ch.load_chunks(act, manifest) for act in bm.ACTS]).model_dump()
    problems += [f"layer {k} differs" for k, v in expected["layer"].items() if layer.get(k) != v]
    return problems


def frozen_files(contract: Mapping[str, Any]) -> dict[str, str]:
    """Every file the contract freezes, by path, with its SHA-256: the code of the systems and
    the evaluation, their configs, the S4 plan, and the construction's code, inputs and
    templates."""
    files = dict(contract["code"]["files_sha256"])
    for system in contract["systems"].values():
        for part in (system, *(v for v in system.values() if isinstance(v, dict))):
            if "config_file" in part:
                files[part["config_file"]] = part["config_sha256"]
    files[contract["analysis"]["plan_file"]] = contract["analysis"]["plan_sha256"]
    construction = contract["construction"]
    files |= construction["code_sha256"] | construction["inputs_sha256"]
    for name, sha in construction["templates_sha256"].items():
        files[construction["template_files"][name]] = sha
    return dict(sorted(files.items()))


def verify(
    pinned_sha256: str,
    path: Path = CONTRACT_PATH,
    repo: Path = Path("."),
    check_corpus: bool = True,
) -> dict[str, Any]:
    """The contract as pinned, every file it freezes with the content it froze, and the corpus
    and chunk layer on disk as it records them: a stop otherwise. Content only: the commit
    checked out is never compared."""
    if _sha(path) != pinned_sha256:
        raise ContractError(f"{path} changed since it was pinned")
    contract: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    try:
        frozen = frozen_files(contract)
    except KeyError as error:
        raise ContractError(f"{path} is not a contract of this schema: no {error}") from None
    changed = [f for f, h in frozen.items() if not (repo / f).exists() or _sha(repo / f) != h]
    if changed:
        raise ContractError(f"files differ from the contract: {changed}")
    if check_corpus:
        problems = corpus_problems(contract)
        if problems:
            raise ContractError(f"the corpus differs from the contract: {problems}")
    return contract
