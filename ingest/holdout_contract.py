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
    "ingest/holdout_broker.py",
    "ingest/holdout_checks.py",
    "ingest/holdout_contract.py",
    "ingest/holdout_executor.py",
    "ingest/holdout_isolation.py",
    "ingest/holdout_judging.py",
    "ingest/holdout_launcher.py",
    "ingest/holdout_ledger.py",
    "ingest/holdout_packets.py",
    "ingest/holdout_pipeline.py",
    "ingest/holdout_roles.py",
    "ingest/holdout_runtime.py",
    "ingest/holdout_spool.py",
    "ingest/holdout_state.py",
    "ingest/psd2_exclusions.py",
    "scripts/evidence_tools.py",
    "scripts/holdout.py",
)
# What else decides how TEST is built, read by that code: the protocol (the judges' rules,
# the rubric, the areas), the diagnostic judge's config, and DEV and the fixtures (the
# gatekeeper's comparisons, the authors' forbidden wording)
# A documented amendment of the construction, public and frozen with the rest: what it
# changes, from which point of the durable state's log, and why (CONSTRUCTION_AMENDMENT_1)
AMENDMENT_PATH = Path("config/holdout/construction-amendment-1.json")
CONSTRUCTION_INPUTS = (
    "data/benchmark/PROTOCOL.md",
    AMENDMENT_PATH.as_posix(),
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
) -> dict[str, Any]:
    """What fixes how TEST is built, also frozen before authoring: the grid, the order of the
    cells, the omission audit, the role contracts, the templates, the broker's limits, and the
    Claude and MLX runtimes (ingest/holdout_runtime.py). The lists themselves stay in the
    private TEST directory; the contract publishes their hashes. The account's email and
    organization never enter it."""
    text = json.dumps([claude_runtime, mlx_runtime]).lower()
    if "email" in text or "organization" in text:
        raise ContractError("a runtime identity carries the account's identity: never public")
    from ingest import holdout_broker as hb
    from ingest import holdout_checks as hk
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
        "code_sha256": {f: _sha(Path(f)) for f in CONSTRUCTION_FILES},
        "inputs_sha256": {f: _sha(Path(f)) for f in CONSTRUCTION_INPUTS},
        "template_files": {n: hr.template_path(n).as_posix() for n in hr.template_hashes()},
        "checks": hk.record(),
        "claude_runtime": dict(claude_runtime),
        "mlx_runtime": dict(mlx_runtime),
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
        "amends": amends(),
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
