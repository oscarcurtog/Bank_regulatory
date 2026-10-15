"""Iteration 5 (config/retrieval/iteration-5.md): G1, G2 and G3 with F's adapter, each evaluated
once on DEV, in that order.

    PYTHONPATH=. caffeinate -i .venv/bin/python scripts/night_iteration.py \\
        --expect-version 0.10.0 --experiment g1|g2|g3

Every (requirement, chunk) score comes from F's adapter through the MLX scorer
(scripts/mlx_rerank.py). Scores already computed (F's DEV run, earlier experiments of the night)
are carried over after a check of every carried row against its input: its id, its document and
the SHA-256 of its input. Each decomposer's requirements are encoded exactly as their D run encoded
them. Before anything changes, the working reference's recorded ranking is rebuilt exactly from the
carried scores. The experiment's ranking is written and its SHA-256 recorded before any label is
read; a rerun must reproduce it. Then it is measured, paired with E5, E and the working reference,
and judged by the protocol; the working reference moves when the protocol says so
(retrieval-iteration-5/state.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import decomposition as dc
from ingest import dense as dn
from ingest import embeddings as em
from ingest import requirement_rerank as rq
from ingest import retrieval as rt
from ingest import run_report as rr

E5 = Path("config/embeddings/multilingual-e5-large.json")
RECIPE = Path("config/retrieval/reranker-finetune.json")
MLX_PYTHON = Path(".cache/mlx-venv/bin/python")
D_SYSTEMS = {"4b": "d-qwen3-4b-k5", "8b": "d-qwen3-8b-k5"}
E_SYSTEM = "e-d4b-qwen3-reranker-k5"
F_SYSTEM = "f-d4b-qwen3-reranker-lora-k5"
ORDER = ("g1", "g2", "g3")
SYSTEMS = {"g1": "g1-global-score", "g2": "g2-two-decomposers", "g3": "g3-depth100"}
CHANGES: dict[str, dict[str, Any]] = {
    "g1": {"rule": "scored"},
    "g2": {"decomposers": ["4b", "8b"]},
    "g3": {"depth": 100},
}
START = {"reference": F_SYSTEM, "rule": "turns", "decomposers": ["4b"], "depth": 50}


def answers_of(model: str, version: str, max_tokens: int, it4: Path) -> dict[str, dc.Requirements]:
    run = rt.run_dir(D_SYSTEMS[model], max_tokens, version)
    record = json.loads((it4 / f"d-qwen3-{model}.json").read_text("utf-8"))
    if (
        hashlib.sha256((run / "prompts.jsonl").read_bytes()).hexdigest()
        != record["prompt"]["sha256"]
    ):
        raise SystemExit(f"D-{model}'s prompts are not the ones its record names")
    rows = map(json.loads, (run / "outputs.jsonl").read_text("utf-8").splitlines())
    return {r["id"].split("#")[1]: dc.parse(r["output"]) for r in rows}


def pair_id(requirement: str, chunk: str) -> str:
    return hashlib.sha256(json.dumps([requirement, chunk]).encode("utf-8")).hexdigest()[:24]


def input_sha(instruction: str, query: str, document: str) -> str:
    """The SHA-256 the MLX scorer stores for an input (scripts/mlx_rerank.py)."""
    fields = [instruction, query, document]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode("utf-8")).hexdigest()


def softmax(ranking: tuple[str, ...], raw: dict[str, float]) -> dict[str, float]:
    """A requirement's scores made comparable across requirements: the softmax over its own list."""
    top = max(raw[c] for c in ranking)
    exp = {c: math.exp(raw[c] - top) for c in ranking}
    total = sum(exp.values())
    return {c: v / total for c, v in exp.items()}


def requirement_lists(
    answers: dict[str, dict[str, dc.Requirements]], decomposers: list[str], qid: str
) -> list[str]:
    """A query's requirements: each decomposer's answer if readable on its own; with two, their
    items interleaved (the 4B's first, then the 8B's first, ...); a text repeated, ignoring case,
    once."""
    lists = []
    for m in decomposers:
        p = answers[m][qid]
        lists.append(list(p.items) if p.readable else [])
    out: list[str] = []
    for group in zip_longest(*lists):
        for x in group:
            if x is not None and x.lower() not in {s.lower() for s in out}:
                out.append(x)
    return out


def columns_of(d: dict[str, Any]) -> dict[str, float]:
    return {
        "nDCG@10": d["global"]["nDCG@10"],
        "Recall@10": d["global"]["Recall@10"],
        "MRR@10": d["global"]["MRR@10"],
        "Recall@20": d["global"]["Recall@20"],
        "Full-evidence@10": d["full_evidence"]["Full-evidence@10"],
        "multi_provision": d["groups"]["multi_provision"]["Recall@10"],
        "cross_reference": d["groups"]["cross_reference"]["Recall@10"],
        "cross_act": d["groups"]["cross_act"]["Recall@10"],
        "dependent": d["dependent_evidence"]["Recall@10"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    parser.add_argument("--experiment", choices=ORDER, required=True)
    args = parser.parse_args(argv)
    frozen = rt.load_frozen_inputs(args.benchmark, args.expect_version, args.max_tokens)
    version = frozen.dev.manifest.benchmark_version
    it4 = args.results / f"dev-{version}" / "retrieval-iteration-4"
    out = args.results / f"dev-{version}" / "retrieval-iteration-5"
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state: dict[str, Any] = (
        json.loads(state_path.read_text("utf-8"))
        if state_path.exists()
        else dict(START, done=[], started={})
    )
    if state.get("stopped"):
        raise SystemExit(f"the iteration stopped: {state['stopped']}")
    if len(state["done"]) >= len(ORDER) or args.experiment != ORDER[len(state["done"])]:
        raise SystemExit(f"the order is {ORDER}; done so far: {state['done']}")
    recipe = json.loads(RECIPE.read_text("utf-8"))
    instruction = recipe["instruction"]

    # 1. requirements, encoded per decomposer exactly as its D run encoded them; no label read
    dense = dn.load_dense(E5, frozen, args.max_tokens)
    e5_ranked, e5_summary = rt.load_system_run(
        "multilingual-e5-large", args.max_tokens, version, args.results
    )
    if dense.rank() != e5_ranked:
        raise SystemExit("the cached E5 vectors do not reproduce E5's recorded run")
    e5_of = {r.query_id: r for r in e5_ranked}
    answers = {m: answers_of(m, version, args.max_tokens, it4) for m in D_SYSTEMS}
    doc_ids = list(dense.doc_ids)
    full: dict[str, tuple[str, ...]] = {}
    for m in ("8b", "4b"):  # the 4B last: a text both share keeps the 4B's ranking, F's
        texts = sorted({x for p in answers[m].values() if p.readable for x in p.items})
        for x, v in zip(texts, dense.encode_queries(texts), strict=True):
            full[x] = rt.rank_by_scores("r", doc_ids, dense.doc_vectors @ v).chunk_ids
    question = [q.query_id for q in frozen.dev.queries]

    # 2. the carried scores, every row checked against its input
    known: dict[str, dict[str, Any]] = {}
    f_run = rt.run_dir("ft-reranker-dev", args.max_tokens, version)
    f_inputs = [
        json.loads(x) for x in (f_run / "adapter-inputs.jsonl").read_text("utf-8").splitlines()
    ]
    f_scores = {
        r["id"]: r
        for r in map(json.loads, (f_run / "adapter-scores.jsonl").read_text("utf-8").splitlines())
    }
    f_pairs = sorted(
        {
            (x, c)
            for p in answers["4b"].values()
            for x in (p.items if p.readable else ())
            for c in full[x][: rq.DEPTH]
        }
    )
    if len(f_pairs) != len(f_inputs) or len(f_scores) != len(f_inputs):
        raise SystemExit("F's DEV inputs are not the pairs this script rebuilds")
    for i, (x, c) in enumerate(f_pairs):
        row, text = f_inputs[i], frozen.chunks[c].text
        s = f_scores[str(i)]
        if (row["id"], row["query"], row["document"]) != (str(i), x, text):
            raise SystemExit(f"F's input {i} is not the pair ({x[:40]}..., {c})")
        if s["input_sha256"] != input_sha(instruction, x, text):
            raise SystemExit(f"F's score {i} was computed from another input")
        known[pair_id(x, c)] = s
    for previous in ORDER[: len(state["done"])]:
        path = rt.run_dir(f"night-{previous}", args.max_tokens, version) / "scores.jsonl"
        night_inputs = {
            r["id"]: r
            for r in map(json.loads, (path.parent / "inputs.jsonl").read_text("utf-8").splitlines())
        }
        for s in map(json.loads, path.read_text("utf-8").splitlines()):
            r = night_inputs[s["id"]]
            if s["input_sha256"] != input_sha(instruction, r["query"], r["document"]):
                raise SystemExit(f"{path}: a score was computed from another input")
            known.setdefault(s["id"], s)

    def build(cfg: dict[str, Any], score: dict[str, float]) -> tuple[list[rt.Ranked], int]:
        ranked = []
        pairs = 0
        for qid in question:
            hits, norm = [], []
            for x in requirement_lists(answers, cfg["decomposers"], qid):
                cand = full[x][: cfg["depth"]]
                raw = {c: score[pair_id(x, c)] for c in cand}
                order = tuple(rq.reorder(cand, raw))
                hits.append(order)
                norm.append(softmax(order, raw))
                pairs += len(cand)
            e5 = e5_of[qid].chunk_ids
            if cfg["rule"] == "scored":
                ranking = dc.assemble_scored(e5, hits, norm).ranking
            else:
                ranking = dc.assemble(e5, hits).ranking
            n = len(ranking)
            ranked.append(rt.Ranked(qid, ranking, tuple(float(n - i) for i in range(n))))
        return ranked, pairs

    def needed(cfg: dict[str, Any]) -> list[tuple[str, str]]:
        return sorted(
            {
                (x, c)
                for qid in question
                for x in requirement_lists(answers, cfg["decomposers"], qid)
                for c in full[x][: cfg["depth"]]
            }
        )

    # 3. the working reference rebuilt exactly from the carried scores
    ref_cfg = {k: state[k] for k in ("rule", "decomposers", "depth")}
    missing = [p for p in needed(ref_cfg) if pair_id(*p) not in known]
    if missing:
        raise SystemExit(f"{len(missing)} pairs of the working reference have no carried score")
    carried = {k: v["logit_diff"] for k, v in known.items()}
    ref_rebuilt, _ = build(ref_cfg, carried)
    ref_ranked, _ = rt.load_system_run(state["reference"], args.max_tokens, version, args.results)
    if [r.chunk_ids for r in ref_rebuilt] != [r.chunk_ids for r in ref_ranked]:
        raise SystemExit(f"{state['reference']}'s ranking is not rebuilt from the carried scores")

    # 4. the variant: the missing pairs scored, the ranking written and fixed before any label
    cfg = dict(ref_cfg, **CHANGES[args.experiment])
    pairs = needed(cfg)
    run = rt.run_dir(f"night-{args.experiment}", args.max_tokens, version)
    run.mkdir(parents=True, exist_ok=True)
    scores_path = run / "scores.jsonl"
    stored = (
        {json.loads(x)["id"] for x in scores_path.read_text("utf-8").splitlines()}
        if scores_path.exists()
        else set()
    )
    carry = [
        dict(known[pair_id(x, c)], id=pair_id(x, c))
        for x, c in pairs
        if pair_id(x, c) in known and pair_id(x, c) not in stored
    ]
    with scores_path.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in carry)
    inputs = [
        {"id": pair_id(x, c), "query": x, "document": frozen.chunks[c].text} for x, c in pairs
    ]
    (run / "inputs.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inputs), "utf-8"
    )
    training = json.loads((it4 / "f-training.json").read_text("utf-8"))
    adapter = Path(training["out_dir"]) / training["best"]["adapter"]
    if not MLX_PYTHON.exists():
        raise SystemExit(f"{MLX_PYTHON} is missing: the MLX environment of D-040 is not set up")
    command = [str(MLX_PYTHON), "scripts/mlx_rerank.py", "--recipe", str(RECIPE)]
    command += ["--inputs", str(run / "inputs.jsonl"), "--out", str(scores_path)]
    command += ["--adapter", str(adapter)]
    subprocess.run(command, check=True, env=os.environ | {"HF_HOME": str(em.hf_home())})
    score = {
        json.loads(x)["id"]: json.loads(x)["logit_diff"]
        for x in scores_path.read_text("utf-8").splitlines()
    }
    ranked, scored_pairs = build(cfg, score)
    system = SYSTEMS[args.experiment]
    sha = rt.write_run(
        rt.run_dir(system, args.max_tokens, version) / rt.RUN_FILE, ranked, frozen.evaluable
    )
    started = state.setdefault("started", {})
    if args.experiment in started and started[args.experiment] != sha:
        raise SystemExit(f"{args.experiment} was started with another ranking: no second version")
    started[args.experiment] = sha
    state_path.write_text(json.dumps(state, indent=1) + "\n", "utf-8")

    # 5. the labels enter here, once the ranking is fixed
    queries = [q for t in frozen.targets for q in t.queries]
    e_ranked, _ = rt.load_system_run(E_SYSTEM, args.max_tokens, version, args.results)
    diag = {
        n: rr.diagnostics(frozen.dev, frozen.layer, r)
        for n, r in (
            ("e5", e5_ranked),
            ("e", e_ranked),
            ("reference", ref_ranked),
            ("variant", ranked),
        )
    }
    cols = {n: columns_of(d) for n, d in diag.items()}

    def paired(b: str) -> dict[str, Any]:
        return {
            m: rr.paired_bootstrap(
                {
                    q.query_id: diag["variant"]["per_query"][m][q.query_id]
                    - diag[b]["per_query"][m][q.query_id]
                    for q in queries
                },
                queries,
            )
            for m in ("Recall@10", "nDCG@10", "Full-evidence@10")
        }

    v, ref = ({k: round(x, 3) for k, x in cols[n].items()} for n in ("variant", "reference"))
    replaces = (
        v["Recall@10"] - ref["Recall@10"] >= 0.02 - 1e-9
        or v["Full-evidence@10"] - ref["Full-evidence@10"] >= 0.03 - 1e-9
    ) and v["nDCG@10"] >= ref["nDCG@10"] - 0.01 - 1e-9
    useful = (
        v["Recall@10"] >= 0.85
        and v["multi_provision"] >= 0.60
        and v["dependent"] >= 0.45
        and v["Full-evidence@10"] >= 0.75
    )
    evaluated = rr.evaluate_ranked(
        frozen.dev, frozen.layer, ranked, {c: x.body for c, x in frozen.chunks.items()}
    )
    summary = {
        "system": {
            "kind": f"iteration_5_{args.experiment}",
            "configuration": cfg,
            "normalization": "softmax over each requirement's list"
            if cfg["rule"] == "scored"
            else None,
            "reranker": f"F's LoRA adapter, {training['best']['adapter']}",
        },
        "identity": {"base_run_sha256": e5_summary["identity"]["run_sha256"], "run_sha256": sha},
        "benchmark_version": version,
        "parser_version": frozen.dev.manifest.parser_version,
        "layer": em.layer_source(list(frozen.docs)).model_dump(mode="json"),
        "queries_sha256": em.query_source(frozen.dev).queries_sha256,
        "metrics": evaluated["metrics"],
        "diagnostics": {k: x for k, x in diag["variant"].items() if k != "per_query"},
        "paired_against_e5": paired("e5"),
        "per_need": evaluated["per_need"],
        "per_query": evaluated["per_query"],
        "failures": evaluated["failures"],
    }
    sys_dir = args.results / f"dev-{version}" / f"{system}-layer-{args.max_tokens}"
    sys_dir.mkdir(parents=True, exist_ok=True)
    (sys_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", "utf-8"
    )
    payload = {
        "experiment": args.experiment,
        "system_name": system,
        "run_sha256": sha,
        "working_reference": state["reference"],
        "configuration": cfg,
        "pairs_ranked": scored_pairs,
        "pairs_carried": len(carry),
        "columns": cols,
        "by_language_recall10": diag["variant"]["by_language"]["Recall@10"],
        "paired_against_e5": paired("e5"),
        "paired_against_e": paired("e"),
        "paired_against_working_reference": paired("reference"),
        "replaces_working_reference": replaces,
        "useful_target": useful,
    }
    (out / f"{args.experiment}.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", "utf-8"
    )
    state["done"].append(args.experiment)
    if replaces:
        state.update(reference=system, **cfg)
    if useful:
        state["stopped"] = f"{args.experiment} reached the useful target"
    state_path.write_text(json.dumps(state, indent=1) + "\n", "utf-8")
    print(json.dumps({"columns": cols, "replaces": replaces, "useful_target": useful}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
