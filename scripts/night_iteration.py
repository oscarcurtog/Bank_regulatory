"""Iteration 5 (config/retrieval/iteration-5.md): G1, G2 and G3 with F's adapter, each evaluated
once on DEV.

    PYTHONPATH=. caffeinate -i .venv/bin/python scripts/night_iteration.py \\
        --expect-version 0.10.0 --experiment g1|g2|g3

Every (requirement, chunk) score comes from F's adapter through the MLX scorer
(scripts/mlx_rerank.py): the pairs F already scored on DEV are carried over, and only the missing
ones are scored. Before anything changes, F's recorded DEV ranking is rebuilt exactly from the
carried scores and D's rule. The experiment's ranking is then written, measured, paired with E5 and
with the reference of the moment, and judged by the protocol; the reference moves when the
protocol says so (retrieval-iteration-5/state.json). The labels are read only after every ranking
exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
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
D4B, D8B = "d-qwen3-4b-k5", "d-qwen3-8b-k5"
F_SYSTEM = "f-d4b-qwen3-reranker-lora-k5"
SYSTEMS = {"g1": "g1-global-score", "g2": "g2-two-decomposers", "g3": "g3-depth100"}
NORMALIZATION = "softmax"  # fixed in the protocol before F's DEV result was read


def answers_of(
    system: str, version: str, max_tokens: int, results: Path
) -> dict[str, dc.Requirements]:
    run = rt.run_dir(system, max_tokens, version)
    model = system.split("-")[2]  # "4b" or "8b"
    record = json.loads((results / f"d-qwen3-{model}.json").read_text("utf-8"))
    if (
        hashlib.sha256((run / "prompts.jsonl").read_bytes()).hexdigest()
        != record["prompt"]["sha256"]
    ):
        raise SystemExit(f"{system}'s prompts are not the ones its record names")
    rows = map(json.loads, (run / "outputs.jsonl").read_text("utf-8").splitlines())
    return {r["id"].split("#")[1]: dc.parse(r["output"]) for r in rows}


def pair_id(requirement: str, chunk: str) -> str:
    return hashlib.sha256(json.dumps([requirement, chunk]).encode("utf-8")).hexdigest()[:24]


def normalized(ranking: tuple[str, ...], raw: dict[str, float]) -> dict[str, float]:
    """A requirement's scores made comparable across requirements: the softmax over its own list."""
    if NORMALIZATION == "raw":
        return {c: raw[c] for c in ranking}
    top = max(raw[c] for c in ranking)
    exp = {c: math.exp(raw[c] - top) for c in ranking}
    total = sum(exp.values())
    return {c: v / total for c, v in exp.items()}


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
    parser.add_argument("--experiment", choices=sorted(SYSTEMS), required=True)
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
        else {
            "reference": F_SYSTEM,
            "rule": "turns",
            "decomposers": ["4b"],
            "depth": 50,
            "done": [],
        }
    )
    if args.experiment in state["done"]:
        raise SystemExit(f"{args.experiment} has run already: one DEV evaluation per experiment")
    if state.get("stopped"):
        raise SystemExit(f"the iteration stopped: {state['stopped']}")

    # 1. the requirements and E5's lists for this experiment; no label is read here
    dense = dn.load_dense(E5, frozen, args.max_tokens)
    e5_ranked, e5_summary = rt.load_system_run(
        "multilingual-e5-large", args.max_tokens, version, args.results
    )
    if dense.rank() != e5_ranked:
        raise SystemExit("the cached E5 vectors do not reproduce E5's recorded run")
    e5_of = {r.query_id: r for r in e5_ranked}
    answers = {"4b": answers_of(D4B, version, args.max_tokens, it4)}
    decomposers = list(state["decomposers"])
    depth = int(state["depth"])
    rule = state["rule"]
    if args.experiment == "g1":
        rule = "scored"
    elif args.experiment == "g2":
        decomposers = ["4b", "8b"]
    else:
        depth = 100
    if "8b" in decomposers:
        answers["8b"] = answers_of(D8B, version, args.max_tokens, it4)
    question = [q.query_id for q in frozen.dev.queries]
    requirements: dict[str, list[str]] = {}
    for qid in question:
        seen: list[str] = []
        for m in decomposers:
            p = answers[m][qid]
            for x in p.items if p.readable else ():
                if x.lower() not in {s.lower() for s in seen}:
                    seen.append(x)
        requirements[qid] = seen
    texts = sorted(
        {x for v in requirements.values() for x in v}
        | {x for p in answers["4b"].values() for x in p.items}
    )
    vectors = dict(zip(texts, dense.encode_queries(texts), strict=True))
    doc_ids = list(dense.doc_ids)
    full = {
        x: rt.rank_by_scores("r", doc_ids, dense.doc_vectors @ vectors[x]).chunk_ids for x in texts
    }

    # 2. F's scores carried over, F's DEV ranking rebuilt exactly, the missing pairs scored
    f_run = rt.run_dir("ft-reranker-dev", args.max_tokens, version)
    f_inputs = [
        json.loads(x) for x in (f_run / "adapter-inputs.jsonl").read_text("utf-8").splitlines()
    ]
    f_scores = {
        json.loads(x)["id"]: json.loads(x)
        for x in (f_run / "adapter-scores.jsonl").read_text("utf-8").splitlines()
    }
    f_pairs = sorted(
        {
            (x, c)
            for p in answers["4b"].values()
            for x in (p.items if p.readable else ())
            for c in full[x][: rq.DEPTH]
        }
    )
    if len(f_pairs) != len(f_inputs) or any(
        f_inputs[i]["query"] != x for i, (x, _) in enumerate(f_pairs)
    ):
        raise SystemExit("F's DEV inputs are not the pairs this script rebuilds")
    known = {pair_id(x, c): f_scores[str(i)] for i, (x, c) in enumerate(f_pairs)}
    f_hits = {
        qid: [
            tuple(
                rq.reorder(
                    full[x][: rq.DEPTH],
                    {c: known[pair_id(x, c)]["logit_diff"] for c in full[x][: rq.DEPTH]},
                )
            )
            for x in (p.items if p.readable else ())
        ]
        for qid, p in answers["4b"].items()
    }
    f_ranked, _ = rt.load_system_run(F_SYSTEM, args.max_tokens, version, args.results)
    for r in f_ranked:
        if dc.assemble(e5_of[r.query_id].chunk_ids, f_hits[r.query_id]).ranking != r.chunk_ids:
            raise SystemExit(f"{r.query_id}: F's DEV ranking is not rebuilt from its scores")
    needed = sorted({(x, c) for v in requirements.values() for x in v for c in full[x][:depth]})
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
        for x, c in needed
        if pair_id(x, c) in known and pair_id(x, c) not in stored
    ]
    with scores_path.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in carry)
    inputs = [
        {"id": pair_id(x, c), "query": x, "document": frozen.chunks[c].text} for x, c in needed
    ]
    (run / "inputs.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inputs), "utf-8"
    )
    adapter = (
        Path(json.loads((it4 / "f-training.json").read_text("utf-8"))["out_dir"])
        / "adapter-step00600.safetensors"
    )
    if not MLX_PYTHON.exists():
        raise SystemExit(f"{MLX_PYTHON} is missing: the MLX environment of D-040 is not set up")
    command = [str(MLX_PYTHON), "scripts/mlx_rerank.py", "--recipe", str(RECIPE)]
    command += [
        "--inputs",
        str(run / "inputs.jsonl"),
        "--out",
        str(scores_path),
        "--adapter",
        str(adapter),
    ]
    subprocess.run(command, check=True, env=os.environ | {"HF_HOME": str(em.hf_home())})
    score = {
        json.loads(x)["id"]: json.loads(x)["logit_diff"]
        for x in scores_path.read_text("utf-8").splitlines()
    }

    # 3. each requirement's list reordered, and the rule of the moment builds the ten
    hits: dict[str, list[tuple[str, ...]]] = {}
    norm: dict[str, list[dict[str, float]]] = {}
    for qid, xs in requirements.items():
        hits[qid], norm[qid] = [], []
        for x in xs:
            cand = full[x][:depth]
            raw = {c: score[pair_id(x, c)] for c in cand}
            order = tuple(rq.reorder(cand, raw))
            hits[qid].append(order)
            norm[qid].append(normalized(order, raw))
    ranked = []
    for qid in question:
        if rule == "scored":
            order = dc.assemble_scored(e5_of[qid].chunk_ids, hits[qid], norm[qid]).ranking
        else:
            order = dc.assemble(e5_of[qid].chunk_ids, hits[qid]).ranking
        ranked.append(
            rt.Ranked(qid, order, tuple(float(len(order) - i) for i in range(len(order))))
        )
    system = SYSTEMS[args.experiment]
    sha = rt.write_run(
        rt.run_dir(system, args.max_tokens, version) / rt.RUN_FILE, ranked, frozen.evaluable
    )

    # 4. the labels enter here, once every ranking exists
    queries = [q for t in frozen.targets for q in t.queries]
    ref_ranked, _ = rt.load_system_run(state["reference"], args.max_tokens, version, args.results)
    diag = {
        n: rr.diagnostics(frozen.dev, frozen.layer, r)
        for n, r in (("e5", e5_ranked), ("reference", ref_ranked), ("variant", ranked))
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
            "rule": rule,
            "decomposers": decomposers,
            "depth": depth,
            "normalization": NORMALIZATION if rule == "scored" else None,
            "reranker": "F's LoRA adapter, step 600",
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
        "reference": state["reference"],
        "configuration": {"rule": rule, "decomposers": decomposers, "depth": depth},
        "pairs_scored": len(needed),
        "pairs_carried_from_f": len(known),
        "columns": cols,
        "by_language_recall10": diag["variant"]["by_language"]["Recall@10"],
        "paired_against_e5": paired("e5"),
        "paired_against_reference": paired("reference"),
        "replaces_reference": replaces,
        "useful_target": useful,
    }
    (out / f"{args.experiment}.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", "utf-8"
    )
    state["done"].append(args.experiment)
    if replaces:
        state.update(reference=system, rule=rule, decomposers=decomposers, depth=depth)
    if useful:
        state["stopped"] = f"{args.experiment} reached the useful target"
    state_path.write_text(json.dumps(state, indent=1) + "\n", "utf-8")
    print(
        json.dumps(
            {"columns": cols, "replaces_reference": replaces, "useful_target": useful}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
