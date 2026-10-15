"""F: train the requirement reranker's LoRA recipe, measure it, and run DEV once
(config/retrieval/reranker-finetuning.md).

    PYTHONPATH=. caffeinate -i .venv/bin/python scripts/reranker_finetune.py \\
        --expect-version 0.10.0 --stage train|component|parity|dev

- train: scripts/mlx_train_reranker.py on the synthetic data (ft-reranker-data); the curve and the
  chosen checkpoint go to retrieval-iteration-4/f-training.json.
- component: VAL-REPORT's questions, E5's 50 candidates each, scored by the base and by the chosen
  adapter (scripts/mlx_rerank.py); P@1, MRR and Recall@3, @5, @10 of the first correct chunk, and
  the MRR difference with a 95% interval over source articles: f-component.json.
- parity: the base MLX scorer on the first 500 pairs of E's record against E's PyTorch scores (no
  label read): f-parity.json.
- dev: once, after the three records above: D-4B's requirements, E5's 50 each, the adapter's
  scores, D's rule unchanged; the run measured as every system and judged by the protocol:
  f-dev.json. The labels are read only after every ranking exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from ingest import reranker_data as rd
from ingest import retrieval as rt
from ingest import run_report as rr

E5 = Path("config/embeddings/multilingual-e5-large.json")
RECIPE = Path("config/retrieval/reranker-finetune.json")
MLX_PYTHON = Path(".cache/mlx-venv/bin/python")
D_SYSTEM = "d-qwen3-4b-k5"
E_SYSTEM = "e-d4b-qwen3-reranker-k5"
SYSTEM = "f-d4b-qwen3-reranker-lora-k5"
PARITY_PAIRS = 500
PARITY_LIMIT = 0.01
TRAINING_GAIN = 0.03


def mlx(script: str, *extra: str) -> None:
    if not MLX_PYTHON.exists():
        raise SystemExit(f"{MLX_PYTHON} is missing: the MLX environment of D-040 is not set up")
    command = [str(MLX_PYTHON), f"scripts/{script}", "--recipe", str(RECIPE), *extra]
    subprocess.run(command, check=True, env=os.environ | {"HF_HOME": str(em.hf_home())})


def score(name: str, inputs: list[dict[str, str]], run: Path, adapter: Path | None) -> dict:
    """The MLX scorer on the inputs (id, query, document): each id's logit difference."""
    run.mkdir(parents=True, exist_ok=True)
    path = run / f"{name}-inputs.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inputs), "utf-8")
    out = run / f"{name}-scores.jsonl"
    mlx(
        "mlx_rerank.py",
        "--inputs",
        str(path),
        "--out",
        str(out),
        *(("--adapter", str(adapter)) if adapter else ()),
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return {r["id"]: r["logit_diff"] for r in rows}


def chosen_adapter(results: Path) -> Path | None:
    record = json.loads((results / "f-training.json").read_text(encoding="utf-8"))
    best = record["best"]["adapter"]
    return None if best is None else Path(record["out_dir"]) / best


def columns_of(d: dict[str, Any]) -> dict[str, float]:
    return {
        "nDCG@10": d["global"]["nDCG@10"],
        "Recall@10": d["global"]["Recall@10"],
        "Recall@20": d["global"]["Recall@20"],
        "MRR@10": d["global"]["MRR@10"],
        "Full-evidence@10": d["full_evidence"]["Full-evidence@10"],
        "multi_provision": d["groups"]["multi_provision"]["Recall@10"],
        "cross_reference": d["groups"]["cross_reference"]["Recall@10"],
        "cross_act": d["groups"]["cross_act"]["Recall@10"],
        "dependent": d["dependent_evidence"]["Recall@10"],
    }


def write(results: Path, name: str, payload: dict[str, Any]) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / name).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def stage_train(args: argparse.Namespace, version: str, results: Path) -> None:
    data = rt.run_dir("ft-reranker-data", args.max_tokens, version)
    out = rt.run_dir("ft-reranker-train", args.max_tokens, version)
    mlx("mlx_train_reranker.py", "--data", str(data), "--out", str(out))
    record = json.loads((out / "training.json").read_text(encoding="utf-8"))
    write(results, "f-training.json", record | {"out_dir": str(out)})


def stage_component(args: argparse.Namespace, version: str, results: Path) -> None:
    data = rt.run_dir("ft-reranker-data", args.max_tokens, version)
    rows = [json.loads(x) for x in (data / "val_report.jsonl").read_text("utf-8").splitlines()]
    inputs = [
        {"id": f"{r['item_id']}|{c}", "query": r["question"], "document": t}
        for r in rows
        for c, t in zip(r["chunks"], r["texts"], strict=True)
    ]
    run = rt.run_dir("ft-reranker-component", args.max_tokens, version)
    adapter = chosen_adapter(results)
    scored = {"zero_shot": score("base", inputs, run, None)}
    scored["fine_tuned"] = (
        score("adapter", inputs, run, adapter) if adapter else dict(scored["zero_shot"])
    )
    correct = {r["item_id"]: frozenset(r["correct"]) for r in rows}
    # the source article, from the source chunk's id ("MiCA:3.1.6#0" is MiCA's article 3)
    cluster = {
        r["item_id"]: ":".join(r["item_id"].split("#")[0].split(":")[:2]).split(".")[0]
        for r in rows
    }
    orders = {"e5": {r["item_id"]: list(r["chunks"]) for r in rows}}
    for name, s in scored.items():
        orders[name] = {
            r["item_id"]: rq.reorder(
                r["chunks"], {c: s[f"{r['item_id']}|{c}"] for c in r["chunks"]}
            )
            for r in rows
        }
    metrics = {name: rd.ranking_metrics(o, correct) for name, o in orders.items()}
    base_rr = rd.reciprocal_ranks(orders["zero_shot"], correct)
    ft_rr = rd.reciprocal_ranks(orders["fine_tuned"], correct)
    paired = rd.cluster_bootstrap({i: ft_rr[i] - base_rr[i] for i in base_rr}, cluster)
    gain = round(metrics["fine_tuned"]["MRR"], 3) - round(metrics["zero_shot"]["MRR"], 3)
    success = gain >= TRAINING_GAIN - 1e-9 and paired["ci95_low"] > 0
    write(
        results,
        "f-component.json",
        {
            "experiment": "F, component on VAL-REPORT",
            "items": len(rows),
            "candidates_per_item": rd.DEPTH,
            "adapter": None if adapter is None else adapter.name,
            "metrics": metrics,
            "mrr_paired_over_articles": paired,
            "mrr_gain": round(gain, 3),
            "rule": f"fine-tuned MRR +{TRAINING_GAIN} with the interval above zero",
            "training_success": success,
        },
    )
    print(json.dumps(metrics, indent=1), f"\nMRR paired {paired}; training success {success}")


def stage_parity(args: argparse.Namespace, frozen: rt.FrozenInputs, results: Path) -> None:
    version = frozen.dev.manifest.benchmark_version
    e_scores = rt.run_dir("e-qwen3-reranker-0.6b", args.max_tokens, version) / "scores.jsonl"
    rows = [json.loads(x) for x in e_scores.read_text("utf-8").splitlines()][:PARITY_PAIRS]
    inputs = [
        {"id": str(i), "query": r["requirement"], "document": frozen.chunks[r["chunk"]].text}
        for i, r in enumerate(rows)
    ]
    mlx_scores = score(
        "parity", inputs, rt.run_dir("ft-reranker-parity", args.max_tokens, version), None
    )
    diffs = [abs(mlx_scores[str(i)] - r["logit_diff"]) for i, r in enumerate(rows)]
    by_req: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_req.setdefault(r["requirement"], []).append(i)
    full = {q: ix for q, ix in by_req.items() if len(ix) == rq.DEPTH}
    same = sum(
        sorted(ix, key=lambda i: -mlx_scores[str(i)])
        == sorted(ix, key=lambda i: -rows[i]["logit_diff"])
        for ix in full.values()
    )
    payload = {
        "experiment": "F, parity of the MLX scorer with E's PyTorch scores",
        "pairs": len(rows),
        "max_abs_difference": max(diffs),
        "median_abs_difference": sorted(diffs)[len(diffs) // 2],
        "requirements_with_all_50": len(full),
        "requirements_ordered_identically": same,
        "limit": PARITY_LIMIT,
        "passes": max(diffs) <= PARITY_LIMIT,
    }
    write(results, "f-parity.json", payload)
    print(payload)


def stage_dev(args: argparse.Namespace, frozen: rt.FrozenInputs, results: Path) -> None:
    version = frozen.dev.manifest.benchmark_version
    component = json.loads((results / "f-component.json").read_text("utf-8"))
    parity = json.loads((results / "f-parity.json").read_text("utf-8"))
    if not parity["passes"]:
        raise SystemExit("parity failed: the protocol asks for a zero-shot MLX DEV baseline first")
    adapter = chosen_adapter(results)

    # 1. D-4B as it ran, rebuilt exactly; then the adapter's scores on the same 50 per requirement
    dense = dn.load_dense(E5, frozen, args.max_tokens)
    e5_ranked, e5_summary = rt.load_system_run(
        "multilingual-e5-large", args.max_tokens, version, args.results
    )
    if dense.rank() != e5_ranked:
        raise SystemExit("the cached E5 vectors do not reproduce E5's recorded run")
    e5_of = {r.query_id: r for r in e5_ranked}
    d_run = rt.run_dir(D_SYSTEM, args.max_tokens, version)
    d_record = json.loads((results / "d-qwen3-4b.json").read_text("utf-8"))
    if (
        hashlib.sha256((d_run / "prompts.jsonl").read_bytes()).hexdigest()
        != d_record["prompt"]["sha256"]
    ):
        raise SystemExit("D-4B's prompts are not the ones its record names")
    answers = {
        r["id"].split("#")[1]: dc.parse(r["output"])
        for r in map(json.loads, (d_run / "outputs.jsonl").read_text("utf-8").splitlines())
    }
    texts = sorted({x for p in answers.values() if p.readable for x in p.items})
    vectors = dict(zip(texts, dense.encode_queries(texts), strict=True))
    doc_ids = list(dense.doc_ids)
    hits = {
        qid: [
            rt.rank_by_scores(qid, doc_ids, dense.doc_vectors @ vectors[x]).chunk_ids[: rq.DEPTH]
            for x in (p.items if p.readable else ())
        ]
        for qid, p in answers.items()
    }
    d_ranked, _ = rt.load_system_run(D_SYSTEM, args.max_tokens, version, args.results)
    for r in d_ranked:
        if dc.assemble(e5_of[r.query_id].chunk_ids, hits[r.query_id]).ranking != r.chunk_ids:
            raise SystemExit(f"{r.query_id}: D-4B's ranking is not rebuilt from its requirements")
    pairs = sorted(
        {
            (x, c)
            for qid, p in answers.items()
            for x, h in zip(p.items if p.readable else (), hits[qid], strict=True)
            for c in h
        }
    )
    key = {pair: str(i) for i, pair in enumerate(pairs)}
    inputs = [{"id": key[(x, c)], "query": x, "document": frozen.chunks[c].text} for x, c in pairs]
    run = rt.run_dir("ft-reranker-dev", args.max_tokens, version)
    if adapter is None:
        raise SystemExit("the chosen checkpoint is the base model: F is E, nothing to run")
    s = score("adapter", inputs, run, adapter)
    hits_ft = {
        qid: [
            tuple(rq.reorder(h, {c: s[key[(x, c)]] for c in h}))
            for x, h in zip(p.items if p.readable else (), hits[qid], strict=True)
        ]
        for qid, p in answers.items()
    }
    question = [q.query_id for q in frozen.dev.queries]
    ranked = []
    for qid in question:
        order = dc.assemble(e5_of[qid].chunk_ids, hits_ft[qid]).ranking
        ranked.append(
            rt.Ranked(qid, order, tuple(float(len(order) - i) for i in range(len(order))))
        )
    sha = rt.write_run(
        rt.run_dir(SYSTEM, args.max_tokens, version) / rt.RUN_FILE, ranked, frozen.evaluable
    )

    # 2. the labels enter here, once every ranking exists
    targets = frozen.targets
    queries = [q for t in targets for q in t.queries]
    rel = {q.query_id: rt.relevant_chunks(t) for t in targets for q in t.queries}
    e_ranked, _ = rt.load_system_run(E_SYSTEM, args.max_tokens, version, args.results)
    runs = {"e5": e5_ranked, "d": d_ranked, "e": e_ranked, "f": ranked}
    diag = {n: rr.diagnostics(frozen.dev, frozen.layer, run) for n, run in runs.items()}
    cols = {n: columns_of(d) for n, d in diag.items()}

    def paired(a: str, b: str) -> dict[str, Any]:
        return {
            m: rr.paired_bootstrap(
                {
                    q.query_id: diag[a]["per_query"][m][q.query_id]
                    - diag[b]["per_query"][m][q.query_id]
                    for q in queries
                },
                queries,
            )
            for m in ("Recall@10", "nDCG@10", "Full-evidence@10")
        }

    e_record = json.loads((results / "e-requirement-reranking.json").read_text("utf-8"))
    component_dev = {
        "e_zero_shot": e_record["requirement_ranking"]["reranked"],
        "f": {f"Recall@{k}": rq.requirement_recall(hits_ft, rel, queries, k)[0] for k in rq.CUTOFFS}
        | {"MRR": rq.requirement_mrr(hits_ft, rel, queries)},
    }
    f, e = ({k: round(v, 3) for k, v in cols[n].items()} for n in ("f", "e"))
    signal = {
        "Recall@10 +0.02 or full-evidence +0.03": f["Recall@10"] - e["Recall@10"] >= 0.02 - 1e-9
        or f["Full-evidence@10"] - e["Full-evidence@10"] >= 0.03 - 1e-9,
        "nDCG@10 not below E's by more than 0.01": f["nDCG@10"] >= e["nDCG@10"] - 0.01 - 1e-9,
    }
    useful = (
        f["Recall@10"] >= 0.85
        and f["multi_provision"] >= 0.60
        and f["dependent"] >= 0.45
        and f["Full-evidence@10"] >= 0.75
    )
    success = component["training_success"] and all(signal.values())
    evaluated = rr.evaluate_ranked(
        frozen.dev, frozen.layer, ranked, {c: x.body for c, x in frozen.chunks.items()}
    )
    summary = {
        "system": {
            "kind": "decomposition_requirements_finetuned_reranker_merge",
            "rule": "D's rule on each requirement's first 50 E5 chunks, reordered by the LoRA",
            "recipe": json.loads(RECIPE.read_text("utf-8")),
            "adapter": adapter.name,
            "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        },
        "identity": {"base_run_sha256": e5_summary["identity"]["run_sha256"], "run_sha256": sha},
        "benchmark_version": version,
        "parser_version": frozen.dev.manifest.parser_version,
        "layer": em.layer_source(list(frozen.docs)).model_dump(mode="json"),
        "queries_sha256": em.query_source(frozen.dev).queries_sha256,
        "metrics": evaluated["metrics"],
        "diagnostics": {k: v for k, v in diag["f"].items() if k != "per_query"},
        "paired_against_e5": paired("f", "e5"),
        "per_need": evaluated["per_need"],
        "per_query": evaluated["per_query"],
        "failures": evaluated["failures"],
    }
    sys_dir = args.results / f"dev-{version}" / f"{SYSTEM}-layer-{args.max_tokens}"
    sys_dir.mkdir(parents=True, exist_ok=True)
    (sys_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", "utf-8"
    )
    payload = {
        "experiment": "F, DEV run, once",
        "system_name": SYSTEM,
        "run_sha256": sha,
        "pairs_scored": len(inputs),
        "columns": cols,
        "by_language_recall10": diag["f"]["by_language"]["Recall@10"],
        "paired_against_e": paired("f", "e"),
        "paired_against_e5": paired("f", "e5"),
        "requirement_ranking_on_dev": component_dev,
        "system_signal_against_e": signal,
        "training_success": component["training_success"],
        "useful_target": useful,
        "domain_adapted_reranker_success": success,
    }
    write(results, "f-dev.json", payload)
    print(
        json.dumps(
            {k: payload[k] for k in ("columns", "paired_against_e", "system_signal_against_e")},
            indent=1,
        )
    )
    print(f"DOMAIN-ADAPTED RERANKER SUCCESS: {'YES' if success else 'NO'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    parser.add_argument("--stage", choices=("train", "component", "parity", "dev"), required=True)
    args = parser.parse_args(argv)
    frozen = rt.load_frozen_inputs(args.benchmark, args.expect_version, args.max_tokens)
    version = frozen.dev.manifest.benchmark_version
    results = args.results / f"dev-{version}" / "retrieval-iteration-4"
    if args.stage == "train":
        stage_train(args, version, results)
    elif args.stage == "component":
        stage_component(args, version, results)
    elif args.stage == "parity":
        stage_parity(args, frozen, results)
    else:
        stage_dev(args, frozen, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
