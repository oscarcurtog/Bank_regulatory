"""C of retrieval iteration 3, step 2: the legal-function set selector over the frozen population
(config/retrieval/necessity-selection.md, section C).

    PYTHONPATH=. .venv/bin/python scripts/c_selection.py --expect-version 0.10.0 --measure
    PYTHONPATH=. caffeinate -i .venv/bin/python scripts/c_selection.py --expect-version 0.10.0

With --measure, a fixed sample of windows (the first window of every tenth query) is generated
and timed, and the projected time of the whole run decides the population by the protocol's rule:
the whole population if it projects to 8 hours or less, otherwise its first 128 members in R1's
order if that does, otherwise C does not run (retrieval-iteration-3/c-timing.json).

Without it, the decided population (checked against the audit's SHA-256) is cut into windows of
8, each window's prompt goes to the MLX worker (scripts/mlx_generate.py, its own environment),
each answer is parsed into judgments, the fixed merge builds each query's evidence set, and the
ranking (the set, then E5's order) is written as a run, measured as every system and paired with
E5. The model's raw answers stay in data/runs (not versioned).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import c_population as cp  # noqa: E402

from ingest import benchmark as bm  # noqa: E402
from ingest import embeddings as em  # noqa: E402
from ingest import functional_selection as fs  # noqa: E402
from ingest import retrieval as rt  # noqa: E402
from ingest import run_report as rr  # noqa: E402

MLX_PYTHON = Path(".cache/mlx-venv/bin/python")
MODEL = Path("config/retrieval/qwen3-4b-instruct-mlx.json")
MAX_NEW_TOKENS = 160
BUDGET_HOURS = 8.0
SAMPLE_EVERY = 10
E5_REFERENCE = {"Recall@10": 0.763, "nDCG@10": 0.528, "multi_provision": 0.481, "dependent": 0.300}


def prompts_for(
    populations: dict[str, list[fs.Member]],
    cap: int,
    question: dict[str, str],
    text: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, list[fs.Member]]]:
    rows, windows = [], {}
    for qid, members in populations.items():
        for i, window in enumerate(fs.windows(cp.capped(members, cap))):
            wid = f"{qid}#w{i:03d}"
            windows[wid] = window
            rows.append({"id": wid, "messages": fs.window_prompt(question[qid], window, text)})
    return rows, windows


def run_worker(prompts: Path, outputs: Path, limit: int = 0) -> None:
    command = [
        str(MLX_PYTHON),
        "scripts/mlx_generate.py",
        "--config",
        str(MODEL),
        "--prompts",
        str(prompts),
        "--out",
        str(outputs),
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
    ]
    if limit:
        command += ["--limit", str(limit)]
    if not MLX_PYTHON.exists():
        raise SystemExit(f"{MLX_PYTHON} is missing: the MLX environment of D-040 is not set up")
    subprocess.run(command, check=True, env=os.environ | {"HF_HOME": str(em.hf_home())})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args(argv)
    frozen = rt.load_frozen_inputs(args.benchmark, args.expect_version, args.max_tokens)
    version = frozen.dev.manifest.benchmark_version
    out_dir = args.results / f"dev-{version}" / "retrieval-iteration-3"
    audit = json.loads((out_dir / "c-eligibility-audit.json").read_text(encoding="utf-8"))
    populations = cp.read_population(
        cp.population_file(args.max_tokens, version), audit["population_sha256"]
    )
    question = {q.query_id: q.text for q in frozen.dev.queries}
    text = {cid: c.text for cid, c in frozen.chunks.items()}
    run_path = rt.run_dir(cp.NAME, args.max_tokens, version)

    if args.measure:
        sample_ids = list(populations)[::SAMPLE_EVERY]
        rows, _ = prompts_for({q: populations[q] for q in sample_ids}, 0, question, text)
        rows = [r for r in rows if r["id"].endswith("#w000")]
        sample_prompts, sample_out = (
            run_path / "sample-prompts.jsonl",
            run_path / "sample-outputs.jsonl",
        )
        write_jsonl(sample_prompts, rows)
        run_worker(sample_prompts, sample_out)
        timed = [json.loads(line) for line in sample_out.read_text(encoding="utf-8").splitlines()]
        per_window = statistics.fmean(r["seconds"] for r in timed)
        projection = {
            name: entry["windows_of_8"] * per_window / 3600
            for name, entry in audit["ladder"].items()
        }
        fits = [
            n for n, h in projection.items() if h <= BUDGET_HOURS and audit["ladder"][n]["passes"]
        ]
        decision = {
            "rule": "the whole population if it projects to 8 hours or less, otherwise its first "
            "128 members in R1's order if that does, otherwise C does not run",
            "seconds_per_window": per_window,
            "projected_hours": projection,
            "population": fits[0] if fits else None,
            "run": bool(fits),
        }
        sample_parsed = [fs.parse_judgments(r["output"], 8) for r in timed]
        timing = {
            "sample": [r["id"] for r in timed],
            "prompt_tokens": [r["prompt_tokens"] for r in timed],
            "new_tokens": [r["new_tokens"] for r in timed],
            "seconds": [r["seconds"] for r in timed],
            "lines_parsed": [len(p) for p in sample_parsed],
            "versions": timed[0]["versions"],
            "decision": decision,
        }
        (out_dir / "c-timing.json").write_text(
            json.dumps(timing, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"{per_window:.1f} s per window; projected {projection}; decision {decision}")
        return 0

    timing = json.loads((out_dir / "c-timing.json").read_text(encoding="utf-8"))
    if not timing["decision"]["run"]:
        raise SystemExit("the timing rule says C does not run")
    name = timing["decision"]["population"]
    cap = 0 if name == "all" else int(name.split()[1])
    rows, windows = prompts_for(populations, cap, question, text)
    prompts_path, outputs_path = run_path / "prompts.jsonl", run_path / "outputs.jsonl"
    prompts_sha = write_jsonl(prompts_path, rows)
    run_worker(prompts_path, outputs_path)
    outputs = {
        r["id"]: r for r in map(json.loads, outputs_path.read_text(encoding="utf-8").splitlines())
    }
    missing = [r["id"] for r in rows if r["id"] not in outputs]
    if missing:
        raise SystemExit(f"{len(missing)} windows have no output")

    # judgments, sets and rankings: no label is read until the evaluation
    judged: dict[str, dict[str, fs.Judgment]] = {qid: {} for qid in populations}
    lines_missing = 0
    for wid, window in windows.items():
        parsed = fs.parse_judgments(outputs[wid]["output"], len(window))
        lines_missing += len(window) - len(parsed)
        qid = wid.split("#")[0]
        for i, m in enumerate(window, start=1):
            if i in parsed:
                judged[qid][m.chunk] = parsed[i]
    e5_ranked, e5_summary = rt.load_system_run(
        "multilingual-e5-large", args.max_tokens, version, args.results
    )
    chosen: dict[str, list[str]] = {}
    ranked = []
    for r in e5_ranked:
        members = cp.capped(populations[r.query_id], cap)
        chosen[r.query_id] = fs.merge(judged[r.query_id], members)
        ranked.append(fs.evidence_ranking(r, chosen[r.query_id]))
    system = f"{cp.NAME}-qwen3-4b"
    sha = rt.write_run(
        rt.run_dir(system, args.max_tokens, version) / rt.RUN_FILE, ranked, frozen.evaluable
    )

    # evaluation, with the labels
    bodies = {cid: c.body for cid, c in frozen.chunks.items()}
    evaluated = rr.evaluate_ranked(frozen.dev, frozen.layer, ranked, bodies)
    diag = rr.diagnostics(frozen.dev, frozen.layer, ranked)
    e5_diag = rr.diagnostics(frozen.dev, frozen.layer, e5_ranked)
    targets = frozen.targets
    queries = [q for t in targets for q in t.queries]
    paired = {
        m: rr.paired_bootstrap(
            {
                q.query_id: diag["per_query"][m][q.query_id] - e5_diag["per_query"][m][q.query_id]
                for q in queries
            },
            queries,
        )
        for m in ("Recall@10", "nDCG@10", "Full-evidence@10")
    }

    def columns_of(d: dict[str, Any]) -> dict[str, float]:
        return {
            "nDCG@10": d["global"]["nDCG@10"],
            "Recall@10": d["global"]["Recall@10"],
            "Recall@20": d["global"]["Recall@20"],
            "MRR@10": d["global"]["MRR@10"],
            "Recall@T=1000": d["global"]["Recall@T=1000"],
            "Recall@T=2000": d["global"]["Recall@T=2000"],
            "Full-evidence@10": d["full_evidence"]["Full-evidence@10"],
            "multi_provision": d["groups"]["multi_provision"]["Recall@10"],
            "cross_reference": d["groups"]["cross_reference"]["Recall@10"],
            "cross_act": d["groups"]["cross_act"]["Recall@10"],
            "dependent": d["dependent_evidence"]["Recall@10"],
        }

    columns = columns_of(diag)
    rounded = {k: round(v, 3) for k, v in columns.items()}
    success = {
        "useful": rounded["Recall@10"] >= 0.85
        and rounded["multi_provision"] >= 0.60
        and rounded["dependent"] >= 0.45
        and rounded["Full-evidence@10"] >= 0.75,
        "strong": rounded["Recall@10"] >= 0.90 and rounded["nDCG@10"] >= 0.65,
    }
    grades = {q.query_id: dict(t.chunk_grades) for t in targets for q in t.queries}
    evaluable = [q.query_id for q in queries]
    relevant_members = [
        (qid, m.chunk)
        for qid in evaluable
        for m in cp.capped(populations[qid], cap)
        if grades[qid].get(m.chunk, 0) >= 2
    ]
    other_members = [
        (qid, m.chunk)
        for qid in evaluable
        for m in cp.capped(populations[qid], cap)
        if grades[qid].get(m.chunk, 0) < 2
    ]

    def necessary(qid: str, c: str) -> bool:
        j = judged[qid].get(c)
        return j is not None and j.necessary and j.function != "merely related"

    selection = {
        "set_size": {
            "mean": statistics.fmean(len(v) for v in chosen.values()),
            "distribution": dict(sorted(Counter(len(v) for v in chosen.values()).items())),
        },
        "functions_judged_necessary": dict(
            Counter(j.function for v in judged.values() for j in v.values() if j.necessary)
        ),
        "lines_missing": lines_missing,
        "judged_necessary_share_of_relevant_members": sum(
            necessary(q, c) for q, c in relevant_members
        )
        / len(relevant_members),
        "judged_necessary_share_of_other_members": sum(necessary(q, c) for q, c in other_members)
        / len(other_members),
        "selected_relevant_share": sum(
            grades[q].get(c, 0) >= 2 for q in evaluable for c in chosen[q]
        )
        / max(1, sum(len(chosen[q]) for q in evaluable)),
    }
    rank_of = {r.query_id: {c: i for i, c in enumerate(r.chunk_ids, 1)} for r in ranked}
    rank_e5 = {r.query_id: {c: i for i, c in enumerate(r.chunk_ids, 1)} for r in e5_ranked}
    followed: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        if target.need.need_id not in cp.FOLLOWED:
            continue
        for q in target.queries:
            qid = q.query_id
            in_pop = {m.chunk for m in cp.capped(populations[qid], cap)}
            followed.setdefault(target.need.need_id, []).append(
                {
                    "query_id": qid,
                    "lang": q.lang,
                    "relevant": [
                        {
                            "chunk": c,
                            "grade": target.chunk_grades[c],
                            "e5_rank": rank_e5[qid][c],
                            "in_population": c in in_pop,
                            "judged": judged[qid][c].__dict__ if c in judged[qid] else None,
                            "selected": c in chosen[qid],
                            "rank": rank_of[qid][c],
                        }
                        for c in sorted(rt.relevant_chunks(target), key=rank_e5[qid].__getitem__)
                    ],
                    "set": [
                        {
                            "chunk": c,
                            "function": judged[qid][c].function,
                            "grade": target.chunk_grades.get(c, 0),
                        }
                        for c in chosen[qid]
                    ],
                    "recall10": {
                        "e5": e5_diag["per_query"]["Recall@10"][qid],
                        "c": diag["per_query"]["Recall@10"][qid],
                    },
                    "full_evidence10": {
                        "e5": e5_diag["per_query"]["Full-evidence@10"][qid],
                        "c": diag["per_query"]["Full-evidence@10"][qid],
                    },
                }
            )
    seconds = [o["seconds"] for o in outputs.values()]
    summary = {
        "system": {"kind": "legal_function_set_selection", "population": name, "window": fs.WINDOW},
        "identity": {
            "base_run_sha256": e5_summary["identity"]["run_sha256"],
            "population_sha256": audit["population_sha256"],
            "prompts_sha256": prompts_sha,
            "run_sha256": sha,
        },
        "benchmark_version": version,
        "parser_version": frozen.dev.manifest.parser_version,
        "layer": em.layer_source(list(frozen.docs)).model_dump(mode="json"),
        "queries_sha256": em.query_source(frozen.dev).queries_sha256,
        "metrics": evaluated["metrics"],
        "diagnostics": {k: v for k, v in diag.items() if k != "per_query"},
        "paired_against_e5": paired,
        "per_need": evaluated["per_need"],
        "per_query": evaluated["per_query"],
        "failures": evaluated["failures"],
    }
    sys_dir = args.results / f"dev-{version}" / f"{system}-layer-{args.max_tokens}"
    sys_dir.mkdir(parents=True, exist_ok=True)
    (sys_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    payload = {
        "experiment": "C",
        "benchmark_version": version,
        "population": name,
        "population_sha256": audit["population_sha256"],
        "prompts_sha256": prompts_sha,
        "run_sha256": sha,
        "prompt": {"system": fs.SYSTEM, "instructions": fs.INSTRUCTIONS, "functions": fs.FUNCTIONS},
        "columns": columns,
        "e5_columns": columns_of(e5_diag),
        "by_language_recall10": diag["by_language"]["Recall@10"],
        "paired_against_e5": paired,
        "success": success,
        "selection": selection,
        "followed_needs": followed,
        "performance": {
            "windows": len(windows),
            "prompt_tokens_total": sum(o["prompt_tokens"] for o in outputs.values()),
            "reused_prefix_tokens_total": sum(o["reused_prefix_tokens"] for o in outputs.values()),
            "new_tokens_total": sum(o["new_tokens"] for o in outputs.values()),
            "seconds_total": sum(seconds),
            "seconds_per_window_median": statistics.median(seconds),
            "versions": next(iter(outputs.values()))["versions"],
        },
    }
    (out_dir / "c-set-selection.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"C: {columns}")
    print(f"paired R@10 {paired['Recall@10']}")
    print(f"success {success}; selection {selection}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
