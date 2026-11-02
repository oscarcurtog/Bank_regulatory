"""Stage B of the S5-S6 retrieval selection: the label-free candidate runs and the single read.

The private document of stage B is pinned by its SHA-256 in
config/retrieval/s5-s6-preregistration-b.json, after stage A and amendment A.1. Subcommands:

    b0     (no label) build the three candidate runs from the frozen historical rankings: RRF,
           and the reranker at N=20 and N=50 from the scores that S5.3 saved. The files are
           written once, read-only, and their SHA-256 is printed. Nothing is ever overwritten.
    read   the one labelled read of the six systems, exactly as preregistered. It refuses to
           run unless the preregistration is pinned and public, the code is committed and
           pushed, the tree is clean and every input hash matches. It shows no figure until the
           file with the six vectors is written and hashed; if that file exists, the looks are
           spent and it refuses to run again (--resume-analysis only rebuilds the analysis from
           the same vectors, which it checks byte by byte).

b0 reads no qrel. read reads the DEV qrels once, through the project's own loader and evaluator.
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
from ingest import chunks as ch
from ingest import cost_protocol as cp
from ingest import download as dl
from ingest import evaluation as ev
from ingest import requirement_rerank as rq
from ingest import selection_read as rd
from ingest import selection_rules as sr

PIN_A = Path("config/retrieval/s5-s6-preregistration-a.json")
PIN_A1 = Path("config/retrieval/s5-s6-preregistration-a1.json")
PIN_B = Path("config/retrieval/s5-s6-preregistration-b.json")
DOC_A = Path("docs/research/s5-2b-a-protocolo-medicion-v1.md")
DOC_A1 = Path("docs/research/s5-2b-a1-enmienda-energia-v1.md")
DOC_B = Path("docs/research/s5-2b-b-preregistro-v1.md")
CODE = ("scripts/s5_selection.py", "ingest/selection_rules.py", "ingest/selection_read.py")
RUNS = Path("data/runs")
RESULTS = Path("data/results/dev-0.10.0/s5-selection")
COST = Path("data/results/dev-0.10.0/s5-cost-m1")
SUFFIX = "layer-500-dev-0.10.0"
STORED = {  # system -> run directory prefix
    "E5": "multilingual-e5-large",
    "BM25": "bm25",
    "D-4B": "d-qwen3-4b-k5",
    "E": "e-d4b-qwen3-reranker-k5",
}
CANDIDATE_RUNS = {"RRF": "s5-rrf-k60-d50", "N20": "s5-e5-rerank-n20", "N50": "s5-e5-rerank-n50"}
COST_SYSTEM = {"E5": "S1", "RRF": "S3", "N20": "S6a", "N50": "S6b", "D-4B": "S4", "E": "S5"}
S6B_SCORES = COST / "S6b" / "scores-S6b-pass1.jsonl"
# the frozen inputs of B0 (the same files and hashes as the preregistration lists)
RUN_SHA256 = {
    "E5": "b4880d4977fb242757849f4b8a20550f566fbad0fd5eda041505e28de83b9921",
    "BM25": "49b9d78fd41c12205e07a878ffeb2260d1110abd5ad99464b2d5f0e282c7aa76",
    "D-4B": "ccdf3e111731c78c19e2bcd616e6c46896527c0cdf8530547df88d9a3ee15317",
    "E": "8ee79746f75e817320228e004344fb6eb09d5fe97fdcb815891834197b7ef521",
}
S6B_SCORES_SHA256 = "5812d94de48c3e0fb02b82915866d7dc6186389d341d90cd5fc4c25291f4a38a"


def out_of(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def clean_tree() -> bool:
    return not out_of(["git", "status", "--porcelain", "--untracked-files=no"]).strip()


def require_tracked(names: list[str]) -> None:
    for name in names:
        done = subprocess.run(
            ["git", "ls-files", "--error-unmatch", name], capture_output=True, check=False
        )
        if done.returncode != 0:
            raise SystemExit(f"{name} is not committed: commit it first")


def checked_bytes(path: Path, sha: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise SystemExit(f"{path} is not the file the preregistration pins")
    return data


def run_path(name: str) -> Path:
    return RUNS / f"{name}-{SUFFIX}" / "run.jsonl"


def read_rows(path: Path, sha: str) -> list[dict[str, Any]]:
    data = checked_bytes(path, sha)
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line]


def write_once(path: Path, data: bytes) -> None:
    """Written once and read-only. The same bytes again are accepted; anything else is refused."""
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"{path} exists with other content: results are never overwritten")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    temporary.chmod(0o444)
    os.replace(temporary, path)


def dump(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# B0: the three label-free candidate runs
# ---------------------------------------------------------------------------


def rrf_scores(first: list[str], second: list[str]) -> list[tuple[str, float]]:
    """The fused chunks with their score, in the order of cost_protocol.rrf_fuse (k = 60, the
    first fifty of each list, equal weights), recomputed here so the run can carry the score."""
    score: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in (first[: cp.RRF_DEPTH], second[: cp.RRF_DEPTH]):
        for position, chunk in enumerate(ranking, start=1):
            score[chunk] = score.get(chunk, 0.0) + 1.0 / (cp.RRF_K + position)
            best[chunk] = min(best.get(chunk, position), position)
    ordered = sorted(score, key=lambda c: (-score[c], best[c], c))[: cp.TOP]
    return [(c, score[c]) for c in ordered]


def b0() -> int:
    require_tracked(list(CODE))
    if not clean_tree():
        raise SystemExit("tracked files changed since the commit: commit the code first")
    e5 = read_rows(run_path(STORED["E5"]), RUN_SHA256["E5"])
    bm25 = {
        r["query_id"]: r["chunk_ids"]
        for r in read_rows(run_path(STORED["BM25"]), RUN_SHA256["BM25"])
    }
    scores: dict[str, dict[str, float]] = {}
    for line in checked_bytes(S6B_SCORES, S6B_SCORES_SHA256).decode("utf-8").splitlines():
        row = json.loads(line)
        scores.setdefault(row["query_id"], {})[row["chunk_id"]] = row["logit_diff"]
    runs: dict[str, list[dict[str, Any]]] = {name: [] for name in CANDIDATE_RUNS}
    for row in e5:
        query_id, ranking, evaluated = row["query_id"], row["chunk_ids"], row["evaluated"]
        fused = rrf_scores(ranking, bm25[query_id])
        if [c for c, _ in fused] != cp.rrf_fuse(ranking, bm25[query_id]):
            raise SystemExit(f"{query_id}: the recomputed RRF differs from rrf_fuse")
        runs["RRF"].append(
            {
                "query_id": query_id,
                "evaluated": evaluated,
                "chunk_ids": [c for c, _ in fused],
                "scores": [s for _, s in fused],
            }
        )
        for name, depth in (("N20", 20), ("N50", 50)):
            candidates = ranking[:depth]
            missing = [c for c in candidates if c not in scores.get(query_id, {})]
            if missing:
                raise SystemExit(f"{query_id}: {len(missing)} candidates without a saved score")
            by_chunk = {c: scores[query_id][c] for c in candidates}
            top = rq.reorder(candidates, by_chunk)[: cp.TOP]
            runs[name].append(
                {
                    "query_id": query_id,
                    "evaluated": evaluated,
                    "chunk_ids": top,
                    "scores": [by_chunk[c] for c in top],
                }
            )
    record: dict[str, Any] = {
        "inputs": {
            "e5_run": RUN_SHA256["E5"],
            "bm25_run": RUN_SHA256["BM25"],
            "s6b_scores": S6B_SCORES_SHA256,
        },
        "commit": out_of(["git", "rev-parse", "HEAD"]).strip(),
        "outputs": {},
    }
    for name, rows in runs.items():
        data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        path = run_path(CANDIDATE_RUNS[name])
        write_once(path, data.encode("utf-8"))
        record["outputs"][name] = {
            "path": str(path),
            "sha256": cp.sha256_file(path),
            "queries": len(rows),
        }
    if not (
        RESULTS / "b0.json"
    ).exists():  # the first record is the provenance; a re-run only verifies
        write_once(RESULTS / "b0.json", dump(record))
    for name, out in record["outputs"].items():
        print(f"{name}  {out['sha256']}  {out['path']}  ({out['queries']} queries)")
    return 0


# ---------------------------------------------------------------------------
# the read
# ---------------------------------------------------------------------------


def guard() -> dict[str, Any]:
    """What must hold before a label is read."""
    require_tracked([*CODE, str(PIN_A), str(PIN_A1), str(PIN_B)])
    if not clean_tree():
        raise SystemExit("tracked files changed since the commit: commit the code first")
    ahead = out_of(["git", "rev-list", "--count", "@{u}..HEAD"]).strip()
    if ahead != "0":
        raise SystemExit(f"the commit is not public yet ({ahead} commits ahead of origin)")
    pin_a, pin_a1, pin_b = (
        json.loads(p.read_text(encoding="utf-8")) for p in (PIN_A, PIN_A1, PIN_B)
    )
    for doc, pin, label in ((DOC_A, pin_a, "A"), (DOC_A1, pin_a1, "A.1"), (DOC_B, pin_b, "B")):
        if cp.sha256_file(doc) != pin["document_sha256"]:
            raise SystemExit(f"stage {label}: {doc} does not hash to its public pin")
    if pin_a1["amends_document_sha256"] != pin_a["document_sha256"]:
        raise SystemExit("amendment A.1 does not amend the sealed stage A")
    text = DOC_B.read_text(encoding="utf-8")
    inputs: dict[str, str] = pin_b["inputs"]
    absent = [k for k, v in inputs.items() if v not in text]
    if absent:
        raise SystemExit(f"the preregistration text does not hold the pinned hashes of {absent}")
    return {
        "preregistration_b_sha256": pin_b["document_sha256"],
        "stage_a_sha256": pin_a["document_sha256"],
        "amendment_a1_sha256": pin_a1["document_sha256"],
        "commit": out_of(["git", "rev-parse", "HEAD"]).strip(),
        "inputs": inputs,
    }


def load_lists(inputs: dict[str, str]) -> sr.Lists:
    def top10(path: Path, sha: str) -> dict[str, tuple[str, ...]]:
        return {r["query_id"]: tuple(r["chunk_ids"][: sr.TOP]) for r in read_rows(path, sha)}

    lists: dict[str, dict[str, tuple[str, ...]]] = {}
    for system in ("E5", "D-4B", "E"):
        lists[system] = top10(run_path(STORED[system]), inputs[f"run_{system}"])
    for system, name in CANDIDATE_RUNS.items():
        lists[system] = top10(run_path(name), inputs[f"run_{system}"])
    return lists


def load_costs(inputs: dict[str, str]) -> dict[str, sr.CostRow]:
    checked_bytes(COST / "cost-summary.json", inputs["cost_summary"])
    manifest = json.loads(checked_bytes(COST / "manifest.json", inputs["cost_manifest"]))
    costs: dict[str, sr.CostRow] = {}
    for system, folder in COST_SYSTEM.items():
        files: dict[str, Any] = {}
        for name in ("summary.json", "main-memory.json"):
            relative = f"{folder}/{name}"
            if cp.sha256_file(COST / relative) != manifest[relative]:
                raise SystemExit(f"{relative} differs from the S5.3 manifest")
            files[name] = json.loads((COST / relative).read_text(encoding="utf-8"))
        worker = files["summary.json"].get("worker_memory") or {}
        memory = files["main-memory.json"]["peak_memory_footprint"] + worker.get(
            "peak_memory_footprint", 0
        )
        complexity, portability = sr.STRUCTURE[system]
        costs[system] = sr.CostRow(
            files["summary.json"]["p95_ns"] / 1e9, memory / 2**30, complexity, portability
        )
    return costs


def read(resume_analysis: bool) -> int:
    seal = guard()
    inputs = seal["inputs"]
    vectors_path = RESULTS / "vectors.json"
    if vectors_path.exists() != resume_analysis:
        raise SystemExit(
            "the six looks are already spent: vectors.json exists (use --resume-analysis to "
            "rebuild the analysis from it)"
            if vectors_path.exists()
            else "--resume-analysis needs the vectors.json of the read"
        )
    for name, path in (
        ("queries", Path("data/benchmark/dev/queries.jsonl")),
        ("dev_manifest", Path("data/benchmark/dev/manifest.json")),
    ):
        checked_bytes(path, inputs[name])
    lists = load_lists(inputs)
    costs = load_costs(inputs)

    # ---- the labelled read: from here on the DEV qrels are open -------------------------------
    dev = bm.load_benchmark(bm.DEV_DIR)
    manifest = dl.load_manifest()
    docs = [ch.load_chunks(act, manifest) for act in bm.ACTS]
    layer = ev.layer_from_documents(docs)
    ev.check_layer(dev.manifest, layer)
    needs = rd.needs_from_targets(ev.targets(dev, layer))
    if len(needs) != 27 or sum(len(n.query_ids) for n in needs) != 81:
        raise SystemExit(f"{len(needs)} needs: the preregistration says 27 needs, 81 queries")
    per_query, per_need = rd.all_vectors(needs, lists)
    vectors = dump(
        {
            "preregistration_b_sha256": seal["preregistration_b_sha256"],
            "commit": seal["commit"],
            "measures": list(rd.METRICS),
            "per_query": per_query,
            "per_need": per_need,
        }
    )
    if resume_analysis:
        if vectors_path.read_bytes() != vectors:
            raise SystemExit("the vectors rebuilt now differ from the stored ones: integrity stop")
    else:
        write_once(vectors_path, vectors)
    vectors_sha = cp.sha256_file(vectors_path)

    analysis = sr.analyse(
        needs,
        lists,
        rd.metric_of(per_need, "Recall@10"),
        rd.metric_of(per_need, "nDCG@10"),
        costs,
    )
    pairs = list(analysis.comparisons)
    coverage = rd.coverage_figures(needs, lists, pairs)
    summary = rd.system_summary(per_need)
    header = {
        "preregistration B": seal["preregistration_b_sha256"],
        "stage A": seal["stage_a_sha256"],
        "amendment A.1": seal["amendment_a1_sha256"],
        "code commit": seal["commit"],
        "vectors sha256": vectors_sha,
        **{k: v for k, v in sorted(inputs.items())},
    }
    report = rd.render_report(analysis, summary, coverage, header, looks=len(sr.SYSTEMS))
    payload = {
        "header": header,
        "looks_consumed": len(sr.SYSTEMS),
        "systems": summary,
        "coverage_figures": coverage,
        "costs": {s: vars(c) for s, c in costs.items()},
        **rd.analysis_to_dict(analysis),
    }
    write_once(RESULTS / "analysis.json", dump(payload))
    write_once(RESULTS / "report.md", report.encode("utf-8"))
    files = {
        name: cp.sha256_file(RESULTS / name)
        for name in ("vectors.json", "analysis.json", "report.md")
    }
    write_once(RESULTS / "manifest.json", dump(files))
    for name, digest in files.items():
        print(f"{digest}  {RESULTS / name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("b0", help="build the three label-free candidate runs")
    reader = sub.add_parser("read", help="the one labelled read of the six systems")
    reader.add_argument("--resume-analysis", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "b0":
        return b0()
    return read(args.resume_analysis)


if __name__ == "__main__":
    sys.exit(main())
