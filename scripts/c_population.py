"""C of retrieval iteration 3, step 1: the expanded candidate population, frozen, and its oracle
audit (config/retrieval/necessity-selection.md, section C).

    PYTHONPATH=. .venv/bin/python scripts/c_population.py --expect-version 0.10.0

Builds every query's population from E5's recorded run, R1's recorded run and provenance (both
checked by SHA-256) and the law's structure (ingest/functional_selection.py), without reading a
label, and freezes it in data/runs/c-functional-layer-<n>-dev-<version>/population.jsonl (not
versioned; its SHA-256 goes into the audit). Then, with the labels, the oracle: the best Recall@10
and full-evidence coverage a selector could reach choosing 10 chunks from the population, for the
whole population and for each deterministic cap of the ladder (the first members in R1's order),
against the gate (Recall@10 >= 0.93 and full-evidence >= 0.85). Writes
data/results/dev-<version>/retrieval-iteration-3/c-eligibility-audit.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import dependencies as dp
from ingest import functional_selection as fs
from ingest import necessity as nc
from ingest import references as xr
from ingest import retrieval as rt

XREF = Path("config/retrieval/xref-expansion.json")
NAME = "c-functional"
LADDER = (0, 128)  # 0: the whole population
GATE = {"Recall@10": 0.93, "Full-evidence@10": 0.85}
FOLLOWED = ("N022", "N024", "N034", "N035")


def population_file(max_tokens: int, version: str) -> Path:
    return rt.run_dir(NAME, max_tokens, version) / "population.jsonl"


def build_all(frozen: rt.FrozenInputs, results: Path, max_tokens: int) -> dict[str, Any]:
    """Every query's population and what it was built from."""
    version = frozen.dev.manifest.benchmark_version
    r1 = json.loads(
        (results / f"dev-{version}" / "retrieval-iteration-2" / "r1-candidates.json").read_text(
            encoding="utf-8"
        )
    )
    r1_name = f"acdr-e5-bridge{r1['decision']['selected_k']}"
    r1_dir = rt.run_dir(r1_name, max_tokens, version)
    r1_lists, _ = rt.read_run(r1_dir / rt.RUN_FILE, r1["configurations"][r1_name]["run_sha256"])
    provenance_bytes = (r1_dir / "provenance.jsonl").read_bytes()
    provenance = {
        row["query_id"]: row
        for row in map(json.loads, provenance_bytes.decode("utf-8").splitlines())
    }
    e5, e5_summary = rt.load_system_run("multilingual-e5-large", max_tokens, version, results)
    acts = xr.ExpansionConfig.model_validate_json(XREF.read_text(encoding="utf-8")).acts
    structure = nc.structure_of(frozen, acts)
    finder = dp.TermFinder(structure.definitions)
    visible = fs.visibility(
        {c: x.context or "" for c, x in frozen.chunks.items()},
        {c: frozen.layer.chunks[c].nodes for c in frozen.chunks},
        structure.node_text,
        structure.article,
    )
    question = {q.query_id: q.text for q in frozen.dev.queries}
    r1_of = {r.query_id: r for r in r1_lists}
    populations = {
        r.query_id: fs.expanded_population(
            r,
            r1_of[r.query_id],
            provenance[r.query_id],
            structure,
            question[r.query_id],
            finder,
            visible,
        )
        for r in e5
    }
    return {
        "populations": populations,
        "e5": e5,
        "e5_run_sha256": e5_summary["identity"]["run_sha256"],
        "r1_name": r1_name,
        "r1_run_sha256": r1["configurations"][r1_name]["run_sha256"],
        "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
    }


def capped(members: list[fs.Member], cap: int) -> list[fs.Member]:
    return list(members) if cap == 0 else list(members[:cap])


def write_population(path: Path, populations: dict[str, list[fs.Member]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for qid, members in populations.items():
        row = {
            "query_id": qid,
            "members": [
                {
                    "chunk": m.chunk,
                    "hop": m.hop,
                    "position": m.position,
                    "routes": [r.__dict__ for r in m.routes],
                    "visible_in": list(m.visible_in),
                }
                for m in members
            ],
        }
        lines.append(json.dumps(row, ensure_ascii=False) + "\n")
    data = "".join(lines).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_population(path: Path, sha256: str) -> dict[str, list[fs.Member]]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"{path}: not the population the audit froze")
    out = {}
    for row in map(json.loads, data.decode("utf-8").splitlines()):
        out[row["query_id"]] = [
            fs.Member(
                m["chunk"],
                m["hop"],
                m["position"],
                tuple(fs.Route(**r) for r in m["routes"]),
                tuple(m["visible_in"]),
            )
            for m in row["members"]
        ]
    return out


def oracle(frozen: rt.FrozenInputs, chosen: dict[str, set[str]]) -> dict[str, Any]:
    """The best a selector could do choosing 10 from each population (labels read here)."""
    targets = frozen.targets
    queries = [q for t in targets for q in t.queries]
    rel = {q.query_id: rt.relevant_chunks(t) for t in targets for q in t.queries}
    dep = rt.dependent_evidence(targets)
    needs = {t.need.need_id: t.need for t in targets}
    recall = {
        q.query_id: min(10, len(rel[q.query_id] & chosen[q.query_id])) / len(rel[q.query_id])
        for q in queries
    }
    full = {
        q.query_id: float(rel[q.query_id] <= chosen[q.query_id] and len(rel[q.query_id]) <= 10)
        for q in queries
    }
    dq = [q for q in queries if q.need_id in dep]
    dependent = {
        q.query_id: len(dep[q.need_id].dependents & chosen[q.query_id])
        / len(dep[q.need_id].dependents)
        for q in dq
    }

    def group(kind: str, label: str) -> list[bm.Query]:
        return [
            q
            for q in queries
            if (
                needs[q.need_id].type == label
                if kind == "type"
                else label in needs[q.need_id].drivers
            )
        ]

    return {
        "Recall@10": rt.macro_by_need(recall, queries),
        "Full-evidence@10": rt.macro_by_need(full, queries),
        "multi_provision": rt.macro_by_need(recall, group("type", "multi_provision")),
        "cross_reference": rt.macro_by_need(recall, group("driver", "cross_reference")),
        "cross_act": rt.macro_by_need(recall, group("driver", "cross_act")),
        "dependent": rt.macro_by_need(dependent, dq),
        "by_language": rt.per_language(recall, queries),
        "followed_needs": {
            n: {
                "Recall@10": [round(recall[q.query_id], 3) for q in queries if q.need_id == n],
                "Full-evidence@10": [full[q.query_id] for q in queries if q.need_id == n],
            }
            for n in FOLLOWED
        },
        "still_ineligible": {
            n: sorted(
                {c for q in queries if q.need_id == n for c in rel[q.query_id] - chosen[q.query_id]}
            )
            for n in sorted({q.need_id for q in queries})
            if any(rel[q.query_id] - chosen[q.query_id] for q in queries if q.need_id == n)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)
    frozen = rt.load_frozen_inputs(args.benchmark, args.expect_version, args.max_tokens)
    version = frozen.dev.manifest.benchmark_version
    built = build_all(frozen, args.results, args.max_tokens)
    populations: dict[str, list[fs.Member]] = built["populations"]
    sha = write_population(population_file(args.max_tokens, version), populations)

    sizes = sorted(len(v) for v in populations.values())
    members = [m for v in populations.values() for m in v]
    audit: dict[str, Any] = {
        "experiment": "C, eligibility audit",
        "benchmark_version": version,
        "e5_run_sha256": built["e5_run_sha256"],
        "candidates": built["r1_name"],
        "candidates_run_sha256": built["r1_run_sha256"],
        "provenance_sha256": built["provenance_sha256"],
        "population_sha256": sha,
        "rules": fs.__doc__.split("Population, per query, built without a label:\n")[1].split(
            "\n\nSelector"
        )[0],
        "restrictions_removed_against_b": [
            "anchors 9 and 10 now count (A and B used anchors 1 to 8 or 1 to n)",
            "E5's ranks 1 to 20 compete as members (B excluded E5's top 10 and kept 6 to 8 fixed)",
            "explicit references are followed for two hops (R1 followed one)",
            "terms of the question and of the supports are resolved to their definitions",
            "no relation to a kept anchor is required, and no position is reserved for E5",
        ],
        "population": {
            "members": len(members),
            "per_query": {
                "median": statistics.median(sizes),
                "p90": sizes[int(0.9 * len(sizes))],
                "max": sizes[-1],
                "min": sizes[0],
            },
            "by_hop": dict(sorted(Counter(m.hop for m in members).items())),
            "by_route_kind": dict(
                sorted(Counter(r.kind for m in members for r in m.routes).items())
            ),
            "visible_in_another_members_context": sum(bool(m.visible_in) for m in members),
        },
        "ladder": {},
    }
    for cap in LADDER:
        chosen = {qid: {m.chunk for m in capped(v, cap)} for qid, v in populations.items()}
        result = oracle(frozen, chosen)
        windows = sum(math.ceil(len(capped(v, cap)) / fs.WINDOW) for v in populations.values())
        gate = {k: round(result[k], 3) >= threshold - 1e-9 for k, threshold in GATE.items()}
        audit["ladder"]["all" if cap == 0 else f"first {cap} in R1's order"] = {
            "windows_of_8": windows,
            "members": sum(len(capped(v, cap)) for v in populations.values()),
            "oracle": result,
            "gate": gate,
            "passes": all(gate.values()),
        }
        print(
            f"{'all' if cap == 0 else cap}: oracle R@10 {result['Recall@10']:.3f} full "
            f"{result['Full-evidence@10']:.3f} multi {result['multi_provision']:.3f} dependent "
            f"{result['dependent']:.3f}; {windows} windows; gate "
            f"{'passes' if all(gate.values()) else 'fails'}"
        )
    out = args.results / f"dev-{version}" / "retrieval-iteration-3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "c-eligibility-audit.json").write_text(
        json.dumps(audit, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"population frozen: {sha[:12]}; {len(members)} members, median {statistics.median(sizes)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
