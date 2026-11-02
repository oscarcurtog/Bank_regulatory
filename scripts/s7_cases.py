"""Materialize the deterministic cases of S7-S8 from the frozen DEV benchmark and rankings.

    PYTHONPATH=. .venv/bin/python scripts/s7_cases.py materialize

It generates no answer and calls no model. It reads the DEV needs, queries and qrels, the chunk
layer and the stored rankings of E5 and E (checked by SHA-256), applies ingest/answer_cases.py and
writes one read-only file, cases.json, whose content depends only on those inputs. Nothing is ever
overwritten. The qrels are read here to build the GOLD_KNOWN context and to find the chunks known
not to be needed; the prompt-development and evaluation needs are fixed before any generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ingest import answer_cases as ac
from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import download as dl
from ingest import evaluation as ev

RUNS = Path("data/runs")
RESULTS = Path("data/results/dev-0.10.0/s7-s8")
RUN_DIRS = {
    "E5": (
        "multilingual-e5-large",
        "b4880d4977fb242757849f4b8a20550f566fbad0fd5eda041505e28de83b9921",
    ),
    "E": (
        "e-d4b-qwen3-reranker-k5",
        "8ee79746f75e817320228e004344fb6eb09d5fe97fdcb815891834197b7ef521",
    ),
}


def read_ranking(system: str) -> dict[str, list[str]]:
    name, sha = RUN_DIRS[system]
    path = RUNS / f"{name}-layer-500-dev-0.10.0" / "run.jsonl"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise SystemExit(f"{path} is not the run the preregistration pins")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    return {r["query_id"]: list(r["chunk_ids"]) for r in rows}


def materialize() -> int:
    dev = bm.load_benchmark(bm.DEV_DIR)
    manifest = dl.load_manifest()
    docs = [ch.load_chunks(act, manifest) for act in bm.ACTS]
    layer = ev.layer_from_documents(docs)
    ev.check_layer(dev.manifest, layer)
    texts = {c.chunk_id: c.text for d in docs for c in d.chunks}
    tokens = {c.chunk_id: c.token_count for d in docs for c in d.chunks}
    layer_sha = hashlib.sha256(
        "\n".join(f"{c}\t{t}" for c, t in texts.items()).encode()
    ).hexdigest()

    infos = [ac.NeedInfo(n.need_id, n.answerability, tuple(n.acts)) for n in dev.needs]
    acts = {i.need_id: i.acts for i in infos}
    targets = {t.need.need_id: t for t in ev.targets(dev, layer)}
    grades = {n: dict(t.chunk_grades) for n, t in targets.items()}
    formal = {
        q.need_id: ac.Question(q.query_id, q.text)
        for q in dev.queries
        if q.lang == "en" and q.style == "formal"
    }
    if len(formal) != len(dev.needs):
        raise SystemExit("every need must have exactly one English formal query")
    e5, e = read_ranking("E5"), read_ranking("E")

    split = ac.make_split(infos)

    def contexts_of(need: str) -> set[str]:
        q = formal[need].query_id
        return set(e5[q][:10]) | set(e[q][:10])

    selection = ac.select_negatives(
        split,
        acts,
        grades,
        contexts_of,
        lambda need: ac.insufficient_context(grades[need], e5[formal[need].query_id]) is not None,
    )
    dev_cases = ac.positive_cases(
        split.prompt_dev,
        formal,
        grades,
        e5,
        e,
        kind=ac.PROMPT_DEV,
        contexts=(ac.GOLD_KNOWN, ac.E5_10),
    )
    positives = ac.positive_cases(split.evaluation, formal, grades, e5, e)
    negatives = ac.negative_cases(split, selection, formal, grades, e5, e)

    def describe(cases: list[ac.Case]) -> list[dict[str, Any]]:
        return [
            {
                **asdict(c),
                "chunk_ids": list(c.chunk_ids),
                "context_tokens": sum(tokens[x] for x in c.chunk_ids),
            }
            for c in cases
        ]

    payload = {
        "seed": ac.SEED,
        "layer_sha256": layer_sha,
        "inputs": {"run_E5": RUN_DIRS["E5"][1], "run_E": RUN_DIRS["E"][1]},
        "split": asdict(split),
        "negatives": {
            "natural": list(split.none),
            "insufficient_context": list(selection.insufficient),
            "mismatched_context": [list(p) for p in selection.mismatched],
        },
        "cases": {
            "prompt_dev": describe(dev_cases),
            "evaluation_positive": describe(positives),
            "evaluation_negative": describe(negatives),
        },
    }
    path = RESULTS / "cases.json"
    data = (json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"{path} exists with other content: results are never overwritten")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o444)
    print(hashlib.sha256(data).hexdigest(), path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["materialize"])
    parser.parse_args(argv)
    return materialize()


if __name__ == "__main__":
    sys.exit(main())
