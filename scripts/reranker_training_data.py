"""Synthetic training data for the requirement reranker, from the corpus alone
(config/retrieval/reranker-finetuning.md).

    PYTHONPATH=. caffeinate -i .venv/bin/python scripts/reranker_training_data.py \\
        --expect-version 0.10.0

Every chunk whose body has at least 20 words gets a prompt; the MLX worker
(scripts/mlx_generate.py, Qwen3-4B-Instruct-2507) writes 2 questions from it; the filters of
ingest/reranker_data.py drop what is ill-formed, leaks a reference, copies, leans on inherited
text or fails the round trip through E5. The kept questions are split by article into TRAIN and
VALIDATION (VALIDATION in two disjoint parts: one to choose the checkpoint, one to report), and
written with their candidates and texts to data/runs/ft-reranker-data-layer-<n>-dev-<version>/.
The statistics go to data/results/dev-<version>/retrieval-iteration-4/ft-data.json. No DEV need,
query, label, requirement or result is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import dense as dn
from ingest import embeddings as em
from ingest import reranker_data as rd
from ingest import retrieval as rt

E5 = Path("config/embeddings/multilingual-e5-large.json")
MLX_PYTHON = Path(".cache/mlx-venv/bin/python")
MODEL = Path("config/retrieval/qwen3-4b-instruct-mlx.json")
NAME = "ft-reranker-data"
MAX_NEW_TOKENS = 160
CHECK_CHUNKS = 10


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", type=Path, default=bm.DEV_DIR)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    parser.add_argument(
        "--check", action="store_true", help="the first 10 eligible chunks only: the form"
    )
    args = parser.parse_args(argv)
    frozen = rt.load_frozen_inputs(args.benchmark, args.expect_version, args.max_tokens)
    version = frozen.dev.manifest.benchmark_version
    chunks = frozen.chunks

    # 1. the questions, written from each eligible chunk's own text
    eligible = [c for c, x in chunks.items() if rd.eligible(x.body)]
    if args.check:
        eligible = eligible[:CHECK_CHUNKS]
    rows = [{"id": f"g#{c}", "messages": rd.generation_prompt(chunks[c].text)} for c in eligible]
    run_path = rt.run_dir(f"{NAME}-check" if args.check else NAME, args.max_tokens, version)
    run_path.mkdir(parents=True, exist_ok=True)
    prompts_sha = write_jsonl(run_path / "prompts.jsonl", rows)
    if not MLX_PYTHON.exists():
        raise SystemExit(f"{MLX_PYTHON} is missing: the MLX environment of D-040 is not set up")
    command = [str(MLX_PYTHON), "scripts/mlx_generate.py", "--config", str(MODEL)]
    command += ["--prompts", str(run_path / "prompts.jsonl"), "--out"]
    command += [str(run_path / "outputs.jsonl"), "--max-new-tokens", str(MAX_NEW_TOKENS)]
    subprocess.run(command, check=True, env=os.environ | {"HF_HOME": str(em.hf_home())})
    outputs = {
        r["id"]: r
        for r in map(json.loads, (run_path / "outputs.jsonl").read_text("utf-8").splitlines())
    }
    if missing := [r["id"] for r in rows if r["id"] not in outputs]:
        raise SystemExit(f"{len(missing)} prompts have no output")

    if args.check:
        for c in eligible:
            for q in rd.questions(outputs[f"g#{c}"]["output"]):
                print(f"{c}: {q}\n    -> {rd.rejection(q, chunks[c].context, chunks[c].body)}")
        return 0

    # 2. the filters before retrieval
    reasons: Counter[str] = Counter()
    kept: list[tuple[str, str, str]] = []  # item id, question, source chunk
    per_chunk: Counter[int] = Counter()
    for c in eligible:
        qs = rd.questions(outputs[f"g#{c}"]["output"])
        per_chunk[len(qs)] += 1
        reasons["fewer than 2 questions read"] += rd.QUESTIONS - len(qs)
        for i, q in enumerate(qs):
            why = rd.rejection(q, chunks[c].context, chunks[c].body)
            if why:
                reasons[why] += 1
            else:
                kept.append((f"{c}#{i}", q, c))

    # 3. E5's first 50 for each question; the round trip; equivalents; the split
    dense = dn.load_dense(E5, frozen, args.max_tokens)
    vectors = dense.encode_queries([q for _, q, _ in kept])
    doc_ids = list(dense.doc_ids)
    equivalent = rd.equivalents({c: x.body for c, x in chunks.items()})
    article = {c: rd.article_key(c, x.article_id) for c, x in chunks.items()}
    items: list[rd.Item] = []
    source_ranks = []
    for (item_id, q, c), v in zip(kept, vectors, strict=True):
        top = rt.rank_by_scores(item_id, doc_ids, dense.doc_vectors @ v).chunk_ids[: rd.DEPTH]
        if c not in top:
            reasons["round trip: source outside E5's first 50"] += 1
            continue
        source_ranks.append(top.index(c) + 1)
        items.append(rd.Item(item_id, q, c, article[c], tuple(top), equivalent[c] | {c}))
    validation_articles = {a for a in set(article.values()) if rd.in_validation(a)}
    excluded = frozenset(c for c, a in article.items() if a in validation_articles)
    train = [it for it in items if it.article not in validation_articles]
    validation = sorted(
        (it for it in items if it.article in validation_articles),
        key=lambda it: hashlib.sha256(f"{rd.SEED}:{it.item_id}".encode()).hexdigest(),
    )
    select = validation[: rd.SELECT_SIZE]
    report = validation[rd.SELECT_SIZE : rd.SELECT_SIZE + rd.REPORT_SIZE]

    # 4. the groups and the files the trainer and the scorer read
    text = {c: x.text for c, x in chunks.items()}
    train_rows: list[dict[str, Any]] = []
    negative_ranks: list[int] = []
    for it in train:
        rng = random.Random(f"{rd.SEED}:{it.item_id}")
        negatives = rd.train_negatives(it, excluded, rng)
        if len(negatives) < rd.TRAIN_NEGATIVES:
            reasons["train: fewer than 3 negatives left"] += 1
            continue
        negative_ranks += [it.candidates.index(n) + 1 for n in negatives]
        train_rows.append(
            {
                "item_id": it.item_id,
                "question": it.question,
                "chunks": [it.positive, *negatives],
                "texts": [text[it.positive], *(text[n] for n in negatives)],
            }
        )
    select_rows: list[dict[str, Any]] = [
        {
            "item_id": it.item_id,
            "question": it.question,
            "chunks": (cands := rd.select_candidates(it)),
            "texts": [text[c] for c in cands],
            "correct": sorted(it.correct),
        }
        for it in select
    ]
    report_rows: list[dict[str, Any]] = [
        {
            "item_id": it.item_id,
            "question": it.question,
            "chunks": list(it.candidates),
            "texts": [text[c] for c in it.candidates],
            "correct": sorted(it.correct),
        }
        for it in report
    ]
    shas = {
        name: write_jsonl(run_path / f"{name}.jsonl", data)
        for name, data in (
            ("train", train_rows),
            ("val_select", select_rows),
            ("val_report", report_rows),
        )
    }
    train_articles = {it.article for it in train}
    leak = {
        "articles_in_both_splits": len(train_articles & validation_articles),
        "validation_chunks_used_as_train_negatives": sum(
            c in excluded for r in train_rows for c in r["chunks"][1:]
        ),
        "validation_positives_used_in_train": len(
            {it.positive for it in validation} & {r["chunks"][0] for r in train_rows}
        ),
    }
    stats = {
        "experiment": "fine-tuning, synthetic data",
        "benchmark_version": version,
        "generator": json.loads(MODEL.read_text(encoding="utf-8")),
        "prompt": {"system": rd.SYSTEM, "task": rd.TASK, "sha256": prompts_sha},
        "rules": {
            "min_body_words": rd.MIN_BODY_WORDS,
            "questions_per_chunk": rd.QUESTIONS,
            "copy_run": rd.COPY_RUN,
            "equivalent_jaccard": rd.EQUIVALENT_JACCARD,
            "shingle": rd.SHINGLE,
            "depth": rd.DEPTH,
            "train_negatives": rd.TRAIN_NEGATIVES,
            "select_candidates": rd.SELECT_CANDIDATES,
            "validation_percent": rd.VALIDATION_PERCENT,
            "seed": rd.SEED,
        },
        "chunks": {"all": len(chunks), "eligible": len(eligible)},
        "generated": len(eligible) * rd.QUESTIONS,
        "questions_read_per_chunk": dict(sorted(per_chunk.items())),
        "rejected": dict(reasons.most_common()),
        "accepted": len(items),
        "source_rank_in_e5": {
            "median": statistics.median(source_ranks),
            "first": sum(r == 1 for r in source_ranks),
            "within_5": sum(r <= 5 for r in source_ranks),
        },
        "chunks_with_an_equivalent": sum(bool(v) for v in equivalent.values()),
        "splits": {
            "train": len(train_rows),
            "val_select": len(select_rows),
            "val_report": len(report_rows),
            "train_articles": len(train_articles),
            "validation_articles": len(validation_articles),
        },
        "train_negative_e5_rank_median": statistics.median(negative_ranks),
        "leakage_audit": leak,
        "fingerprints": {
            "train": rd.fingerprint([r["item_id"] for r in train_rows]),
            "val_select": rd.fingerprint([r["item_id"] for r in select_rows]),
            "val_report": rd.fingerprint([r["item_id"] for r in report_rows]),
        },
        "files_sha256": shas,
        "examples": [{"chunk": it.positive, "question": it.question} for it in train[:12]],
        "generation": {
            "seconds_total": sum(o["seconds"] for o in outputs.values()),
            "versions": next(iter(outputs.values()))["versions"],
        },
    }
    out = args.results / f"dev-{version}" / "retrieval-iteration-4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ft-data.json").write_text(
        json.dumps(stats, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{len(eligible)} chunks, {stats['generated']} questions asked, {len(items)} accepted; "
        f"train {len(train_rows)}, val_select {len(select_rows)}, val_report {len(report_rows)}; "
        f"rejected {dict(reasons.most_common())}; leakage {leak}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
