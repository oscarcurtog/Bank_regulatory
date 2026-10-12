"""Validate, seal, or prompt for a retrieval benchmark directory (D-019).

    PYTHONPATH=. .venv/bin/python scripts/benchmark.py validate data/benchmark/dev
    PYTHONPATH=. .venv/bin/python scripts/benchmark.py seal data/benchmark/dev \\
        --role dev --version 0.2.0
    PYTHONPATH=. .venv/bin/python scripts/benchmark.py prompt data/benchmark/dev N004

validate runs the three boundaries (the seal, the dataset rules, the corpus) and
reports the lexical overlap of every query with its evidence, how the labels were
adjudicated (judges A, B and C), and the definitions graded 2 or 3, each of which
must pass the necessity test of the protocol. It reads the processed layer, so it
is a local command; CI checks the first two boundaries in the test suite.

seal writes data/benchmark/<dir>/manifest.json only if every check passes. It
never fixes anything in passing: a stale qrel is reported with its need and path.

prompt prints the query-variant prompt of one need (data/benchmark/PROTOCOL.md,
step 2). It reads the need statement and nothing else, so it also works for a
draft need that is not sealed yet.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import get_args

from ingest import benchmark as bm
from ingest import processed as pr

DEFAULT_VARIANTS = ["en:formal", "en:natural", "es:natural"]


def summary(benchmark: bm.Benchmark) -> str:
    m = benchmark.manifest
    status = Counter(n.status for n in benchmark.needs)
    answerability = Counter(n.answerability for n in benchmark.needs)
    langs = Counter(q.lang for q in benchmark.queries)
    grades = Counter(r.grade for r in benchmark.qrels)
    return (
        f"{benchmark.directory}: role {m.role}, version {m.benchmark_version}, parser "
        f"{m.parser_version}\n"
        f"  needs {len(benchmark.needs)} {dict(sorted(status.items()))} "
        f"{dict(sorted(answerability.items()))}\n"
        f"  queries {len(benchmark.queries)} {dict(sorted(langs.items()))}\n"
        f"  qrels {len(benchmark.qrels)} by grade {dict(sorted(grades.items()))}, "
        f"ambiguous {sum(r.ambiguous for r in benchmark.qrels)}"
    )


def overlaps(benchmark: bm.Benchmark, docs: dict[str, pr.ProcessedDocument]) -> str:
    lines = ["  lexical overlap with the evidence (share of content words, longest verbatim run):"]
    for need in benchmark.needs:
        if need.status == "draft" or need.answerability == "none":
            continue
        evidence = bm.evidence_texts(benchmark, need.need_id, docs)
        for query in benchmark.queries_of(need.need_id):
            o = bm.lexical_overlap(query.text, evidence)
            flag = "  POSSIBLE COPY" if o.possible_copy else ""
            lines.append(f"    {query.query_id:<12} {o.share:5.2f} {o.longest_run:3}{flag}")
    return "\n".join(lines)


def _c_outcome(record: bm.Adjudication) -> str:
    """What judge C decided, seen from outside: C itself never knew which was A."""
    assert record.c is not None
    if record.c.ambiguous:
        return "ambiguous"
    if record.c.label == record.a.label:
        return "judge A's label"
    if record.c.label == record.b.label:
        return "judge B's label"
    return "another label"


def adjudication(benchmark: bm.Benchmark) -> str:
    records = benchmark.adjudications
    if not records:
        return "  adjudication record: none"
    disagreements = [a for a in records if not a.agree]
    outcome = Counter(_c_outcome(a) for a in records if a.c is not None)
    lines = [
        f"  adjudication: {len(records)} labels judged by A and B, "
        f"{len(records) - len(disagreements)} agreed, {len(disagreements)} went to judge C "
        f"({sum(a.crosses_threshold for a in disagreements)} across the relevance line), "
        f"{sum(a.escalated is not None for a in records)} agreements escalated",
        f"    judge C chose: {dict(sorted(outcome.items()))}",
    ]
    return "\n".join(lines)


def definitions(benchmark: bm.Benchmark) -> str:
    graded = bm.definition_qrels(benchmark.qrels)
    lines = [f"  definitions graded 2 or 3 (necessity test, PROTOCOL.md section 2): {len(graded)}"]
    lines += [f"    {r.need_id} {r.act}:{r.path} grade {r.grade}" for r in graded]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="check the seal, the dataset and the corpus")
    validate.add_argument("directory", type=Path)
    seal = sub.add_parser("seal", help="check everything, then write manifest.json")
    seal.add_argument("directory", type=Path)
    seal.add_argument("--role", choices=get_args(bm.Role), required=True)
    seal.add_argument("--version", required=True, help="benchmark_version, x.y.z")
    prompt = sub.add_parser("prompt", help="print the query-variant prompt of one need")
    prompt.add_argument("directory", type=Path)
    prompt.add_argument("need_id")
    prompt.add_argument("--variant", action="append", help="lang:style, repeatable")
    args = parser.parse_args(argv)

    if args.command == "prompt":
        needs = {n.need_id: n for n in bm.read_jsonl(args.directory / bm.NEEDS_FILE, bm.Need)}
        if args.need_id not in needs:
            raise bm.BenchmarkError(f"{args.directory}: no need {args.need_id}")
        pairs = (v.partition(":") for v in args.variant or DEFAULT_VARIANTS)
        variants = [(lang, style) for lang, _, style in pairs]
        print(bm.render_query_prompt(needs[args.need_id].statement, variants), end="")
        return 0

    docs = bm.load_corpus()
    if args.command == "seal":
        manifest = bm.seal(args.directory, args.version, args.role, docs)
        print(f"sealed {args.directory} at benchmark_version {manifest.benchmark_version}")
    benchmark = bm.load_benchmark(args.directory)
    bm.check_corpus(benchmark, docs)
    print(summary(benchmark))
    print(adjudication(benchmark))
    print(definitions(benchmark))
    print(overlaps(benchmark, docs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
