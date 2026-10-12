"""Validate or seal a retrieval benchmark directory (D-019).

    PYTHONPATH=. .venv/bin/python scripts/benchmark.py validate data/benchmark/dev
    PYTHONPATH=. .venv/bin/python scripts/benchmark.py seal data/benchmark/dev \\
        --role dev --version 0.2.0

validate runs the three boundaries: the seal, the dataset rules, and the corpus.
The last one reads the processed layer, so this is a local command; CI checks the
first two in the test suite.

seal writes data/benchmark/<dir>/manifest.json only if every check passes. It
never fixes anything in passing: a stale qrel is reported with its need and path.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import get_args

from ingest import benchmark as bm


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="check the seal, the dataset and the corpus")
    validate.add_argument("directory", type=Path)
    seal = sub.add_parser("seal", help="check everything, then write manifest.json")
    seal.add_argument("directory", type=Path)
    seal.add_argument("--role", choices=get_args(bm.Role), required=True)
    seal.add_argument("--version", required=True, help="benchmark_version, x.y.z")
    args = parser.parse_args(argv)

    docs = bm.load_corpus()
    if args.command == "seal":
        manifest = bm.seal(args.directory, args.version, args.role, docs)
        print(f"sealed {args.directory} at benchmark_version {manifest.benchmark_version}")
    benchmark = bm.load_benchmark(args.directory)
    bm.check_corpus(benchmark, docs)
    print(summary(benchmark))
    return 0


if __name__ == "__main__":
    sys.exit(main())
