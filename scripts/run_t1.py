"""Orchestrates T1: validated manifest -> parse -> processed layer -> the four
numbers of the T1 definition of done (project plan).

It downloads nothing. Downloading is a separate step: ingest.download.fetch_all()
writes data/raw/ and write_manifest() records it. This script only reads what is
already on disk, so it needs no network.

Usage (with the EF-1 environment active, "ingest" installed in editable mode):
    .venv/bin/python scripts/run_t1.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Literal

from ingest import download as dl
from ingest import hierarchy as h
from ingest import processed as pr

REPORT_PATH = Path("data/t1_report.json")

# article title class, per family: a signal INDEPENDENT of the eli-subdivision
# divs that hierarchy.py uses, to cross-check the count (number 2 of the T1
# definition of done) without the parser validating itself.
TITLE_CLASS = {"doue": "oj-ti-art", "consolidado": "title-article-norm"}


def independent_article_count(xhtml_path: str, family: str) -> int:
    raw = open(xhtml_path, encoding="utf-8").read()
    cls = TITLE_CLASS[family]
    return len(re.findall(rf'class="[^"]*\b{re.escape(cls)}\b[^"]*"', raw))


def run() -> dict:
    # load_manifest validates every entry before returning anything: if the
    # manifest is wrong, this aborts here and no document gets chosen.
    docs = dl.load_manifest()

    # Payload bound for JSON, heterogeneous values. Any is the honest type:
    # the EF-3 inventory decided that report dicts carry no schema because
    # they have no code consumer.
    per_act: dict[str, dict[str, Any]] = {}
    all_articles = {}  # act -> list[Article]

    for act_name in ("DORA", "MiCA", "PSD2"):
        source = dl.parsed_source_for(act_name, docs)
        family: Literal["doue", "consolidado"] = (
            "consolidado" if source.role == "consolidated" else "doue"
        )

        articles = h.parse(source.local_path, family)
        all_articles[act_name] = articles

        indep = independent_article_count(source.local_path, family)
        n = len(articles)

        # Building the model IS the validation at the write boundary: if
        # anything breaks the contract, no half-written file is produced.
        pr.write_processed(
            pr.ProcessedDocument(
                source_celex=source.celex,
                source_role=source.role,
                source_sha256=source.sha256,
                parser_version=h.PARSER_VERSION,
                family=family,
                articles=articles,
            ),
            act_name,
        )

        n_deep = sum(1 for a in articles if a.max_depth > 3)
        per_act[act_name] = {
            "source_celex": source.celex,
            "source_role": source.role,
            "n_articles_eli_subdivision": n,
            "n_articles_independent_title": indep,
            "counts_match": n == indep,
            "with_more_than_3_levels": n_deep,
            "pct_more_than_3_levels": round(100 * n_deep / n, 1) if n else None,
        }

    # -- random sample for human review (number 3 of the definition of done).
    #    Fixed seed so that the sample is reproducible: unseeded random is not
    #    used because then every run would show different articles and
    #    "reviewed by hand" would stop meaning anything.
    rng = random.Random(20260905)
    sample = []
    for act_name, articles in all_articles.items():
        chosen = rng.sample(articles, k=min(14, len(articles)))
        for a in chosen:
            sample.append(
                {
                    "act": act_name,
                    "article_id": a.article_id,
                    "title": a.title,
                    "max_depth": a.max_depth,
                    "nodes": [nd.path for nd in a.nodes],
                    "text_preview": a.text[:220],
                }
            )

    # -- automatic structural invariant check (it does NOT replace the manual
    #    review: it only rules out obvious mechanical bugs, such as a node
    #    whose path is not a child of any existing one).
    invariant_failures = []
    for act_name, articles in all_articles.items():
        for a in articles:
            paths = {a.path} | {nd.path for nd in a.nodes}
            for nd in a.nodes:
                parent = nd.path.rsplit(".", 1)[0]
                if parent not in paths:
                    invariant_failures.append(
                        f"{act_name}/{a.article_id}: {nd.path} has no parent {parent}"
                    )

    report = {
        "per_act": per_act,
        "total_articles": sum(v["n_articles_eli_subdivision"] for v in per_act.values()),
        "total_more_than_3_levels": sum(v["with_more_than_3_levels"] for v in per_act.values()),
        "invariant_check": {
            "description": "every node must have its parent in the article's own tree",
            "failures": invariant_failures,
        },
        "sample_for_human_review": sample,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["per_act"], indent=2, ensure_ascii=False))
    print()
    print("total_articles:", r["total_articles"])
    print("total_more_than_3_levels:", r["total_more_than_3_levels"])
    print("structural invariant failures:", len(r["invariant_check"]["failures"]))
    print(
        "sample for human review:",
        len(r["sample_for_human_review"]),
        "articles ->",
        REPORT_PATH,
    )
