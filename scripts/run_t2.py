"""T2, chunking: the processed layer into data/chunks/{act}.json (not versioned,
regenerable) plus a small versioned profile, data/t2_chunk_profile.json.

max_tokens is a run parameter. The default, 500, is the official run. Any other
value is a measurement: write it somewhere else, so that the official artifacts
stay as they are.

    PYTHONPATH=. .venv/bin/python scripts/run_t2.py
    PYTHONPATH=. .venv/bin/python scripts/run_t2.py --max-tokens 300 \\
        --chunks-dir /tmp/chunks300 --profile /tmp/chunks300/profile.json

It reads the processed layer through load_processed(), so it aborts on a stale
layer, and it checks ownership against it before writing the profile: every
non-empty own_text must be in exactly one body. tiktoken downloads cl100k_base the
first time.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Any

from ingest import chunks as ch
from ingest import download as dl
from ingest import processed as pr

ACTS = ("DORA", "MiCA", "PSD2")
PROFILE_OUT = Path("data/t2_chunk_profile.json")
RANGES = (
    (0, 19, "<20"),
    (20, 49, "20-49"),
    (50, 99, "50-99"),
    (100, 199, "100-199"),
    (200, 299, "200-299"),
    (300, 399, "300-399"),
    (400, 500, "400-500"),
)
# English names for the node kinds in this report, so that the legacy schema codes
# of D-012 do not spread to a new versioned artifact.
KIND_NAMES = {
    "articulo": "article",
    "apartado": "paragraph",
    "punto": "point",
    "subparagraph": "subparagraph",
    "unnumbered_paragraph": "unnumbered_paragraph",
}
TINY_TEXT_TOKENS = 50
TINY_BODY_TOKENS = 20


def pct95(v: list[int]) -> int:
    v = sorted(v)
    return v[int(0.95 * len(v)) - 1] if v else 0


def stats(v: list[int]) -> dict[str, Any]:
    return {"min": min(v), "median": st.median(v), "p95": pct95(v), "max": max(v)}


def token_range(tokens: int) -> str:
    return next((label for lo, hi, label in RANGES if lo <= tokens <= hi), ">500")


def profile(
    processed: dict[str, pr.ProcessedDocument],
    chunked: dict[str, ch.ChunkedDocument],
    max_tokens: int,
) -> dict[str, Any]:
    count = ch.cl100k_counter()
    rows = [(act, c) for act in ACTS for c in chunked[act].chunks]
    nodes = {(act, n.path): n for act in ACTS for a in processed[act].articles for n in a.nodes}
    articles = [(act, a) for act in ACTS for a in processed[act].articles]

    # ownership, checked against the processed layer itself
    sources = {k for k, n in nodes.items() if n.own_text}
    owned = Counter((act, p) for act, c in rows for p in c.source_paths)
    missing = sorted(sources - set(owned))
    duplicated = sorted(k for k, v in owned.items() if v > 1)
    not_a_source = sorted(set(owned) - sources)
    body_mismatch = [
        c.chunk_id
        for act, c in rows
        if c.body != ch.render_body([nodes[(act, p)] for p in c.source_paths])
    ]

    per_article = Counter((act, c.article_id) for act, c in rows)
    whole = sum(
        1 for act, c in rows if c.coverage == "subtree" and c.root_path == c.root_path.split(".")[0]
    )
    tokens = [c.token_count for _, c in rows]
    text_tiny = sorted((c.token_count, c.chunk_id) for _, c in rows)
    exceptions = [(c.chunk_id, c.budget_exception) for _, c in rows if c.budget_exception]
    root_kinds = Counter(KIND_NAMES[nodes[(act, c.root_path)].kind] for act, c in rows)
    return {
        "chunker_version": ch.CHUNKER_VERSION,
        "parser_version": next(iter(chunked.values())).parser_version,
        "tokenizer": ch.TOKENIZER,
        "max_tokens": max_tokens,
        "articles": len(articles),
        "chunks": {"total": len(rows), "by_act": {a: len(chunked[a].chunks) for a in ACTS}},
        "token_count": stats(tokens),
        "token_ranges": {
            label: sum(1 for t in tokens if token_range(t) == label)
            for label in [r[2] for r in RANGES] + [">500"]
        },
        "articles_whole": whole,
        "articles_split": len(articles) - whole,
        "root_level": dict(sorted(Counter(c.root_path.count(".") + 1 for _, c in rows).items())),
        "root_kind": dict(sorted(root_kinds.items())),
        "coverage": dict(sorted(Counter(c.coverage for _, c in rows).items())),
        "tiny": {
            f"text_tokens_below_{TINY_TEXT_TOKENS}": sum(1 for t in tokens if t < TINY_TEXT_TOKENS),
            f"body_tokens_below_{TINY_BODY_TOKENS}": sum(
                1 for _, c in rows if count(c.body) < TINY_BODY_TOKENS
            ),
            "smallest": [{"chunk_id": i, "token_count": t} for t, i in text_tiny[:10]],
        },
        "budget_exceptions": {
            "leaf_over_budget": sum(1 for _, e in exceptions if e == "leaf_over_budget"),
            "own_over_budget": sum(1 for _, e in exceptions if e == "own_over_budget"),
            "chunks": [{"chunk_id": i, "budget_exception": e} for i, e in exceptions],
        },
        "most_chunks": [
            {"act": act, "article_id": art, "chunks": n}
            for (act, art), n in sorted(per_article.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        ],
        "context_tokens": stats([count(c.context) for _, c in rows]),
        "body_tokens": stats([count(c.body) for _, c in rows]),
        "source_nodes": {
            "with_own_text": len(sources),
            "owned_exactly_once": sum(1 for k in sources if owned.get(k) == 1),
            "missing": [f"{a}:{p}" for a, p in missing],
            "duplicated": [f"{a}:{p}" for a, p in duplicated],
            "owned_but_not_a_source": [f"{a}:{p}" for a, p in not_a_source],
            "bodies_not_rebuilt_from_their_sources": body_mismatch,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--chunks-dir", type=Path, default=ch.CHUNKS_DIR)
    parser.add_argument("--profile", type=Path, default=PROFILE_OUT)
    args = parser.parse_args()

    manifest = dl.load_manifest()
    processed = {act: pr.load_processed(act, manifest) for act in ACTS}
    chunked = {act: ch.chunk_document(processed[act], act, args.max_tokens) for act in ACTS}
    report = profile(processed, chunked, args.max_tokens)
    owned = report["source_nodes"]
    if owned["missing"] or owned["duplicated"] or owned["owned_but_not_a_source"]:
        raise SystemExit(f"ownership is broken, nothing written: {owned}")
    if owned["bodies_not_rebuilt_from_their_sources"]:
        raise SystemExit("a body is not the rendering of its sources, nothing written")

    for doc in chunked.values():
        ch.write_chunks(doc, args.chunks_dir)
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    args.profile.write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"max_tokens={args.max_tokens} chunker {ch.CHUNKER_VERSION}, tokenizer {ch.TOKENIZER}")
    print(f"chunks: {report['chunks']}")
    print(f"token_count: {report['token_count']}  ranges: {report['token_ranges']}")
    print(f"articles whole/split: {report['articles_whole']}/{report['articles_split']}")
    print(f"coverage: {report['coverage']}  root_level: {report['root_level']}")
    print(f"budget_exceptions: {report['budget_exceptions']['chunks']}")
    print(
        f"source nodes owned exactly once: {owned['owned_exactly_once']}/{owned['with_own_text']}"
    )
    print(f"written: {args.chunks_dir}/{{act}}.json, {args.profile}")


if __name__ == "__main__":
    main()
