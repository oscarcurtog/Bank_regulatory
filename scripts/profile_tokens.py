"""Token profile per article, with two tokenizers from different families.

It serves the first T2 decision (which tokenizer to use as a provisional
yardstick while the embedding model is still unchosen, see the design document,
stage 3). It measures the WHOLE corpus, not a sample: tokenizing 7 MB takes
seconds, and there is no reason to sample what can be measured in full.

Two yardsticks, one per family:
  - cl100k_base (tiktoken): OpenAI's BPE, the "neutral reference".
  - bert-base-uncased (tokenizers): WordPiece, the vocabulary most open
    embedding models derive from. It is the family's REPRESENTATIVE, not a
    model choice.

This is the script that got burnt by the stale layer: M-005 was measured on
text written by a parser with the three defects of M-004, and the result looked
normal (median 327; on correct text it is 349). It now loads through
ingest.processed.load_processed, which aborts if the layer is out of date.

Dependencies: tiktoken, tokenizers (they need the project venv).
Usage: .venv/bin/python scripts/profile_tokens.py  -> data/t2_token_profile.json
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Any

import tiktoken
from tokenizers import Tokenizer

from ingest import download as dl
from ingest import processed as pr

OUT = Path("data/t2_token_profile.json")
ACTS = ("DORA", "MiCA", "PSD2")


def main() -> None:
    bpe = tiktoken.get_encoding("cl100k_base")
    wp = Tokenizer.from_pretrained("bert-base-uncased")

    manifest = dl.load_manifest()
    rows: list[dict[str, Any]] = []
    for act in ACTS:
        doc = pr.load_processed(act, manifest)
        for a in doc.articles:
            t = a.text
            rows.append(
                {
                    "act": act,
                    "article_id": a.article_id,
                    "words": len(t.split()),
                    "cl100k": len(bpe.encode(t)),
                    "wordpiece": len(wp.encode(t, add_special_tokens=False).ids),
                }
            )

    OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    ratios = [r["wordpiece"] / r["cl100k"] for r in rows]
    print(f"{len(rows)} articles")
    print(
        f"WordPiece/cl100k: median {st.median(ratios):.3f}, min {min(ratios):.3f}, max {max(ratios):.3f}"
    )
    print(
        f"diverge >10%: {sum(abs(r - 1) > 0.10 for r in ratios)}   >20%: {sum(abs(r - 1) > 0.20 for r in ratios)}"
    )
    for cap in (300, 400, 500):
        n_bpe = sum(r["cl100k"] > cap for r in rows)
        n_wp = sum(r["wordpiece"] > cap for r in rows)
        print(f"cap {cap}: above the cap {n_bpe} (cl100k) / {n_wp} (WordPiece)")


if __name__ == "__main__":
    main()
