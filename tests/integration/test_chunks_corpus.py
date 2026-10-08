"""Chunker tests on the full corpus (T2). Marked "corpus": they need data/raw/,
which is not versioned (D-002), so they skip themselves where it is missing.

They check every invariant on all 331 articles at the official budget and at two
smaller ones, and they pin the official run, so that any change in the chunks has
to come with a new CHUNKER_VERSION or a new parser, never by accident.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest
import tiktoken

from ingest import chunks as ch
from ingest import download as dl
from ingest import hierarchy as h
from tests.conftest import MANIFEST_PATH, assert_chunk_invariants

pytestmark = pytest.mark.corpus

ACTS = ("DORA", "MiCA", "PSD2")
ENCODING = tiktoken.get_encoding("cl100k_base")


def count(text: str) -> int:
    return len(ENCODING.encode(text))


def _articles(act: str) -> list[h.Article]:
    source = dl.parsed_source_for(act, dl.load_manifest(MANIFEST_PATH))
    family = "consolidado" if source.role == "consolidated" else "doue"
    return h.parse(source.local_path, family)


@pytest.mark.parametrize("max_tokens", [500, 400, 300])
def test_invariants_on_every_article(max_tokens):
    counter = ch.cl100k_counter()
    for act in ACTS:
        for article in _articles(act):
            chunks = ch.chunk_article(article, act, max_tokens, counter)
            assert_chunk_invariants(article, act, chunks, max_tokens, count)


def test_official_run_at_500():
    """The official run, pinned (T2, max_tokens=500)."""
    counter = ch.cl100k_counter()
    by_act: dict[str, list[ch.Chunk]] = {
        act: [c for a in _articles(act) for c in ch.chunk_article(a, act, 500, counter)]
        for act in ACTS
    }
    assert {act: len(chunks) for act, chunks in by_act.items()} == {
        "DORA": 322,
        "MiCA": 759,
        "PSD2": 273,
    }
    every = [c for chunks in by_act.values() for c in chunks]
    # the T2 phase 1 prediction: with parser 1.4.0 no unit is too big for 500 tokens
    assert Counter(c.budget_exception for c in every) == {None: 1354}
    assert Counter(c.coverage for c in every) == {"subtree": 1329, "own": 25}
    assert max(c.token_count for c in every) <= 500
    sources = sum(1 for act in ACTS for a in _articles(act) for n in a.nodes if n.own_text)
    assert sum(len(c.source_paths) for c in every) == sources == 3580


def test_the_full_corpus_chunks_deterministically():
    counter = ch.cl100k_counter()

    def run() -> str:
        return json.dumps(
            [
                c.model_dump(mode="json")
                for act in ACTS
                for a in _articles(act)
                for c in ch.chunk_article(a, act, 500, counter)
            ],
            ensure_ascii=False,
        )

    assert run() == run()
