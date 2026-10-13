"""BM25 (ingest/lexical.py): the pinned contract, the analyzer, and the library's scores
against the formula computed independently. No download: bm25s and PyStemmer are in
requirements.lock.txt."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest

from ingest import lexical as lx

CONFIG_PATH = Path(__file__).parents[2] / "config" / "retrieval" / "bm25.json"
CONFIG = lx.load_config(CONFIG_PATH)
DOCS = [
    ("X:3", "The issuer's recovery plan shall be maintained."),
    ("X:1", "Credit institutions and payment institutions report major incidents."),
    ("X:2", "A recovery plan, a redemption plan: plans of issuers of asset-referenced tokens."),
    ("X:4", "Nothing here about tokens, tokens or tokens."),
]


@pytest.fixture(scope="module")
def index():
    return lx.build_index(CONFIG, DOCS)


def test_the_versioned_config_is_lucene_bm25_with_an_english_analyzer():
    assert (CONFIG.library, CONFIG.library_version, CONFIG.method) == ("bm25s", "0.3.11", "lucene")
    assert (CONFIG.k1, CONFIG.b) == (1.2, 0.75)
    a = CONFIG.analyzer
    assert (a.lowercase, a.strip_possessive, a.stopwords, a.stemmer) == (
        True,
        True,
        "lucene_english",
        "porter",
    )
    assert a.stemmer_library_version == "3.1.0"
    assert len(lx.LUCENE_ENGLISH_STOP_WORDS) == 33


def test_the_config_refuses_unknown_keys_and_hashes_its_content():
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="surprise"):
        lx.BM25Config.model_validate(raw | {"surprise": 1})
    assert lx.BM25Config.model_validate(raw | {"k1": 0.9}).config_hash() != CONFIG.config_hash()


def test_the_analyzer_lowercases_splits_drops_stop_words_and_stems():
    analyze = lx.Analyzer(CONFIG.analyzer)
    assert analyze("The Issuer's Recovery-Plans under Regulation (EU) 2023/1114") == [
        "issuer",
        "recoveri",
        "plan",
        "under",
        "regul",
        "eu",
        "2023",
        "1114",
    ]
    # "no" and "not" are Lucene stop words: a negation is invisible to this analyzer
    assert analyze("It is not a payment institution; no licence") == [
        "payment",
        "institut",
        "licenc",
    ]


def test_spanish_queries_go_through_the_same_english_analyzer():
    """The Spanish text is read from DEV, the only place the repository keeps it."""
    queries = Path(__file__).parents[2] / "data" / "benchmark" / "dev" / "queries.jsonl"
    rows = [json.loads(line) for line in queries.read_text(encoding="utf-8").splitlines()]
    text = next(r["text"] for r in rows if r["lang"] == "es" and not r["text"].isascii())
    terms = lx.Analyzer(CONFIG.analyzer)(text)
    assert terms and all(t == t.lower() for t in terms)
    # accented letters are kept, not folded: no English stemmer rule removes them
    assert any(not t.isascii() for t in terms)
    assert not set(terms) & lx.LUCENE_ENGLISH_STOP_WORDS


def test_library_scores_are_the_lucene_formula(index):
    texts = [t for _, t in DOCS]
    for query in ("recovery plan of an issuer", "payment incidents", "tokens tokens"):
        assert np.abs(index.scores(query) - lx.reference_scores(index, texts, query)).max() < 1e-12


def test_a_repeated_query_term_counts_each_time(index):
    once, twice = index.scores("tokens"), index.scores("tokens tokens")
    assert np.allclose(twice, 2 * once) and once.max() > 0


def test_a_query_without_a_known_term_scores_zero_everywhere(index):
    for query in ("zzz qqq", "the of and", ""):
        assert not index.scores(query).any()


def test_ranking_is_by_score_then_chunk_id(index):
    [ranked] = index.rank([("q", "zzz")])
    assert ranked.chunk_ids == ("X:1", "X:2", "X:3", "X:4")
    [ranked] = index.rank([("q", "recovery plan of an issuer")])
    assert ranked.chunk_ids[:2] == ("X:3", "X:2")
    assert list(ranked.scores) == sorted(ranked.scores, reverse=True)


def test_another_installed_version_fails_explicitly(monkeypatch):
    real = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, "version", lambda p: "9.9.9" if p == "bm25s" else real(p)
    )
    with pytest.raises(lx.LexicalError, match=r"bm25s 9\.9\.9 \(contract 0\.3\.11\)"):
        lx.build_index(CONFIG, DOCS)


def test_an_index_refuses_repeated_ids_and_documents_without_terms():
    with pytest.raises(lx.LexicalError, match="twice"):
        lx.build_index(CONFIG, [("X:1", "plan"), ("X:1", "plan")])
    with pytest.raises(lx.LexicalError, match="no index term"):
        lx.build_index(CONFIG, [("X:1", "plan"), ("X:2", "the of")])
