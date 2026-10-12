"""The real benchmark directories (D-019): the three fixtures from the design study
(N001-N003) and the DEV skeleton.

The fixtures are TEST DATA for the pipeline, not DEV: their queries were written
knowing the evidence (origin 'evidence_aware'), so they never count among the DEV
needs. Their seal and their dataset rules are checked here in any checkout; their
agreement with the corpus needs data/raw/ and is marked "corpus".
"""

from __future__ import annotations

from collections import Counter

import pytest

from ingest import benchmark as bm


def test_the_fixtures_meet_the_contract_and_their_seal():
    fixtures = bm.load_benchmark(bm.FIXTURES_DIR)
    assert fixtures.manifest.role == "fixture"
    assert [n.need_id for n in fixtures.needs] == ["N001", "N002", "N003"]
    assert {q.origin for q in fixtures.queries} == {"evidence_aware"}


def test_the_fixtures_cover_what_the_design_asked_them_to_show():
    fixtures = bm.load_benchmark(bm.FIXTURES_DIR)
    kinds = {n.need_id: (n.type, n.answerability) for n in fixtures.needs}
    assert kinds == {
        "N001": ("direct", "complete"),
        "N002": ("exception_confusion", "complete"),
        "N003": ("partial_out_of_scope", "partial"),
    }
    assert Counter(r.grade for r in fixtures.qrels) == {0: 7, 1: 11, 2: 6, 3: 3}
    hard_negatives = {(r.need_id, r.act, r.path) for r in fixtures.qrels if r.grade == 0}
    assert ("N001", "DORA", "64") in hard_negatives  # the cross-act lexical twin
    assert ("N002", "MiCA", "40.2") in hard_negatives  # the twin regime
    assert ("N003", "DORA", "20.unp_2") in hard_negatives  # the wrong date
    spans = [(r.act, r.path, r.span.start, r.span.end) for r in fixtures.qrels if r.span]
    assert spans == [("DORA", "20.a", 57, 638)]
    assert all(len(fixtures.queries_of(n.need_id)) == 2 for n in fixtures.needs)


def test_the_dev_skeleton_is_sealed_and_empty():
    dev = bm.load_benchmark(bm.DEV_DIR)
    assert dev.manifest.role == "dev"
    assert (dev.needs, dev.queries, dev.qrels) == ([], [], [])


@pytest.mark.corpus
def test_the_fixtures_and_dev_still_match_the_corpus():
    docs = bm.load_corpus()
    bm.check_corpus(bm.load_benchmark(bm.FIXTURES_DIR), docs)
    bm.check_corpus(bm.load_benchmark(bm.DEV_DIR), docs)


@pytest.mark.corpus
def test_the_lexical_twin_of_n001_is_byte_identical():
    """The claim behind the N001 hard negative, checked on the real text."""
    texts = bm.node_texts(bm.load_corpus())
    assert texts[("DORA", "64")] == texts[("MiCA", "149.1")]
    qrels = {(r.act, r.path): r for r in bm.load_benchmark(bm.FIXTURES_DIR).qrels}
    assert qrels[("DORA", "64")].text_sha256 == qrels[("MiCA", "149.1")].text_sha256


@pytest.mark.corpus
def test_the_overlap_of_the_fixture_queries():
    """The diagnostic on the real evidence: the expert English query of N003 shares
    most of its words, the user-vocabulary query of N002 shares none."""
    docs = bm.load_corpus()
    fixtures = bm.load_benchmark(bm.FIXTURES_DIR)
    evidence = bm.evidence_texts(fixtures, "N002", docs)
    assert "MiCA Article 50: Prohibition of granting interest" in evidence
    share = {
        q.query_id: bm.lexical_overlap(q.text, bm.evidence_texts(fixtures, q.need_id, docs)).share
        for q in fixtures.queries
    }
    assert share["N002-en-1"] == 0.0
    assert share["N003-en-1"] > 0.5
    assert max(share[q] for q in share if q.endswith("-es-1")) < 0.2
