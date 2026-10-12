"""The evaluation logic on the real benchmark fixtures (D-019, D-020).

The known case of fixture N002 runs in any checkout on an EXCERPT of the real
500-token layer: chunk ids, owned paths and token counts, no text. A corpus test
checks that the excerpt still matches the real layer, and the regressions that
need whole layers (500 against 300 tokens) are marked "corpus".
"""

from __future__ import annotations

import pytest

from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import evaluation as ev

# MiCA chunks of the official 500-token layer that fixture N002 touches, plus the
# two that close its known ranking. Checked against the real layer below.
N002_EXCERPT = (
    ("MiCA:40", ("40.1", "40.2", "40.3"), 167),
    ("MiCA:50", ("50.1", "50.2", "50.3"), 170),
    ("MiCA:3.1.6", ("3.1.6",), 73),
    ("MiCA:3.1.7", ("3.1.7",), 61),
    ("MiCA:3.1.5", ("3.1.5",), 58),
    (
        "MiCA:39",
        ("39.1", "39.2", "39.2.a", "39.2.b", "39.2.c", "39.2.d", "39.2.e", "39.2.sub_2", "39.3"),
        442,
    ),
    ("MiCA:3.1.15", ("3.1.15",), 78),
)
N002_RANKING = ["MiCA:40", "MiCA:50", "MiCA:3.1.6", "MiCA:3.1.7", "MiCA:3.1.5", "MiCA:39"]


def _excerpt_layer() -> ev.ChunkLayer:
    refs = [
        ev.ChunkRef(cid, tuple(("MiCA", p) for p in paths), tokens)
        for cid, paths, tokens in N002_EXCERPT
    ]
    return ev.layer_from_refs(refs, max_tokens=500, parser_version="1.4.0", sources={})


def _n002() -> tuple[dict, frozenset]:
    fixtures = bm.load_benchmark(bm.FIXTURES_DIR)
    judged = ev.judged_nodes(fixtures.qrels_of("N002"))
    return judged, frozenset(n for n, g in judged.items() if g >= bm.RELEVANT_GRADE)


def test_n002_projects_onto_its_chunks():
    judged, relevant = _n002()
    assert ev.project(judged, _excerpt_layer()) == {
        "MiCA:50": 3,  # owns 50.1 (1), 50.2 (3) and 50.3 (2): the maximum
        "MiCA:3.1.7": 2,
        "MiCA:3.1.6": 1,
        "MiCA:3.1.15": 1,
        "MiCA:40": 0,  # the twin regime: judged, not relevant
    }
    assert relevant == {("MiCA", "50.2"), ("MiCA", "50.3"), ("MiCA", "3.1.7")}


@pytest.mark.parametrize(
    "budget,covered,tokens,chunks",
    [(300, 0, 167, 1), (1000, 3, 971, 6)],
)
def test_n002_recall_at_tokens(budget, covered, tokens, chunks):
    """The values computed by hand in the design study: with 300 tokens the twin
    MiCA:40 (167) fits and MiCA:50 (170) does not, so the walk stops at 0/3."""
    _, relevant = _n002()
    r = ev.recall_at_tokens(
        N002_RANKING, _excerpt_layer(), relevant, budget, depth=ev.MIN_RUN_DEPTH
    )
    assert (r.covered, r.relevant, r.tokens_used, r.chunks_used) == (covered, 3, tokens, chunks)


# ---------------------------------------------------------------------------
# against the real layers
# ---------------------------------------------------------------------------


def _layers(*budgets: int) -> dict[int, ev.ChunkLayer]:
    docs = bm.load_corpus()
    return {
        budget: ev.layer_from_documents(
            ch.chunk_document(docs[act], act, budget) for act in bm.ACTS
        )
        for budget in budgets
    }


@pytest.mark.corpus
def test_the_n002_excerpt_matches_the_real_layer():
    layer = _layers(500)[500]
    for chunk_id, paths, tokens in N002_EXCERPT:
        real = layer.chunks[chunk_id]
        assert (real.nodes, real.token_count) == (tuple(("MiCA", p) for p in paths), tokens)


@pytest.mark.corpus
def test_the_fixtures_project_onto_500_and_300_tokens():
    """The number of relevant CHUNKS of a need depends on the chunking; the number
    of relevant NODES does not. That is why Recall@K never compares chunk sizes."""
    fixtures = bm.load_benchmark(bm.FIXTURES_DIR)
    projected = {
        budget: {t.need.need_id: t for t in ev.targets(fixtures, layer)}
        for budget, layer in _layers(500, 300).items()
    }

    def relevant_chunks(budget: int, need_id: str) -> dict[str, int]:
        grades = projected[budget][need_id].chunk_grades
        return {c: g for c, g in grades.items() if g >= bm.RELEVANT_GRADE}

    assert relevant_chunks(500, "N001") == {"MiCA:149": 3}
    assert relevant_chunks(300, "N001") == {"MiCA:149.2": 3, "MiCA:149.3": 2, "MiCA:149.4": 2}
    assert (
        len(projected[500]["N001"].relevant_nodes)
        == len(projected[300]["N001"].relevant_nodes)
        == 3
    )
    assert relevant_chunks(500, "N002") == {"MiCA:50": 3, "MiCA:3.1.7": 2}
    assert relevant_chunks(500, "N003") == {"DORA:19.4": 3, "DORA:20.a": 2}
    assert projected[500]["N003"].need.answerability == "partial"


@pytest.mark.corpus
def test_n002_recall_at_tokens_on_the_real_layer():
    layer = _layers(500)[500]
    _, relevant = _n002()
    values = [ev.recall_at_tokens(N002_RANKING, layer, relevant, t, depth=100) for t in (300, 1000)]
    assert [(r.covered, r.tokens_used) for r in values] == [(0, 167), (3, 971)]
