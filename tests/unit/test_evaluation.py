"""Unit tests for the Fundamento-specific evaluation logic (D-019, D-020): node to
chunk projection, the run contract, Recall@T and the aggregation. Synthetic
layers only, no corpus, no ir_measures."""

from __future__ import annotations

import pytest

from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import evaluation as ev
from ingest import hierarchy as h

LEAD_IN = "Member States shall require the following:"


def _words(text: str) -> int:
    return len(text.split())


def _chunked_article() -> ch.ChunkedDocument:
    """Article 7 with a lead-in and two points, cut so that the lead-in has its
    own chunk and travels as inherited context in the chunks of its points."""
    texts = {
        "7.1": LEAD_IN,
        "7.1.a": "a first requirement;",
        "7.1.b": "a second requirement.",
        "7.2": "Competent authorities shall supervise compliance.",
    }
    nodes = [h.Node(path="7", level=1, marker="7", kind="articulo")] + [
        h.Node(
            path=p,
            level=len(p.split(".")),
            marker=p.split(".")[-1],
            kind="punto" if p.count(".") == 2 else "apartado",
            own_text=t,
        )
        for p, t in texts.items()
    ]
    article = h.Article(
        article_id="art_7",
        path="7",
        title="Title",
        text=" ".join(texts.values()),
        max_depth=3,
        nodes=nodes,
    )
    return ch.ChunkedDocument(
        act="DORA",
        source_celex="32022R2554",
        source_role="original",
        source_sha256="1" * 64,
        parser_version=h.PARSER_VERSION,
        chunker_version=ch.CHUNKER_VERSION,
        tokenizer="cl100k_base",
        max_tokens=15,
        chunks=ch.chunk_article(article, "DORA", 15, _words),
    )


def _layer(*refs: tuple[str, tuple[str, ...], int], act: str = "X") -> ev.ChunkLayer:
    return ev.layer_from_refs(
        [ev.ChunkRef(cid, tuple((act, p) for p in paths), tokens) for cid, paths, tokens in refs],
        max_tokens=500,
        parser_version=h.PARSER_VERSION,
        sources={},
    )


LAYER = _layer(
    ("X:1", ("1.1", "1.2"), 100),
    ("X:2", ("2",), 50),
    ("X:3", ("3",), 80),
    ("X:4", ("4",), 40),
)


def _qrel(path: str, grade: int, **overrides) -> bm.Qrel:
    return bm.Qrel.model_validate(
        {
            "need_id": "N001",
            "act": "MiCA",
            "path": path,
            "grade": grade,
            "rationale": "r",
            "text_sha256": "a" * 64,
        }
        | overrides
    )


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------


def test_inherited_context_is_not_ownership():
    doc = _chunked_article()
    by_id = {c.chunk_id: c for c in doc.chunks}
    assert set(by_id) == {"DORA:7.1:own", "DORA:7.1.a", "DORA:7.1.b", "DORA:7.2"}
    assert LEAD_IN in by_id["DORA:7.1.a"].context  # the lead-in is visible there...
    layer = ev.layer_from_documents([doc])
    # ...but only the chunk that owns it is credited with it
    assert ev.project({("DORA", "7.1"): 3}, layer) == {"DORA:7.1:own": 3}


def test_a_chunk_takes_the_maximum_grade_it_owns_never_the_sum():
    assert ev.project({("X", "1.1"): 1, ("X", "1.2"): 3}, LAYER) == {"X:1": 3}
    assert ev.project({("X", "1.1"): 3, ("X", "1.2"): 1}, LAYER) == {"X:1": 3}


def test_grade_zero_is_judged_and_unjudged_is_absent():
    grades = ev.project({("X", "2"): 0}, LAYER)
    assert grades == {"X:2": 0}  # present: judged not relevant
    assert "X:3" not in grades  # absent: never judged


def test_an_ambiguous_judgment_counts_as_unjudged():
    qrels = [_qrel("149.2", 3, ambiguous=True), _qrel("149.1", 1)]
    assert ev.judged_nodes(qrels) == {("MiCA", "149.1"): 1}


def test_several_relevant_nodes_in_one_chunk_stay_several_nodes():
    judged = {("X", "1.1"): 3, ("X", "1.2"): 2}
    assert ev.project(judged, LAYER) == {"X:1": 3}  # one relevant chunk
    relevant = frozenset(n for n, g in judged.items() if g >= bm.RELEVANT_GRADE)
    assert len(relevant) == 2  # two relevant nodes for node-level recall


def test_a_judged_node_that_no_chunk_owns_fails():
    with pytest.raises(ev.EvaluationError, match="no chunk owns the judged node X:9"):
        ev.project({("X", "9"): 2}, LAYER)


def test_a_layer_where_a_node_has_two_owners_fails():
    with pytest.raises(ev.EvaluationError, match="owned by X:1 and X:5"):
        _layer(("X:1", ("1",), 10), ("X:5", ("1",), 10))


# ---------------------------------------------------------------------------
# the run contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "run,message",
    [
        (ev.Run(10, {"q1": ["X:1"]}), "depth 10 is below the minimum of 100"),
        (ev.Run(100, {"q9": ["X:1"]}), "queries that the benchmark does not evaluate: ['q9']"),
        (ev.Run(100, {"q1": ["X:1", "X:1"]}), "chunks ranked twice ['X:1']"),
        (ev.Run(100, {"q1": ["X:7"]}), "chunks not in the layer ['X:7']"),
    ],
)
def test_check_run_rejects(run, message):
    with pytest.raises(ev.EvaluationError, match=message.replace("[", r"\[").replace("]", r"\]")):
        ev.check_run(run, LAYER, ["q1"])


def test_a_query_absent_from_the_run_is_allowed():
    ev.check_run(ev.Run(100, {}), LAYER, ["q1", "q2"])


# ---------------------------------------------------------------------------
# Recall@T
# ---------------------------------------------------------------------------

RELEVANT = frozenset({("X", "1.1"), ("X", "4")})


def test_a_chunk_enters_only_if_it_fits_whole_and_the_walk_stops_there():
    # budget 150: X:1 (100) fits; X:3 (80) does not (180); X:4 (40) WOULD fit (140),
    # and holds a relevant node, but the walk already stopped at X:3
    r = ev.recall_at_tokens(["X:1", "X:3", "X:4"], LAYER, RELEVANT, budget=150, depth=100)
    assert (r.covered, r.relevant, r.tokens_used, r.chunks_used) == (1, 2, 100, 1)
    assert r.value == 0.5


def test_every_relevant_node_counts_once():
    r = ev.recall_at_tokens(
        ["X:1", "X:4"], LAYER, RELEVANT | {("X", "1.2")}, budget=1000, depth=100
    )
    assert (r.covered, r.relevant) == (3, 3)


def test_a_budget_filled_exactly_is_reached():
    r = ev.recall_at_tokens(["X:1", "X:2"], LAYER, RELEVANT, budget=150, depth=2)
    assert (r.tokens_used, r.chunks_used) == (150, 2)


def test_a_ranking_cut_at_the_depth_before_the_budget_fails():
    with pytest.raises(ev.InsufficientRunDepthError, match="cut at depth 2 with 150 of 1000"):
        ev.recall_at_tokens(["X:1", "X:2"], LAYER, RELEVANT, budget=1000, depth=2)


def test_a_ranking_shorter_than_the_depth_is_what_the_system_returned():
    r = ev.recall_at_tokens(["X:1"], LAYER, RELEVANT, budget=1000, depth=100)
    assert (r.covered, r.tokens_used) == (1, 100)


def test_a_ranking_of_the_whole_layer_is_never_too_short():
    r = ev.recall_at_tokens(["X:1", "X:2", "X:3", "X:4"], LAYER, RELEVANT, budget=1000, depth=4)
    assert (r.covered, r.tokens_used) == (2, 270)


def test_an_empty_ranking_recalls_nothing():
    r = ev.recall_at_tokens([], LAYER, RELEVANT, budget=1000, depth=100)
    assert (r.value, r.tokens_used) == (0.0, 0)


def test_recall_without_relevant_nodes_is_refused():
    with pytest.raises(ev.EvaluationError, match="at least one relevant node"):
        ev.recall_at_tokens(["X:1"], LAYER, frozenset(), budget=1000, depth=100)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _score(query_id: str, value: float, answerability: str = "complete") -> ev.QueryScores:
    lang = query_id.split("-")[1]
    query = bm.Query.model_validate(
        {
            "query_id": query_id,
            "need_id": query_id.split("-")[0],
            "lang": lang,
            "style": "formal" if lang == "en" else "natural",
            "origin": "human",
            "text": f"text {query_id}",
        }
    )
    return ev.QueryScores(query, answerability, {"m": value})


def test_means_go_per_need_first_and_languages_stay_apart():
    scores = [
        _score("N001-en-1", 1.0),
        _score("N001-en-2", 0.0),  # N001 in English is 0.5: one need, not two queries
        _score("N002-en-1", 1.0),
        _score("N001-es-1", 0.0),
        _score("N002-es-1", 0.5),
    ]
    report = ev.aggregate(scores, [s.query.query_id for s in scores])
    assert report.per_need[("N001", "en")] == {"m": 0.5}
    assert report.by_language["en"].means == {"m": 0.75}  # (0.5 + 1.0) / 2 needs
    assert (report.by_language["en"].needs, report.by_language["en"].queries) == (2, 3)
    assert report.by_language["es"].means == {"m": 0.25}
    assert set(report.by_language) == {"en", "es"}  # no pooled number


def test_partial_needs_stay_marked():
    scores = [_score("N001-en-1", 1.0), _score("N002-en-1", 0.0, "partial")]
    report = ev.aggregate(scores, [s.query.query_id for s in scores])
    assert report.by_language_and_answerability[("en", "complete")].means == {"m": 1.0}
    assert report.by_language_and_answerability[("en", "partial")].means == {"m": 0.0}


def test_no_query_may_disappear_from_the_denominator():
    scores = [_score("N001-en-1", 1.0)]
    with pytest.raises(ev.EvaluationError, match=r"absent \['N001-es-1'\]"):
        ev.aggregate(scores, ["N001-en-1", "N001-es-1"])
    with pytest.raises(ev.EvaluationError, match=r"repeated \['N001-en-1'\]"):
        ev.aggregate(scores * 2, ["N001-en-1"])


def test_queries_scored_with_different_metrics_are_refused():
    a = _score("N001-en-1", 1.0)
    b = ev.QueryScores(_score("N002-en-1", 1.0).query, "complete", {"other": 1.0})
    with pytest.raises(ev.EvaluationError, match="different metrics"):
        ev.aggregate([a, b], ["N001-en-1", "N002-en-1"])
