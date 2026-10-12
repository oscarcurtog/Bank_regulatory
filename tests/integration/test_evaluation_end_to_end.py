"""The whole evaluation (D-019, D-020) on deterministic runs that need no retriever.

ORACLE: relevant chunks first, in grade order; where the mathematics allows it,
every metric is perfect. ADVERSARIAL: hard negatives first, relevant last; the
expected values are computed by hand. Both run on a synthetic benchmark in any
checkout, and on the real fixtures and the real 500-token layer with the corpus.
"""

from __future__ import annotations

import math

import pytest

from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import evaluation as ev
from ingest import hierarchy as h

pytestmark = pytest.mark.evaluator

SHAS: dict[str, str] = {act: str(i) * 64 for i, act in enumerate(bm.ACTS, start=1)}

# chunk -> (owned MiCA paths, tokens). MiCA:3 is the heavy hard negative of N001.
CHUNKS = {
    "MiCA:1": (("1.1",), 100),
    "MiCA:2": (("2.1", "2.2"), 120),
    "MiCA:3": (("3",), 990),
    "MiCA:4": (("4",), 60),
    "MiCA:5": (("5",), 90),
    "MiCA:6": (("6",), 70),
    "MiCA:7": (("7",), 50),
}
GRADES = {
    "N001": {"1.1": 3, "2.1": 2, "2.2": 1, "3": 0},  # complete: two relevant chunks
    "N002": {"5": 3, "6": 1},  # partial: evaluated on the evidence the corpus has
    "N003": {"7": 1},  # none: never enters the metrics
}


def _benchmark() -> bm.Benchmark:
    needs = [
        bm.Need.model_validate(
            {
                "need_id": need_id,
                "split": "dev",
                "statement": f"Need {need_id}.",
                "type": "direct" if need_id == "N001" else "partial_out_of_scope",
                "drivers": [],
                "difficulty": "medium",
                "acts": ["MiCA"],
                "answerability": answerability,
                "gap_note": None if answerability == "complete" else "Something is missing.",
                "status": "ready",
            }
        )
        for need_id, answerability in (("N001", "complete"), ("N002", "partial"), ("N003", "none"))
    ]
    queries = [
        bm.Query.model_validate(
            {
                "query_id": f"{need_id}-{lang}-1",
                "need_id": need_id,
                "lang": lang,
                "style": "formal" if lang == "en" else "natural",
                "origin": "human",
                "text": f"{need_id} in {lang}",
            }
        )
        for need_id in GRADES
        for lang in ("en", "es")
    ]
    qrels = [
        bm.Qrel(
            need_id=need_id,
            act="MiCA",
            path=path,
            grade=grade,
            rationale="Synthetic.",
            text_sha256="a" * 64,
        )
        for need_id, grades in GRADES.items()
        for path, grade in grades.items()
    ]
    manifest = bm.BenchmarkManifest(
        schema_version="1",
        benchmark_version="0.1.0",
        role="dev",
        parser_version=h.PARSER_VERSION,
        corpus={
            act: bm.CorpusSource(source_celex="32022R2554", source_sha256=SHAS[act])
            for act in bm.ACTS
        },
        files={name: "b" * 64 for name in bm.DATA_FILES},
        templates={},
    )
    return bm.Benchmark(bm.DEV_DIR, manifest, needs, queries, qrels)


def _layer() -> ev.ChunkLayer:
    refs = [
        ev.ChunkRef(c, tuple(("MiCA", p) for p in paths), t) for c, (paths, t) in CHUNKS.items()
    ]
    return ev.layer_from_refs(refs, max_tokens=500, parser_version=h.PARSER_VERSION, sources=SHAS)


def _run(n001: list[str], n002: list[str], skip: tuple[str, ...] = ()) -> ev.Run:
    rankings = {
        f"{need}-{lang}-1": ranking
        for need, ranking in (("N001", n001), ("N002", n002))
        for lang in ("en", "es")
    }
    return ev.Run(depth=100, rankings={q: r for q, r in rankings.items() if q not in skip})


ORACLE_RANKINGS = (["MiCA:1", "MiCA:2", "MiCA:3", "MiCA:4"], ["MiCA:5", "MiCA:6"])
ORACLE = _run(*ORACLE_RANKINGS)
ADVERSARIAL = _run(["MiCA:3", "MiCA:4", "MiCA:2", "MiCA:1"], ["MiCA:6", "MiCA:5"])


def test_the_oracle_is_perfect_where_the_mathematics_allows_it():
    report = ev.evaluate(_benchmark(), _layer(), ORACLE)
    n001 = report.per_need[("N001", "en")]
    assert (n001["Recall@5"], n001["MRR@10"]) == (1.0, 1.0)
    assert n001["nDCG@10"] == pytest.approx(1.0, abs=1e-12)
    assert n001["Judged@10"] == 0.75  # MiCA:4 was never judged
    assert n001["Recall@T=1000"] == 1.0  # MiCA:1 + MiCA:2 = 220 tokens; MiCA:3 stops the walk
    n002 = report.per_need[("N002", "es")]
    assert (n002["Recall@5"], n002["MRR@10"]) == (1.0, 1.0)
    assert n002["nDCG@10"] == pytest.approx(1.0, abs=1e-12)


def test_the_adversarial_run_scores_what_was_computed_by_hand():
    report = ev.evaluate(_benchmark(), _layer(), ADVERSARIAL)
    n001 = report.per_need[("N001", "en")]
    assert n001["MRR@10"] == pytest.approx(1 / 3, abs=1e-12)  # first relevant chunk at rank 3
    assert (n001["Recall@5"], n001["Judged@10"]) == (1.0, 0.75)
    # gains 0, unjudged, 2, 3 against the ideal 3, 2
    expected = (2 / math.log2(4) + 3 / math.log2(5)) / (3 + 2 / math.log2(3))
    assert n001["nDCG@10"] == pytest.approx(expected, abs=1e-9)
    # the 990-token hard negative fills 1,000 tokens; with 2,000 everything fits
    assert (n001["Recall@T=1000"], n001["Recall@T=2000"]) == (0.0, 1.0)
    n002 = report.per_need[("N002", "en")]
    assert n002["MRR@10"] == 0.5
    assert n002["nDCG@10"] == pytest.approx(
        (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)), abs=1e-9
    )


def test_none_needs_never_enter_and_partial_needs_stay_marked():
    report = ev.evaluate(_benchmark(), _layer(), ORACLE)
    assert {need for need, _ in report.per_need} == {"N001", "N002"}
    assert set(report.by_language) == {"en", "es"}
    assert report.by_language_and_answerability[("en", "partial")].needs == 1


def test_an_absent_query_scores_zero_and_stays_in_the_denominator():
    report = ev.evaluate(_benchmark(), _layer(), _run(*ORACLE_RANKINGS, skip=("N002-es-1",)))
    assert report.missing_rankings == ("N002-es-1",)
    assert set(report.per_need[("N002", "es")].values()) == {0.0}
    es = report.by_language["es"]
    assert (es.needs, es.queries) == (2, 2)
    assert es.means["MRR@10"] == 0.5  # (1.0 for N001 + 0.0 for N002) / 2 needs
    assert report.by_language["en"].means["MRR@10"] == 1.0


def test_a_run_for_a_none_need_is_refused():
    run = ev.Run(depth=100, rankings={"N003-en-1": ["MiCA:7"]})
    with pytest.raises(ev.EvaluationError, match="queries that the benchmark does not evaluate"):
        ev.evaluate(_benchmark(), _layer(), run)


# ---------------------------------------------------------------------------
# the real fixtures on the real 500-token layer
# ---------------------------------------------------------------------------


def _real() -> tuple[bm.Benchmark, ev.ChunkLayer]:
    docs = bm.load_corpus()
    layer = ev.layer_from_documents(ch.chunk_document(docs[act], act, 500) for act in bm.ACTS)
    return bm.load_benchmark(bm.FIXTURES_DIR), layer


def _ordered(grades: dict[str, int], best_first: bool) -> list[str]:
    return sorted(grades, key=lambda c: (-grades[c] if best_first else grades[c], c))


@pytest.mark.corpus
def test_the_fixtures_oracle_and_adversarial_runs_on_the_real_layer():
    benchmark, layer = _real()
    targets = ev.targets(benchmark, layer)

    def run(best_first: bool) -> ev.Run:
        return ev.Run(
            depth=100,
            rankings={
                q.query_id: _ordered(dict(t.chunk_grades), best_first)
                for t in targets
                for q in t.queries
            },
        )

    oracle = ev.evaluate(benchmark, layer, run(best_first=True))
    for scores in oracle.per_need.values():
        assert (scores["Recall@5"], scores["MRR@10"]) == (1.0, 1.0)
        assert scores["nDCG@10"] == pytest.approx(1.0, abs=1e-12)
        assert scores["Judged@10"] == 1.0

    adversarial = ev.evaluate(benchmark, layer, run(best_first=False)).per_need[("N002", "en")]
    # 0 (MiCA:40), 1 (MiCA:3.1.15), 1 (MiCA:3.1.6), 2 (MiCA:3.1.7), 3 (MiCA:50)
    assert (adversarial["MRR@10"], adversarial["Recall@5"]) == (0.25, 1.0)
