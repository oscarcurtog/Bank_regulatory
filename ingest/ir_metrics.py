"""The only module that imports ir_measures (D-020): the standard retrieval metrics,
computed on chunk-level qrels projected from the benchmark.

A correct library with a wrong integration is still a bug. Each point below was
measured on ir_measures 0.4.3 with pytrec-eval-terrier 0.5.10, and each one is
pinned by tests/integration/test_ir_metrics.py:

- The provider is chosen here, never by the automatic fallback of ir_measures:
  pytrec_eval for Recall, MRR and nDCG, judged for Judged. Provider.evaluator()
  does not check that a measure is supported, so this module calls supports()
  itself and fails on a measure the provider cannot compute.
- RR@10 is not supported by the pytrec_eval provider: trec_eval's recip_rank has
  no cutoff, and evaluator() silently computes it over the whole ranking (a first
  hit at rank 11 scores 1/11). MRR@10 is therefore RR(rel=2) on the ranking cut at
  10.
- trec_eval breaks score ties by document id, in reverse order, not by our rank.
  Scores are derived from the rank and strictly decreasing, so ties cannot happen.
- The judged provider divides by zero on an empty ranking. Empty rankings are not
  passed to the library: they score 0 here, explicitly.
- Binary measures use rel=2 (grade >= 2). nDCG uses the grades as linear gains
  (trec_eval: grade / log2(rank + 1)) and runs in an evaluator of its own, so that
  it never shares another measure's relevance level.
- On short rankings Judged@k divides by the number of results returned, not by k.

ir_measures is a development dependency (requirements-dev.lock.txt). It is not in
the Docker images: its backend publishes no Linux ARM64 wheel. Importing this
module never imports ir_measures; computing a metric without it fails.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from ingest import benchmark as bm

MRR_CUTOFF = 10
NDCG_CUTOFF = 10
JUDGED_CUTOFF = 10
DEFAULT_RECALL_CUTOFFS = (5, 10, 20)


class MetricError(Exception):
    """The metric backend cannot compute what was asked, or returned something else."""


@dataclass(frozen=True)
class Backend:
    ir_measures: str
    pytrec_eval_terrier: str


def _ir_measures() -> ModuleType:
    try:
        import ir_measures
    except ImportError as e:
        raise ImportError(
            "ir_measures is not installed. Install requirements-dev.lock.txt (D-020); it "
            "is deliberately absent from the Docker images."
        ) from e
    return ir_measures


def backend() -> Backend:
    """The versions that computed the metrics, for the record of every result."""
    _ir_measures()
    return Backend(
        importlib.metadata.version("ir-measures"),
        importlib.metadata.version("pytrec-eval-terrier"),
    )


def strict_scores(ranking: Sequence[str]) -> dict[str, float]:
    """The rank turned into strictly decreasing scores: first n, last 1."""
    n = len(ranking)
    return {chunk_id: float(n - i) for i, chunk_id in enumerate(ranking)}


def _require_support(provider: Any, measures: Sequence[Any]) -> None:
    unsupported = [str(m) for m in measures if not provider.supports(m)]
    if unsupported:
        raise MetricError(f"provider {provider.NAME} does not support {unsupported}")


def standard_scores(
    qrels: Mapping[str, Mapping[str, int]],
    rankings: Mapping[str, Sequence[str]],
    recall_cutoffs: Sequence[int] = DEFAULT_RECALL_CUTOFFS,
) -> dict[str, dict[str, float]]:
    """query_id -> metric -> value, for EVERY query in qrels.

    qrels are the projected chunk grades of each query's need (grade 0 included:
    judged and not relevant). rankings are chunk ids, best first; a query may be
    absent or have an empty ranking, and then it scores 0 on every metric."""
    irm = _ir_measures()
    unknown = sorted(set(rankings) - set(qrels))
    if unknown:
        raise MetricError(f"rankings for queries without qrels: {unknown}")
    if any(not grades for grades in qrels.values()):
        raise MetricError("a query without any judged chunk cannot be scored")

    recall = {f"Recall@{k}": irm.R(rel=bm.RELEVANT_GRADE) @ k for k in recall_cutoffs}
    mrr = {f"MRR@{MRR_CUTOFF}": irm.RR(rel=bm.RELEVANT_GRADE)}
    ndcg = {f"nDCG@{NDCG_CUTOFF}": irm.nDCG @ NDCG_CUTOFF}
    judged = {f"Judged@{JUDGED_CUTOFF}": irm.Judged @ JUDGED_CUTOFF}

    plain = {q: dict(grades) for q, grades in qrels.items()}
    full = {q: strict_scores(r) for q, r in rankings.items() if r}
    top = {q: strict_scores(list(r)[:MRR_CUTOFF]) for q, r in rankings.items() if r}
    values: dict[str, dict[str, float]] = {q: {} for q in qrels}

    def collect(
        provider: Any, named: Mapping[str, Any], run: Mapping[str, Mapping[str, float]]
    ) -> None:
        _require_support(provider, list(named.values()))
        name_of: dict[Any, str] = {m: n for n, m in named.items()}
        for metric in provider.evaluator(list(named.values()), plain).iter_calc(run):
            scores = values.get(metric.query_id)
            name = name_of.get(metric.measure)
            if scores is None or name is None or name in scores:
                raise MetricError(f"unexpected result {metric.query_id} {metric.measure}")
            scores[name] = float(metric.value)

    steps: list[tuple[Any, Mapping[str, Any], Mapping[str, Mapping[str, float]]]] = [
        (irm.pytrec_eval, recall, full),
        (irm.pytrec_eval, mrr, top),
        (irm.pytrec_eval, ndcg, full),
        (irm.judged, judged, full),
    ]
    for provider, named, run in steps:
        collect(provider, named, run)

    names = [*recall, *mrr, *ndcg, *judged]
    for query_id, scores in values.items():
        for name in names:
            if name not in scores:
                if rankings.get(query_id):
                    raise MetricError(f"{query_id}: {name} was not computed")
                scores[name] = 0.0  # absent or empty ranking: 0, and it still counts
    return values


def binary_scores(
    relevant: Mapping[str, Collection[str]],
    rankings: Mapping[str, Sequence[str]],
    recall: Sequence[int],
    ndcg: Sequence[int],
    rr: Sequence[int],
    ap: Sequence[int],
) -> dict[str, dict[str, float]]:
    """query_id -> metric -> value, for an external benchmark with binary, positive-only
    judgments (T4, D-053): every listed document is relevant with gain 1, every other one is
    not, so there is no Judged@k. The names are Recall@k, nDCG@k, MRR@k and mAP@k.

    The integration rules of standard_scores hold: the provider is chosen here, ranks become
    strictly decreasing scores, and MRR@k is RR on the ranking cut at k. AP@k is trec_eval's
    map_cut_k, which divides by all of the query's relevant documents, as JuriFindIT defines
    it (paper, appendix F); a relevant document the corpus does not hold still counts there."""
    irm = _ir_measures()
    unknown = sorted(set(rankings) - set(relevant))
    if unknown:
        raise MetricError(f"rankings for queries without qrels: {unknown}")
    if any(not docs for docs in relevant.values()):
        raise MetricError("a query without any relevant document cannot be scored")
    for query_id, ranking in rankings.items():
        if len(set(ranking)) != len(ranking):
            raise MetricError(f"{query_id}: a document ranked twice")

    plain = {q: dict.fromkeys(docs, 1) for q, docs in relevant.items()}
    full = {q: strict_scores(r) for q, r in rankings.items() if r}
    values: dict[str, dict[str, float]] = {q: {} for q in relevant}

    def collect(named: Mapping[str, Any], run: Mapping[str, Mapping[str, float]]) -> None:
        _require_support(irm.pytrec_eval, list(named.values()))
        name_of: dict[Any, str] = {m: n for n, m in named.items()}
        for metric in irm.pytrec_eval.evaluator(list(named.values()), plain).iter_calc(run):
            scores = values.get(metric.query_id)
            name = name_of.get(metric.measure)
            if scores is None or name is None or name in scores:
                raise MetricError(f"unexpected result {metric.query_id} {metric.measure}")
            scores[name] = float(metric.value)

    collect({f"Recall@{k}": irm.R(rel=1) @ k for k in recall}, full)
    collect({f"nDCG@{k}": irm.nDCG @ k for k in ndcg}, full)
    collect({f"mAP@{k}": irm.AP(rel=1) @ k for k in ap}, full)
    for k in rr:
        cut = {q: strict_scores(list(r)[:k]) for q, r in rankings.items() if r}
        collect({f"MRR@{k}": irm.RR(rel=1)}, cut)

    names = [
        *(f"Recall@{k}" for k in recall),
        *(f"nDCG@{k}" for k in ndcg),
        *(f"mAP@{k}" for k in ap),
        *(f"MRR@{k}" for k in rr),
    ]
    for query_id, scores in values.items():
        for name in names:
            if name not in scores:
                if rankings.get(query_id):
                    raise MetricError(f"{query_id}: {name} was not computed")
                scores[name] = 0.0  # absent or empty ranking: 0, and it still counts
    return values
