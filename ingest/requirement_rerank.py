"""E of retrieval iteration 4: each requirement's first 50 E5 results reranked by
Qwen3-Reranker-0.6B, zero-shot, as classic relevance (does this passage answer this question?),
then D's rule unchanged (config/retrieval/requirement-reranking.md).

D's requirements find most of the evidence E5 misses somewhere in their first 50 results, but
less than half of it in their first 5, the only places D's rule reads. E asks one thing: does a
cross-encoder lift that evidence into the first 5? The requirements are D-4B's, read by D's
parser; the reranker sees one requirement and one chunk at a time, never the original question,
an anchor, another requirement or a label. Its order replaces E5's within the same 50 chunks, so
nothing outside them can enter and Recall@50 cannot change; ties keep E5's order. D's rule then
builds the ten with the reranked first 5 where D used E5's. Nothing here reads a label, except
the measures, which take the labels as arguments once every ranking exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ingest import benchmark as bm
from ingest import retrieval as rt

INSTRUCTION = (
    "Given a question about EU financial regulation, "
    "retrieve the passages of the law that answer it"
)
DEPTH = 50
CUTOFFS = (1, 3, 5, 10, 20, 50)
SIGNAL_GAIN = 0.03  # requirement Recall@5, against E5's order of the same 50


def reorder(candidates: Sequence[str], scores: Mapping[str, float]) -> list[str]:
    """The candidates by descending score; equal scores keep their order in `candidates` (E5's)."""
    position = {c: i for i, c in enumerate(candidates)}
    return sorted(candidates, key=lambda c: (-scores[c], position[c]))


def best_rank(hits: Sequence[Sequence[str]]) -> dict[str, int]:
    """Each chunk's best rank (1 is first) over a query's requirements."""
    out: dict[str, int] = {}
    for ranking in hits:
        for i, c in enumerate(ranking, start=1):
            if i < out.get(c, i + 1):
                out[c] = i
    return out


def requirement_recall(
    hits: Mapping[str, Sequence[Sequence[str]]],
    relevant: Mapping[str, frozenset[str]],
    queries: Sequence[bm.Query],
    k: int,
) -> tuple[float, dict[str, float]]:
    """Recall@k of a query's requirements together: the share of the query's relevant chunks that
    some requirement ranks within its first k, macro by need, and the per-query values."""
    values = {}
    for q in queries:
        best = best_rank(hits[q.query_id])
        found = {c for c in relevant[q.query_id] if best.get(c, k + 1) <= k}
        values[q.query_id] = len(found) / len(relevant[q.query_id])
    return rt.macro_by_need(values, queries), values


def requirement_mrr(
    hits: Mapping[str, Sequence[Sequence[str]]],
    relevant: Mapping[str, frozenset[str]],
    queries: Sequence[bm.Query],
) -> float:
    """The reciprocal of the best rank of any relevant chunk over a query's requirements (0 when
    none is in their lists), macro by need."""
    values = {}
    for q in queries:
        best = best_rank(hits[q.query_id])
        ranks = [best[c] for c in relevant[q.query_id] if c in best]
        values[q.query_id] = 1 / min(ranks) if ranks else 0.0
    return rt.macro_by_need(values, queries)
