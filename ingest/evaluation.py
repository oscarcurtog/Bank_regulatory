"""Evaluation of ranked runs against the retrieval benchmark (D-019, D-020).

What lives here is the logic specific to Fundamento, pure and without third-party
dependencies:

- node -> chunk projection. A chunk is relevant through the nodes it OWNS
  (source_paths), never through its inherited context. Its grade is the maximum
  of the judged nodes it owns: summing would reward big chunks. A chunk that owns
  no judged node stays unjudged, which is not the same as judged with grade 0.
- Recall@T: recall of relevant nodes under a token budget, the fair way to compare
  chunkings of different sizes. Recall@K does not compare chunkings: the number of
  relevant chunks of a need changes with the chunk size.
- the run contract and its depth guard;
- the aggregation: query -> (need, language) -> language. English and Spanish are
  never mixed into one number.

The standard metrics (Recall@K, MRR, nDCG, Judged) come from ingest/ir_metrics.py,
the only module that imports ir_measures.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from ingest import benchmark as bm
from ingest import chunks as ch

# Recall@T budgets for v1, in cl100k tokens of the chunk text (D-020).
TOKEN_BUDGETS = (1000, 2000)
# Results a run must request per query. Recall@20 and Judged@10 need far less; the
# margin is for the token budgets, and the exact guard is in recall_at_tokens().
MIN_RUN_DEPTH = 100


class EvaluationError(Exception):
    """A run, a chunk layer or a benchmark cannot be evaluated together."""


class InsufficientRunDepthError(EvaluationError):
    """A ranking was cut at the requested depth before a token budget was reached:
    evaluating it would lower the recall without warning."""


# ---------------------------------------------------------------------------
# the chunk layer, reduced to what evaluation needs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkRef:
    chunk_id: str
    nodes: tuple[bm.NodeKey, ...]  # the nodes it owns, from source_paths; never its context
    token_count: int


@dataclass(frozen=True)
class ChunkLayer:
    """One chunking of the corpus: every chunk, and the chunk that owns each node."""

    max_tokens: int
    parser_version: str
    sources: Mapping[str, str]  # act -> source_sha256
    chunks: Mapping[str, ChunkRef]
    owner: Mapping[bm.NodeKey, str]


def layer_from_refs(
    refs: Iterable[ChunkRef], max_tokens: int, parser_version: str, sources: Mapping[str, str]
) -> ChunkLayer:
    chunks: dict[str, ChunkRef] = {}
    owner: dict[bm.NodeKey, str] = {}
    for ref in refs:
        if ref.chunk_id in chunks:
            raise EvaluationError(f"chunk {ref.chunk_id} appears twice in the layer")
        chunks[ref.chunk_id] = ref
        for node in ref.nodes:
            if node in owner:
                raise EvaluationError(f"node {node} is owned by {owner[node]} and {ref.chunk_id}")
            owner[node] = ref.chunk_id
    return ChunkLayer(max_tokens, parser_version, dict(sources), chunks, owner)


def layer_from_documents(docs: Iterable[ch.ChunkedDocument]) -> ChunkLayer:
    """The layer of a set of ChunkedDocuments, one per act, all from one run of the
    chunker (same budget, same parser)."""
    docs = list(docs)
    budgets = {d.max_tokens for d in docs}
    parsers = {d.parser_version for d in docs}
    if len(budgets) != 1 or len(parsers) != 1:
        raise EvaluationError(
            f"one layer needs one max_tokens and one parser, got {sorted(budgets)} and "
            f"{sorted(parsers)}"
        )
    refs = [
        ChunkRef(c.chunk_id, tuple((d.act, p) for p in c.source_paths), c.token_count)
        for d in docs
        for c in d.chunks
    ]
    return layer_from_refs(
        refs, budgets.pop(), parsers.pop(), {d.act: d.source_sha256 for d in docs}
    )


def check_layer(manifest: bm.BenchmarkManifest, layer: ChunkLayer) -> None:
    """The layer must come from the snapshot the benchmark was sealed against:
    otherwise its paths may not be the judged ones."""
    problems: list[str] = []
    if layer.parser_version != manifest.parser_version:
        problems.append(
            f"layer from parser {layer.parser_version}, benchmark sealed with "
            f"{manifest.parser_version}"
        )
    for act in bm.ACTS:
        sealed = manifest.corpus[act].source_sha256
        if layer.sources.get(act) != sealed:
            problems.append(f"{act}: layer source {layer.sources.get(act)}, benchmark {sealed}")
    if problems:
        raise EvaluationError(
            "the chunk layer and the benchmark come from different snapshots:\n"
            + "\n".join(f"- {p}" for p in problems)
        )


# ---------------------------------------------------------------------------
# node -> chunk projection
# ---------------------------------------------------------------------------


def judged_nodes(qrels: Iterable[bm.Qrel]) -> dict[bm.NodeKey, int]:
    """node -> grade for the qrels of ONE need. Ambiguous judgments are left out:
    they count as unjudged (D-019)."""
    return {r.node: r.grade for r in qrels if not r.ambiguous}


def project(judged: Mapping[bm.NodeKey, int], layer: ChunkLayer) -> dict[str, int]:
    """chunk_id -> grade, for every chunk that owns at least one judged node: the
    maximum grade it owns. Chunks that own no judged node are absent: unjudged."""
    grades: dict[str, int] = {}
    for node, grade in judged.items():
        chunk_id = layer.owner.get(node)
        if chunk_id is None:
            raise EvaluationError(
                f"no chunk owns the judged node {node[0]}:{node[1]}: the layer was built "
                f"from another parser or corpus"
            )
        grades[chunk_id] = max(grade, grades.get(chunk_id, grade))
    return grades


@dataclass(frozen=True)
class Target:
    """One need that enters Recall, MRR and nDCG, projected onto one layer."""

    need: bm.Need
    queries: tuple[bm.Query, ...]
    chunk_grades: Mapping[str, int]
    relevant_nodes: frozenset[bm.NodeKey]  # grade >= 2, each counted once


def targets(benchmark: bm.Benchmark, layer: ChunkLayer) -> list[Target]:
    """The needs that enter Recall, MRR and nDCG: status 'ready', answerability
    'complete' or 'partial'. A 'partial' need is evaluated on the evidence the
    corpus has and keeps its answerability for the report. A 'none' need never
    enters: it has no evidence by definition and is kept for abstention."""
    check_layer(benchmark.manifest, layer)
    out: list[Target] = []
    for need in benchmark.needs:
        if need.status != "ready" or need.answerability == "none":
            continue
        judged = judged_nodes(benchmark.qrels_of(need.need_id))
        relevant = frozenset(n for n, g in judged.items() if g >= bm.RELEVANT_GRADE)
        if not relevant:
            raise EvaluationError(f"need {need.need_id}: no relevant node to evaluate")
        out.append(
            Target(
                need, tuple(benchmark.queries_of(need.need_id)), project(judged, layer), relevant
            )
        )
    if not out:
        raise EvaluationError(f"{benchmark.directory}: no ready need with evidence to evaluate")
    return out


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """A system's output: for each query, chunk ids in rank order, best first.

    depth is how many results were requested per query. A ranking shorter than
    depth means the system returned fewer, which is legitimate (a lexical
    retriever with no match). A query absent from the run scores 0 and still
    counts in every mean."""

    depth: int
    rankings: Mapping[str, Sequence[str]]


def check_run(run: Run, layer: ChunkLayer, query_ids: Collection[str]) -> None:
    problems: list[str] = []
    if run.depth < MIN_RUN_DEPTH:
        problems.append(f"depth {run.depth} is below the minimum of {MIN_RUN_DEPTH}")
    unknown = sorted(set(run.rankings) - set(query_ids))
    if unknown:
        problems.append(f"queries that the benchmark does not evaluate: {unknown}")
    for query_id, ranking in sorted(run.rankings.items()):
        if len(ranking) > run.depth:
            problems.append(f"{query_id}: {len(ranking)} results with depth {run.depth}")
        twice = sorted(c for c, n in Counter(ranking).items() if n > 1)
        if twice:
            problems.append(f"{query_id}: chunks ranked twice {twice}")
        alien = [c for c in ranking if c not in layer.chunks]
        if alien:
            problems.append(f"{query_id}: chunks not in the layer {alien[:5]}")
    if problems:
        raise EvaluationError(
            "the run does not meet the contract:\n" + "\n".join(f"- {p}" for p in problems)
        )


# ---------------------------------------------------------------------------
# Recall@T
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetRecall:
    budget: int
    covered: int
    relevant: int
    tokens_used: int
    chunks_used: int

    @property
    def value(self) -> float:
        return self.covered / self.relevant


def recall_at_tokens(
    ranking: Sequence[str],
    layer: ChunkLayer,
    relevant: frozenset[bm.NodeKey],
    budget: int,
    depth: int,
) -> BudgetRecall:
    """Recall of the relevant nodes within a token budget (D-020).

    The ranking is walked in order and a chunk is taken only if it fits whole:
    tokens_used + token_count <= budget. The first chunk that does not fit ends
    the walk: chunks are never split, and smaller chunks further down are not used
    to fill the gap. Each relevant node counts once.

    A ranking cut at exactly the requested depth, before the budget was reached
    and before the layer ran out, raises InsufficientRunDepthError instead of
    returning a recall that the depth, not the system, made lower."""
    if not relevant:
        raise EvaluationError("Recall@T needs at least one relevant node")
    used = taken = 0
    covered: set[bm.NodeKey] = set()
    for chunk_id in ranking:
        chunk = layer.chunks[chunk_id]
        if used + chunk.token_count > budget:
            break
        used += chunk.token_count
        taken += 1
        covered.update(n for n in chunk.nodes if n in relevant)
    else:
        if used < budget and len(ranking) >= depth and len(ranking) < len(layer.chunks):
            raise InsufficientRunDepthError(
                f"the ranking was cut at depth {depth} with {used} of {budget} tokens used: "
                f"request a deeper run"
            )
    return BudgetRecall(budget, len(covered), len(relevant), used, taken)


# ---------------------------------------------------------------------------
# aggregation: query -> (need, language) -> language
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryScores:
    query: bm.Query
    answerability: str
    values: Mapping[str, float]


@dataclass(frozen=True)
class Summary:
    needs: int
    queries: int
    means: Mapping[str, float]


@dataclass(frozen=True)
class Report:
    per_need: Mapping[tuple[str, str], Mapping[str, float]]  # (need_id, lang) -> metric -> value
    by_language: Mapping[str, Summary]
    by_language_and_answerability: Mapping[tuple[str, str], Summary]
    missing_rankings: tuple[str, ...]  # queries absent from the run, scored 0


def aggregate(
    scores: Sequence[QueryScores], expected: Collection[str], missing_rankings: Iterable[str] = ()
) -> Report:
    """Means per (need, language) first, so that a need with more formulations does
    not weigh more; then per language over needs. Fails unless the scores cover
    exactly the expected queries, each once, with the same metrics."""
    ids = [s.query.query_id for s in scores]
    absent = sorted(set(expected) - set(ids))
    extra = sorted(set(ids) - set(expected))
    twice = sorted(i for i, n in Counter(ids).items() if n > 1)
    if absent or extra or twice:
        raise EvaluationError(
            f"the scores must cover each expected query once: absent {absent}, unexpected "
            f"{extra}, repeated {twice}"
        )
    metrics = {tuple(sorted(s.values)) for s in scores}
    if len(metrics) > 1:
        raise EvaluationError(f"queries scored with different metrics: {sorted(metrics)}")

    groups: dict[tuple[str, str], list[QueryScores]] = {}
    for s in scores:
        groups.setdefault((s.query.need_id, s.query.lang), []).append(s)
    names = sorted(next(iter(metrics))) if metrics else []
    per_need = {
        key: {m: statistics.fmean(s.values[m] for s in group) for m in names}
        for key, group in sorted(groups.items())
    }
    answerability: dict[tuple[str, str], str] = {
        (s.query.need_id, s.query.lang): s.answerability for s in scores
    }

    def summary(keys: list[tuple[str, str]]) -> Summary:
        return Summary(
            needs=len(keys),
            queries=sum(len(groups[k]) for k in keys),
            means={m: statistics.fmean(per_need[k][m] for k in keys) for m in names},
        )

    langs = sorted({lang for _, lang in per_need})
    by_language = {lang: summary([k for k in per_need if k[1] == lang]) for lang in langs}
    pairs = sorted({(k[1], answerability[k]) for k in per_need})
    by_pair = {
        (lang, a): summary([k for k in per_need if k[1] == lang and answerability[k] == a])
        for lang, a in pairs
    }
    return Report(per_need, by_language, by_pair, tuple(sorted(missing_rankings)))
