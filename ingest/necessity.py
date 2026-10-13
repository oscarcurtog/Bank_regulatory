"""Anchor-conditioned necessity scoring and evidence-set selection: retrieval iteration 3
(config/retrieval/necessity-selection.md).

The failure iteration 2 isolated: E5 finds the anchor, R1's pool holds most of the evidence the
anchor depends on, and no score tried orders that evidence. Here a reranker is asked another
question: not "is this candidate relevant to the question?" but "is this candidate necessary to
apply this anchor to the question?".

- The model, Qwen/Qwen3-Reranker-0.6B, under an explicit contract
  (config/retrieval/qwen3-reranker-0.6b.json): the model card's own prompt, read at the pinned
  revision, with a custom instruction in its <Instruct> field. The score is the logit of "yes"
  minus the logit of "no" at the last position; its logistic is the card's probability of
  "yes". A prompt over the limit loses the tail of the candidate and is recorded, never cut
  silently.
- The population: for each query, each anchor among E5's first chunks and each candidate of R1's
  pool outside E5's top 10 that the anchor reaches by one of R1's relations (a reference, a
  defined term, a definition by reference) or by a bridge about a term the anchor uses.
- Three inputs: A1 the question and the candidate; A2 adds the anchor; A3 adds the relation and
  its citation context, the clause of the anchor (or of the definition) where the relation is
  written.
- Evidence sets: E5's first n chunks, then at most k supports whose best probability under a
  kept anchor reaches a threshold, then E5's order.

Inputs come from the question, the chunks' text, R1's provenance and the law's structure;
nothing here reads a label. Labels enter only the diagnostics, once every score exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ingest import dependencies as dp
from ingest import embeddings as em
from ingest import evaluation as ev
from ingest import references as xr
from ingest import retrieval as rt

Variant = Literal["A1", "A2", "A3"]
VARIANTS: tuple[Variant, ...] = ("A1", "A2", "A3")


class NecessityError(Exception):
    """The structure, a population or a score does not meet its contract."""


# ---------------------------------------------------------------------------
# the model's contract
# ---------------------------------------------------------------------------


class NecessityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    schema_version: Literal["1"]
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    model_id: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    revision: str = Field(pattern=em.REVISION_PATTERN)
    tokenizer_file: str
    prefix: str  # the card's system turn and the opening of the user turn, verbatim
    body: str  # the card's "<Instruct>: ...\n<Query>: ...\n<Document>: ..." template
    suffix: str  # the close of the user turn and an assistant turn with an empty think block
    yes_token: str
    no_token: str
    pad_token: str
    max_input_tokens: int = Field(ge=64)
    truncation: Literal["explicit_document_tail"]
    score: Literal["logit_yes_minus_no"]
    batch_size: int = Field(ge=1)
    device: Literal["cpu", "mps"]
    dtype: Literal["float32"]

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: Path) -> NecessityConfig:
    return NecessityConfig.model_validate_json(path.read_text(encoding="utf-8"))


def p_yes(logit_diff: float) -> float:
    """The card's probability of "yes" (a softmax over "no" and "yes"): the logistic of the
    logit difference, computed without overflow."""
    if logit_diff >= 0:
        return 1.0 / (1.0 + math.exp(-logit_diff))
    e = math.exp(logit_diff)
    return e / (1.0 + e)


@dataclass(frozen=True)
class Prompt:
    ids: tuple[int, ...]
    original_tokens: int
    truncated: bool


class NecessityReranker:
    """Qwen3-Reranker as its card runs it: the prompt template, left padding, the last
    position's logits of "yes" and "no". torch and transformers are the optional 'embed' extra,
    loaded only when one is built."""

    def __init__(self, config: NecessityConfig) -> None:
        try:
            import torch
            import transformers
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise em.ModelUnavailableError(
                "torch and transformers are not installed: install requirements-embed.lock.txt"
            ) from e
        self.config = config
        self._torch = torch
        cache = em.hf_home() / "hub"
        os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
        if config.device == "mps" and not torch.backends.mps.is_available():
            raise em.ModelUnavailableError("the contract asks for MPS, which is not available")
        try:
            path = hf_hub_download(
                config.model_id, config.tokenizer_file, revision=config.revision, cache_dir=cache
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                config.model_id, revision=config.revision, cache_dir=cache
            )
            model: Any = transformers.AutoModelForCausalLM.from_pretrained(
                config.model_id, revision=config.revision, cache_dir=cache, dtype=torch.float32
            )
        except Exception as e:
            raise em.ModelUnavailableError(
                f"{config.model_id}@{config.revision[:12]}: cannot load the reranker from the "
                f"cache {cache} or the network"
            ) from e
        self.tokenizer: Any = tokenizer
        self.tokenizer_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        ids = {}
        for token in (config.yes_token, config.no_token, config.pad_token):
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1:
                raise em.EmbeddingError(f"{token!r} is not one token of the reranker")
            ids[token] = encoded[0]
        self.yes_id, self.no_id = ids[config.yes_token], ids[config.no_token]
        self.pad_id = ids[config.pad_token]
        if model.config.max_position_embeddings < config.max_input_tokens:
            raise em.EmbeddingError(
                f"{model.config.max_position_embeddings} positions, the contract says more"
            )
        self._prefix = list(tokenizer.encode(config.prefix, add_special_tokens=False))
        self._suffix = list(tokenizer.encode(config.suffix, add_special_tokens=False))
        self.model = model.to(config.device).eval()

    def prompt(self, instruction: str, query: str, document: str) -> Prompt:
        """The card's prompt: its prefix, the body as one text, its suffix. Over the limit the
        body loses its tail, which is the document's; a body whose query alone is over it is
        refused."""
        body = self.config.body.format(instruction=instruction, query=query, document=document)
        mid = list(self.tokenizer(body)["input_ids"])
        n = len(self._prefix) + len(mid) + len(self._suffix)
        room = self.config.max_input_tokens - len(self._prefix) - len(self._suffix)
        if len(mid) <= room:
            return Prompt(tuple(self._prefix + mid + self._suffix), n, False)
        head = self.config.body.format(instruction=instruction, query=query, document="")
        if len(self.tokenizer(head)["input_ids"]) >= room:
            raise em.TruncationRefusedError("the instruction and the query fill the prompt alone")
        return Prompt(tuple(self._prefix + mid[:room] + self._suffix), n, True)

    def score(self, prompts: Sequence[Prompt]) -> np.ndarray:
        """The logit of "yes" minus the logit of "no" at each prompt's last position, in input
        order. Batches go longest first, padded on the left as the card pads them."""
        torch = self._torch
        out = np.empty(len(prompts), dtype=np.float64)
        order = sorted(range(len(prompts)), key=lambda i: (-len(prompts[i].ids), i))
        device = self.config.device
        with torch.inference_mode():
            for start in range(0, len(order), self.config.batch_size):
                batch = order[start : start + self.config.batch_size]
                width = max(len(prompts[i].ids) for i in batch)
                ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for row, i in enumerate(batch):
                    n = len(prompts[i].ids)
                    ids[row, width - n :] = torch.tensor(prompts[i].ids, dtype=torch.long)
                    mask[row, width - n :] = 1
                logits = self.model(
                    input_ids=ids.to(device), attention_mask=mask.to(device), logits_to_keep=1
                ).logits[:, -1, :]
                diff = logits[:, self.yes_id] - logits[:, self.no_id]
                # to the CPU first: on MPS, .to("cpu", torch.float64) returns zeros silently
                out[batch] = diff.to("cpu").to(torch.float64).numpy()
        self._release()
        return out

    def score_shared(self, prompts: Sequence[Prompt]) -> np.ndarray:
        """The scores of score(), for prompts that open with the same tokens (one query and
        one anchor): the tokens all of them share run once, and their keys and values serve
        every prompt's remaining tokens, padded on the right. The common prefix is read from
        the token ids themselves, so each prompt is still the exact sequence score() reads."""
        if len(prompts) < 2:
            return self.score(prompts)
        torch = self._torch
        from transformers.cache_utils import DynamicCache

        shortest = min(len(p.ids) for p in prompts)
        common = 0
        while common < shortest - 1 and all(
            p.ids[common] == prompts[0].ids[common] for p in prompts
        ):
            common += 1
        device = self.config.device
        out = np.empty(len(prompts), dtype=np.float64)
        order = sorted(range(len(prompts)), key=lambda i: (-len(prompts[i].ids), i))
        weight = self.model.lm_head.weight
        with torch.inference_mode():
            shared = torch.tensor([prompts[0].ids[:common]], dtype=torch.long, device=device)
            past = self.model.model(input_ids=shared, use_cache=True).past_key_values
            layers = [(layer.keys, layer.values) for layer in past.layers]
            for start in range(0, len(order), self.config.batch_size):
                batch = order[start : start + self.config.batch_size]
                rest = [prompts[i].ids[common:] for i in batch]
                width = max(len(r) for r in rest)
                ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
                for row, r in enumerate(rest):
                    ids[row, : len(r)] = torch.tensor(r, dtype=torch.long)
                cache = DynamicCache()
                for i, (k, v) in enumerate(layers):
                    cache.update(
                        k.expand(len(batch), -1, -1, -1), v.expand(len(batch), -1, -1, -1), i
                    )
                hidden = self.model.model(
                    input_ids=ids.to(device), past_key_values=cache, use_cache=True
                ).last_hidden_state
                rows = torch.arange(len(batch), device=device)
                ends = torch.tensor([len(r) - 1 for r in rest], device=device)
                last = hidden[rows, ends]
                diff = last @ (weight[self.yes_id] - weight[self.no_id])
                out[batch] = diff.to("cpu").to(torch.float64).numpy()
                del cache, hidden, last, diff
            del past, layers
        self._release()
        return out

    def _release(self) -> None:
        """Return the device's cached blocks: the expanded keys and values of a group hold
        gigabytes that MPS otherwise keeps, and the machine swaps."""
        if self.config.device == "mps":
            self._torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# the three inputs, fixed before any score
# ---------------------------------------------------------------------------

_VERDICT = (
    'Reply "yes" if the candidate is necessary supporting evidence, and "no" if it is merely '
    "related context or unnecessary."
)
INSTRUCTIONS: Mapping[Variant, str] = {
    "A1": "Determine whether the candidate provision is necessary to correctly address the "
    f"user's question. {_VERDICT}",
    "A2": "Determine whether the candidate provision is necessary to correctly apply the anchor "
    f"provision to the user's question. {_VERDICT}",
    "A3": "Determine whether the candidate provision is necessary to correctly apply the anchor "
    f"provision to the user's question. {_VERDICT}",
}


@dataclass(frozen=True)
class Triple:
    """One candidate of one anchor of one query, with what links them."""

    query_id: str
    anchor: str
    anchor_rank: int
    candidate: str
    kinds: tuple[str, ...]
    relation: str
    context: str


def model_input(
    variant: Variant, question: str, triple: Triple, text: Mapping[str, str]
) -> tuple[str, str, str]:
    """(instruction, query, document) of one variant; `text` is each chunk's text, with its
    act, article and title."""
    query = f"QUESTION: {question}"
    if variant in ("A2", "A3"):
        query += f"\nANCHOR: {text[triple.anchor]}"
    if variant == "A3":
        query += f"\nRELATION: {triple.relation}\nCITATION CONTEXT: {triple.context}"
    return INSTRUCTIONS[variant], query, f"CANDIDATE: {text[triple.candidate]}"


# ---------------------------------------------------------------------------
# relations and their citation contexts, read from the law's structure
# ---------------------------------------------------------------------------

CONTEXT_CHARS = 300
CUE_WINDOW = 80
MAX_CONTEXTS = 3
_SEGMENT = re.compile(r"(?<=[.;:])\s+")
EXCEPTION_CUES = re.compile(
    r"\b(?:by way of derogation from|derogat(?:ion|ing|es?) from|except|notwithstanding|"
    r"save (?:as|for|where)|other than|exempt(?:ed)? from|exemptions? from)\b",
    re.IGNORECASE,
)
CONDITION_CUES = re.compile(
    r"\b(?:subject to|in accordance with|pursuant to|provided that|on condition that|"
    r"in compliance with|compl(?:y|ies|ying) with|fulfil(?:s|ling)?|"
    r"(?:conditions?|requirements?) (?:laid down|set out|referred to|specified) in)\b",
    re.IGNORECASE,
)


def segment_at(text: str, start: int, end: int, limit: int = CONTEXT_CHARS) -> tuple[int, str]:
    """The clause of `text` (cut after '.', ';' or ':' and a space) that holds [start, end),
    with its first offset; over `limit` characters, about `limit` characters around the
    mention, cut at spaces and marked with '...'."""
    cuts = [0, *(m.end() for m in _SEGMENT.finditer(text)), len(text)]
    lo = max(c for c in cuts if c <= start)
    hi = min(c for c in cuts if c >= end)
    if hi - lo <= limit:
        return lo, text[lo:hi].strip()
    width = max(limit, end - start)
    side = (width - (end - start)) // 2
    a, b = max(lo, start - side), min(hi, end + side)
    if a == lo:
        b = min(hi, lo + width)
    elif b == hi:
        a = max(lo, hi - width)
    cut_left, cut_right = a > lo, b < hi
    if cut_left and (space := text.find(" ", a, start)) >= 0:
        a = space + 1
    if cut_right and (space := text.rfind(" ", end, b)) >= 0:
        b = space
    snippet = text[a:b].strip()
    return lo, ("..." if cut_left else "") + snippet + ("..." if cut_right else "")


def cue(text: str, start: int, lo: int) -> str | None:
    """'exception' or 'condition' when the words just before a reference say so (within its
    clause, at most CUE_WINDOW characters back)."""
    before = text[max(lo, start - CUE_WINDOW) : start]
    if EXCEPTION_CUES.search(before):
        return "exception"
    if CONDITION_CUES.search(before):
        return "condition"
    return None


def act_of(chunk_id: str) -> str:
    return chunk_id.split(":", 1)[0]


@dataclass(frozen=True)
class Structure:
    """R1's structure, rebuilt as R1 built it, and each chunk's article."""

    node_text: Mapping[dp.NodeKey, str]
    layer: ev.ChunkLayer
    graph: xr.ReferenceGraph
    index: dp.DependencyIndex
    definitions: tuple[dp.Definition, ...]
    article: Mapping[str, str]  # chunk id -> "<act>:<article id>"


@dataclass(frozen=True)
class Link:
    kind: str
    label: str
    context: str


def _scope(anchor: str, candidate: str, structure: Structure) -> list[str]:
    if act_of(candidate) != act_of(anchor):
        return ["cross-act"]
    if structure.article.get(candidate) == structure.article.get(anchor):
        return ["hierarchy: same article"]
    return []


def _with(label: str, scope: Sequence[str]) -> str:
    return f"{label} ({', '.join(scope)})" if scope else label


def term_context(structure: Structure, anchor: str, term: str) -> str:
    """The clause of the anchor's own text where it first uses a defined term."""
    pattern = dp.term_pattern(term)
    for act, path in structure.layer.chunks[anchor].nodes:
        text = structure.node_text[(act, path)]
        if m := pattern.search(text):
            return segment_at(text, m.start(), m.end())[1]
    raise NecessityError(f"{anchor}: the term {term!r} is not in its own text")


def reference_link(
    structure: Structure, anchor: str, detail: str, target_node: str, candidate: str
) -> Link:
    for edge in structure.graph.edges.get(anchor, ()):
        ref = edge.reference
        if (
            ref.text == detail
            and f"{ref.act}:{edge.resolved_path}" == target_node
            and candidate in edge.targets
        ):
            text = structure.node_text[(ref.source_act, ref.source_path)]
            lo, context = segment_at(text, ref.start, ref.start + len(ref.text))
            kind = cue(text, ref.start, lo)
            label = {
                None: "reference: the anchor cites the candidate",
                "exception": "exception: the anchor cites the candidate as an exception or "
                "derogation",
                "condition": "condition: the anchor cites the candidate as a condition",
            }[kind]
            return Link("reference", _with(label, _scope(anchor, candidate, structure)), context)
    raise NecessityError(f"{anchor} -> {candidate}: no reference edge for {detail!r}")


def definition_link(structure: Structure, anchor: str, term: str, candidate: str) -> Link:
    label = f"definition: the candidate defines '{term}', a term the anchor uses"
    return Link("definition", label, term_context(structure, anchor, term))


def definition_reference_link(
    structure: Structure, anchor: str, term: str, target_node: str, candidate: str
) -> Link:
    for d in structure.definitions:
        if d.term != term or d.act != act_of(anchor):
            continue
        owner = structure.layer.owner.get((d.act, d.path))
        for edge in structure.graph.edges.get(owner or "", ()):
            ref = edge.reference
            if (
                (ref.source_act, ref.source_path) == (d.act, d.path)
                and f"{ref.act}:{edge.resolved_path}" == target_node
                and candidate in edge.targets
            ):
                text = structure.node_text[(d.act, d.path)]
                defined = segment_at(text, ref.start, ref.start + len(ref.text))[1]
                label = _with(
                    f"definition by reference: the definition of '{term}', a term the anchor "
                    "uses, refers to the candidate",
                    ["cross-act"] if act_of(candidate) != act_of(anchor) else [],
                )
                context = f"{term_context(structure, anchor, term)} / {defined}"
                return Link("definition_reference", label, context)
    raise NecessityError(f"{anchor} -> {candidate}: no definition by reference for {term!r}")


def bridge_link(structure: Structure, anchor: str, term: str, candidate: str) -> Link:
    label = _with(
        f"term: the candidate was found by searching \"{dp.bridge_text(term)}\", '{term}' "
        "being a term the anchor uses",
        ["cross-act"] if act_of(candidate) != act_of(anchor) else [],
    )
    return Link("bridge", label, term_context(structure, anchor, term))


_BRIDGE_HEAD, _BRIDGE_TAIL = dp.BRIDGE_TEMPLATE.split("{term}")


def term_of_bridge(bridge: str) -> str:
    if not (bridge.startswith(_BRIDGE_HEAD) and bridge.endswith(_BRIDGE_TAIL)):
        raise NecessityError(f"not a bridge: {bridge!r}")
    return bridge[len(_BRIDGE_HEAD) : len(bridge) - len(_BRIDGE_TAIL)]


def population(
    e5: rt.Ranked,
    record: Mapping[str, Any],
    structure: Structure,
    anchors: int,
    outside: int,
) -> list[Triple]:
    """One query's (anchor, candidate) pairs: every anchor among E5's first `anchors` chunks,
    every candidate of R1's pool (its provenance record) outside E5's first `outside` that the
    anchor reaches by one of R1's relations, or by a bridge about a term the anchor uses.
    Ordered by anchor rank, then candidate id; every link of a pair kept, in R1's order."""
    if record["query_id"] != e5.query_id:
        raise NecessityError(f"provenance of {record['query_id']}, ranking of {e5.query_id}")
    rank = {a: i for i, a in enumerate(e5.chunk_ids[:anchors], start=1)}
    top = set(e5.chunk_ids[:outside])
    terms = {a: {d.term for d in structure.index.terms.get(a, ())} for a in rank}
    found: dict[tuple[str, str], list[Link]] = {}
    for candidate, relations in sorted(record["candidates"].items()):
        if candidate in top:
            continue
        for rel in relations:
            kind, anchor = rel["kind"], rel["anchor"]
            if kind == "bridge":
                term = term_of_bridge(rel["detail"])
                for a in rank:
                    if term in terms[a] and a != candidate:
                        link = bridge_link(structure, a, term, candidate)
                        found.setdefault((a, candidate), []).append(link)
                continue
            if anchor not in rank or anchor == candidate:
                continue
            if kind == "reference":
                link = reference_link(
                    structure, anchor, rel["detail"], rel["target_node"], candidate
                )
            elif kind == "definition":
                link = definition_link(structure, anchor, rel["detail"], candidate)
            elif kind == "definition_reference":
                link = definition_reference_link(
                    structure, anchor, rel["detail"], rel["target_node"], candidate
                )
            else:
                raise NecessityError(f"unknown relation kind {kind!r}")
            found.setdefault((anchor, candidate), []).append(link)
    out = []
    for (a, c), links in sorted(found.items(), key=lambda kv: (rank[kv[0][0]], kv[0][1])):
        labels = list(dict.fromkeys(link.label for link in links))
        contexts = list(dict.fromkeys(link.context for link in links))[:MAX_CONTEXTS]
        out.append(
            Triple(
                e5.query_id,
                a,
                rank[a],
                c,
                tuple(dict.fromkeys(link.kind for link in links)),
                "; ".join(labels),
                " | ".join(f'"{x}"' for x in contexts),
            )
        )
    return out


# ---------------------------------------------------------------------------
# evidence sets (experiment B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Support:
    candidate: str
    anchor: str
    probability: float


def evidence_set(
    e5: rt.Ranked,
    probability: Mapping[tuple[str, str], float],
    head: int,
    supports: int,
    threshold: float,
) -> tuple[rt.Ranked, list[Support]]:
    """E5's first `head` chunks; then at most `supports` candidates whose best probability under
    a kept anchor (one of those `head`) reaches `threshold`, best first (ties: the anchor's E5
    rank, then the candidate's id); then every other chunk in E5's order.
    `probability` maps (anchor, candidate) to the necessity probability of one query."""
    kept = e5.chunk_ids[:head]
    rank = {a: i for i, a in enumerate(kept, start=1)}
    best: dict[str, Support] = {}
    for (anchor, candidate), p in probability.items():
        if anchor not in rank or candidate in rank:
            continue
        current = best.get(candidate)
        if current is None or (p, -rank[anchor]) > (current.probability, -rank[current.anchor]):
            best[candidate] = Support(candidate, anchor, p)
    chosen = sorted(
        (s for s in best.values() if s.probability >= threshold),
        key=lambda s: (-s.probability, rank[s.anchor], s.candidate),
    )[:supports]
    order = [*kept, *(s.candidate for s in chosen)]
    placed = set(order)
    order += [c for c in e5.chunk_ids if c not in placed]
    n = len(order)
    return rt.Ranked(e5.query_id, tuple(order), tuple(float(n - i) for i in range(n))), chosen


# ---------------------------------------------------------------------------
# discrimination, measured after scoring
# ---------------------------------------------------------------------------


def average_precision(ranked_labels: Sequence[bool]) -> float:
    """Average precision of a ranked list that holds every positive of its population."""
    hits, total = 0, 0.0
    for i, positive in enumerate(ranked_labels, start=1):
        if positive:
            hits += 1
            total += hits / i
    if not hits:
        raise NecessityError("average precision needs at least one positive")
    return total / hits


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve: the chance that a positive outscores a negative, ties
    counted as half (the Mann-Whitney statistic over average ranks)."""
    pairs = sorted(zip(scores, labels, strict=True), key=lambda p: p[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        raise NecessityError("AUROC needs positives and negatives")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        average = (i + j) / 2 + 1
        rank_sum += average * sum(1 for k in range(i, j + 1) if pairs[k][1])
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ranked_by_score(scores: Mapping[str, float]) -> list[str]:
    """Candidates best first, ties by id."""
    return sorted(scores, key=lambda c: (-scores[c], c))


GRADE_BANDS = (3, 2, 1, 0)
THRESHOLDS = (0.50, 0.70, 0.85)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    i = (len(sorted_values) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (i - lo)


def discrimination(
    triples: Mapping[str, Sequence[Triple]],
    score: Mapping[tuple[str, str, str], float],
    grades: Mapping[str, Mapping[str, int]],
    missing: Mapping[str, frozenset[str]],
    dependents: Mapping[str, frozenset[str]],
    queries: Sequence[Any],
    k: int = 4,
) -> dict[str, Any]:
    """How well one variant's scores separate the needed evidence, read with the labels once
    every score exists.

    triples: each query's population; score: (query, anchor, candidate) -> logit difference;
    grades: each query's projected chunk grades (absent: unjudged); missing and dependents:
    each query's relevant chunks, and its dependent evidence, outside E5's top 10; queries:
    the evaluable ones, for the macro means by need. A candidate's score for a query is its
    best over the query's anchors."""
    best: dict[str, dict[str, float]] = {}
    for qid, rows in triples.items():
        for t in rows:
            s = score[(qid, t.anchor, t.candidate)]
            best.setdefault(qid, {})[t.candidate] = max(best.get(qid, {}).get(t.candidate, s), s)
    measured = [q for q in queries if q.query_id in best]
    # every query counts for the recalls: one without candidates misses all its evidence

    def relevant(qid: str, c: str) -> bool:
        return grades[qid].get(c, 0) >= 2

    ap: dict[str, float] = {}
    p1: dict[str, float] = {}
    p3: dict[str, float] = {}
    support: dict[str, float] = {}
    dependent: dict[str, float] = {}
    ceiling_support: dict[str, float] = {}
    ceiling_dependent: dict[str, float] = {}
    pooled: list[tuple[float, bool]] = []
    bands: dict[int, list[float]] = {b: [] for b in GRADE_BANDS}
    groups_measured = 0
    for q in queries:
        qid = q.query_id
        order = ranked_by_score(best.get(qid, {}))
        labels = [relevant(qid, c) for c in order]
        if any(labels):
            ap[qid] = average_precision(labels)
        pooled += [(best[qid][c], relevant(qid, c)) for c in order]
        for c in order:
            bands[min(max(grades[qid].get(c, 0), 0), 3)].append(p_yes(best[qid][c]))
        groups: dict[str, dict[str, float]] = {}
        for t in triples.get(qid, ()):
            groups.setdefault(t.anchor, {})[t.candidate] = score[(qid, t.anchor, t.candidate)]
        at1, at3 = [], []
        for group in groups.values():
            ranked = ranked_by_score(group)
            if any(relevant(qid, c) for c in ranked):
                at1.append(float(relevant(qid, ranked[0])))
                at3.append(sum(relevant(qid, c) for c in ranked[:3]) / 3)
        if at1:
            groups_measured += len(at1)
            p1[qid] = sum(at1) / len(at1)
            p3[qid] = sum(at3) / len(at3)
        first = set(order[:k])
        if missing.get(qid):
            support[qid] = len(missing[qid] & first) / len(missing[qid])
            ceiling_support[qid] = len(missing[qid] & set(order)) / len(missing[qid])
        if dependents.get(qid):
            dependent[qid] = len(dependents[qid] & first) / len(dependents[qid])
            ceiling_dependent[qid] = len(dependents[qid] & set(order)) / len(dependents[qid])

    def macro(values: Mapping[str, float]) -> float | None:
        return rt.macro_by_need(values, queries) if values else None

    distribution = {}
    for b, values in bands.items():
        v = sorted(values)
        distribution[str(b) if b else "0 or unjudged"] = (
            {
                "candidates": len(v),
                "mean": sum(v) / len(v),
                "p25": _quantile(v, 0.25),
                "median": _quantile(v, 0.5),
                "p75": _quantile(v, 0.75),
            }
            | {f"share >= {t:.2f}": sum(x >= t for x in v) / len(v) for t in THRESHOLDS}
            if v
            else {"candidates": 0}
        )
    return {
        "queries_measured": len(measured),
        "average_precision": macro(ap),
        "average_precision_queries": len(ap),
        "auroc": auroc([s for s, _ in pooled], [r for _, r in pooled]),
        "precision_at_1": macro(p1),
        "precision_at_3": macro(p3),
        "anchor_groups_with_a_relevant_candidate": groups_measured,
        f"support_recall@{k}": macro(support),
        "support_recall_ceiling": macro(ceiling_support),
        f"dependent_support_recall@{k}": macro(dependent),
        "dependent_support_recall_ceiling": macro(ceiling_dependent),
        "probability_by_grade": distribution,
        "per_query": {"average_precision": ap, f"dependent_support_recall@{k}": dependent},
    }


def structure_of(frozen: rt.FrozenInputs, acts: Mapping[str, str]) -> Structure:
    """R1's structure, built as scripts/dependency_retrieval.py builds it, from the parsed
    corpus; `acts` names the acts a reference may point into (xref-expansion.json)."""
    from ingest import benchmark as bm

    corpus = bm.load_corpus()
    texts = {
        (act, n.path): n.own_text for act, d in corpus.items() for a in d.articles for n in a.nodes
    }
    graph = xr.build_graph(frozen.layer, texts, list(frozen.chunks), acts)
    definitions = dp.glossary(texts)
    index = dp.build_index(frozen.layer, texts, graph, definitions)
    article = {cid: f"{act_of(cid)}:{c.article_id}" for cid, c in frozen.chunks.items()}
    return Structure(texts, frozen.layer, graph, index, tuple(definitions), article)
