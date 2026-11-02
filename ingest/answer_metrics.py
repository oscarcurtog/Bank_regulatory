"""The metrics, the cascade rule, the 8B trigger and the human anchor of S7-S8, as functions of
outputs that were already produced.

Nothing here runs a model or reads a file. It turns (case, raw output) into an `Outcome`, counts,
and applies the rules that were fixed before the evaluation. No single score is built: each
quantity is reported on its own, with its numerator and denominator.

The two rules that decide are A (SELF: the model says its sources are sufficient) and C
(COMBINED: it says so, has at least one claim, and every claim passes layer 0). Rule B (layer 0
alone) is computed only as a reference and diagnostic; it decides nothing. On the negative cases
the metric is the *undesired answer rate on preregistered negative contexts*: how often a rule lets
an answer through where abstaining was the aim. It does not say that any answer is legally false.

Known evidence is what the DEV qrels judge relevant (grade 2 or 3). It is not the complete legal
evidence: a chunk nobody judged is neither relevant nor irrelevant, and it is counted apart as
`unjudged`, never as an error and never as demonstrated relevance.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ingest import answer_cases as ac
from ingest.answer_format import ParsedOutput, parse_output
from ingest.answer_verifier import (
    AnswerVerdict,
    Decisions,
    decide,
    normalize,
    verify_answer,
)

RULES = ("A", "B", "C")
DECISIVE_RULES = ("A", "C")
DIAGNOSTIC_RULES = ("B",)
RESCUE_THRESHOLD = 2
CASCADE_MAX_LAYER0_DROP = 0.05
HUMAN_SUPPORT_THRESHOLD = 0.90
GOLD_ANSWER_RATE_FLOOR = 0.80
STRUCTURAL_FAILURES_MATERIAL = 2  # of the 18 GOLD outputs (11 %)
LAYER0_FAILURE_MATERIAL = 0.10
HUMAN_CLAIMS = 40
HUMAN_PER_CONTEXT = 10
HUMAN_LABELS = ("SUPPORTED", "UNSUPPORTED", "UNCLEAR")
HUMAN_CONTEXTS = (ac.GOLD_KNOWN, ac.E5_10, ac.E_10, ac.E5_5)
HUMAN_SEED = "s7-s8-human-v1"


@dataclass(frozen=True)
class Outcome:
    case: ac.Case
    parsed: ParsedOutput
    verdict: AnswerVerdict | None
    decisions: Decisions


def make_outcome(case: ac.Case, raw_output: str, texts: Mapping[str, str]) -> Outcome:
    """Parse one raw output, verify it against the context it was given, apply the three rules."""
    parsed = parse_output(raw_output)
    verdict = (
        verify_answer(parsed.answer, case.chunk_ids, texts) if parsed.answer is not None else None
    )
    return Outcome(case, parsed, verdict, decide(parsed, verdict))


def answered(outcome: Outcome, rule: str) -> bool:
    d = outcome.decisions
    return {"A": d.a_self_only, "B": d.b_verifier_only, "C": d.c_combined}[rule]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a proportion; descriptive, never a gate."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def _share(k: int, n: int) -> dict[str, float | int]:
    low, high = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "rate": k / n if n else float("nan"),
        "wilson_low": low,
        "wilson_high": high,
    }


# ---------------------------------------------------------------------------
# positives
# ---------------------------------------------------------------------------


def cited_instances(outcome: Outcome) -> list[str]:
    """Every cited chunk id of every claim, with repeats."""
    if outcome.parsed.answer is None:
        return []
    return [c.chunk_id for claim in outcome.parsed.answer.claims for c in claim.citations]


def known_hit(outcome: Outcome, grades: Mapping[str, int]) -> bool:
    """The output cites at least one chunk the DEV qrels judge relevant (grade 2 or 3)."""
    return any(grades.get(c, -1) >= 2 for c in cited_instances(outcome))


def known_coverage(outcome: Outcome, grades: Mapping[str, int]) -> float:
    """The share of the known relevant chunks that the output cites. Known evidence coverage,
    not complete legal evidence."""
    relevant = ac.known_relevant(grades)
    return len(relevant.keys() & set(cited_instances(outcome))) / len(relevant) if relevant else 0.0


def positive_summary(
    outcomes: Sequence[Outcome], grades: Mapping[str, Mapping[str, int]], rule: str = "C"
) -> dict[str, object]:
    """The CORE metrics of one context over the positive needs, under one rule."""
    n = len(outcomes)
    shown = [o for o in outcomes if answered(o, rule)]
    claims = [v for o in outcomes if o.verdict for v in o.verdict.claims]
    citations = [c for v in claims for c in v.citations]
    instances_shown = [(c, o.case.need_id) for o in shown for c in cited_instances(o)]
    instances_all = [(c, o.case.need_id) for o in outcomes for c in cited_instances(o)]
    return {
        "rule": rule,
        "cases": n,
        "answer_rate": _share(len(shown), n),
        "structural_failures": _share(sum(1 for o in outcomes if not o.parsed.usable), n),
        "soft_violations": _count_soft(outcomes),
        "verified_claim_rate": _share(sum(v.passed for v in claims), len(claims)),
        "verified_citation_rate": _share(sum(c.passed for c in citations), len(citations)),
        "non_literal_citations_literal_once_case_and_glyphs_are_folded": _share(
            sum(c.near_literal for c in citations), sum(not c.passed for c in citations)
        ),
        "known_relevant_citation_hit": _share(
            sum(known_hit(o, grades[o.case.need_id]) for o in shown), len(shown)
        ),
        "known_evidence_coverage_mean": (
            sum(known_coverage(o, grades[o.case.need_id]) for o in shown) / len(shown)
            if shown
            else float("nan")
        ),
        "unjudged_citation_rate_answered": _share(
            sum(1 for c, need in instances_shown if c not in grades[need]), len(instances_shown)
        ),
        "unjudged_citation_rate_all_emitted": _share(
            sum(1 for c, need in instances_all if c not in grades[need]), len(instances_all)
        ),
    }


def _count_soft(outcomes: Sequence[Outcome]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for o in outcomes:
        for flag in o.parsed.soft:
            counts[flag] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# negatives, FAST, QUALITY and the cascade
# ---------------------------------------------------------------------------


def _by_context(outcomes: Sequence[Outcome]) -> dict[tuple[str, str | None], dict[str, Outcome]]:
    grouped: dict[tuple[str, str | None], dict[str, Outcome]] = defaultdict(dict)
    for o in outcomes:
        grouped[(o.case.need_id, o.case.source_need_id)][o.case.context_name] = o
    return grouped


def fast_quality(
    outcomes: Sequence[Outcome],
) -> dict[tuple[str, str | None], tuple[Outcome, Outcome]]:
    """For each question, the FAST output (E5@10) and the QUALITY output (E@10). A case with a
    single context (insufficient context) serves both policies."""
    pairs = {}
    for key, contexts in _by_context(outcomes).items():
        single = contexts.get(ac.INSUFFICIENT)
        fast = single or contexts.get(ac.E5_10)
        quality = single or contexts.get(ac.E_10)
        if fast is not None and quality is not None:
            pairs[key] = (fast, quality)
    return pairs


def policy_answered(fast: Outcome, quality: Outcome, policy: str, rule: str = "C") -> bool:
    """FAST uses E5's output, QUALITY uses E's, CASCADE uses E5's and, only when the rule abstains
    on it, E's output that was already generated."""
    if policy == "FAST":
        return answered(fast, rule)
    if policy == "QUALITY":
        return answered(quality, rule)
    if policy == "CASCADE":
        return answered(fast, rule) or answered(quality, rule)
    raise ValueError(policy)


def negative_summary(outcomes: Sequence[Outcome]) -> dict[str, object]:
    """Undesired answer rate on the preregistered negative contexts: for every policy and rule, how
    many negative cases the rule lets through, in total (the 15) and for each kind (natural,
    constructed insufficient context, mismatched context). 0 of 15 under the combined rule is the
    engineering aim of this exploratory bank, not a universal claim of safety."""
    pairs = fast_quality(outcomes)
    kinds = sorted({o.case.kind for o in outcomes})
    sizes = {kind: sum(1 for f, _ in pairs.values() if f.case.kind == kind) for kind in kinds}
    result: dict[str, object] = {"cases": len(pairs), "cases_by_kind": sizes}
    for policy in ("FAST", "QUALITY", "CASCADE"):
        for rule in RULES:
            hits = {k: policy_answered(f, q, policy, rule) for k, (f, q) in pairs.items()}
            by_kind = {
                kind: {
                    "undesired_answers": sum(
                        h for k, h in hits.items() if pairs[k][0].case.kind == kind
                    ),
                    "of": sizes[kind],
                }
                for kind in kinds
            }
            result[f"{policy}/{rule}"] = {
                "undesired_answers": sum(hits.values()),
                "of": len(hits),
                "wilson_95": wilson(sum(hits.values()), len(hits)),
                "by_kind": by_kind,
                "diagnostic_only": rule in DIAGNOSTIC_RULES,
            }
    return result


def cascade_rescues(
    positives: Sequence[Outcome], grades: Mapping[str, Mapping[str, int]]
) -> dict[str, object]:
    """A rescue is a need where the combined rule abstains on E5@10 and lets E@10 through with an
    answer that cites known relevant evidence. The broader count (E5@10 abstains or answers
    without citing known relevant evidence) is descriptive."""
    pairs = fast_quality([o for o in positives if o.case.context_name in (ac.E5_10, ac.E_10)])
    rescued, broader, regressions = [], [], []
    for (need, _), (fast, quality) in sorted(pairs.items()):
        g = grades[need]
        fast_ok = answered(fast, "C") and known_hit(fast, g)
        quality_ok = answered(quality, "C") and known_hit(quality, g)
        if not answered(fast, "C") and quality_ok:
            rescued.append(need)
        if not fast_ok and quality_ok:
            broader.append(need)
        if fast_ok and not quality_ok:
            regressions.append(need)
    return {
        "rescues": rescued,
        "rescue_count": len(rescued),
        "rescue_threshold": RESCUE_THRESHOLD,
        "descriptive_broader_count": len(broader),
        "descriptive_regressions": regressions,
    }


def layer0_claim_pass_rate(outcomes: Sequence[Outcome]) -> float:
    claims = [v for o in outcomes if o.verdict for v in o.verdict.claims]
    return sum(v.passed for v in claims) / len(claims) if claims else float("nan")


def cascade_decision(
    positives: Sequence[Outcome],
    negatives: Sequence[Outcome],
    grades: Mapping[str, Mapping[str, int]],
    human_global_support: float | None = None,
    human_by_context: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """The preregistered rule for exploratory evidence in favour of an automatic cascade. E must
    rescue at least RESCUE_THRESHOLD positive cases; the cascade must let through no negative case
    that FAST does not; the share of E's claims that pass layer 0 must not fall more than 5 points
    below E5@10's; and the global human support (SUPPORTED among SUPPORTED + UNSUPPORTED, over the
    40 claims) must reach the guardrail of 90 %. The support of E's claims against E5@10's is
    reported, not decided on. With fewer rescues there is no automatic cascade and E stays a
    manual or on-demand QUALITY PATH."""
    rescues = cascade_rescues(positives, grades)
    negs = negative_summary(negatives)
    fast_hits = negs["FAST/C"]["undesired_answers"]  # type: ignore[index]
    cascade_hits = negs["CASCADE/C"]["undesired_answers"]  # type: ignore[index]
    e5 = [o for o in positives if o.case.context_name == ac.E5_10]
    e = [o for o in positives if o.case.context_name == ac.E_10]
    drop = layer0_claim_pass_rate(e5) - layer0_claim_pass_rate(e)
    criteria = {
        "enough_rescues": rescues["rescue_count"] >= RESCUE_THRESHOLD,  # type: ignore[operator]
        "no_extra_undesired_answer": cascade_hits <= fast_hits,
        "layer0_not_degraded": (drop <= CASCADE_MAX_LAYER0_DROP) if drop == drop else False,
        "human_support_meets_guardrail": (
            human_global_support >= HUMAN_SUPPORT_THRESHOLD
            if human_global_support is not None
            else None
        ),
    }
    decided = all(v is True for v in criteria.values())
    return {
        **rescues,
        "fast_undesired_answers": fast_hits,
        "cascade_undesired_answers": cascade_hits,
        "layer0_pass_rate_drop": drop,
        "criteria": criteria,
        "automatic_cascade_justified": decided,
        "unresolved": [k for k, v in criteria.items() if v is None],
        "human_support_by_context_descriptive": dict(human_by_context or {}),
    }


def gold_trigger(
    gold: Sequence[Outcome], human_support_rate: float | None = None
) -> dict[str, object]:
    """Whether the 4B with GOLD_KNOWN shows that the generator may be the bottleneck. If any
    condition holds, STOP and decide whether to try the 8B; it never runs by itself."""
    n = len(gold)
    rate = sum(answered(o, "C") for o in gold) / n if n else float("nan")
    structural = sum(1 for o in gold if not o.parsed.usable)
    failure = 1 - layer0_claim_pass_rate(gold)
    conditions = {
        "combined_answer_rate_below_80": rate < GOLD_ANSWER_RATE_FLOOR,
        "material_structural_failures": structural >= STRUCTURAL_FAILURES_MATERIAL,
        "material_layer0_claim_failures": failure > LAYER0_FAILURE_MATERIAL,
        "human_support_below_90": (
            human_support_rate < HUMAN_SUPPORT_THRESHOLD if human_support_rate is not None else None
        ),
    }
    triggered = any(v is True for v in conditions.values())
    pending = conditions["human_support_below_90"] is None
    status = (
        "8B_TRIGGERED_FOR_USER_DECISION"
        if triggered
        else "8B_NOT_TRIGGERED_ON_AUTOMATIC_CONDITIONS_HUMAN_SUPPORT_PENDING"
        if pending
        else "8B_NOT_TRIGGERED"
    )
    return {
        "combined_answer_rate": rate,
        "structural_failures": structural,
        "layer0_claim_failure_rate": failure,
        "conditions": conditions,
        "triggered": triggered,
        "status": status,
        "note": "The 8B is never run automatically: a trigger only asks for a decision.",
    }


# ---------------------------------------------------------------------------
# the human anchor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimRef:
    case_id: str
    context_name: str
    need_id: str
    index: int
    text: str
    citations: tuple[tuple[str, str], ...]  # (chunk_id, quote)

    @property
    def dedup_key(self) -> str:
        quotes = sorted((c, normalize(q)) for c, q in self.citations)
        return f"{self.need_id}|{normalize(self.text)}|{quotes}"


def eligible_claims(positives: Sequence[Outcome]) -> dict[str, list[ClaimRef]]:
    """Per context, the claims that passed layer 0 in the outputs where the model said its sources
    were sufficient. These are the claims the deterministic layer accepts: the question is how many
    of them the cited text really supports."""
    out: dict[str, list[ClaimRef]] = defaultdict(list)
    for o in positives:
        if o.parsed.answer is None or o.verdict is None or not o.parsed.answer.sufficiency:
            continue
        for i, (claim, verdict) in enumerate(
            zip(o.parsed.answer.claims, o.verdict.claims, strict=True)
        ):
            if verdict.passed:
                refs = tuple((c.chunk_id, c.quote) for c in claim.citations)
                out[o.case.context_name].append(
                    ClaimRef(
                        o.case.case_id, o.case.context_name, o.case.need_id, i, claim.text, refs
                    )
                )
    return dict(out)


def _rank(seed: str, ref: ClaimRef) -> str:
    return hashlib.sha256(f"{seed}|{ref.case_id}|{ref.index}".encode()).hexdigest()


def _round_robin(pool: Sequence[ClaimRef], seed: str, wanted: int) -> list[ClaimRef]:
    """One claim per need in turn, needs and claims ordered by hash, until `wanted` or empty."""
    by_need: dict[str, list[ClaimRef]] = defaultdict(list)
    for ref in sorted(pool, key=lambda r: _rank(seed, r)):
        by_need[ref.need_id].append(ref)
    needs = sorted(by_need, key=lambda n: hashlib.sha256(f"{seed}|need|{n}".encode()).hexdigest())
    chosen: list[ClaimRef] = []
    depth = 0
    while len(chosen) < wanted and any(depth < len(v) for v in by_need.values()):
        for need in needs:
            if depth < len(by_need[need]) and len(chosen) < wanted:
                chosen.append(by_need[need][depth])
        depth += 1
    return chosen


def sample_human_claims(
    eligible: Mapping[str, Sequence[ClaimRef]],
    seed: str = HUMAN_SEED,
    per_context: int = HUMAN_PER_CONTEXT,
    total: int = HUMAN_CLAIMS,
) -> list[ClaimRef]:
    """10 claims per context (GOLD_KNOWN, E5@10, E@10, E5@5), one per need in turn, an identical
    claim only once across contexts. A context with fewer than 10 lets the others fill up to 40,
    in the same context order. Fixed before the claims are seen."""
    chosen: list[ClaimRef] = []
    taken: set[str] = set()
    remaining: dict[str, list[ClaimRef]] = {}
    for context in HUMAN_CONTEXTS:
        pool = [r for r in eligible.get(context, ()) if r.dedup_key not in taken]
        picked = _round_robin(pool, f"{seed}|{context}", per_context)
        chosen += picked
        taken.update(r.dedup_key for r in picked)
        remaining[context] = [r for r in pool if r not in picked]
    for context in HUMAN_CONTEXTS:
        if len(chosen) >= total:
            break
        pool = [r for r in remaining[context] if r.dedup_key not in taken]
        picked = _round_robin(pool, f"{seed}|fill|{context}", total - len(chosen))
        chosen += picked
        taken.update(r.dedup_key for r in picked)
    return chosen


def build_packet(
    sample: Sequence[ClaimRef],
    eligible: Mapping[str, Sequence[ClaimRef]],
    texts: Mapping[str, str],
    seed: str = HUMAN_SEED,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """The items to label, shuffled and without any mention of the context or the system, and
    the key that maps each item to its case. An identical claim from another context shares the
    label."""
    order = sorted(
        sample, key=lambda r: hashlib.sha256(f"{seed}|shuffle|{r.dedup_key}".encode()).hexdigest()
    )
    packet: list[dict[str, object]] = []
    key: list[dict[str, object]] = []
    for n, ref in enumerate(order, start=1):
        item_id = f"H{n:02d}"
        packet.append(
            {
                "item": item_id,
                "claim": ref.text,
                "citations": [
                    {"chunk_id": c, "quote": q, "chunk_text": texts[c]} for c, q in ref.citations
                ],
            }
        )
        also = sorted(
            {
                r.context_name
                for pool in eligible.values()
                for r in pool
                if r.dedup_key == ref.dedup_key
            }
        )
        key.append(
            {
                "item": item_id,
                "case_id": ref.case_id,
                "context": ref.context_name,
                "need_id": ref.need_id,
                "same_claim_in_contexts": also,
            }
        )
    return packet, key


def score_human(
    labels: Mapping[str, str], key: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Counts of the human labels, the support rate excluding UNCLEAR, the number of UNCLEAR, and
    the same by context (an identical claim counts for every context it appears in)."""
    unknown = sorted(set(labels) - {str(k["item"]) for k in key})
    missing = sorted({str(k["item"]) for k in key} - set(labels))
    bad = sorted(i for i, label in labels.items() if label not in HUMAN_LABELS)
    if unknown or missing or bad:
        raise ValueError(
            f"labels do not match the packet: unknown {unknown}, missing {missing}, bad {bad}"
        )

    def tally(items: Sequence[str]) -> dict[str, object]:
        counts = {label: sum(labels[i] == label for i in items) for label in HUMAN_LABELS}
        evaluable = counts["SUPPORTED"] + counts["UNSUPPORTED"]
        return {
            **counts,
            "items": len(items),
            "support_rate_excluding_unclear": counts["SUPPORTED"] / evaluable
            if evaluable
            else float("nan"),
            "wilson": wilson(counts["SUPPORTED"], evaluable),
        }

    overall = tally([str(k["item"]) for k in key])
    by_context = {
        context: tally(
            [
                str(k["item"])
                for k in key
                if context == k["context"] or context in k["same_claim_in_contexts"]  # type: ignore[operator]
            ]
        )
        for context in HUMAN_CONTEXTS
    }
    rate = overall["support_rate_excluding_unclear"]
    return {
        "overall": overall,
        "by_context": by_context,
        "meets_90_percent": bool(rate >= HUMAN_SUPPORT_THRESHOLD) if rate == rate else False,  # type: ignore[operator]
    }


# ---------------------------------------------------------------------------
# the human check at the level of the answer
# ---------------------------------------------------------------------------

ANSWER_LABELS = ("ANSWERED", "PARTIAL", "NOT_ANSWERED", "UNCLEAR")
ANSWER_SEED = "s7-s8-answers-v1"


@dataclass(frozen=True)
class AnswerItem:
    case_id: str
    context_name: str
    need_id: str
    kind: str  # primary (E5@10 answered under the combined rule) | rescue_check (E answered)


def select_answer_items(positives: Sequence[Outcome]) -> tuple[list[AnswerItem], list[str]]:
    """One item for each positive need whose E5@10 output is let through by the combined rule, and,
    where E5@10 abstains and E@10 is let through, that E answer as a `rescue_check` item. The needs
    where nothing is answered have nothing to label and are returned apart."""
    pairs = fast_quality([o for o in positives if o.case.context_name in (ac.E5_10, ac.E_10)])
    items: list[AnswerItem] = []
    abstained: list[str] = []
    for (need, _), (fast, quality) in sorted(pairs.items()):
        if answered(fast, "C"):
            items.append(AnswerItem(fast.case.case_id, fast.case.context_name, need, "primary"))
        elif answered(quality, "C"):
            items.append(
                AnswerItem(quality.case.case_id, quality.case.context_name, need, "rescue_check")
            )
        else:
            abstained.append(need)
    return items, abstained


def build_answer_packet(
    items: Sequence[AnswerItem],
    outcomes: Mapping[str, Outcome],
    texts: Mapping[str, str],
    seed: str = ANSWER_SEED,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """The blind packet (question, the claims with their citation ids, the text of the cited
    chunks) and the key. Order shuffled; no context or system is named."""
    order = sorted(items, key=lambda i: hashlib.sha256(f"{seed}|{i.case_id}".encode()).hexdigest())
    packet: list[dict[str, object]] = []
    key: list[dict[str, object]] = []
    for n, item in enumerate(order, start=1):
        outcome = outcomes[item.case_id]
        answer = outcome.parsed.answer
        assert answer is not None
        cited = sorted({c.chunk_id for claim in answer.claims for c in claim.citations})
        packet.append(
            {
                "item": f"A{n:02d}",
                "question": outcome.case.question,
                "claims": [
                    {"text": claim.text, "cites": [c.chunk_id for c in claim.citations]}
                    for claim in answer.claims
                ],
                "cited_texts": {c: texts[c] for c in cited},
            }
        )
        key.append(
            {
                "item": f"A{n:02d}",
                "case_id": item.case_id,
                "context": item.context_name,
                "need_id": item.need_id,
                "kind": item.kind,
            }
        )
    return packet, key


def score_answers(
    labels: Mapping[str, str], key: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Counts of the answer-level labels overall and for the primary and rescue-check items."""
    items = {str(k["item"]) for k in key}
    unknown, missing = sorted(set(labels) - items), sorted(items - set(labels))
    bad = sorted(i for i, label in labels.items() if label not in ANSWER_LABELS)
    if unknown or missing or bad:
        raise ValueError(f"labels do not match the packet: {unknown} {missing} {bad}")

    def tally(selected: Sequence[str]) -> dict[str, object]:
        counts = {label: sum(labels[i] == label for i in selected) for label in ANSWER_LABELS}
        return {
            **counts,
            "items": len(selected),
            "answered_share": counts["ANSWERED"] / len(selected) if selected else float("nan"),
            "answered_or_partial_share": (
                (counts["ANSWERED"] + counts["PARTIAL"]) / len(selected)
                if selected
                else float("nan")
            ),
        }

    return {
        "overall": tally(sorted(items)),
        "primary_E5_at_10": tally([str(k["item"]) for k in key if k["kind"] == "primary"]),
        "rescue_check_E_at_10": tally([str(k["item"]) for k in key if k["kind"] == "rescue_check"]),
    }
