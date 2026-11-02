"""The automatic report of S7-S8 and the files that the human labels are written in.

Pure functions of outputs that were already produced: they build the analysis exactly as the
preregistration fixed it (no metric is added after the results), render it, and render the two
blind packets and the label templates for the human anchor. Reading is separate (the script).
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ingest import answer_cases as ac
from ingest import answer_metrics as am
from ingest.cost_protocol import nearest_rank_percentile

NEGATIVE_KINDS = (ac.NEG_NATURAL, ac.NEG_INSUFFICIENT, ac.NEG_MISMATCHED)
POLICIES = ("FAST", "QUALITY", "CASCADE")


def clean(value: Any) -> Any:
    """Turn NaN into None and tuples into lists, so that the analysis is plain JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def latency_summary(
    records: Mapping[str, Mapping[str, Any]], outcomes: Sequence[am.Outcome]
) -> dict[str, Any]:
    """Generation time and tokens per context and kind, `measured on M1` in whatever conditions
    the machine was in: a description of this run, not a measurement protocol."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for o in outcomes:
        groups.setdefault(f"{o.case.kind}/{o.case.context_name}", []).append(
            records[o.case.case_id]
        )
    out: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        seconds = [float(r["seconds"]) for r in rows]
        out[name] = {
            "cases": len(rows),
            "seconds_p50": nearest_rank_percentile(seconds, 50),
            "seconds_p95": nearest_rank_percentile(seconds, 95),
            "seconds_total": sum(seconds),
            "prompt_tokens_mean": sum(r["prompt_tokens"] for r in rows) / len(rows),
            "new_tokens_mean": sum(r["new_tokens"] for r in rows) / len(rows),
            "hit_token_limit": sum(bool(r["hit_token_limit"]) for r in rows),
        }
    return out


def _policy_rows(
    positives: Sequence[am.Outcome], grades: Mapping[str, Mapping[str, int]]
) -> dict[str, Any]:
    pairs = am.fast_quality([o for o in positives if o.case.context_name in (ac.E5_10, ac.E_10)])
    rows: dict[str, Any] = {}
    for policy in POLICIES:
        for rule in am.DECISIVE_RULES:
            used = []
            for (need, _), (fast, quality) in sorted(pairs.items()):
                if policy == "FAST":
                    out = fast
                elif policy == "QUALITY":
                    out = quality
                else:
                    out = fast if am.answered(fast, rule) else quality
                used.append((need, out, am.answered(out, rule)))
            n = len(used)
            answered = [(need, o) for need, o, ok in used if ok]
            hits = sum(am.known_hit(o, grades[need]) for need, o in answered)
            rows[f"{policy}/{rule}"] = {
                "answered": am._share(len(answered), n),
                "answered_citing_known_relevant_evidence": am._share(hits, n),
            }
    return rows


def context_comparison(
    positives: Sequence[am.Outcome],
    grades: Mapping[str, Mapping[str, int]],
    small: str = ac.E5_5,
    large: str = ac.E5_10,
) -> dict[str, Any]:
    """E5@5 against E5@10 under the combined rule: the summaries side by side and, need by need,
    where the answer was kept, lost or gained. Descriptive: it decides nothing by itself."""
    by = {(o.case.need_id, o.case.context_name): o for o in positives}
    needs = sorted({n for n, _ in by})
    kept = lost = gained = neither = 0
    lost_known_hit: list[str] = []
    for need in needs:
        a, b = by[(need, small)], by[(need, large)]
        sa, sb = am.answered(a, "C"), am.answered(b, "C")
        if sa and sb:
            kept += 1
        elif sb and not sa:
            lost += 1
        elif sa and not sb:
            gained += 1
        else:
            neither += 1
        if sb and am.known_hit(b, grades[need]) and not (sa and am.known_hit(a, grades[need])):
            lost_known_hit.append(need)
    return {
        small: am.positive_summary([o for o in positives if o.case.context_name == small], grades),
        large: am.positive_summary([o for o in positives if o.case.context_name == large], grades),
        "need_transitions_combined_rule": {
            "answered_at_both": kept,
            f"answered_only_at_{large}": lost,
            f"answered_only_at_{small}": gained,
            "answered_at_neither": neither,
        },
        f"needs_where_{small}_loses_known_relevant_citation": lost_known_hit,
    }


def analyse(
    outcomes: Sequence[am.Outcome],
    grades: Mapping[str, Mapping[str, int]],
    records: Mapping[str, Mapping[str, Any]],
    human_claims: Mapping[str, Any] | None = None,
    human_answers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Every preregistered quantity, from the outputs, the qrels grades and (when they exist) the
    scores of the human labels."""
    positives = [o for o in outcomes if o.case.kind == ac.POSITIVE]
    negatives = [o for o in outcomes if o.case.kind in NEGATIVE_KINDS]
    per_context = {
        context: {
            rule: am.positive_summary(
                [o for o in positives if o.case.context_name == context], grades, rule
            )
            for rule in am.RULES
        }
        for context in ac.CONTEXTS
    }
    gold = [o for o in positives if o.case.context_name == ac.GOLD_KNOWN]
    overall_support = None
    by_context_support: dict[str, float] | None = None
    if human_claims is not None:
        overall_support = human_claims["overall"]["support_rate_excluding_unclear"]
        by_context_support = {
            c: v["support_rate_excluding_unclear"] for c, v in human_claims["by_context"].items()
        }
    errors: dict[str, dict[str, int]] = {}
    for o in outcomes:
        bucket = errors.setdefault(f"{o.case.kind}/{o.case.context_name}", {})
        key = o.parsed.error or "usable"
        bucket[key] = bucket.get(key, 0) + 1
    failure_codes: dict[str, int] = {}
    for o in outcomes:
        if o.verdict:
            for claim in o.verdict.claims:
                for code in claim.all_failures():
                    failure_codes[code] = failure_codes.get(code, 0) + 1
    result: dict[str, Any] = clean(
        {
            "cases": {
                "positive": len(positives),
                "negative": len(negatives),
                "total": len(outcomes),
            },
            "decisive_rules": list(am.DECISIVE_RULES),
            "diagnostic_rules": list(am.DIAGNOSTIC_RULES),
            "positives_by_context": per_context,
            "policies_on_positives": _policy_rows(positives, grades),
            "e5_at_5_against_e5_at_10": context_comparison(positives, grades),
            "negatives": am.negative_summary(negatives),
            "cascade": am.cascade_decision(
                positives, negatives, grades, overall_support, by_context_support
            ),
            "gold_trigger_8B": am.gold_trigger(gold, overall_support),
            "parse_outcomes_by_kind_and_context": errors,
            "layer0_failure_codes": failure_codes,
            "generation": latency_summary(records, outcomes),
            "human_claims": human_claims,
            "human_answers": human_answers,
        }
    )
    return result


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _pct(share: Mapping[str, Any]) -> str:
    if not share or share["n"] == 0:
        return "n/a"
    return f"{share['k']}/{share['n']} ({100 * share['rate']:.0f} %)"


def render_report(analysis: Mapping[str, Any], header: Mapping[str, str]) -> str:
    lines = ["# S7-S8 evaluation: automatic report", ""]
    lines += [f"- {k}: `{v}`" for k, v in header.items()]
    lines += [
        "",
        "`explored on DEV, not independently confirmed`. The layer-0 verifier proves citation "
        "provenance and integrity, not legal entailment. Latencies: measured on M1, a "
        "description of this run.",
        "",
    ]
    lines += [
        "## Positives, by context (18 needs each)",
        "",
        "| context | rule | answer rate | structural failures | verified claims | verified "
        "citations | known relevant citation hit | known evidence coverage (mean) | "
        "unjudged citations (answered) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for context, rules in analysis["positives_by_context"].items():
        for rule in am.RULES:
            s = rules[rule]
            tag = f"{rule}" + (" (diagnostic)" if rule in am.DIAGNOSTIC_RULES else "")
            cov = s["known_evidence_coverage_mean"]
            lines.append(
                f"| {context} | {tag} | {_pct(s['answer_rate'])} | "
                f"{_pct(s['structural_failures'])} | "
                f"{_pct(s['verified_claim_rate'])} | {_pct(s['verified_citation_rate'])} | "
                f"{_pct(s['known_relevant_citation_hit'])} | "
                f"{'n/a' if cov is None else f'{cov:.2f}'} | "
                f"{_pct(s['unjudged_citation_rate_answered'])} |"
            )
    lines += [
        "",
        "Rule A = SELF (the model says its sources suffice). Rule C = COMBINED (and at "
        "least one claim, and every claim passes layer 0). Rule B (layer 0 alone) is a "
        "reference and decides nothing.",
        "",
    ]
    negs = analysis["negatives"]
    lines += [
        "## Undesired answer rate on the preregistered negative contexts",
        "",
        "| policy / rule | all 15 | natural (3) | constructed insufficient context (6) | "
        "mismatched context (6) |",
        "|---|---|---|---|---|",
    ]
    for key, value in negs.items():
        if "/" not in key:
            continue
        kinds = value["by_kind"]
        cells = [
            f"{kinds[k]['undesired_answers']}/{kinds[k]['of']}" if k in kinds else "n/a"
            for k in NEGATIVE_KINDS
        ]
        tag = key + (" (diagnostic)" if value["diagnostic_only"] else "")
        lines.append(
            f"| {tag} | {value['undesired_answers']}/{value['of']} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "Aim for the combined rule: 0 of 15, an engineering criterion on this exploratory "
        "bank. It is not a claim of safety.",
        "",
    ]
    lines += [
        "## FAST, QUALITY and CASCADE on the positives",
        "",
        "| policy / rule | answered | answered and cites known relevant evidence |",
        "|---|---|---|",
    ]
    for key, value in analysis["policies_on_positives"].items():
        lines.append(
            f"| {key} | {_pct(value['answered'])} | "
            f"{_pct(value['answered_citing_known_relevant_evidence'])} |"
        )
    c = analysis["cascade"]
    lines += [
        "",
        "## Cascade (rescues)",
        "",
        f"- rescues: {c['rescue_count']} (threshold {c['rescue_threshold']}): {c['rescues']}",
        f"- descriptive broader count: {c['descriptive_broader_count']}; "
        f"descriptive regressions: {c['descriptive_regressions']}",
        f"- undesired answers on negatives, FAST/C {c['fast_undesired_answers']}, "
        f"CASCADE/C {c['cascade_undesired_answers']}",
        f"- layer-0 pass rate of claims, E5@10 minus E@10: {c['layer0_pass_rate_drop']}",
        f"- criteria: {c['criteria']}",
        f"- automatic cascade justified (needs the human guardrail): "
        f"**{c['automatic_cascade_justified']}**; unresolved: {c['unresolved']}",
        "",
    ]
    t = analysis["e5_at_5_against_e5_at_10"]["need_transitions_combined_rule"]
    lines += [
        "## E5@5 against E5@10 (combined rule)",
        "",
        f"- need transitions: {t}",
        f"- needs where E5@5 loses the known relevant citation: "
        f"{analysis['e5_at_5_against_e5_at_10']['needs_where_E5@5_loses_known_relevant_citation']}",
        "",
    ]
    g = analysis["gold_trigger_8B"]
    lines += [
        "## 8B trigger (never run automatically)",
        "",
        f"- **{g['status']}**",
        f"- GOLD_KNOWN combined answer rate {g['combined_answer_rate']}, structural failures "
        f"{g['structural_failures']}, layer-0 claim failure rate {g['layer0_claim_failure_rate']}",
        f"- conditions: {g['conditions']}",
        "",
    ]
    lines += [
        "## Structure and generation",
        "",
        f"- parse outcomes: {analysis['parse_outcomes_by_kind_and_context']}",
        f"- layer-0 failure codes: {analysis['layer0_failure_codes']}",
        "",
        "| group | cases | seconds p50 | p95 | prompt tokens | new tokens | hit token limit |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, v in analysis["generation"].items():
        lines.append(
            f"| {name} | {v['cases']} | {v['seconds_p50']:.1f} | {v['seconds_p95']:.1f} | "
            f"{v['prompt_tokens_mean']:.0f} | {v['new_tokens_mean']:.0f} | {v['hit_token_limit']} |"
        )
    if analysis.get("human_claims"):
        h = analysis["human_claims"]["overall"]
        lines += [
            "",
            "## Human anchor: claims",
            "",
            f"- {h}",
            f"- meets 90 %: {analysis['human_claims']['meets_90_percent']}",
        ]
    if analysis.get("human_answers"):
        lines += ["", "## Human check: answers", "", f"- {analysis['human_answers']}"]
    return "\n".join(lines) + "\n"


def _quoted(text: str) -> str:
    """A Markdown block quote that keeps every line of a multi-line text inside the quote."""
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.splitlines()) or ">"


def render_claims_packet(packet: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Ancla humana: 40 afirmaciones",
        "",
        "Para cada afirmación, lee el texto citado y responde SOLO a esto: "
        "**¿El texto citado sostiene realmente la afirmación?** No tienes que decidir si es "
        "jurídicamente correcta. Etiqueta en `claims-labels.csv`.",
        "",
    ]
    for item in packet:
        lines += [f"## {item['item']}", "", f"**Afirmación:** {item['claim']}", ""]
        for cite in item["citations"]:
            lines += [
                f"**Cita literal** (fragmento `{cite['chunk_id']}`):",
                "",
                _quoted(cite["quote"]),
                "",
                f"Texto completo del fragmento `{cite['chunk_id']}`:",
                "",
                _quoted(cite["chunk_text"]),
                "",
            ]
        lines += ["Etiqueta: SUPPORTED / UNSUPPORTED / UNCLEAR", "", "---", ""]
    return "\n".join(lines)


def render_answers_packet(packet: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Evaluación humana de respuestas",
        "",
        "Para cada elemento, dada la pregunta y **únicamente** las citas y textos mostrados, "
        "responde: **¿la respuesta aborda realmente lo que se pregunta?** No es una "
        "evaluación jurídica. Etiqueta en `answers-labels.csv`.",
        "",
    ]
    for item in packet:
        lines += [
            f"## {item['item']}",
            "",
            f"**Pregunta:** {item['question']}",
            "",
            "**Respuesta del sistema:**",
            "",
        ]
        for n, claim in enumerate(item["claims"], start=1):
            cites = ", ".join(f"`{c}`" for c in claim["cites"])
            lines.append(f"{n}. {claim['text']} [{cites}]")
        lines += ["", "**Textos citados:**", ""]
        for chunk_id, text in item["cited_texts"].items():
            lines += [f"`{chunk_id}`:", "", _quoted(text), ""]
        lines += ["Etiqueta: ANSWERED / PARTIAL / NOT_ANSWERED / UNCLEAR", "", "---", ""]
    return "\n".join(lines)


def labels_template(packet: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["item", "label"])
    for item in packet:
        writer.writerow([item["item"], ""])
    return buffer.getvalue()


def read_labels(text: str) -> dict[str, str]:
    """The labels of a filled CSV. A blank or repeated item is an error: nothing is guessed."""
    labels: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(text)):
        item, label = row["item"].strip(), (row.get("label") or "").strip().upper()
        if item in labels:
            raise ValueError(f"item {item} appears twice")
        if not label:
            raise ValueError(f"item {item} has no label")
        labels[item] = label
    return labels
