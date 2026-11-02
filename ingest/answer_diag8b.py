"""The 8B capability diagnostic of S7-S8, as functions of outputs that were already produced.

The question is narrow: is the main bottleneck the capacity or the obedience of the 4B generator
even when the known relevant evidence is already in its context? The diagnostic runs a larger model
on the 18 evaluation needs with the GOLD_KNOWN context only, under exactly the same prompt, schema,
verifier and decoding as the 4B run, and compares it with the stored 4B GOLD_KNOWN outputs. It is a
capability diagnostic, not a new campaign.

Nothing here runs a model or reads a file except `write_audit_inputs`, which writes the blind
packet. The decision rule was fixed before any 8B output existed:

1. the combined answer rate on GOLD_KNOWN is at least 80 %;
2. fewer than 2 of the 18 outputs are structural failures;
3. the verified citation rate is at least 10 points above the 4B's (the materiality band of the
   preregistration);
4. no new systematic failure: no type of failure that never occurs in the 4B GOLD_KNOWN outputs
   appears in 3 or more of the 18 outputs of the 8B.

All four must hold for `8B_DIAGNOSTIC_PASSED_READY_FOR_BLIND_AUDIT`; otherwise the conclusion is
`8B_DIAGNOSTIC_FAILED_STOP`. Passing the rule is not a result about the answers: it only allows a
small blind audit of their claims, which is still a cross-model audit and not a human validation.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ingest import answer_cases as ac
from ingest import answer_metrics as am
from ingest import answer_report as rep

SEED = "s7-s8-8b-v1"
CONTEXT = ac.GOLD_KNOWN
ANSWER_RATE_FLOOR = am.GOLD_ANSWER_RATE_FLOOR  # 0.80
STRUCTURAL_FAILURES_LIMIT = am.STRUCTURAL_FAILURES_MATERIAL  # fewer than 2 of the 18
INTEGRITY_MARGIN = 0.10
SYSTEMATIC_MIN_OUTPUTS = 3
AUDIT_CLAIMS = 40
PASSED = "8B_DIAGNOSTIC_PASSED_READY_FOR_BLIND_AUDIT"
FAILED = "8B_DIAGNOSTIC_FAILED_STOP"


def failure_types(outcome: am.Outcome, record: Mapping[str, Any]) -> set[str]:
    """Every type of failure of one output, once each: a structural error of the parser, a soft
    violation of the format, words around the JSON, a cut at the token limit and each failure
    code of layer 0 on any claim or citation."""
    parsed = outcome.parsed
    types: set[str] = set()
    if parsed.error:
        types.add(f"structural:{parsed.error}")
    types.update(f"soft:{flag}" for flag in parsed.soft)
    if parsed.usable and parsed.extra_text:
        types.add("extra_text_around_json")
    if record["hit_token_limit"]:
        types.add("hit_token_limit")
    if outcome.verdict is not None:
        for claim in outcome.verdict.claims:
            types.update(f"layer0:{code}" for code in claim.all_failures())
    return types


def summary(
    outcomes: Sequence[am.Outcome],
    records: Mapping[str, Mapping[str, Any]],
    grades: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """The preregistered metrics of one model on the GOLD_KNOWN context, nothing else."""
    claims = [v for o in outcomes if o.verdict for v in o.verdict.claims]
    codes = Counter(code for claim in claims for code in claim.all_failures())
    per_output = Counter(t for o in outcomes for t in failure_types(o, records[o.case.case_id]))
    latency = rep.latency_summary(records, outcomes)
    return {
        "cases": len(outcomes),
        "SELF": am.positive_summary(outcomes, grades, "A"),
        "COMBINED": am.positive_summary(outcomes, grades, "C"),
        "claims_emitted": len(claims),
        "layer0_failure_codes": dict(sorted(codes.items())),
        "failure_types_by_outputs": dict(sorted(per_output.items())),
        "latency": latency[f"{ac.POSITIVE}/{CONTEXT}"],
    }


def judge(four: Mapping[str, Any], eight: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen rule to the 4B and 8B summaries."""
    answer = eight["COMBINED"]["answer_rate"]
    structural = eight["COMBINED"]["structural_failures"]
    rate8 = eight["COMBINED"]["verified_citation_rate"]
    rate4 = four["COMBINED"]["verified_citation_rate"]
    improvement = (
        rate8["rate"] - rate4["rate"] if rate8["n"] and rate4["n"] else float("nan")
    )  # NaN compares False below
    four_types = four["failure_types_by_outputs"]
    new_systematic = sorted(
        t
        for t, n in eight["failure_types_by_outputs"].items()
        if n >= SYSTEMATIC_MIN_OUTPUTS and four_types.get(t, 0) == 0
    )
    criteria = {
        "combined_answer_rate_at_least_80_percent": bool(
            answer["n"] and answer["k"] / answer["n"] >= ANSWER_RATE_FLOOR
        ),
        "fewer_than_2_structural_failures": structural["k"] < STRUCTURAL_FAILURES_LIMIT,
        "verified_citation_rate_10_points_above_4b": bool(improvement >= INTEGRITY_MARGIN - 1e-9),
        "no_new_systematic_failure": not new_systematic,
    }
    passed = all(criteria.values())
    return {
        "criteria": criteria,
        "verified_citation_rate_improvement": improvement,
        "new_systematic_failure_types": new_systematic,
        "stop_8b_on_criteria_1_or_2": not (
            criteria["combined_answer_rate_at_least_80_percent"]
            and criteria["fewer_than_2_structural_failures"]
        ),
        "conclusion": PASSED if passed else FAILED,
    }


def _share(s: Mapping[str, Any]) -> str:
    return f"{s['k']}/{s['n']} ({s['rate']:.0%})" if s["n"] else "0/0"


def render(four: Mapping[str, Any], eight: Mapping[str, Any], verdict: Mapping[str, Any]) -> str:
    """The table of the 4B and the 8B on GOLD_KNOWN, the failure types and the decision."""

    def row(label: str, a: str, b: str) -> str:
        return f"| {label} | {a} | {b} |"

    lat4, lat8 = four["latency"], eight["latency"]
    lines = [
        "# S7-S8: 8B capability diagnostic on GOLD_KNOWN",
        "",
        "`explored on DEV, not independently confirmed`. Same prompt, schema, verifier, decoding, "
        "questions and contexts; only the model differs. The verifier proves citation provenance "
        "and integrity, not semantic support. Latencies: measured on M1, a description of each "
        "run, not a protocol.",
        "",
        "| GOLD_KNOWN, 18 evaluation needs | 4B | 8B |",
        "|---|---|---|",
        row(
            "SELF answer rate",
            _share(four["SELF"]["answer_rate"]),
            _share(eight["SELF"]["answer_rate"]),
        ),
        row(
            "Combined answer rate",
            _share(four["COMBINED"]["answer_rate"]),
            _share(eight["COMBINED"]["answer_rate"]),
        ),
        row(
            "structural failures",
            _share(four["COMBINED"]["structural_failures"]),
            _share(eight["COMBINED"]["structural_failures"]),
        ),
        row(
            "verified citation rate",
            _share(four["COMBINED"]["verified_citation_rate"]),
            _share(eight["COMBINED"]["verified_citation_rate"]),
        ),
        row(
            "verified claim rate",
            _share(four["COMBINED"]["verified_claim_rate"]),
            _share(eight["COMBINED"]["verified_claim_rate"]),
        ),
        row("claims emitted", str(four["claims_emitted"]), str(eight["claims_emitted"])),
        row(
            "known relevant citation hit (Combined)",
            _share(four["COMBINED"]["known_relevant_citation_hit"]),
            _share(eight["COMBINED"]["known_relevant_citation_hit"]),
        ),
        row(
            "known evidence coverage, mean (Combined)",
            f"{four['COMBINED']['known_evidence_coverage_mean']:.2f}",
            f"{eight['COMBINED']['known_evidence_coverage_mean']:.2f}",
        ),
        row(
            "generation seconds, p50 / p95",
            f"{lat4['seconds_p50']:.1f} / {lat4['seconds_p95']:.1f}",
            f"{lat8['seconds_p50']:.1f} / {lat8['seconds_p95']:.1f}",
        ),
        row(
            "new tokens, mean; cut at the limit",
            f"{lat4['new_tokens_mean']:.0f}; {lat4['hit_token_limit']}",
            f"{lat8['new_tokens_mean']:.0f}; {lat8['hit_token_limit']}",
        ),
        "",
        f"- layer-0 failure codes on claims, 4B: {four['layer0_failure_codes']}",
        f"- layer-0 failure codes on claims, 8B: {eight['layer0_failure_codes']}",
        f"- failure types by number of outputs, 4B: {four['failure_types_by_outputs']}",
        f"- failure types by number of outputs, 8B: {eight['failure_types_by_outputs']}",
        "",
        "## Frozen rule",
        "",
    ]
    lines += [f"- {name}: **{ok}**" for name, ok in verdict["criteria"].items()]
    lines += [
        f"- verified citation rate, 8B minus 4B: {verdict['verified_citation_rate_improvement']}",
        f"- new systematic failure types: {verdict['new_systematic_failure_types']}",
        "",
        f"**{verdict['conclusion']}**",
        "",
    ]
    return "\n".join(lines)


def write_audit_inputs(
    outcomes: Sequence[am.Outcome],
    texts: Mapping[str, str],
    folder: Path,
    key_folder: Path,
) -> dict[str, Any]:
    """The blind packet of the audit of the 8B claims: the claims that passed layer 0 in outputs
    that declared their sources sufficient, all of them if there are at most 40, otherwise 40 chosen
    by the fixed seed, one per need in turn. The packet names no model. The key stays apart."""
    eligible = am.eligible_claims(outcomes)
    sample = am.sample_human_claims(
        eligible, seed=SEED, per_context=AUDIT_CLAIMS, total=AUDIT_CLAIMS
    )
    packet, key = am.build_packet(sample, eligible, texts, seed=SEED)
    # the renderer titles its packet with a fixed count of 40: only the number is changed
    title, _, rest = rep.render_claims_packet(packet).partition("\n")
    if not re.search(r"\d+", title):
        raise ValueError("the packet renderer changed: its title carries no count")
    text = re.sub(r"\d+", str(len(packet)), title, count=1) + "\n" + rest
    folder.mkdir(parents=True, exist_ok=True)
    key_folder.mkdir(parents=True, exist_ok=True)
    files = {
        folder / "claims-packet.md": text,
        folder / "claims-labels.csv": rep.labels_template(packet),
        key_folder / "claims-key.json": json.dumps(key, indent=1, sort_keys=True) + "\n",
    }
    for path, content in files.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"{path} exists with other content: results are never overwritten")
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            path.chmod(0o444)
    return {
        "eligible_claims": sum(len(v) for v in eligible.values()),
        "items": len(packet),
        "all_eligible_claims_included": len(packet) == sum(len(v) for v in eligible.values()),
    }
