"""The automatic report and the human files of S7-S8, on synthetic outputs."""

from __future__ import annotations

import json

import pytest

from ingest import answer_cases as ac
from ingest import answer_metrics as am
from ingest import answer_report as rep

TEXTS = {
    "K:1": "Financial entities shall notify the supervisory authority within 72 hours of becoming "
    "aware of a major incident.",
    "K:2": "Providers must be listed in the register of information kept by each entity.",
}
GOOD_QUOTE = "notify the supervisory authority within 72 hours of becoming aware"
ABSTAIN = json.dumps({"sufficiency": False, "claims": [], "missing": "The sources do not say."})
WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
    "oscar papa quebec romeo"
).split()
PREFIX = {ac.GOLD_KNOWN: "gold", ac.E5_10: "fast", ac.E_10: "quality", ac.E5_5: "short"}


def _answer(text: str) -> str:
    claim = {"text": text, "citations": [{"chunk_id": "K:1", "quote": GOOD_QUOTE}]}
    return json.dumps({"sufficiency": True, "claims": [claim], "missing": ""})


def _case(need: str, context: str, kind: str = ac.POSITIVE, source: str | None = None) -> ac.Case:
    return ac.Case(
        f"{kind}|{need}|{context}",
        kind,
        need,
        f"{need}-en-1",
        "Q?",
        context,
        ("K:1", "K:2"),
        source,
    )


def _world() -> tuple[list[am.Outcome], dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    """18 positive needs in four contexts (E5@10 abstains in two, E@10 rescues them) and 15
    negatives that are all refused."""
    outcomes: list[am.Outcome] = []
    grades: dict[str, dict[str, int]] = {}
    for i in range(18):
        need = f"N{i + 4:03d}"
        grades[need] = {"K:1": 3, "K:2": 1}
        for context in ac.CONTEXTS:
            fast_abstains = context == ac.E5_10 and i < 2
            raw = (
                ABSTAIN
                if fast_abstains
                else _answer(f"Entities notify {PREFIX[context]} {WORDS[i]}.")
            )
            outcomes.append(am.make_outcome(_case(need, context), raw, TEXTS))
    for i in range(3):
        need = f"N{i + 30:03d}"
        for ctx in (ac.E5_10, ac.E_10):
            outcomes.append(am.make_outcome(_case(need, ctx, ac.NEG_NATURAL), ABSTAIN, TEXTS))
    for i in range(6):
        need = f"N{i + 40:03d}"
        outcomes.append(
            am.make_outcome(_case(need, ac.INSUFFICIENT, ac.NEG_INSUFFICIENT), ABSTAIN, TEXTS)
        )
        for ctx in (ac.E5_10, ac.E_10):
            outcomes.append(
                am.make_outcome(_case(need, ctx, ac.NEG_MISMATCHED, "N099"), ABSTAIN, TEXTS)
            )
    records = {
        o.case.case_id: {
            "seconds": 20.0 + i,
            "prompt_tokens": 1000,
            "new_tokens": 100,
            "hit_token_limit": False,
        }
        for i, o in enumerate(outcomes)
    }
    return outcomes, grades, records


def test_nan_becomes_null_and_tuples_become_lists() -> None:
    assert rep.clean({"a": float("nan"), "b": (1, 2), "c": [float("nan")]}) == {
        "a": None,
        "b": [1, 2],
        "c": [None],
    }


def test_the_analysis_has_every_preregistered_section_and_is_plain_json() -> None:
    outcomes, grades, records = _world()
    analysis = rep.analyse(outcomes, grades, records)
    json.dumps(analysis, allow_nan=False)  # strict JSON: no NaN left
    assert analysis["cases"] == {"positive": 72, "negative": 24, "total": 96}
    assert set(analysis["positives_by_context"]) == set(ac.CONTEXTS)
    assert analysis["decisive_rules"] == ["A", "C"] and analysis["diagnostic_rules"] == ["B"]
    assert analysis["cascade"]["rescue_count"] == 2
    assert analysis["cascade"]["automatic_cascade_justified"] is False  # human guardrail pending
    assert analysis["cascade"]["unresolved"] == ["human_support_meets_guardrail"]
    assert analysis["negatives"]["CASCADE/C"]["undesired_answers"] == 0
    assert analysis["negatives"]["CASCADE/C"]["of"] == 15
    assert analysis["gold_trigger_8B"]["status"].startswith("8B_NOT_TRIGGERED")
    assert analysis["policies_on_positives"]["CASCADE/C"]["answered"]["k"] == 18
    assert analysis["policies_on_positives"]["FAST/C"]["answered"]["k"] == 16
    assert analysis["generation"]["positive/E5@10"]["cases"] == 18


def test_the_analysis_uses_the_human_scores_when_they_exist() -> None:
    outcomes, grades, records = _world()
    claims = {
        "overall": {"support_rate_excluding_unclear": 0.95},
        "by_context": {c: {"support_rate_excluding_unclear": 0.95} for c in am.HUMAN_CONTEXTS},
        "meets_90_percent": True,
    }
    analysis = rep.analyse(outcomes, grades, records, claims, None)
    assert analysis["cascade"]["automatic_cascade_justified"] is True
    assert analysis["gold_trigger_8B"]["status"] == "8B_NOT_TRIGGERED"


def test_e5_at_5_against_e5_at_10_counts_what_is_kept_lost_and_gained() -> None:
    outcomes, grades, _ = _world()
    positives = [o for o in outcomes if o.case.kind == ac.POSITIVE]
    transitions = rep.context_comparison(positives, grades)["need_transitions_combined_rule"]
    assert transitions == {
        "answered_at_both": 16,
        "answered_only_at_E5@10": 0,
        "answered_only_at_E5@5": 2,
        "answered_at_neither": 0,
    }


def test_the_report_names_the_rules_the_aim_and_the_non_claims() -> None:
    outcomes, grades, records = _world()
    text = rep.render_report(rep.analyse(outcomes, grades, records), {"prompt sha256": "x" * 64})
    assert "not legal entailment" in text and "explored on DEV" in text
    assert "Undesired answer rate on the preregistered negative contexts" in text
    assert "constructed insufficient context (6)" in text
    assert "(diagnostic)" in text and "8B trigger (never run automatically)" in text
    assert "0 of 15" in text and "not a claim of safety" in text


def test_the_packets_are_readable_and_carry_no_system_name() -> None:
    outcomes, _, _ = _world()
    positives = [o for o in outcomes if o.case.kind == ac.POSITIVE]
    eligible = am.eligible_claims(positives)
    sample = am.sample_human_claims(eligible)
    packet, _ = am.build_packet(sample, eligible, TEXTS)
    markdown = rep.render_claims_packet(packet)
    assert "## H01" in markdown and "SUPPORTED / UNSUPPORTED / UNCLEAR" in markdown
    assert "E5@10" not in markdown and "GOLD" not in markdown
    items, _ = am.select_answer_items(positives)
    answers, _ = am.build_answer_packet(items, {o.case.case_id: o for o in outcomes}, TEXTS)
    text = rep.render_answers_packet(answers)
    assert "## A01" in text and "ANSWERED / PARTIAL / NOT_ANSWERED / UNCLEAR" in text
    assert "E@10" not in text


def test_label_files_are_written_and_read_without_guessing() -> None:
    packet = [{"item": "H01"}, {"item": "H02"}]
    template = rep.labels_template(packet)
    assert template == "item,label\nH01,\nH02,\n"
    with pytest.raises(ValueError, match="no label"):
        rep.read_labels(template)
    assert rep.read_labels("item,label\nH01,supported\nH02, UNCLEAR \n") == {
        "H01": "SUPPORTED",
        "H02": "UNCLEAR",
    }
    with pytest.raises(ValueError, match="twice"):
        rep.read_labels("item,label\nH01,SUPPORTED\nH01,UNCLEAR\n")


def test_a_multi_line_chunk_stays_inside_its_block_quote() -> None:
    assert rep._quoted("one\n\ntwo") == "> one\n>\n> two"
    packet = [
        {
            "item": "H01",
            "claim": "C.",
            "citations": [{"chunk_id": "K:1", "quote": "q", "chunk_text": "a\n\nb"}],
        }
    ]
    assert "> a\n>\n> b" in rep.render_claims_packet(packet)
    answers = [
        {
            "item": "A01",
            "question": "Q?",
            "claims": [{"text": "T.", "cites": ["K:1"]}],
            "cited_texts": {"K:1": "x\ny"},
        }
    ]
    assert "> x\n> y" in rep.render_answers_packet(answers)
