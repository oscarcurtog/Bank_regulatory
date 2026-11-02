"""Metrics, cascade rule, 8B trigger and human anchor of S7-S8, on synthetic outputs."""

from __future__ import annotations

import json

import pytest

from ingest import answer_cases as ac
from ingest import answer_metrics as am

TEXTS = {
    "K:1": "Financial entities shall notify the supervisory authority within 72 hours of becoming "
    "aware of a major incident.",
    "K:2": "Providers must be listed in the register of information kept by each entity.",
    "U:1": "An unjudged chunk says that the entity should document its internal procedures well.",
}
GOOD_QUOTE = "notify the supervisory authority within 72 hours of becoming aware"
CONTEXT = ("K:1", "K:2", "U:1")


def grades() -> dict[str, dict[str, int]]:
    return {f"N{i:03d}": {"K:1": 3, "K:2": 1} for i in range(4, 22)}  # U:1 is unjudged


def _answer(
    text: str = "Entities must notify the authority.", chunk: str = "K:1", quote: str = GOOD_QUOTE
) -> str:
    claim = {"text": text, "citations": [{"chunk_id": chunk, "quote": quote}]}
    return json.dumps({"sufficiency": True, "claims": [claim], "missing": ""})


ABSTAIN = json.dumps({"sufficiency": False, "claims": [], "missing": "The sources do not say."})
BROKEN = '{"sufficiency": tru'
FABRICATED = _answer(quote="this sentence is nowhere in the supervisory sources")
UNJUDGED = _answer(
    "Entities should document procedures.", "U:1", "should document its internal procedures well"
)


def _case(need: str, context: str, kind: str = ac.POSITIVE, source: str | None = None) -> ac.Case:
    return ac.Case(
        f"{kind}|{need}|{context}", kind, need, f"{need}-en-1", "Q?", context, CONTEXT, source
    )


def _out(
    need: str, context: str, raw: str, kind: str = ac.POSITIVE, source: str | None = None
) -> am.Outcome:
    return am.make_outcome(_case(need, context, kind, source), raw, TEXTS)


def test_wilson_interval_of_zero_in_fifteen_and_of_nothing() -> None:
    low, high = am.wilson(0, 15)
    assert low == 0.0 and high == pytest.approx(3.8416 / 18.8416, abs=1e-4)
    assert am.wilson(0, 0) == (0.0, 1.0)
    low, high = am.wilson(36, 40)
    assert 0.76 < low < 0.78 and 0.95 < high < 0.97


def test_positive_summary_counts_each_quantity_with_its_denominator() -> None:
    g = grades()
    outcomes = [
        _out("N004", ac.E5_10, _answer()),  # answered, cites known relevant evidence
        _out("N005", ac.E5_10, UNJUDGED),  # answered, cites an unjudged chunk
        _out("N006", ac.E5_10, ABSTAIN),
        _out("N007", ac.E5_10, BROKEN),  # structural failure
        _out("N008", ac.E5_10, FABRICATED),  # a claim that fails layer 0: abstain under C
    ]
    s = am.positive_summary(outcomes, g)
    assert s["answer_rate"]["k"] == 2 and s["answer_rate"]["n"] == 5  # type: ignore[index]
    assert s["structural_failures"]["k"] == 1  # type: ignore[index]
    assert (s["verified_claim_rate"]["k"], s["verified_claim_rate"]["n"]) == (2, 3)  # type: ignore[index]
    assert s["known_relevant_citation_hit"]["k"] == 1 and s["known_relevant_citation_hit"]["n"] == 2  # type: ignore[index]
    assert s["known_evidence_coverage_mean"] == pytest.approx(0.5)  # K:1 of {K:1}: 1 and 0 -> 0.5
    assert s["unjudged_citation_rate_answered"]["k"] == 1  # type: ignore[index]
    assert s["unjudged_citation_rate_answered"]["n"] == 2  # type: ignore[index]
    b = am.positive_summary(outcomes, g, rule="B")
    assert b["answer_rate"]["k"] == 2  # type: ignore[index]


def test_known_coverage_is_the_share_of_known_relevant_chunks_cited() -> None:
    g = {"K:1": 3, "K:2": 2, "K:3": 1}
    both = {
        "sufficiency": True,
        "missing": "",
        "claims": [
            {"text": "A.", "citations": [{"chunk_id": "K:1", "quote": GOOD_QUOTE}]},
            {
                "text": "B.",
                "citations": [
                    {"chunk_id": "K:2", "quote": "must be listed in the register of information"}
                ],
            },
        ],
    }
    o = _out("N004", ac.GOLD_KNOWN, json.dumps(both))
    assert am.known_coverage(o, g) == 1.0
    assert am.known_coverage(_out("N004", ac.GOLD_KNOWN, _answer()), g) == 0.5


def _negatives(e5_answers: list[bool], e_answers: list[bool]) -> list[am.Outcome]:
    """Three natural and three mismatched items, E5 and E outputs; plus one insufficient item
    (single output) that abstains."""
    out: list[am.Outcome] = []
    items: list[tuple[str, str, str | None]] = [
        (ac.NEG_NATURAL, f"N{i:03d}", None) for i in range(4, 7)
    ]
    items += [(ac.NEG_MISMATCHED, f"N{i:03d}", f"N{i + 10:03d}") for i in range(7, 10)]
    for (kind, need, source), a5, a_e in zip(items, e5_answers, e_answers, strict=True):
        out.append(_out(need, ac.E5_10, _answer() if a5 else ABSTAIN, kind, source))
        out.append(_out(need, ac.E_10, _answer() if a_e else ABSTAIN, kind, source))
    out.append(_out("N020", ac.INSUFFICIENT, ABSTAIN, ac.NEG_INSUFFICIENT))
    return out


def test_negative_summary_counts_undesired_answers_per_rule_policy_and_kind() -> None:
    outcomes = _negatives(
        [False, False, True, False, False, False], [False, True, False, False, False, False]
    )
    s = am.negative_summary(outcomes)
    assert s["cases"] == 7
    assert s["cases_by_kind"] == {ac.NEG_INSUFFICIENT: 1, ac.NEG_MISMATCHED: 3, ac.NEG_NATURAL: 3}
    assert s["FAST/C"]["undesired_answers"] == 1 and s["QUALITY/C"]["undesired_answers"] == 1  # type: ignore[index]
    assert s["CASCADE/C"]["undesired_answers"] == 2  # type: ignore[index]
    assert s["FAST/C"]["by_kind"][ac.NEG_NATURAL] == {"undesired_answers": 1, "of": 3}  # type: ignore[index]
    assert s["FAST/C"]["by_kind"][ac.NEG_MISMATCHED]["undesired_answers"] == 0  # type: ignore[index]
    assert s["CASCADE/A"]["of"] == 7  # type: ignore[index]
    assert s["FAST/B"]["diagnostic_only"] is True and s["FAST/C"]["diagnostic_only"] is False  # type: ignore[index]
    assert am.DECISIVE_RULES == ("A", "C") and am.DIAGNOSTIC_RULES == ("B",)


def test_the_self_rule_answers_where_the_combined_rule_would_not() -> None:
    sufficient_but_fabricated = [
        _out("N004", ac.E5_10, FABRICATED, ac.NEG_NATURAL),
        _out("N004", ac.E_10, FABRICATED, ac.NEG_NATURAL),
    ]
    s = am.negative_summary(sufficient_but_fabricated)
    assert s["FAST/A"]["undesired_answers"] == 1  # type: ignore[index]
    assert s["FAST/B"]["undesired_answers"] == 0 and s["FAST/C"]["undesired_answers"] == 0  # type: ignore[index]


def _positives(
    e5_answers: list[bool], e_answers: list[bool], e_hits: list[bool] | None = None
) -> list[am.Outcome]:
    out: list[am.Outcome] = []
    for i, (a5, a_e) in enumerate(zip(e5_answers, e_answers, strict=True)):
        need = f"N{i + 4:03d}"
        out.append(_out(need, ac.E5_10, _answer() if a5 else ABSTAIN))
        hit = True if e_hits is None else e_hits[i]
        out.append(_out(need, ac.E_10, (_answer() if hit else UNJUDGED) if a_e else ABSTAIN))
    return out


def test_a_rescue_is_a_need_where_e5_abstains_and_e_answers_with_known_evidence() -> None:
    e5 = [False, False, False] + [True] * 15
    e = [True, True, False] + [True] * 15
    r = am.cascade_rescues(_positives(e5, e), grades())
    assert r["rescue_count"] == 2 and r["rescues"] == ["N004", "N005"]
    only_unjudged = am.cascade_rescues(_positives(e5, e, [False] + [True] * 17), grades())
    assert only_unjudged["rescue_count"] == 1  # N004's answer cites nothing known: not a rescue
    assert only_unjudged["descriptive_broader_count"] == 1


def test_the_cascade_is_justified_only_if_every_preregistered_criterion_holds() -> None:
    g = grades()
    positives = _positives([False, False] + [True] * 16, [True, True] + [True] * 16)
    clean = _negatives([False] * 6, [False] * 6)
    ok = am.cascade_decision(positives, clean, g, 0.93)
    assert ok["automatic_cascade_justified"] is True and ok["unresolved"] == []
    # without the human labels the decision is unresolved, not justified
    pending = am.cascade_decision(positives, clean, g)
    assert pending["unresolved"] == ["human_support_meets_guardrail"]
    assert pending["automatic_cascade_justified"] is False
    # the global human support must reach 90 %
    low = am.cascade_decision(positives, clean, g, 0.85)
    assert low["criteria"]["human_support_meets_guardrail"] is False  # type: ignore[index]
    # one rescue is not enough
    one = _positives([False] + [True] * 17, [True] * 18)
    assert am.cascade_decision(one, clean, g, 0.95)["automatic_cascade_justified"] is False
    # E letting through a negative case that E5 had refused
    worse = _negatives([False] * 6, [True] + [False] * 5)
    result = am.cascade_decision(positives, worse, g, 0.95)
    assert result["criteria"]["no_extra_undesired_answer"] is False  # type: ignore[index]
    # E's support against E5's is reported and decides nothing
    reported = am.cascade_decision(positives, clean, g, 0.95, {ac.E5_10: 0.9, ac.E_10: 0.7})
    assert reported["automatic_cascade_justified"] is True
    assert reported["human_support_by_context_descriptive"][ac.E_10] == 0.7  # type: ignore[index]


def test_the_cascade_does_not_pass_if_e_degrades_layer_0() -> None:
    g = grades()
    positives = _positives([False, False] + [True] * 16, [True, True] + [True] * 16)
    degraded = [o for o in positives if o.case.context_name == ac.E5_10]
    degraded += [
        _out(o.case.need_id, ac.E_10, FABRICATED if i % 2 else _answer())
        for i, o in enumerate(o for o in positives if o.case.context_name == ac.E_10)
    ]
    result = am.cascade_decision(degraded, _negatives([False] * 6, [False] * 6), g, 0.95)
    assert result["criteria"]["layer0_not_degraded"] is False  # type: ignore[index]


def test_the_gold_trigger_fires_on_any_of_its_conditions() -> None:
    fine = [_out(f"N{i:03d}", ac.GOLD_KNOWN, _answer()) for i in range(4, 22)]
    assert am.gold_trigger(fine, 0.95)["triggered"] is False
    assert am.gold_trigger(fine, 0.95)["status"] == "8B_NOT_TRIGGERED"
    pending = am.gold_trigger(fine)["status"]
    assert pending == "8B_NOT_TRIGGERED_ON_AUTOMATIC_CONDITIONS_HUMAN_SUPPORT_PENDING"
    low_rate = fine[:14] + [_out(f"N{i:03d}", ac.GOLD_KNOWN, ABSTAIN) for i in range(18, 22)]
    assert am.gold_trigger(low_rate)["status"] == "8B_TRIGGERED_FOR_USER_DECISION"
    assert am.gold_trigger(low_rate)["conditions"]["combined_answer_rate_below_80"] is True  # type: ignore[index]
    broken = fine[:16] + [_out("N020", ac.GOLD_KNOWN, BROKEN), _out("N021", ac.GOLD_KNOWN, BROKEN)]
    assert am.gold_trigger(broken)["conditions"]["material_structural_failures"] is True  # type: ignore[index]
    assert am.gold_trigger(fine, 0.8)["conditions"]["human_support_below_90"] is True  # type: ignore[index]
    assert am.gold_trigger(fine)["conditions"]["human_support_below_90"] is None  # type: ignore[index]


# ---------------------------------------------------------------------------
# the human anchor
# ---------------------------------------------------------------------------


LETTERS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()
PREFIX = {ac.GOLD_KNOWN: "gold", ac.E5_10: "fast", ac.E_10: "quality", ac.E5_5: "short"}


def _claim_output(words: str) -> str:
    return _answer(f"Entities must notify the authority {words}.")


def _eligible_world(per_context: int = 12, shared: int = 0) -> list[am.Outcome]:
    """Claims without digits, so that layer 0 accepts them; `shared` of them identical across
    contexts."""
    out = []
    for context in am.HUMAN_CONTEXTS:
        for i in range(per_context):
            need = f"N{i + 4:03d}"
            words = f"same {LETTERS[i]}" if i < shared else f"{PREFIX[context]} {LETTERS[i]}"
            out.append(_out(need, context, _claim_output(words)))
    return out


def test_the_sample_has_ten_claims_per_context_and_is_deterministic() -> None:
    eligible = am.eligible_claims(_eligible_world())
    sample = am.sample_human_claims(eligible)
    assert len(sample) == 40
    assert {
        c: sum(r.context_name == c for r in sample) for c in am.HUMAN_CONTEXTS
    } == dict.fromkeys(am.HUMAN_CONTEXTS, 10)
    assert sample == am.sample_human_claims(eligible)
    assert len({(r.case_id, r.index) for r in sample}) == 40
    first = [r.need_id for r in sample if r.context_name == ac.GOLD_KNOWN]
    assert len(set(first)) == 10  # one claim per need in turn


def test_only_claims_that_pass_layer_0_from_sufficient_outputs_are_eligible() -> None:
    outcomes = [
        _out("N004", ac.E5_10, _answer()),
        _out("N005", ac.E5_10, FABRICATED),
        _out("N006", ac.E5_10, ABSTAIN),
        _out("N007", ac.E5_10, BROKEN),
    ]
    assert [r.need_id for r in am.eligible_claims(outcomes)[ac.E5_10]] == ["N004"]


def test_an_identical_claim_is_sampled_once_across_contexts() -> None:
    eligible = am.eligible_claims(_eligible_world(shared=12))  # every context says the same claims
    sample = am.sample_human_claims(eligible)
    assert len({r.dedup_key for r in sample}) == len(sample) == 12


def test_a_short_context_lets_the_others_fill_up_to_forty() -> None:
    world = [
        o
        for o in _eligible_world()
        if not (o.case.context_name == ac.E5_5 and int(o.case.need_id[1:]) > 9)
    ]
    sample = am.sample_human_claims(am.eligible_claims(world))
    assert len(sample) == 40
    assert sum(r.context_name == ac.E5_5 for r in sample) == 6  # all it has
    assert sum(r.context_name == ac.GOLD_KNOWN for r in sample) > 10


def test_the_packet_hides_the_context_and_the_key_maps_every_item() -> None:
    eligible = am.eligible_claims(_eligible_world())
    sample = am.sample_human_claims(eligible)
    packet, key = am.build_packet(sample, eligible, TEXTS)
    assert [p["item"] for p in packet] == [f"H{i:02d}" for i in range(1, 41)]
    blob = json.dumps(packet)
    assert all(name not in blob for name in ("GOLD_KNOWN", "E5@10", "E@10", "E5@5"))
    assert {k["context"] for k in key} == set(am.HUMAN_CONTEXTS)
    assert packet[0]["citations"][0]["chunk_text"] == TEXTS["K:1"]  # type: ignore[index]


def test_the_human_labels_are_scored_with_unclear_apart() -> None:
    eligible = am.eligible_claims(_eligible_world())
    sample = am.sample_human_claims(eligible)
    _, key = am.build_packet(sample, eligible, TEXTS)
    labels = {str(k["item"]): "SUPPORTED" for k in key}
    for item in list(labels)[:2]:
        labels[item] = "UNSUPPORTED"
    for item in list(labels)[2:6]:
        labels[item] = "UNCLEAR"
    score = am.score_human(labels, key)
    overall = score["overall"]
    assert (overall["SUPPORTED"], overall["UNSUPPORTED"], overall["UNCLEAR"]) == (34, 2, 4)  # type: ignore[index]
    assert overall["support_rate_excluding_unclear"] == pytest.approx(34 / 36)  # type: ignore[index]
    assert score["meets_90_percent"] is True
    labels[str(key[10]["item"])] = "UNSUPPORTED"
    labels[str(key[11]["item"])] = "UNSUPPORTED"  # 32 supported of 36 evaluable: 88.9 %
    assert am.score_human(labels, key)["meets_90_percent"] is False


def test_labels_that_do_not_match_the_packet_are_refused() -> None:
    eligible = am.eligible_claims(_eligible_world())
    sample = am.sample_human_claims(eligible)
    _, key = am.build_packet(sample, eligible, TEXTS)
    labels = {str(k["item"]): "SUPPORTED" for k in key}
    with pytest.raises(ValueError):
        am.score_human({**labels, "H99": "SUPPORTED"}, key)
    with pytest.raises(ValueError):
        am.score_human({k: v for k, v in labels.items() if k != "H01"}, key)
    with pytest.raises(ValueError):
        am.score_human({**labels, "H01": "MAYBE"}, key)


# ---------------------------------------------------------------------------
# the human check at the level of the answer
# ---------------------------------------------------------------------------


def _answer_world() -> list[am.Outcome]:
    """Eight needs: E5 answers in five; in two E5 abstains and E answers (rescue checks); in one
    both abstain."""
    out = []
    for i in range(8):
        need = f"N{i + 4:03d}"
        e5_raw = (
            ABSTAIN if i in (5, 6, 7) else _answer(f"Entities notify the authority {LETTERS[i]}.")
        )
        e_raw = (
            ABSTAIN if i == 7 else _answer(f"Entities must notify the authority {LETTERS[i + 1]}.")
        )
        out.append(_out(need, ac.E5_10, e5_raw))
        out.append(_out(need, ac.E_10, e_raw))
    return out


def test_the_answer_items_are_the_e5_answers_and_the_rescued_e_answers() -> None:
    items, abstained = am.select_answer_items(_answer_world())
    assert [i.kind for i in items].count("primary") == 5
    assert [i.kind for i in items].count("rescue_check") == 2
    assert abstained == ["N011"]  # the eighth need: nothing was answered, nothing to label
    assert {i.context_name for i in items if i.kind == "primary"} == {ac.E5_10}
    assert {i.context_name for i in items if i.kind == "rescue_check"} == {ac.E_10}


def test_the_answer_packet_is_blind_shuffled_and_shows_only_cited_text() -> None:
    outcomes = _answer_world()
    items, _ = am.select_answer_items(outcomes)
    by_case = {o.case.case_id: o for o in outcomes}
    packet, key = am.build_answer_packet(items, by_case, TEXTS)
    assert [p["item"] for p in packet] == [f"A{n:02d}" for n in range(1, 8)]
    blob = json.dumps(packet)
    assert all(name not in blob for name in ("E5@10", "E@10", "GOLD", "rescue"))
    assert set(packet[0]["cited_texts"]) == {"K:1"}  # type: ignore[call-overload]
    assert packet[0]["question"] == "Q?"
    assert {k["kind"] for k in key} == {"primary", "rescue_check"}
    again = am.build_answer_packet(items, by_case, TEXTS)
    assert again == (packet, key)  # deterministic


def test_the_answer_labels_are_scored_overall_and_by_kind() -> None:
    outcomes = _answer_world()
    items, _ = am.select_answer_items(outcomes)
    packet, key = am.build_answer_packet(items, {o.case.case_id: o for o in outcomes}, TEXTS)
    labels = {str(k["item"]): "ANSWERED" for k in key}
    first = next(str(k["item"]) for k in key if k["kind"] == "primary")
    labels[first] = "PARTIAL"
    rescue = next(str(k["item"]) for k in key if k["kind"] == "rescue_check")
    labels[rescue] = "NOT_ANSWERED"
    score = am.score_answers(labels, key)
    assert score["overall"]["items"] == 7  # type: ignore[index]
    assert score["primary_E5_at_10"]["PARTIAL"] == 1  # type: ignore[index]
    assert score["primary_E5_at_10"]["answered_or_partial_share"] == 1.0  # type: ignore[index]
    assert score["rescue_check_E_at_10"]["NOT_ANSWERED"] == 1  # type: ignore[index]
    with pytest.raises(ValueError):
        am.score_answers({**labels, first: "MAYBE"}, key)
    with pytest.raises(ValueError):
        am.score_answers({k: v for k, v in labels.items() if k != first}, key)
