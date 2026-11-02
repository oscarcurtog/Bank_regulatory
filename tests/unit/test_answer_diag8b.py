"""The 8B diagnostic of S7-S8: failure types, the frozen rule and the blind packet, on synthetic
outputs."""

from __future__ import annotations

import importlib
import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pytest

from ingest import answer_cases as ac
from ingest import answer_diag8b as d8
from ingest import answer_metrics as am

TEXTS = {
    "K:1": "Financial entities shall notify the supervisory authority within 72 hours of becoming "
    "aware of a major incident.",
    "K:2": "Providers must be listed in the register of information kept by each entity.",
}
GRADES = {f"N{i + 4:03d}": {"K:1": 3, "K:2": 1} for i in range(18)}
GOOD_QUOTE = "notify the supervisory authority within 72 hours of becoming aware"
BAD_QUOTE = "notify the supervisory authority within 99 hours of becoming aware"
ABSTAIN = json.dumps({"sufficiency": False, "claims": [], "missing": "The sources do not say."})
WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
    "oscar papa quebec romeo"
).split()


def _raw(text: str, quote: str = GOOD_QUOTE) -> str:
    claim = {"text": text, "citations": [{"chunk_id": "K:1", "quote": quote}]}
    return json.dumps({"sufficiency": True, "claims": [claim], "missing": ""})


def _case(need: str) -> ac.Case:
    return ac.Case(
        f"positive|{need}|{ac.GOLD_KNOWN}",
        ac.POSITIVE,
        need,
        f"{need}-en-1",
        "Q?",
        ac.GOLD_KNOWN,
        ("K:1", "K:2"),
        None,
    )


def _world(
    raws: list[str], limit: Collection[int] = ()
) -> tuple[list[am.Outcome], dict[str, dict[str, Any]]]:
    outcomes = [am.make_outcome(_case(f"N{i + 4:03d}"), raw, TEXTS) for i, raw in enumerate(raws)]
    records = {
        o.case.case_id: {
            "seconds": 10.0 + i,
            "prompt_tokens": 1000,
            "new_tokens": 100,
            "hit_token_limit": i in limit,
        }
        for i, o in enumerate(outcomes)
    }
    return outcomes, records


def _good(n: int = 18) -> list[str]:
    return [_raw(f"Entities notify {WORDS[i]}.") for i in range(n)]


def _summary(raws: list[str], limit: Collection[int] = ()) -> dict[str, Any]:
    outcomes, records = _world(raws, limit)
    return d8.summary(outcomes, records, GRADES)


def test_failure_types_name_each_kind_of_failure_once_per_output() -> None:
    outcomes, records = _world(
        [
            "not json at all",
            json.dumps({"sufficiency": 3}),
            _raw("Entities notify alpha.", BAD_QUOTE),
            _raw("Entities notify bravo."),
        ],
        limit={3},
    )
    types = [d8.failure_types(o, records[o.case.case_id]) for o in outcomes]
    assert types[0] == {"structural:no_json_object"}
    assert types[1] == {"structural:schema_invalid"}
    assert types[2] == {"layer0:quote_not_literal"}
    assert types[3] == {"hit_token_limit"}


def test_the_summary_holds_the_preregistered_metrics_only() -> None:
    s = _summary(_good(16) + [ABSTAIN, "garbage"])
    assert s["cases"] == 18
    assert s["COMBINED"]["answer_rate"]["k"] == 16
    assert s["SELF"]["answer_rate"]["k"] == 16
    assert s["COMBINED"]["structural_failures"]["k"] == 1
    assert s["claims_emitted"] == 16
    assert s["failure_types_by_outputs"] == {"structural:no_json_object": 1}
    assert s["latency"]["cases"] == 18


def test_the_rule_passes_only_when_all_four_criteria_hold() -> None:
    four = _summary(_good(8) + [_raw("Entities notify z.", BAD_QUOTE)] * 10)
    eight = _summary(_good(16) + [ABSTAIN, ABSTAIN])
    verdict = d8.judge(four, eight)
    assert verdict["criteria"] == {
        "combined_answer_rate_at_least_80_percent": True,
        "fewer_than_2_structural_failures": True,
        "verified_citation_rate_10_points_above_4b": True,
        "no_new_systematic_failure": True,
    }
    assert verdict["conclusion"] == d8.PASSED == "8B_DIAGNOSTIC_PASSED_READY_FOR_BLIND_AUDIT"
    assert verdict["stop_8b_on_criteria_1_or_2"] is False


def test_an_answer_rate_below_80_percent_stops() -> None:
    four = _summary(_good(8) + [ABSTAIN] * 10)
    at_14 = d8.judge(four, _summary(_good(14) + [ABSTAIN] * 4))
    at_15 = d8.judge(four, _summary(_good(15) + [ABSTAIN] * 3))
    assert at_14["criteria"]["combined_answer_rate_at_least_80_percent"] is False
    assert at_14["conclusion"] == d8.FAILED == "8B_DIAGNOSTIC_FAILED_STOP"
    assert at_14["stop_8b_on_criteria_1_or_2"] is True
    assert at_15["criteria"]["combined_answer_rate_at_least_80_percent"] is True


def test_two_structural_failures_stop_even_with_a_high_answer_rate() -> None:
    four = _summary(_good(8) + [ABSTAIN] * 10)
    eight = _summary(_good(16) + ["garbage", "garbage"])
    verdict = d8.judge(four, eight)
    assert verdict["criteria"]["combined_answer_rate_at_least_80_percent"] is True
    assert verdict["criteria"]["fewer_than_2_structural_failures"] is False
    assert verdict["conclusion"] == d8.FAILED
    one = d8.judge(four, _summary(_good(16) + ["garbage", ABSTAIN]))
    assert one["criteria"]["fewer_than_2_structural_failures"] is True


def test_no_clear_improvement_of_citation_integrity_does_not_pass() -> None:
    same = _summary(_good(15) + [_raw("Entities notify z.", BAD_QUOTE)] * 3)
    verdict = d8.judge(same, same)
    assert verdict["criteria"]["verified_citation_rate_10_points_above_4b"] is False
    assert verdict["conclusion"] == d8.FAILED
    nothing = d8.judge(same, _summary([ABSTAIN] * 18))
    assert nothing["criteria"]["verified_citation_rate_10_points_above_4b"] is False  # no claims


def test_a_new_failure_type_in_three_outputs_is_systematic_but_two_or_a_known_one_is_not() -> None:
    four = _summary(_good(18))
    three = _good(15) + [_raw("Entities notify z.", BAD_QUOTE)] * 3
    two = _good(16) + [_raw("Entities notify z.", BAD_QUOTE)] * 2
    new_three = d8.judge(four, _summary(three))
    new_two = d8.judge(four, _summary(two))
    assert new_three["new_systematic_failure_types"] == ["layer0:quote_not_literal"]
    assert new_three["criteria"]["no_new_systematic_failure"] is False
    assert new_two["criteria"]["no_new_systematic_failure"] is True
    known = _summary(_good(15) + [_raw("Entities notify z.", BAD_QUOTE)] * 3)
    assert d8.judge(known, _summary(three))["criteria"]["no_new_systematic_failure"] is True


def test_cut_outputs_are_a_failure_type_too() -> None:
    four = _summary(_good(18))
    cut = _summary(_good(18), limit={0, 1, 2})
    verdict = d8.judge(four, cut)
    assert verdict["new_systematic_failure_types"] == ["hit_token_limit"]
    assert verdict["conclusion"] == d8.FAILED


def test_the_report_puts_both_models_side_by_side_with_the_decision() -> None:
    four = _summary(_good(8) + [_raw("Entities notify z.", BAD_QUOTE)] * 10)
    eight = _summary(_good(16) + [ABSTAIN, ABSTAIN])
    verdict = d8.judge(four, eight)
    text = d8.render(four, eight, verdict)
    assert "| Combined answer rate | 8/18 (44%) | 16/18 (89%) |" in text
    assert "| SELF answer rate | 18/18 (100%) | 16/18 (89%) |" in text
    assert "not independently confirmed" in text
    assert text.rstrip().endswith("**8B_DIAGNOSTIC_PASSED_READY_FOR_BLIND_AUDIT**")


def test_the_blind_packet_names_no_model_and_is_deterministic(tmp_path: Path) -> None:
    outcomes, _ = _world(_good(16) + [ABSTAIN, ABSTAIN])
    first = d8.write_audit_inputs(outcomes, TEXTS, tmp_path / "a", tmp_path / "ka")
    second = d8.write_audit_inputs(outcomes, TEXTS, tmp_path / "b", tmp_path / "kb")
    assert (
        first
        == second
        == {
            "eligible_claims": 16,
            "items": 16,
            "all_eligible_claims_included": True,
        }
    )
    packet = (tmp_path / "a" / "claims-packet.md").read_text(encoding="utf-8")
    first_line = packet.splitlines()[0]
    assert "16" in first_line and "40" not in first_line
    assert (tmp_path / "a" / "claims-packet.md").read_bytes() == (
        tmp_path / "b" / "claims-packet.md"
    ).read_bytes()
    lowered = packet.lower()
    assert not any(word in lowered for word in ("8b", "4b", "qwen", "gold", "e5", "fast"))
    labels = (tmp_path / "a" / "claims-labels.csv").read_text(encoding="utf-8").splitlines()
    assert labels[0] == "item,label" and len(labels) == 17
    key = json.loads((tmp_path / "ka" / "claims-key.json").read_text(encoding="utf-8"))
    assert len(key) == 16 and {k["context"] for k in key} == {ac.GOLD_KNOWN}


def test_more_than_forty_eligible_claims_are_sampled_down_to_forty(tmp_path: Path) -> None:
    claims = [
        {"text": f"Entities notify {w}.", "citations": [{"chunk_id": "K:1", "quote": GOOD_QUOTE}]}
        for w in WORDS[:6]
    ]
    big = json.dumps({"sufficiency": True, "claims": claims, "missing": ""})
    outcomes, _ = _world([big] * 18)  # 108 eligible claims, 6 per need
    info = d8.write_audit_inputs(outcomes, TEXTS, tmp_path / "a", tmp_path / "k")
    assert info["items"] == 40 and info["eligible_claims"] == 108
    assert info["all_eligible_claims_included"] is False


def test_the_packet_is_never_overwritten_with_other_content(tmp_path: Path) -> None:
    outcomes, _ = _world(_good(16) + [ABSTAIN, ABSTAIN])
    d8.write_audit_inputs(outcomes, TEXTS, tmp_path / "a", tmp_path / "k")
    other, _ = _world(_good(10) + [ABSTAIN] * 8)
    with pytest.raises(ValueError, match="never overwritten"):
        d8.write_audit_inputs(other, TEXTS, tmp_path / "a", tmp_path / "k")


def test_words_around_a_usable_json_are_a_failure_type_but_garbage_is_not_double_counted() -> None:
    wrapped = "Here is the answer:\n" + _raw("Entities notify alpha.")
    outcomes, records = _world([wrapped, "garbage"])
    types = [d8.failure_types(o, records[o.case.case_id]) for o in outcomes]
    assert types[0] == {"extra_text_around_json"}
    assert types[1] == {"structural:no_json_object"}


def test_the_weights_name_is_the_target_of_the_snapshot_link_even_if_the_blob_is_a_link_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend("scripts")  # scripts/ is no package
    module = importlib.import_module("s7_diag8b")
    cache = tmp_path / "hub"
    revision = "abc"
    root = cache / "models--mlx-community--Qwen3-8B-4bit"
    (root / "snapshots" / revision).mkdir(parents=True)
    (root / "blobs" / "09").mkdir(parents=True)
    (root / "blobs" / "09" / "stored-elsewhere").write_bytes(b"weights")
    (root / "blobs" / "content-hash").symlink_to(Path("09") / "stored-elsewhere")
    (root / "snapshots" / revision / "model.safetensors").symlink_to(
        Path("..") / ".." / "blobs" / "content-hash"
    )
    monkeypatch.setattr(module, "HF_CACHE", cache)
    assert module.weights_blob(revision) == "content-hash"
