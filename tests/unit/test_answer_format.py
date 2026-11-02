"""The answer format of S7-S8: prompt, strict schema and parser, on synthetic text."""

from __future__ import annotations

import json
import re

from ingest import answer_format as af
from ingest import answer_verifier as av


def _valid() -> dict[str, object]:
    return {
        "sufficiency": True,
        "claims": [{"text": "A claim.", "citations": [{"chunk_id": "X:1", "quote": "a quote"}]}],
        "missing": "",
    }


def test_the_messages_carry_the_system_prompt_the_sources_and_the_question() -> None:
    messages = af.build_messages("What is due?", [("X:1", "First text."), ("X:2", "Second text.")])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == af.SYSTEM_PROMPT
    user = messages[1]["content"]
    assert '<source id="X:1">\nFirst text.\n</source>' in user
    assert user.index('id="X:1"') < user.index('id="X:2"') < user.index("Question: What is due?")


def test_the_prompt_hash_is_stable_and_changes_with_the_text() -> None:
    assert af.prompt_sha256() == af.prompt_sha256()
    assert re.fullmatch(r"[0-9a-f]{64}", af.prompt_sha256())
    original = af.SYSTEM_PROMPT
    try:
        af.SYSTEM_PROMPT = original + " "
        assert af.prompt_sha256() != _hash_of(original)
    finally:
        af.SYSTEM_PROMPT = original


def _hash_of(system: str) -> str:
    saved = af.SYSTEM_PROMPT
    af.SYSTEM_PROMPT = system
    try:
        return af.prompt_sha256()
    finally:
        af.SYSTEM_PROMPT = saved


def test_the_two_examples_of_the_prompt_obey_the_format_and_the_first_passes_layer_0() -> None:
    examples = re.findall(
        r'^\{"sufficiency": (?:true|false), "claims": \[.*$', af.SYSTEM_PROMPT, re.M
    )
    examples = [e for e in examples if "true or false" not in e]
    assert len(examples) == 2
    first, second = (af.parse_output(e) for e in examples)
    assert first.answer is not None and first.answer.sufficiency and not first.soft
    assert second.answer is not None and not second.answer.sufficiency and not second.soft
    verdict = av.verify_answer(first.answer, ["EXAMPLE:1"], af.EXAMPLE_SOURCES)
    assert verdict.all_pass


def test_a_clean_object_parses_without_flags() -> None:
    parsed = af.parse_output(json.dumps(_valid()))
    assert parsed.usable and parsed.error is None and not parsed.extra_text and not parsed.soft


def test_a_code_fence_or_words_around_the_object_are_tolerated_and_noted() -> None:
    fenced = "```json\n" + json.dumps(_valid()) + "\n```"
    assert af.parse_output(fenced).usable and not af.parse_output(fenced).extra_text
    chatty = "Here is the answer: " + json.dumps(_valid()) + " Hope it helps."
    parsed = af.parse_output(chatty)
    assert parsed.usable and parsed.extra_text


def test_structural_defects_make_the_output_unusable_with_a_reason() -> None:
    assert af.parse_output("").error == "empty_output"
    assert af.parse_output("no object here").error == "no_json_object"
    assert af.parse_output('{"sufficiency": true, "claims": [').error == "invalid_json"
    extra = {**_valid(), "confidence": 0.9}
    assert af.parse_output(json.dumps(extra)).error == "schema_invalid"
    assert (
        af.parse_output(json.dumps({**_valid(), "sufficiency": "true"})).error == "schema_invalid"
    )
    no_citations = {**_valid(), "claims": [{"text": "A claim."}]}
    assert af.parse_output(json.dumps(no_citations)).error == "schema_invalid"
    empty_claim = {**_valid(), "claims": [{"text": "", "citations": []}]}
    assert af.parse_output(json.dumps(empty_claim)).error == "schema_invalid"
    assert not af.parse_output("[1, 2]").usable


def test_soft_violations_are_recorded_and_the_output_is_still_read() -> None:
    insufficient_with_claims = {**_valid(), "sufficiency": False, "missing": "something"}
    parsed = af.parse_output(json.dumps(insufficient_with_claims))
    assert parsed.usable and parsed.soft == ("sufficiency_false_with_claims",)
    sufficient_without = {**_valid(), "claims": []}
    assert af.parse_output(json.dumps(sufficient_without)).soft == (
        "sufficiency_true_without_claims",
    )
    no_reason = {"sufficiency": False, "claims": [], "missing": "  "}
    assert af.parse_output(json.dumps(no_reason)).soft == ("insufficient_without_missing",)
    clean_abstention = {"sufficiency": False, "claims": [], "missing": "No evidence."}
    assert af.parse_output(json.dumps(clean_abstention)).soft == ()
