"""The query-variant prompt and the lexical overlap diagnostic (D-019).

The prompt is built from the need statement alone: the renderer takes a string,
so nothing else about the need can reach the model that writes the queries."""

from __future__ import annotations

import re

import pytest

from ingest import benchmark as bm

VARIANTS = [("en", "formal"), ("en", "natural"), ("es", "natural")]


def test_the_template_is_versioned_and_sealable():
    path = bm.template_path(bm.QUERY_TEMPLATE_ID)
    assert re.fullmatch(bm.TEMPLATE_ID_PATTERN, bm.QUERY_TEMPLATE_ID)
    assert path.read_text(encoding="utf-8").count(bm.PROMPT_MARKER) == 1


def test_the_prompt_carries_the_statement_and_the_formulations_only():
    prompt = bm.render_query_prompt("A bank wants to know X.", VARIANTS)
    assert "Information need:\nA bank wants to know X.\n" in prompt
    assert "- en formal\n- en natural\n- es natural" in prompt
    assert bm.PROMPT_MARKER not in prompt and "Never edit this file" not in prompt  # no header
    assert "{statement}" not in prompt and "{variants}" not in prompt


def test_a_statement_is_inserted_literally():
    prompt = bm.render_query_prompt("Does {variants} appear?", [("en", "natural")])
    assert "Does {variants} appear?" in prompt


@pytest.mark.parametrize(
    "variants,message",
    [
        ([("es", "formal")], "es formal is not allowed"),
        ([("fr", "natural")], "fr natural is not allowed"),
        ([], "at least one formulation"),
        ([("en", "formal"), ("en", "formal")], "each once"),
    ],
)
def test_the_prompt_refuses(variants, message):
    with pytest.raises(bm.BenchmarkError, match=message):
        bm.render_query_prompt("A need.", variants)


def test_the_prompt_of_a_fixture_reveals_none_of_its_evidence():
    fixtures = bm.load_benchmark(bm.FIXTURES_DIR)
    for need in fixtures.needs:
        prompt = bm.render_query_prompt(need.statement, VARIANTS)
        assert need.statement in prompt
        if need.gap_note:
            assert need.gap_note not in prompt
        for r in fixtures.qrels_of(need.need_id):
            assert r.rationale not in prompt
            assert f"{r.act}:{r.path}" not in prompt


def test_overlap_counts_content_words_and_the_longest_verbatim_run():
    evidence = ["Financial entities shall submit an initial notification;"]
    o = bm.lexical_overlap("When must the initial notification be submitted?", evidence)
    # content words: initial, notification, submitted ("when", "must", "the", "be" are
    # stopwords); two are shared, and "submitted" is not "submit": there is no stemming
    assert o.share == pytest.approx(2 / 3) and o.longest_run == 2 and not o.possible_copy


def test_overlap_reads_diacritics_and_flags_a_copy():
    o = bm.lexical_overlap("A naïve façade rule", ["Rules on the façade."])
    assert o.share == pytest.approx(1 / 3)  # façade, out of naïve, façade and rule (no stemming)
    copied = "financial entities shall submit an initial notification to the authority"
    o = bm.lexical_overlap(copied, ["Financial entities shall submit an initial notification;"])
    assert o.longest_run == 7 and o.possible_copy
