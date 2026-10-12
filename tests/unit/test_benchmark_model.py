"""Contract tests for the retrieval benchmark records (D-019): what each model
accepts and what it rejects. No files, no corpus."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from ingest import benchmark as bm
from ingest import download as dl

SHA = "a" * 64


def _need(**overrides) -> dict:
    need = {
        "need_id": "N001",
        "split": "dev",
        "statement": "When does the Regulation apply?",
        "type": "direct",
        "drivers": ["date"],
        "difficulty": "easy",
        "acts": ["MiCA"],
        "answerability": "complete",
        "gap_note": None,
        "ambiguity_note": None,
        "status": "ready",
    }
    return need | overrides


def _query(**overrides) -> dict:
    query = {
        "query_id": "N001-en-1",
        "need_id": "N001",
        "lang": "en",
        "style": "formal",
        "origin": "human",
        "template": None,
        "text": "From what date does MiCA apply?",
    }
    return query | overrides


def _qrel(**overrides) -> dict:
    qrel = {
        "need_id": "N001",
        "act": "MiCA",
        "path": "149.2",
        "grade": 3,
        "rationale": "States the date of application.",
        "text_sha256": SHA,
        "span": None,
        "ambiguous": False,
    }
    return qrel | overrides


def _manifest(**overrides) -> dict:
    manifest = {
        "schema_version": "1",
        "benchmark_version": "0.1.0",
        "role": "dev",
        "parser_version": "1.4.0",
        "corpus": {
            act: {"source_celex": "32022R2554", "source_sha256": SHA}
            for act in ("DORA", "MiCA", "PSD2")
        },
        "files": {name: SHA for name in ("needs.jsonl", "queries.jsonl", "qrels.jsonl")},
        "templates": {},
    }
    return manifest | overrides


# ---------------------------------------------------------------------------
# the closed sets agree with the rest of the repository
# ---------------------------------------------------------------------------


def test_the_acts_are_the_acts_of_the_corpus():
    assert bm.ACTS == get_args(bm.Act)
    assert set(bm.ACTS) == {a.name for a in dl.ACTS}


def test_the_data_files_are_the_closed_set():
    assert bm.DATA_FILES == get_args(bm.DataFile)
    assert bm.DATA_FILES == (bm.NEEDS_FILE, bm.QUERIES_FILE, bm.QRELS_FILE)


# ---------------------------------------------------------------------------
# acceptance: every record round-trips in declaration order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,record",
    [
        (bm.Need, _need()),
        (bm.Query, _query()),
        (bm.Qrel, _qrel()),
        (bm.BenchmarkManifest, _manifest()),
    ],
)
def test_a_valid_record_round_trips(model, record):
    parsed = model.model_validate(record)
    assert parsed.model_dump(mode="json") == record
    assert list(parsed.model_dump()) == list(record)


def test_a_partial_need_with_its_gap_and_a_none_need():
    bm.Need.model_validate(
        _need(
            type="partial_out_of_scope",
            answerability="partial",
            gap_note="The number is in Level 2.",
        )
    )
    bm.Need.model_validate(
        _need(type="partial_out_of_scope", answerability="none", gap_note="AMLD is not ingested.")
    )


def test_a_generated_query_names_its_template():
    q = bm.Query.model_validate(_query(origin="llm_from_need", template="query_variants.v1"))
    assert q.template == "query_variants.v1"


def test_a_qrel_with_a_span_and_an_ambiguous_flag():
    r = bm.Qrel.model_validate(_qrel(grade=1, span={"start": 57, "end": 638}, ambiguous=True))
    assert r.span is not None and (r.span.start, r.span.end) == (57, 638)
    assert r.node == ("MiCA", "149.2")


# ---------------------------------------------------------------------------
# rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"need_id": "N1"},
        {"need_id": "Q001"},
        {"split": "train"},
        {"type": "cross_act"},  # cross_act is a driver, never a type
        {"drivers": ["paraphrase"]},  # a property of a formulation, not of the evidence
        {"drivers": ["date", "date"]},
        {"difficulty": "trivial"},
        {"acts": []},
        {"acts": ["CRR"]},
        {"acts": ["MiCA", "MiCA"]},
        {"status": "done"},
        {"statement": ""},
        {"statement": " When does it apply?"},
        {"gap_note": "Something is missing."},  # complete needs have no gap
        {"type": "partial_out_of_scope"},  # with answerability complete
        {"answerability": "partial", "gap_note": "Level 2."},  # partial needs partial_out_of_scope
        {"answerability": "none", "type": "partial_out_of_scope"},  # none without gap_note
        {"extra": 1},
    ],
)
def test_need_rejects(overrides):
    with pytest.raises(ValidationError):
        bm.Need.model_validate(_need(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"query_id": "N002-en-1"},  # not the id of its need
        {"query_id": "N001-es-1"},  # not the id of its language
        {"query_id": "N001-en"},
        {"lang": "fr", "query_id": "N001-fr-1"},
        {"lang": "es", "query_id": "N001-es-1", "style": "formal"},  # Spanish is natural
        {"style": "casual"},
        {"origin": "llm"},
        {"origin": "llm_from_need"},  # without its template
        {"template": "query_variants.v1"},  # a human query names no template
        {"origin": "llm_from_need", "template": "Query Variants v1"},
        {"text": ""},
        {"text": "From what date?\n"},
        {"split": "dev"},  # a query has no split of its own
    ],
)
def test_query_rejects(overrides):
    with pytest.raises(ValidationError):
        bm.Query.model_validate(_query(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"grade": 4},
        {"grade": -1},
        {"grade": True},  # strict: a boolean is not a grade
        {"grade": 2.0},
        {"grade": "2"},
        {"act": "CRR"},
        {"path": "149-2"},
        {"path": ""},
        {"text_sha256": "abc"},
        {"rationale": ""},
        {"span": {"start": 10, "end": 10}},
        {"span": {"start": -1, "end": 5}},
        {"span": {"start": 1, "end": 5, "text": "x"}},
        {"ambiguous": "true"},  # strict: a string is not a flag
        {"ambiguous": 1},
        {"ambigous": True},  # the typo that would otherwise be dropped in silence
    ],
)
def test_qrel_rejects(overrides):
    with pytest.raises(ValidationError):
        bm.Qrel.model_validate(_qrel(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": "2"},
        {"benchmark_version": "1"},
        {"role": "train"},
        {"parser_version": "latest"},
        {"corpus": {"DORA": {"source_celex": "32022R2554", "source_sha256": SHA}}},
        {"files": {"needs.jsonl": SHA}},
        {
            "files": {
                name: SHA for name in ("needs.jsonl", "queries.jsonl", "qrels.jsonl", "x.jsonl")
            }
        },
        {"templates": {"Bad Id": SHA}},
        {"extra": 1},
    ],
)
def test_manifest_rejects(overrides):
    with pytest.raises(ValidationError):
        bm.BenchmarkManifest.model_validate(_manifest(**overrides))
