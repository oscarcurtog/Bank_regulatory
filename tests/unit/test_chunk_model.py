"""Unit tests for the chunk layer contract (T2): Chunk and ChunkedDocument in
isolation, without touching disk or a tokenizer.

Each rejection pins one invariant of the design: text is exactly the rendering of
context and body, an "own" chunk holds only its root, a budget overrun is never
silent, a chunk_id is its act, root and coverage, and no source text is owned by
two chunks.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ingest.chunks import CHUNKER_VERSION, Chunk, ChunkedDocument

CONTEXT = "DORA Article 45: Information-sharing arrangements"
BODY = "1. Financial entities may exchange information.\n(a) aims to enhance resilience;"

VALID_CHUNK = {
    "chunk_id": "DORA:45",
    "article_id": "art_45",
    "root_path": "45",
    "coverage": "subtree",
    "source_paths": ["45.1", "45.1.a"],
    "context": CONTEXT,
    "body": BODY,
    "text": CONTEXT + "\n\n" + BODY,
    "token_count": 20,
    "budget_exception": None,
}

VALID_DOC = {
    "act": "DORA",
    "source_celex": "32022R2554",
    "source_role": "original",
    "source_sha256": "cf50d8f023d23b2a56f535aaee0760eba7f1925ef08a9858c5c31d43d414f46b",
    "parser_version": "1.4.0",
    "chunker_version": CHUNKER_VERSION,
    "tokenizer": "cl100k_base",
    "max_tokens": 500,
    "chunks": [VALID_CHUNK],
}


def _chunk(**overrides):
    return {**VALID_CHUNK, **overrides}


def _doc(**overrides):
    return {**VALID_DOC, **overrides}


def _own_chunk():
    body = "1. Financial entities may exchange information."
    return _chunk(
        chunk_id="DORA:45.1:own",
        root_path="45.1",
        coverage="own",
        source_paths=["45.1"],
        body=body,
        text=CONTEXT + "\n\n" + body,
    )


# ---------------------------------------------------------------------------
# what is accepted
# ---------------------------------------------------------------------------


def test_valid_document_round_trip_keeps_shape_and_order():
    doc = ChunkedDocument.model_validate(VALID_DOC)
    as_json = doc.model_dump(mode="json")
    assert as_json == VALID_DOC
    assert list(as_json) == list(VALID_DOC)
    assert list(as_json["chunks"][0]) == list(VALID_CHUNK)
    assert ChunkedDocument.model_validate(as_json) == doc


def test_over_budget_chunk_with_an_explicit_exception_is_accepted():
    over = _chunk(token_count=600, budget_exception="leaf_over_budget")
    assert ChunkedDocument.model_validate(_doc(chunks=[over])).chunks[0].token_count == 600


def test_own_chunk_is_accepted():
    assert Chunk.model_validate(_own_chunk()).coverage == "own"


# ---------------------------------------------------------------------------
# Chunk: what is rejected
# ---------------------------------------------------------------------------


def test_text_must_be_exactly_the_rendering_of_context_and_body():
    with pytest.raises(ValidationError) as excinfo:
        Chunk.model_validate(_chunk(text=CONTEXT + "\n" + BODY))
    assert "render_text" in str(excinfo.value)


def test_article_id_must_match_the_root():
    with pytest.raises(ValidationError):
        Chunk.model_validate(_chunk(article_id="art_46"))


@pytest.mark.parametrize("paths", [["45.1", "45.2"], ["45.2"]])
def test_an_own_chunk_holds_only_its_root(paths):
    with pytest.raises(ValidationError):
        Chunk.model_validate({**_own_chunk(), "source_paths": paths})


@pytest.mark.parametrize("paths", [["46.1"], ["451"], ["45.1", "45.1"], []])
def test_source_paths_must_be_inside_the_root_once_and_not_empty(paths):
    with pytest.raises(ValidationError):
        Chunk.model_validate(_chunk(root_path="45", source_paths=paths))


@pytest.mark.parametrize(
    "coverage,exception",
    [("own", "leaf_over_budget"), ("subtree", "own_over_budget"), ("subtree", "too_big")],
)
def test_the_budget_exception_must_match_the_coverage(coverage, exception):
    base = _own_chunk() if coverage == "own" else VALID_CHUNK
    with pytest.raises(ValidationError):
        Chunk.model_validate({**base, "token_count": 600, "budget_exception": exception})


@pytest.mark.parametrize(
    "chunk_id", ["DORA:45:part1", "DORA 45", "45", "DORA:45:OWN", "DORA:45.1:own:own"]
)
def test_malformed_chunk_ids_are_rejected(chunk_id):
    """ ":part<n>" is reserved for a future textual fallback and not produced yet."""
    with pytest.raises(ValidationError):
        Chunk.model_validate(_chunk(chunk_id=chunk_id))


# ---------------------------------------------------------------------------
# ChunkedDocument: what is rejected
# ---------------------------------------------------------------------------


def test_duplicate_chunk_ids_are_rejected():
    with pytest.raises(ValidationError) as excinfo:
        ChunkedDocument.model_validate(_doc(chunks=[VALID_CHUNK, VALID_CHUNK]))
    assert "duplicate chunk_id" in str(excinfo.value)


@pytest.mark.parametrize(
    "act,chunk_id", [("MiCA", "DORA:45"), ("DORA", "DORA:45.1"), ("DORA", "DORA:45:own")]
)
def test_chunk_id_must_be_its_act_root_and_coverage(act, chunk_id):
    with pytest.raises(ValidationError) as excinfo:
        ChunkedDocument.model_validate(_doc(act=act, chunks=[_chunk(chunk_id=chunk_id)]))
    assert "not the id of its act, root and coverage" in str(excinfo.value)


def test_a_budget_overrun_is_never_silent():
    with pytest.raises(ValidationError) as excinfo:
        ChunkedDocument.model_validate(_doc(chunks=[_chunk(token_count=501)]))
    assert "budget_exception=None" in str(excinfo.value)


def test_an_exception_within_the_budget_is_rejected():
    within = _chunk(token_count=500, budget_exception="leaf_over_budget")
    with pytest.raises(ValidationError):
        ChunkedDocument.model_validate(_doc(chunks=[within]))


def test_the_budget_is_the_documents_max_tokens():
    """The same chunk is valid or not depending on the max_tokens it was made
    with: the budget is a run parameter recorded in the document."""
    chunk = _chunk(token_count=350)
    assert ChunkedDocument.model_validate(_doc(max_tokens=400, chunks=[chunk]))
    with pytest.raises(ValidationError):
        ChunkedDocument.model_validate(_doc(max_tokens=300, chunks=[chunk]))


def test_source_text_cannot_be_owned_by_two_chunks():
    second = _own_chunk()  # owns 45.1, already in the body of DORA:45
    with pytest.raises(ValidationError) as excinfo:
        ChunkedDocument.model_validate(_doc(chunks=[VALID_CHUNK, second]))
    assert "already owned by another chunk" in str(excinfo.value)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("tokenizer", "bert-base-uncased"),
        ("chunker_version", "0.1"),
        ("parser_version", "v1.4.0"),
        ("max_tokens", 0),
        ("act", "DO:RA"),
        ("source_sha256", "cf50d8f0"),
        ("source_role", "consolidate"),
    ],
)
def test_malformed_provenance_is_rejected(field, bad):
    with pytest.raises(ValidationError) as excinfo:
        ChunkedDocument.model_validate(_doc(**{field: bad}))
    assert field in str(excinfo.value)


def test_a_document_without_chunks_is_rejected():
    with pytest.raises(ValidationError):
        ChunkedDocument.model_validate(_doc(chunks=[]))
