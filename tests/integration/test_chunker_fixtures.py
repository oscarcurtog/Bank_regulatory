"""Integration tests for the chunker (T2) on real EUR-Lex excerpts, with the real
tokenizer (cl100k_base; tiktoken downloads it the first time).

Every fixture is checked against all the chunk layer invariants, at the official
budget and at smaller ones that force deeper splits and budget exceptions on real
text. The concrete cases pin the structure the design asks for.
"""

from __future__ import annotations

import json

import pytest
import tiktoken

from ingest import chunks as ch
from ingest import download as dl
from ingest import hierarchy as h
from ingest import processed as pr
from tests.conftest import MANIFEST_PATH, assert_chunk_invariants, fixture_path

ENCODING = tiktoken.get_encoding("cl100k_base")


def count(text: str) -> int:  # the oracle counts with tiktoken directly
    return len(ENCODING.encode(text))


FIXTURES = [
    ("dora_art45", "doue", "DORA"),
    ("dora_art60", "doue", "DORA"),
    ("dora_art15", "doue", "DORA"),
    ("dora_art36", "doue", "DORA"),
    ("psd2_art9_original", "doue", "PSD2"),
    ("mica_art3", "consolidado", "MiCA"),
    ("psd2_art69", "consolidado", "PSD2"),
    ("psd2_art111", "consolidado", "PSD2"),
    ("psd2_art9", "consolidado", "PSD2"),
]


def _parse(name: str, family: str) -> h.Article:
    return h.parse(fixture_path(name), family)[0]


def _chunks(name: str, family: str, act: str, max_tokens: int) -> list[ch.Chunk]:
    return ch.chunk_article(_parse(name, family), act, max_tokens, ch.cl100k_counter())


def _by_id(chunks: list[ch.Chunk]) -> dict[str, ch.Chunk]:
    return {c.chunk_id: c for c in chunks}


def _source(act: str, family: str) -> dl.FetchResult:
    role = "original" if family == "doue" else "consolidated"
    return next(d for d in dl.load_manifest(MANIFEST_PATH) if d.act == act and d.role == role)


def _document(name: str, family: str, act: str, max_tokens: int) -> ch.ChunkedDocument:
    source = _source(act, family)
    processed = pr.ProcessedDocument(
        source_celex=source.celex,
        source_role=source.role,
        source_sha256=source.sha256,
        parser_version=h.PARSER_VERSION,
        family=family,  # type: ignore[arg-type]
        articles=[_parse(name, family)],
    )
    return ch.chunk_document(processed, act, max_tokens)


# ---------------------------------------------------------------------------
# every invariant, on every fixture, at several budgets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_tokens", [500, 300, 100, 30])
@pytest.mark.parametrize("name,family,act", FIXTURES)
def test_chunk_layer_invariants(name, family, act, max_tokens):
    article = _parse(name, family)
    chunks = ch.chunk_article(article, act, max_tokens, ch.cl100k_counter())
    assert_chunk_invariants(article, act, chunks, max_tokens, count)


@pytest.mark.parametrize("name,family,act", FIXTURES)
def test_document_contract_and_determinism(name, family, act):
    """The ChunkedDocument validators pass, and two runs give identical JSON."""
    first = _document(name, family, act, 500).model_dump(mode="json")
    second = _document(name, family, act, 500).model_dump(mode="json")
    assert json.dumps(first, ensure_ascii=False) == json.dumps(second, ensure_ascii=False)
    assert first["chunker_version"] == ch.CHUNKER_VERSION
    assert first["parser_version"] == h.PARSER_VERSION


# ---------------------------------------------------------------------------
# concrete cases
# ---------------------------------------------------------------------------


def test_an_article_that_fits_is_one_chunk():
    assert [c.chunk_id for c in _chunks("dora_art45", "doue", "DORA", 500)] == ["DORA:45"]
    assert [c.chunk_id for c in _chunks("psd2_art69", "consolidado", "PSD2", 500)] == ["PSD2:69"]


def test_a_smaller_budget_goes_down_to_paragraphs():
    assert [c.chunk_id for c in _chunks("dora_art45", "doue", "DORA", 300)] == [
        "DORA:45.1",
        "DORA:45.2",
        "DORA:45.3",
    ]


@pytest.mark.parametrize(
    "name,family", [("psd2_art9", "consolidado"), ("psd2_art9_original", "doue")]
)
def test_psd2_art9(name, family):
    """The article does not fit; paragraph 1 does not either. Its own text (the
    lead-in, its implicit first subparagraph) gets its own chunk, and each later
    subparagraph is a chunk whose context is only the heading: the lead-in of
    9(1) is a sibling of theirs, not an ancestor."""
    chunks = _chunks(name, family, "PSD2", 500)
    assert [c.chunk_id for c in chunks] == [
        "PSD2:9.1:own",
        "PSD2:9.1.sub_2",
        "PSD2:9.1.sub_3",
        "PSD2:9.1.sub_4",
        "PSD2:9.1.sub_5",
        "PSD2:9.1.sub_6",
        "PSD2:9.1.sub_7",
        "PSD2:9.2",
        "PSD2:9.3",
    ]
    by = _by_id(chunks)
    heading = "PSD2 Article 9: Calculation of own funds"
    for k in range(2, 8):
        assert by[f"PSD2:9.1.sub_{k}"].context == heading
    own = by["PSD2:9.1:own"]
    assert own.body.startswith("1. Notwithstanding the initial capital requirements")
    assert own.body.endswith("in accordance with national legislation:")
    assert by["PSD2:9.1.sub_7"].source_paths[:3] == ["9.1.sub_7", "9.1.sub_7.a", "9.1.sub_7.a.i"]


def test_psd2_art9_gives_the_same_chunks_in_both_markup_families():
    consolidated = [c.model_dump() for c in _chunks("psd2_art9", "consolidado", "PSD2", 500)]
    original = [c.model_dump() for c in _chunks("psd2_art9_original", "doue", "PSD2", 500)]
    assert original == consolidated


def test_mica_art3_definitions():
    """Paragraph 1 holds 51 definitions and does not fit: each definition is a
    chunk, every one of them repeats the lead-in as context, and the lead-in itself
    is owned once, by MiCA:3.1:own."""
    chunks = _chunks("mica_art3", "consolidado", "MiCA", 500)
    assert [c.chunk_id for c in chunks] == (
        ["MiCA:3.1:own"] + [f"MiCA:3.1.{i}" for i in range(1, 52)] + ["MiCA:3.2"]
    )
    lead_in = (
        "MiCA Article 3: Definitions\n\n"
        "1. For the purposes of this Regulation, the following definitions apply:"
    )
    by = _by_id(chunks)
    assert all(by[f"MiCA:3.1.{i}"].context == lead_in for i in range(1, 52))
    assert by["MiCA:3.1:own"].context == "MiCA Article 3: Definitions"
    crypto = by["MiCA:3.1.5"]
    assert crypto.text == (
        lead_in + "\n\n(5) 'crypto-asset' means a digital representation of a value or of a "
        "right that is able to be transferred and stored electronically using distributed "
        "ledger technology or similar technology;"
    )
    assert crypto.token_count == 58


def test_dora_art36_a_subparagraph_inside_and_outside_a_subtree():
    heading = "DORA Article 36: Exercise of the powers of the Lead Overseer outside the Union"
    at_500 = _by_id(_chunks("dora_art36", "doue", "DORA", 500))
    assert list(at_500) == ["DORA:36.1", "DORA:36.2", "DORA:36.3"]
    assert "36.1.sub_2" in at_500["DORA:36.1"].source_paths  # owned inside the subtree

    at_300 = _by_id(_chunks("dora_art36", "doue", "DORA", 300))
    assert at_300["DORA:36.1.sub_2"].context == heading  # 36.1's own text is its sibling
    assert at_300["DORA:36.1.a"].context.startswith(heading + "\n\n1. When oversight objectives")

    at_100 = _by_id(_chunks("dora_art36", "doue", "DORA", 100))
    point = at_100["DORA:36.1.sub_2.i"]
    assert point.context.startswith(
        heading + "\n\nThe powers referred to in the first subparagraph"
    )
    assert "When oversight objectives" not in point.context


def test_dora_art15_unnumbered_paragraphs():
    heading = (
        "DORA Article 15: Further harmonisation of ICT risk management tools, methods, "
        "processes and policies"
    )
    by = _by_id(_chunks("dora_art15", "doue", "DORA", 500))
    assert list(by) == (
        ["DORA:15:own"]
        + [f"DORA:15.{m}" for m in "abcdefg"]
        + ["DORA:15.unp_2", "DORA:15.unp_3", "DORA:15.unp_4"]
    )
    assert by["DORA:15.a"].context.startswith(heading + "\n\nThe ESAs shall, through the Joint")
    assert by["DORA:15.unp_2"].context == heading


def test_a_very_small_chunk_is_kept():
    """The heading line "Method A" of PSD2 Art. 9(1) is a structural unit of its
    own: 14 tokens with its context, kept as it is (no minimum, no packing)."""
    tiny = _by_id(_chunks("psd2_art9", "consolidado", "PSD2", 500))["PSD2:9.1.sub_2"]
    assert tiny.body == "Method A"
    assert tiny.token_count == 14


def test_budget_exceptions_on_real_text():
    """At a budget far below the official one, real leaves and real own texts do
    not fit: they are kept whole and flagged, never silently over budget."""
    chunks = _chunks("psd2_art9", "consolidado", "PSD2", 30)
    kinds = {c.budget_exception for c in chunks}
    assert {"leaf_over_budget", "own_over_budget"} <= kinds
    assert all((c.token_count > 30) == (c.budget_exception is not None) for c in chunks)


def test_max_tokens_is_configuration_and_is_persisted():
    at_500 = _document("dora_art36", "doue", "DORA", 500)
    at_300 = _document("dora_art36", "doue", "DORA", 300)
    assert (at_500.max_tokens, at_300.max_tokens) == (500, 300)
    assert at_500.chunker_version == at_300.chunker_version == ch.CHUNKER_VERSION
    assert len(at_500.chunks) < len(at_300.chunks)


def test_the_embedded_text_carries_no_technical_metadata():
    doc = _document("psd2_art9", "consolidado", "PSD2", 500)
    banned = [
        doc.source_celex,
        doc.source_sha256,
        doc.parser_version,
        ch.CHUNKER_VERSION,
        "chunk_id",
        "source_paths",
        "token_count",
        "budget_exception",
    ]
    for c in doc.chunks:
        assert not [b for b in banned if b in c.text], c.chunk_id


# ---------------------------------------------------------------------------
# persistence: write and load boundaries
# ---------------------------------------------------------------------------


def test_write_then_load_round_trip_is_byte_stable(tmp_path):
    doc = _document("mica_art3", "consolidado", "MiCA", 500)
    ch.write_chunks(doc, tmp_path)
    first = (tmp_path / "MiCA.json").read_bytes()
    loaded = ch.load_chunks("MiCA", dl.load_manifest(MANIFEST_PATH), tmp_path)
    assert loaded == doc
    ch.write_chunks(loaded, tmp_path)
    assert (tmp_path / "MiCA.json").read_bytes() == first


@pytest.mark.parametrize("field,value", [("chunker_version", "0.0.9"), ("parser_version", "1.3.0")])
def test_a_layer_from_another_chunker_or_parser_is_stale(tmp_path, field, value):
    doc = _document("psd2_art69", "consolidado", "PSD2", 500).model_dump(mode="json")
    doc[field] = value
    (tmp_path / "PSD2.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ch.StaleChunksError) as excinfo:
        ch.load_chunks("PSD2", dl.load_manifest(MANIFEST_PATH), tmp_path)
    assert value in str(excinfo.value)


def test_a_layer_from_another_corpus_snapshot_is_stale(tmp_path):
    ch.write_chunks(_document("psd2_art69", "consolidado", "PSD2", 500), tmp_path)
    manifest = [d.model_copy(update={"sha256": "0" * 64}) for d in dl.load_manifest(MANIFEST_PATH)]
    with pytest.raises(ch.StaleChunksError):
        ch.load_chunks("PSD2", manifest, tmp_path)


def test_a_broken_layer_is_a_contract_violation_not_staleness(tmp_path):
    doc = _document("psd2_art69", "consolidado", "PSD2", 500).model_dump(mode="json")
    doc["chunks"][0]["text"] += " tampered"
    (tmp_path / "PSD2.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ch.ChunksError) as excinfo:
        ch.load_chunks("PSD2", dl.load_manifest(MANIFEST_PATH), tmp_path)
    assert not isinstance(excinfo.value, ch.StaleChunksError)


def test_a_missing_layer_says_what_to_run(tmp_path):
    with pytest.raises(ch.ChunksError) as excinfo:
        ch.load_chunks("PSD2", dl.load_manifest(MANIFEST_PATH), tmp_path)
    assert "run_t2" in str(excinfo.value)
