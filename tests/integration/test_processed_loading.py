"""Integration tests for the processed layer boundaries (EF-3, contract 2).

The core of this file is staleness detection, which is what was missing when
M-005 was measured on corrupted text without anything raising a warning.

Almost all of them use a minimal synthetic document in tmp_path, so they run in
a clean checkout. Only the last one needs the real processed layer, which is in
.gitignore under D-002, and it is marked "corpus".
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from ingest import download as dl
from ingest import hierarchy as h
from ingest import processed as pr
from tests.conftest import MANIFEST_PATH, RAW_DIR

SHA_A = "cf50d8f023d23b2a56f535aaee0760eba7f1925ef08a9858c5c31d43d414f46b"
SHA_B = "0000000000000000000000000000000000000000000000000000000000000000"


def _manifest_entry(sha: str = SHA_A) -> dl.FetchResult:
    return dl.FetchResult(
        act="DORA",
        celex="32022R2554",
        role="original",
        url="https://publications.europa.eu/resource/celex/32022R2554",
        local_path="data/raw/32022R2554.xhtml",
        sha256=sha,
        fetched_at=datetime(2026, 9, 5, 16, 17, 15, tzinfo=UTC),
        http_status=200,
        size_bytes=746866,
    )


def _document(parser_version: str = h.PARSER_VERSION, sha: str = SHA_A) -> pr.ProcessedDocument:
    return pr.ProcessedDocument(
        source_celex="32022R2554",
        source_role="original",
        source_sha256=sha,
        parser_version=parser_version,
        family="doue",
        articles=[
            h.Article(
                article_id="art_1",
                path="1",
                title="Subject matter",
                text="Article 1 Subject matter 1. This Regulation lays down uniform requirements.",
                max_depth=2,
                nodes=[
                    h.Node(path="1", level=1, marker="1", kind="articulo"),
                    h.Node(
                        path="1.1",
                        level=2,
                        marker="1",
                        kind="apartado",
                        own_text="This Regulation lays down uniform requirements.",
                    ),
                ],
            )
        ],
    )


# ---------------------------------------------------------------------------
# round trip through disk
# ---------------------------------------------------------------------------


def test_write_then_load_round_trip(tmp_path):
    original = _document()
    pr.write_processed(original, "DORA", tmp_path)
    assert pr.load_processed("DORA", [_manifest_entry()], tmp_path) == original


def test_written_file_has_the_expected_shape(tmp_path):
    pr.write_processed(_document(), "DORA", tmp_path)
    raw = json.loads((tmp_path / "DORA.json").read_text(encoding="utf-8"))
    assert list(raw) == [
        "source_celex",
        "source_role",
        "source_sha256",
        "parser_version",
        "family",
        "articles",
    ]
    assert list(raw["articles"][0]) == [
        "article_id",
        "path",
        "title",
        "text",
        "max_depth",
        "nodes",
    ]
    assert list(raw["articles"][0]["nodes"][0]) == [
        "path",
        "level",
        "marker",
        "kind",
        "own_text",
    ]


# ---------------------------------------------------------------------------
# STALENESS, axis 1: the code changed
# ---------------------------------------------------------------------------


def test_stale_parser_version_aborts(tmp_path):
    """REGRESSION of the M-005 incident.

    profile_tokens.py read a layer written by a parser with the three text
    defects of M-004 and measured it without noticing. The result did not look
    broken: median 327 tokens, when on correct text it is 349. The
    parser_version field had been there from the start and nobody compared it.
    """
    pr.write_processed(_document(parser_version="1.2.0"), "DORA", tmp_path)

    with pytest.raises(pr.StaleLayerError) as excinfo:
        pr.load_processed("DORA", [_manifest_entry()], tmp_path)

    message = str(excinfo.value)
    assert "1.2.0" in message
    assert h.PARSER_VERSION in message
    assert "run_t1" in message  # it says what to do


def test_older_layer_that_breaks_todays_schema_is_reported_as_stale(tmp_path):
    """The real case of 1.4.0: the 1.3.0 layer of PSD2 has duplicate paths (Art.
    9), which Article now rejects. Loading it must say "stale, rerun run_t1",
    not "broken contract": the version is compared before the schema."""
    old = _document().model_dump(mode="json")
    old["parser_version"] = "1.3.0"
    nodes = old["articles"][0]["nodes"]
    nodes.append(dict(nodes[1]))  # the same path twice, as PSD2 Art. 9 had
    (tmp_path / "DORA.json").write_text(json.dumps(old), encoding="utf-8")

    with pytest.raises(pr.StaleLayerError) as excinfo:
        pr.load_processed("DORA", [_manifest_entry()], tmp_path)
    assert "1.3.0" in str(excinfo.value)
    assert "run_t1" in str(excinfo.value)


def test_malformed_parser_version_is_a_contract_violation(tmp_path):
    """Only a well-formed version is compared before the schema. A malformed one
    is a broken contract, not staleness."""
    bad = _document().model_dump(mode="json")
    bad["parser_version"] = "one"
    (tmp_path / "DORA.json").write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(pr.ProcessedError) as excinfo:
        pr.load_processed("DORA", [_manifest_entry()], tmp_path)
    assert not isinstance(excinfo.value, pr.StaleLayerError)
    assert "parser_version" in str(excinfo.value)


def test_current_parser_version_is_accepted(tmp_path):
    pr.write_processed(_document(parser_version=h.PARSER_VERSION), "DORA", tmp_path)
    doc = pr.load_processed("DORA", [_manifest_entry()], tmp_path)
    assert doc.parser_version == h.PARSER_VERSION


# ---------------------------------------------------------------------------
# STALENESS, axis 2: the corpus changed
# ---------------------------------------------------------------------------


def test_stale_source_sha256_aborts(tmp_path):
    """The layer came from one snapshot and the manifest has another: the
    corpus was downloaded again and changed. The text of the layer is no longer
    that of the file which the D-002 traceability says produced it."""
    pr.write_processed(_document(sha=SHA_A), "DORA", tmp_path)

    with pytest.raises(pr.StaleLayerError) as excinfo:
        pr.load_processed("DORA", [_manifest_entry(sha=SHA_B)], tmp_path)

    assert "corpus" in str(excinfo.value)


def test_celex_missing_from_the_manifest_aborts(tmp_path):
    """The layer claims to come from a document that is no longer in the
    manifest: there is no way to check its provenance, so it is not used."""
    pr.write_processed(_document(), "DORA", tmp_path)
    other = _manifest_entry()
    other = other.model_copy(update={"celex": "32015L2366"})

    with pytest.raises(pr.StaleLayerError) as excinfo:
        pr.load_processed("DORA", [other], tmp_path)

    assert "manifest" in str(excinfo.value)


# ---------------------------------------------------------------------------
# file errors, which are not staleness
# ---------------------------------------------------------------------------


def test_missing_file_gives_a_clear_error(tmp_path):
    with pytest.raises(pr.ProcessedError) as excinfo:
        pr.load_processed("DORA", [_manifest_entry()], tmp_path)
    assert "run_t1" in str(excinfo.value)


def test_schema_violation_is_not_reported_as_staleness(tmp_path):
    """A file that breaks the contract gives ProcessedError, not
    StaleLayerError: they are two different failures and it pays to tell them
    apart."""
    bad = _document().model_dump(mode="json")
    bad["articles"][0]["nodes"][0]["kind"] = "invented"
    (tmp_path / "DORA.json").write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(pr.ProcessedError) as excinfo:
        pr.load_processed("DORA", [_manifest_entry()], tmp_path)
    assert not isinstance(excinfo.value, pr.StaleLayerError)
    assert "kind" in str(excinfo.value)


def test_malformed_json_gives_a_clear_error(tmp_path):
    (tmp_path / "DORA.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(pr.ProcessedError) as excinfo:
        pr.load_processed("DORA", [_manifest_entry()], tmp_path)
    assert "is not valid JSON" in str(excinfo.value)


# ---------------------------------------------------------------------------
# the REAL processed layer (data/processed/ is in .gitignore under D-002)
# ---------------------------------------------------------------------------


@pytest.mark.corpus
@pytest.mark.parametrize("act", ["DORA", "MiCA", "PSD2"])
def test_real_processed_layer_validates_and_is_current(act):
    """The three real documents meet the contract and are up to date on both
    axes. If this fails, run_t1.py has to be run again, or a schema constraint
    is not backed by the data."""
    if not pr.path_for(act).exists():
        pytest.skip("data/processed/ not generated in this checkout")
    manifest = dl.load_manifest(MANIFEST_PATH)
    doc = pr.load_processed(act, manifest)
    assert doc.parser_version == h.PARSER_VERSION
    assert len(doc.articles) == {"DORA": 64, "MiCA": 150, "PSD2": 117}[act]
    assert RAW_DIR.exists()  # consistency: the layer came from a corpus that is present
