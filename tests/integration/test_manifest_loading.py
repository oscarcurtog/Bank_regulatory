"""Integration tests for the manifest READ boundary (EF-3).

This tests load_manifest() against real files on disk: the manifest the repo
versions, and manifests tampered with on purpose in a temporary directory.
data/raw/ is not needed: data/manifest.json IS versioned, so these tests also
run in a clean checkout.
"""

from __future__ import annotations

import json

import pytest

from ingest import download as dl
from tests.conftest import MANIFEST_PATH


def _write_manifest(tmp_path, documents):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"generated_at": "2026-09-05T16:17:15Z", "documents": documents}, indent=2),
        encoding="utf-8",
    )
    return path


def _real_documents():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["documents"]


# ---------------------------------------------------------------------------
# the repo's real manifest
# ---------------------------------------------------------------------------


def test_current_manifest_validates():
    """The five real entries meet the contract. If this fails, a constraint of
    the model is not backed by the data."""
    docs = dl.load_manifest(MANIFEST_PATH)
    assert len(docs) == 5
    assert {d.act for d in docs} == {"DORA", "MiCA", "PSD2"}
    assert all(d.http_status == 200 for d in docs)


def test_current_manifest_round_trips_byte_for_byte():
    """Compatibility: serializing the validated manifest again reproduces the
    EXACT file in the repo. It is the proof that introducing Pydantic has not
    changed the shape of the JSON."""
    original_text = MANIFEST_PATH.read_text(encoding="utf-8")
    generated_at = json.loads(original_text)["generated_at"]

    docs = dl.load_manifest(MANIFEST_PATH)
    payload = dl.manifest_payload(docs, generated_at)
    rebuilt = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    assert rebuilt == original_text


def test_parsed_source_picks_consolidated_when_available():
    """D-003: the parsed layer comes from the consolidated version if it exists.
    It is the behaviour that the 'role' bug silently broke."""
    docs = dl.load_manifest(MANIFEST_PATH)
    assert dl.parsed_source_for("PSD2", docs).celex == "02015L2366-20151223"
    assert dl.parsed_source_for("MiCA", docs).celex == "02023R1114-20240109"
    assert dl.parsed_source_for("DORA", docs).celex == "32022R2554"  # no consolidated version


# ---------------------------------------------------------------------------
# regression of the serious failure
# ---------------------------------------------------------------------------


def test_role_typo_is_blocked_at_the_validation_boundary(tmp_path):
    """REGRESSION of the failure that motivated EF-3.

    Before: writing "consolidate" instead of "consolidated" made
    parsed_source_for miss the PSD2 consolidated version and return the
    original. Parsing the original is valid in itself, so nothing fired: 117
    articles as before, 770 nodes against 763, 30,302 words against 30,312. The
    damage does not show in any count: 10 of 117 articles were left with the
    LEGAL TEXT from before the amendments that the consolidated version already
    includes (M-001), which is exactly what D-003 exists to prevent. Neither
    the structural invariant of run_t1.py, nor the article counts, nor the
    reference control caught it.

    Now: loading aborts before anyone can pick a document.
    """
    documents = _real_documents()
    for d in documents:
        if d["act"] == "PSD2" and d["role"] == "consolidated":
            d["role"] = "consolidate"  # the typo, a single letter
    path = _write_manifest(tmp_path, documents)

    with pytest.raises(dl.ManifestError) as excinfo:
        dl.load_manifest(path)

    message = str(excinfo.value)
    assert "role" in message
    assert "consolidate" in message
    assert "02015L2366-20151223" in message  # it says WHICH entry


def test_nothing_is_returned_when_one_entry_is_invalid(tmp_path):
    """Loading is all or nothing: one bad entry invalidates the whole
    operation; the four good ones are not returned for the pipeline to carry
    on with incomplete data."""
    documents = _real_documents()
    documents[0]["sha256"] = "not-a-hash"
    path = _write_manifest(tmp_path, documents)

    with pytest.raises(dl.ManifestError):
        dl.load_manifest(path)


# ---------------------------------------------------------------------------
# errors of the file, not of its entries
# ---------------------------------------------------------------------------


def test_missing_manifest_gives_a_clear_error(tmp_path):
    with pytest.raises(dl.ManifestError) as excinfo:
        dl.load_manifest(tmp_path / "missing.json")
    assert "fetch_all" in str(excinfo.value)


def test_malformed_json_gives_a_clear_error(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(dl.ManifestError) as excinfo:
        dl.load_manifest(path)
    assert "is not valid JSON" in str(excinfo.value)


def test_missing_documents_key_gives_a_clear_error(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"generated_at": "2026-09-05T16:17:15Z"}), encoding="utf-8")
    with pytest.raises(dl.ManifestError) as excinfo:
        dl.load_manifest(path)
    assert "documents" in str(excinfo.value)
