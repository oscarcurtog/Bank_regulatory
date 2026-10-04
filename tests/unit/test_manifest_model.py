"""Unit tests for the manifest contract (EF-3): the FetchResult model in
isolation, without touching disk.

Every constraint tested here is backed by data measured in the repo, not by
intuition. See the EF-3 contract inventory and M-002.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ingest.download import FetchResult

# A valid entry, copied from data/manifest.json. The tests modify it field by
# field to isolate which constraint fires in each case.
VALID = {
    "act": "PSD2",
    "celex": "02015L2366-20151223",
    "role": "consolidated",
    "url": "https://publications.europa.eu/resource/celex/02015L2366-20151223",
    "local_path": "data/raw/02015L2366-20151223.xhtml",
    "sha256": "e9063e8919bf93bd2b92ae0c6f816ad83bcf6ca42a5b97cdc277214e5080e1c3",
    "fetched_at": "2026-09-05T16:17:15Z",
    "http_status": 200,
    "size_bytes": 567712,
}


def _with(**overrides) -> dict:
    return {**VALID, **overrides}


def test_valid_entry_is_accepted():
    entry = FetchResult.model_validate(VALID)
    assert entry.role == "consolidated"
    assert entry.act == "PSD2"
    assert entry.size_bytes == 567712


@pytest.mark.parametrize("role", ["original", "consolidated"])
def test_both_valid_roles_are_accepted(role):
    assert FetchResult.model_validate(_with(role=role)).role == role


# ---------------------------------------------------------------------------
# role: the closed set that used to live in a comment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_role",
    [
        "consolidate",  # the real typo behind all of EF-3
        "Consolidated",  # capital letter
        "consolidated ",  # trailing space
        "",
        "parsed",
    ],
)
def test_invalid_role_is_rejected(bad_role):
    with pytest.raises(ValidationError) as excinfo:
        FetchResult.model_validate(_with(role=bad_role))
    assert "role" in str(excinfo.value)


# ---------------------------------------------------------------------------
# http_status and size_bytes: they encode what fetch_document already rejected,
# and the incident of the 214-byte 404 page (M-002)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [404, 301, 500, 0, 201])
def test_non_200_status_is_rejected(status):
    with pytest.raises(ValidationError) as excinfo:
        FetchResult.model_validate(_with(http_status=status))
    assert "http_status" in str(excinfo.value)


@pytest.mark.parametrize("size", [214, 0, 9_999])
def test_size_below_minimum_is_rejected(size):
    """214 bytes is the real size of the EUR-Lex error page that, in M-002, was
    compared as if it were a document."""
    with pytest.raises(ValidationError) as excinfo:
        FetchResult.model_validate(_with(size_bytes=size))
    assert "size_bytes" in str(excinfo.value)


def test_size_exactly_at_minimum_is_accepted():
    assert FetchResult.model_validate(_with(size_bytes=10_000)).size_bytes == 10_000


# ---------------------------------------------------------------------------
# sha256: SHAPE only. Whether it matches the file is a domain validator and
# deliberately does NOT live here.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_sha",
    [
        "e9063e8919bf",  # truncated
        "E9063E8919BF93BD2B92AE0C6F816AD83BCF6CA42A5B97CDC277214E5080E1C3",  # uppercase
        "g" * 64,  # not hexadecimal
        "e9063e8919bf93bd2b92ae0c6f816ad83bcf6ca42a5b97cdc277214e5080e1c",  # 63
        "",
    ],
)
def test_malformed_sha256_is_rejected(bad_sha):
    with pytest.raises(ValidationError) as excinfo:
        FetchResult.model_validate(_with(sha256=bad_sha))
    assert "sha256" in str(excinfo.value)


# ---------------------------------------------------------------------------
# celex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "celex",
    [
        "32022R2554",  # DORA
        "32023R1114",  # MiCA original
        "02023R1114-20240109",  # MiCA consolidated
        "32015L2366",  # PSD2
        "02015L2366-20151223",  # PSD2 consolidated
        "32013R0575",  # CRR, from the roadmap: it must keep fitting
        "32015L0849",  # AMLD, same
    ],
)
def test_real_celex_identifiers_are_accepted(celex):
    assert FetchResult.model_validate(_with(celex=celex)).celex == celex


@pytest.mark.parametrize(
    "bad_celex",
    [
        "32022R25",  # short number
        "2022R2554",  # no sector
        "32022X2554",  # type not covered
        "02023R1114-2024",  # incomplete date suffix
        "celex:32022R2554",  # with a prefix
        "",
    ],
)
def test_malformed_celex_is_rejected(bad_celex):
    with pytest.raises(ValidationError) as excinfo:
        FetchResult.model_validate(_with(celex=bad_celex))
    assert "celex" in str(excinfo.value)


# ---------------------------------------------------------------------------
# fetched_at: real ISO 8601, not just any string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_date",
    ["yesterday", "2026-13-05T16:17:15Z", "2026-09-32T16:17:15Z", "", "05/09/2026"],
)
def test_invalid_date_is_rejected(bad_date):
    with pytest.raises(ValidationError) as excinfo:
        FetchResult.model_validate(_with(fetched_at=bad_date))
    assert "fetched_at" in str(excinfo.value)


# ---------------------------------------------------------------------------
# required fields and empty strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", list(VALID))
def test_missing_required_field_is_rejected(field):
    incomplete = {k: v for k, v in VALID.items() if k != field}
    with pytest.raises(ValidationError) as excinfo:
        FetchResult.model_validate(incomplete)
    assert field in str(excinfo.value)


@pytest.mark.parametrize("field", ["act", "url", "local_path"])
def test_empty_string_is_rejected(field):
    with pytest.raises(ValidationError) as excinfo:
        FetchResult.model_validate(_with(**{field: ""}))
    assert field in str(excinfo.value)


def test_act_is_not_a_closed_set():
    """act is NOT closed to DORA/MiCA/PSD2: the roadmap includes CRR and AMLD,
    and a model that rejected them would force a schema change to add an act,
    which is exactly what the project expects to do."""
    assert FetchResult.model_validate(_with(act="CRR", celex="32013R0575")).act == "CRR"


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def test_round_trip_model_json_model():
    """model -> JSON -> model gives back exactly the same thing. It is what
    guarantees that validating does not distort the manifest when it is
    written again."""
    original = FetchResult.model_validate(VALID)
    as_json = original.model_dump(mode="json")
    assert as_json == VALID  # same shape as the input, field by field
    assert FetchResult.model_validate(as_json) == original
