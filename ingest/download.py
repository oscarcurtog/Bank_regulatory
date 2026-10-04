"""Download from CELLAR/EUR-Lex, with a reproducible manifest.

Implements D-001/D-002/D-003 (decision log):

- The ORIGINAL from the Official Journal is always downloaded (role "original"):
  it is the authentic artifact that anchors the snapshot hash (D-001) and lives
  in the immutable raw layer (D-002).
- Where a DOWNLOADABLE consolidated version exists, it is downloaded as well
  (role "consolidated"): it is the source of the parsed layer (D-003), because
  EUR-Lex has already applied the corrigenda to it. Where none exists, the
  parsed layer comes from the original, and that is only safe if the act has no
  corrigendum in the language served (verified for the three acts in M-002,
  measurement log).

Every document is validated by status code AND minimum size before it is
accepted: comparing against a 404 error page without noticing was a real
failure during the research (see the method note of M-002).
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

CELLAR_BASE = "https://publications.europa.eu/resource/celex/"
MIN_VALID_BYTES = 10_000  # a real response is never smaller than this; an EUR-Lex 404 is ~214 bytes
HEADERS = {"Accept": "application/xhtml+xml", "Accept-Language": "eng"}

RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.json")

# CELEX: sector + year + type + number, with a date suffix on consolidated
# versions. Sector 3 = original act, 0 = consolidated (D-001). Checked against the
# five current documents and against CRR (32013R0575) and AMLD (32015L0849), which
# the roadmap already anticipates. Deliberately narrow: if a future act does not
# fit, the manifest fails to load and this pattern is widened consciously, which
# is better than silently accepting an identifier that does not resolve in CELLAR.
CELEX_PATTERN = r"^[03]\d{4}[RLD]\d{4}(-\d{8})?$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


@dataclass
class Act:
    name: str
    original_celex: str
    consolidated_celex: str | None  # None if no consolidated version can be downloaded (D-003)


# Config derived from D-001/D-003. The chosen consolidated version is the OLDEST
# in the series that answers 200, not the oldest one EUR-Lex declares in the
# graph: for MiCA the 2023-06-09 one is declared but returns 404 (see M-002).
ACTS = [
    Act("DORA", "32022R2554", None),
    Act("MiCA", "32023R1114", "02023R1114-20240109"),
    Act("PSD2", "32015L2366", "02015L2366-20151223"),
]


class FetchResult(BaseModel):
    """One manifest entry: both what the download returns and what gets
    persisted. A single model, not an internal object plus a separate contract.

    Validating here matters because the manifest decides WHICH file is parsed. A
    misspelt `role` ("consolidate") made parsed_source_for miss the PSD2
    consolidated version and return the original. That parse is valid in
    itself, so nothing fired: 117 articles as before, 770 nodes against 763,
    30,302 words against 30,312. The real consequence does not show in any
    count: 10 of 117 articles kept the LEGAL TEXT from before the amendments
    that the consolidated version already includes (M-001), which is exactly
    what D-003 exists to prevent. Measured.

    Where each constraint comes from:
    - role and http_status are closed sets because fetch_document already
      rejects anything else before building this; the manifest should not be
      able to contain anything the download would not have accepted.
    - size_bytes >= MIN_VALID_BYTES puts the M-002 incident into the schema,
      where a 214-byte 404 error page was compared as if it were a real
      document.
    - act is NOT a closed set: the roadmap includes CRR and AMLD.

    What is deliberately NOT validated here: that sha256 really is the hash of
    the file on disk (that needs reading the file; it is a domain validator,
    not a shape one) and any rule that looks at more than one entry at a time.
    """

    act: str = Field(min_length=1)
    celex: str = Field(pattern=CELEX_PATTERN)
    role: Literal["original", "consolidated"]
    url: str = Field(min_length=1)
    local_path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    fetched_at: datetime
    http_status: Literal[200]
    size_bytes: int = Field(ge=MIN_VALID_BYTES)


class FetchError(Exception):
    pass


class ManifestError(Exception):
    """The manifest exists but does not meet the FetchResult contract."""


def _fetch(celex: str) -> tuple[bytes, int]:
    url = CELLAR_BASE + celex.replace("(", "%28").replace(")", "%29")
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read(), resp.status
    except urllib.error.HTTPError as e:
        return e.read(), e.code


def fetch_document(
    celex: str,
    role: Literal["original", "consolidated"],
    act: str,
    fetched_at: datetime,
) -> FetchResult:
    """Downloads a CELEX and validates it. Raises FetchError if it is not a real document."""
    body, status = _fetch(celex)
    if status != 200 or len(body) < MIN_VALID_BYTES:
        raise FetchError(
            f"{celex}: invalid response (status={status}, {len(body)} bytes, "
            f"expected at least {MIN_VALID_BYTES})"
        )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_path = RAW_DIR / f"{celex}.xhtml"
    local_path.write_bytes(body)
    return FetchResult(
        act=act,
        celex=celex,
        role=role,
        url=CELLAR_BASE + celex,
        local_path=str(local_path),
        sha256=hashlib.sha256(body).hexdigest(),
        fetched_at=fetched_at,
        http_status=200,  # the guard above already rejected anything else
        size_bytes=len(body),
    )


def fetch_all(fetched_at: datetime) -> list[FetchResult]:
    """Downloads everything T1 needs for the three acts. Idempotent: it can be
    run again, and it validates every document again."""
    results: list[FetchResult] = []
    for act in ACTS:
        results.append(fetch_document(act.original_celex, "original", act.name, fetched_at))
        if act.consolidated_celex:
            try:
                results.append(
                    fetch_document(act.consolidated_celex, "consolidated", act.name, fetched_at)
                )
            except FetchError as e:
                raise FetchError(
                    f"{act.name}: the consolidated version {act.consolidated_celex} was "
                    f"registered as downloadable and no longer responds. Review ACTS in "
                    f"ingest/download.py before continuing; do not ignore the failure. Detail: {e}"
                ) from e
    return results


def manifest_payload(results: list[FetchResult], generated_at: str) -> dict:
    """The exact structure that gets persisted. mode="json" is what turns
    fetched_at from a datetime into its ISO string, and what keeps the file
    identical to the one the dataclass used to write."""
    return {
        "generated_at": generated_at,
        "documents": [r.model_dump(mode="json") for r in results],
    }


def write_manifest(results: list[FetchResult], generated_at: str) -> None:
    """WRITE boundary. Nothing gets here without being a FetchResult, and a
    FetchResult does not exist unless it validated when it was built."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest_payload(results, generated_at), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path = MANIFEST_PATH) -> list[FetchResult]:
    """READ boundary: validates every entry BEFORE anyone uses it.

    Aborts with ManifestError as soon as one entry breaks the contract. It never
    returns a partial list, nor lets a consumer pick a document from data that
    has not been checked.
    """
    if not path.exists():
        raise ManifestError(f"{path} does not exist. Run ingest.download.fetch_all first.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict) or not isinstance(raw.get("documents"), list):
        raise ManifestError(f"{path}: the 'documents' key is missing or is not a list")

    docs: list[FetchResult] = []
    for i, entry in enumerate(raw["documents"]):
        try:
            docs.append(FetchResult.model_validate(entry))
        except ValidationError as e:
            celex = entry.get("celex", "?") if isinstance(entry, dict) else "?"
            raise ManifestError(
                f"{path}: entry {i} (celex {celex}) does not meet the manifest "
                f"contract, so none of the entries is used.\n{e}"
            ) from e
    return docs


def parsed_source_for(act_name: str, manifest_docs: list[FetchResult]) -> FetchResult:
    """Of the documents downloaded for an act, the one that feeds the parsed
    layer according to D-003: the consolidated version if it exists, otherwise
    the original."""
    docs = [d for d in manifest_docs if d.act == act_name]
    consolidated = [d for d in docs if d.role == "consolidated"]
    if consolidated:
        return consolidated[0]
    original = [d for d in docs if d.role == "original"]
    if not original:
        raise FetchError(f"{act_name}: no document has been downloaded")
    return original[0]
