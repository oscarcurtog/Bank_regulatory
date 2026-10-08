"""Contract of the processed layer: data/processed/{act}.json (EF-3, contract 2).

It lives in its own module because it sits ABOVE the other two: it needs
ingest.hierarchy (the articles and nodes) and ingest.download (the manifest, to
check provenance). Putting it in hierarchy.py would force the parser to import
the download module, which is a dependency in the wrong direction.

The processed document quotes three manifest fields (source_celex,
source_role, source_sha256): they are the provenance link. THE SAME
constraints that FetchResult already uses are reused here, so that the two
contracts cannot disagree.

Staleness. The file already carried everything needed to detect that it was
out of date, and nobody looked at it. It really happened: M-005 was measured on
a layer written by a parser with the three text defects of M-004, and the
result looked perfectly normal (median 327 tokens; redone on correct text it
came out at 349). load_processed() compares both axes and ABORTS:

  1. the CODE changed:    parser_version != hierarchy.PARSER_VERSION
  2. the CORPUS changed:  source_sha256 != the manifest's for that celex

Pydantic validates the shape of those fields. Not the comparison: it is made
against external state, so it lives in the reader. Same boundary as with the
manifest's sha256, where the model validates that it LOOKS like a hash and the
domain checks that it IS one.

Axis 1 is checked BEFORE the schema (1.4.0). A layer written by another parser
may break today's schema: the 1.3.0 layer of PSD2 has duplicate paths, which
Article rejects since 1.4.0. That is staleness, not a broken contract, and the
reader says so instead of reporting a contract violation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ingest import download as dl
from ingest import hierarchy as h

PROCESSED_DIR = Path("data/processed")
# Shared by the model and by the early staleness check in load_processed(), so that
# the two cannot disagree on what a well-formed version is.
VERSION_PATTERN = r"^\d+\.\d+\.\d+$"


class ProcessedError(Exception):
    """The processed layer exists but does not meet the contract."""


class StaleLayerError(ProcessedError):
    """The processed layer is valid but out of date: another parser wrote it,
    or it came from a different corpus than the one now in the manifest."""


class ProcessedDocument(BaseModel):
    """A whole act, as it is persisted after parsing.

    Declaration order is serialization order, and it deliberately matches the
    order of the file that the dict literal in run_t1.py used to write.
    """

    source_celex: str = Field(pattern=dl.CELEX_PATTERN)
    source_role: Literal["original", "consolidated"]
    source_sha256: str = Field(pattern=dl.SHA256_PATTERN)
    parser_version: str = Field(pattern=VERSION_PATTERN)
    # Legacy serialized schema codes for the markup family, kept on purpose, with
    # the same status as Node.kind in ingest/hierarchy.py:
    #   "doue"         original text as published in the Official Journal
    #                  (DOUE is the Spanish acronym of the Official Journal)
    #   "consolidado"  consolidated text
    #
    # Known debt, deliberately not addressed yet: moving these codes and the three
    # legacy Node.kind codes to English names (article/paragraph/point and English
    # family names) is a schema migration, not a cleanup. It needs a PARSER_VERSION
    # bump and regenerating every derived layer (data/processed/,
    # data/t2_nodes.json). The node kinds added in 1.4.0 are already English. Since
    # 1.4.0, load_processed() compares parser_version before validating the schema,
    # so a layer written with the old codes would be reported as stale
    # (StaleLayerError), not as a contract violation (ProcessedError).
    family: Literal["doue", "consolidado"]
    articles: list[h.Article] = Field(min_length=1)


def path_for(act: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    return processed_dir / f"{act}.json"


def write_processed(doc: ProcessedDocument, act: str, processed_dir: Path = PROCESSED_DIR) -> None:
    """WRITE boundary. Nothing gets here without being a ProcessedDocument, and
    a ProcessedDocument does not exist unless it validated when it was built."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    path_for(act, processed_dir).write_text(
        json.dumps(doc.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_processed(
    act: str,
    manifest: list[dl.FetchResult],
    processed_dir: Path = PROCESSED_DIR,
) -> ProcessedDocument:
    """READ boundary: validates the schema AND checks that the layer is up to
    date before returning anything.

    Aborts with ProcessedError if the file does not meet the contract, and with
    StaleLayerError if it is out of date. It never returns data that a consumer
    could measure believing it is the current one.
    """
    path = path_for(act, processed_dir)
    if not path.exists():
        raise ProcessedError(f"{path} does not exist. Run scripts/run_t1.py first.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ProcessedError(f"{path} is not valid JSON: {e}") from e
    # --- axis 1, before the schema: the code that wrote it changed. Only a
    # well-formed version is compared here; a malformed one is a contract violation
    # and is left to the schema below.
    version = raw.get("parser_version") if isinstance(raw, dict) else None
    if (
        isinstance(version, str)
        and re.fullmatch(VERSION_PATTERN, version)
        and version != h.PARSER_VERSION
    ):
        raise StaleLayerError(
            f"{path} was written by parser {version} and the installed one is "
            f"{h.PARSER_VERSION}. Measuring on it would give another parser's numbers: "
            f"rerun scripts/run_t1.py before using it."
        )

    try:
        doc = ProcessedDocument.model_validate(raw)
    except ValidationError as e:
        raise ProcessedError(f"{path} does not meet the processed layer contract.\n{e}") from e

    # --- axis 2: the corpus it came from changed
    entry = next((d for d in manifest if d.celex == doc.source_celex), None)
    if entry is None:
        raise StaleLayerError(
            f"{path} claims to come from celex {doc.source_celex}, which is no longer in "
            f"the manifest. Rerun the download and scripts/run_t1.py."
        )
    if entry.sha256 != doc.source_sha256:
        raise StaleLayerError(
            f"{path} came from snapshot {doc.source_sha256[:12]} and the manifest now "
            f"has {entry.sha256[:12]} for {doc.source_celex}: the corpus was downloaded "
            f"again and changed. Rerun scripts/run_t1.py."
        )
    return doc
