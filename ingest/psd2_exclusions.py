"""PSD2 articles amended after the snapshot that Fundamento parses (D-019).

The corpus parses PSD2 from its oldest consolidated version, 02015L2366-20151223
(D-003), which already carries the corrigenda (M-002) but none of the later
amendments. A benchmark need that depends on an amended article could be right for
this corpus and wrong for the law in force, so no qrel may point into one.

The list is derived, never typed by hand. The base is compared article by article
with a later consolidated version, both through the same parser, and two controls
must hold, as in M-001:

1. every article whose text differs carries an EUR-Lex amendment marker (M<n>);
2. every article that carries an amendment marker has a different text.

scripts/build_psd2_exclusions.py writes data/benchmark/psd2_excluded_articles.json
from a reference document kept in data/reference/ (not versioned). The ingestion
manifest is never touched, so the version of PSD2 that Fundamento parses does not
change: with two consolidated versions in data/manifest.json, parsed_source_for()
would pick one by list order.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ingest import download as dl
from ingest import hierarchy as h
from ingest import processed as pr

PSD2_EXCLUSIONS_PATH = Path("data/benchmark/psd2_excluded_articles.json")
REFERENCE_DIR = Path("data/reference")
REFERENCE_CELEX = "02015L2366-20250117"

# Consolidation markers. A "▼" opens a block of text that comes from one act and
# runs until the next "▼": ▼B for the base act, ▼M<n> for an amending act, ▼C<n>
# for a corrigendum. "►M<n> ... ◄" marks a span inserted inside a block. The
# anchor's title starts with the CELEX of the act.
_BLOCK = re.compile(r"^▼[BMC]\d*$")
_AMENDMENT = re.compile(r"^[▼►]M\d+$")


class ExclusionsError(Exception):
    """The PSD2 exclusion list is missing, broken, or cannot be derived reliably."""


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    celex: str = Field(pattern=dl.CELEX_PATTERN)
    sha256: str = Field(pattern=dl.SHA256_PATTERN)


class ExcludedArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article: str = Field(pattern=r"^\d+[a-z]?$")
    change: Literal["modified", "inserted"]
    # CELEX of the acts that amended it, read from the consolidation markers
    amended_by: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _celex(self) -> Self:
        bad = [c for c in self.amended_by if not re.fullmatch(dl.CELEX_PATTERN, c)]
        if bad:
            raise ValueError(f"article {self.article}: not CELEX numbers {bad}")
        return self


class Psd2Exclusions(BaseModel):
    """Declaration order is serialization order."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    base: Document  # the PSD2 source the corpus parses
    reference: Document  # the later consolidated version it was compared with
    parser_version: str = Field(pattern=pr.VERSION_PATTERN)
    articles: list[ExcludedArticle]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        names = [a.article for a in self.articles]
        if len(set(names)) != len(names):
            raise ValueError("an article appears twice")
        return self

    def excluded(self) -> frozenset[str]:
        return frozenset(a.article for a in self.articles)


def load_psd2_exclusions(path: Path = PSD2_EXCLUSIONS_PATH) -> Psd2Exclusions:
    if not path.exists():
        raise ExclusionsError(f"{path} does not exist. Run scripts/build_psd2_exclusions.py.")
    try:
        return Psd2Exclusions.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ExclusionsError(f"{path} does not meet the exclusion list contract.\n{e}") from e


def write_psd2_exclusions(exclusions: Psd2Exclusions, path: Path = PSD2_EXCLUSIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(exclusions.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def amendment_markers(xhtml_path: Path) -> dict[str, set[str]]:
    """article -> CELEX of the amending acts behind its text: the amendment markers
    inside the article, and, for an article with no block marker of its own, the
    block in force where it starts. An inserted article usually opens inside the
    block that inserted it (PSD2 35a); an article that opens with its own marker,
    such as a ▼B back to the base text, inherits nothing."""
    root = ET.fromstring(xhtml_path.read_text(encoding="utf-8"))
    found: dict[str, set[str]] = {}
    in_force: str | None = None  # the last block marker in document order
    for element in root.iter():
        if element.tag == h.NS + "a":
            text = (element.text or "").strip()
            if _BLOCK.match(text):
                title = (element.get("title") or "").split(":")[0].strip()
                in_force = title if text.startswith("▼M") else None
            continue
        div_id = element.get("id", "") if element.tag == h.NS + "div" else ""
        if not div_id.startswith("art_") or "#" in div_id:
            continue
        if "eli-subdivision" not in (element.get("class", "") or ""):
            continue
        article = div_id[len("art_") :]
        inside = [a for a in element.iter(h.NS + "a") if (a.text or "").strip()[:1] in "▼►"]
        acts = {
            (a.get("title") or "").split(":")[0].strip()
            for a in inside
            if _AMENDMENT.match((a.text or "").strip())
        }
        if in_force and not any(_BLOCK.match((a.text or "").strip()) for a in inside):
            acts.add(in_force)
        if acts:
            found[article] = acts
    return found


def fetch_reference(celex: str = REFERENCE_CELEX, directory: Path = REFERENCE_DIR) -> Path:
    """Downloads a reference version from CELLAR with the same checks as the corpus
    download (status 200 and a realistic size), into data/reference/, never into
    data/raw/ and never into the manifest."""
    body, status = dl._fetch(celex)
    if status != 200 or len(body) < dl.MIN_VALID_BYTES:
        raise ExclusionsError(f"{celex}: invalid response (status={status}, {len(body)} bytes)")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{celex}.xhtml"
    path.write_bytes(body)
    return path


def _article_order(article: str) -> tuple[int, str]:
    digits = re.match(r"\d+", article)
    return (int(digits.group()) if digits else 0, article)


def amended_articles(
    base: Sequence[h.Article], reference: Sequence[h.Article], markers: dict[str, set[str]]
) -> list[ExcludedArticle]:
    """The articles of the base that the reference changed or added, each one
    explained by its markers. Raises if either control of M-001 fails: a derived
    list that a control contradicts is not used."""
    base_text = {a.path: a.text for a in base}
    ref_text = {a.path: a.text for a in reference}
    removed = sorted(set(base_text) - set(ref_text))
    if removed:
        raise ExclusionsError(f"articles of the base missing from the reference: {removed}")
    changed = {p for p in ref_text if base_text.get(p) != ref_text[p]}
    unexplained = sorted(changed - set(markers))
    silent = sorted(set(markers) - changed)
    if unexplained or silent:
        raise ExclusionsError(
            f"the controls fail: changed without an amendment marker {unexplained}; with a "
            f"marker but unchanged {silent}"
        )
    return [
        ExcludedArticle(
            article=p,
            change="inserted" if p not in base_text else "modified",
            amended_by=sorted(markers[p]),
        )
        for p in sorted(changed, key=_article_order)
    ]
