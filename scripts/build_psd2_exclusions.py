"""Derive data/benchmark/psd2_excluded_articles.json (D-019): the PSD2 articles that
later acts amended after the snapshot Fundamento parses.

    PYTHONPATH=. .venv/bin/python scripts/build_psd2_exclusions.py
    PYTHONPATH=. .venv/bin/python scripts/build_psd2_exclusions.py --check

The base is the PSD2 source of the parsed layer, as data/manifest.json names it
(D-003). The reference, a later consolidated version, is downloaded once from
CELLAR into data/reference/ (not versioned) and its SHA-256 goes into the list. If
a versioned list already exists, the reference on disk must have its SHA-256: a
different file from EUR-Lex is reported, not silently compared.

--check rebuilds the list from the local files, downloads nothing, and fails if
the result differs from the versioned list. Nothing here touches the ingestion
manifest, so the version of PSD2 that Fundamento parses does not change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ingest import benchmark as bm
from ingest import download as dl
from ingest import hierarchy as h
from ingest import psd2_exclusions as px


def build(check: bool) -> px.Psd2Exclusions:
    base = dl.parsed_source_for("PSD2", dl.load_manifest())
    base_path = Path(base.local_path)
    if bm.sha256_file(base_path) != base.sha256:
        raise px.ExclusionsError(f"{base_path} is not the file the manifest recorded")
    reference = px.REFERENCE_DIR / f"{px.REFERENCE_CELEX}.xhtml"
    current = px.load_psd2_exclusions() if px.PSD2_EXCLUSIONS_PATH.exists() else None
    if not reference.exists():
        if check:
            raise px.ExclusionsError(f"{reference} is missing; run without --check to download it")
        reference = px.fetch_reference()
    reference_sha = bm.sha256_file(reference)
    if current is not None and current.reference.sha256 != reference_sha:
        raise px.ExclusionsError(
            f"{reference} has SHA-256 {reference_sha[:12]}, the versioned list was built from "
            f"{current.reference.sha256[:12]}: EUR-Lex served another file; review it first"
        )
    articles = px.amended_articles(
        h.parse(str(base_path), "consolidado"),
        h.parse(str(reference), "consolidado"),
        px.amendment_markers(reference),
    )
    return px.Psd2Exclusions(
        schema_version="1",
        base=px.Document(celex=base.celex, sha256=base.sha256),
        reference=px.Document(celex=px.REFERENCE_CELEX, sha256=reference_sha),
        parser_version=h.PARSER_VERSION,
        articles=articles,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="rebuild and compare, write nothing")
    args = parser.parse_args(argv)
    exclusions = build(args.check)
    for a in exclusions.articles:
        print(f"  PSD2 Article {a.article}: {a.change} by {', '.join(a.amended_by)}")
    if args.check:
        if exclusions != px.load_psd2_exclusions():
            print(f"{px.PSD2_EXCLUSIONS_PATH} differs from the rebuilt list", file=sys.stderr)
            return 1
        print(f"{px.PSD2_EXCLUSIONS_PATH} matches the rebuilt list")
        return 0
    px.write_psd2_exclusions(exclusions)
    print(f"written {px.PSD2_EXCLUSIONS_PATH}: {len(exclusions.articles)} articles excluded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
