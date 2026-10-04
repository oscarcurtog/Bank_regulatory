"""Tests marked "corpus": they need data/raw/, which is in .gitignore under
D-002 and therefore does NOT exist in a clean checkout or on the GitHub Actions
runner (EF-7). tests/conftest.py skips them automatically when the directory is
missing, instead of failing for a reason that is not a bug.

On this machine, with the corpus downloaded, they run and validate against the
331 real articles, not only against the 5 fixtures.
"""

from __future__ import annotations

import re

import pytest

from ingest import download as dl
from ingest import hierarchy as h
from tests.conftest import MANIFEST_PATH, fixture_path

pytestmark = pytest.mark.corpus

_REF_RX = re.compile(r"Articles? \d+[a-z]?\(\d{1,2}\)")

# fixture name -> (act, article_id, family). The same mapping that was used to
# extract them (see tests/fixtures/PROVENANCE.md).
FIXTURES = [
    ("dora_art45", "DORA", "art_45", "doue"),
    ("dora_art60", "DORA", "art_60", "doue"),
    ("psd2_art69", "PSD2", "art_69", "consolidado"),
    ("psd2_art111", "PSD2", "art_111", "consolidado"),
    ("psd2_art9", "PSD2", "art_9", "consolidado"),
]


def _docs():
    """Manifest already validated (EF-3). If an entry breaks the contract,
    load_manifest aborts and no test gets to parse anything."""
    return dl.load_manifest(MANIFEST_PATH)


def _source_and_family(act):
    source = dl.parsed_source_for(act, _docs())
    return source, ("consolidado" if source.role == "consolidated" else "doue")


@pytest.mark.parametrize("fixture_name,act,article_id,family", FIXTURES)
def test_fixtures_match_corpus(fixture_name, act, article_id, family):
    """A fixture is a real excerpt, not an approximation, only if it keeps
    giving the same result as the full document. This is what stops the
    fixtures from silently ageing if the corpus is downloaded again."""
    fixture_article = h.parse(fixture_path(fixture_name), family)[0]

    source = dl.parsed_source_for(act, _docs())
    full_article = next(a for a in h.parse(source.local_path, family) if a.article_id == article_id)

    assert fixture_article.text == full_article.text
    assert [n.path for n in fixture_article.nodes] == [n.path for n in full_article.nodes]


@pytest.mark.parametrize(
    "act,expected_refs",
    [("DORA", 138), ("MiCA", 217), ("PSD2", 122)],
)
def test_reference_control_full_corpus(act, expected_refs):
    """The independent control of M-004/M-006 at full scale: it counts
    "Article N(M)" in the Article.text of ALL the act's articles, not only in
    the two regression fixtures. It is the number that would have caught the
    M-004 bug without relying on someone looking at DORA Art. 60 by hand."""
    source, family = _source_and_family(act)
    articles = h.parse(source.local_path, family)
    total = sum(len(_REF_RX.findall(a.text)) for a in articles)
    assert total == expected_refs


@pytest.mark.parametrize(
    "act,expected_count",
    [("DORA", 64), ("MiCA", 150), ("PSD2", 117)],
)
def test_article_counts(act, expected_count):
    """Number 1 of the T1 definition of done (project plan), restated as a
    test instead of living only in a report JSON."""
    source, family = _source_and_family(act)
    articles = h.parse(source.local_path, family)
    assert len(articles) == expected_count


def test_total_article_count():
    total = 0
    for act in ("DORA", "MiCA", "PSD2"):
        source, family = _source_and_family(act)
        total += len(h.parse(source.local_path, family))
    assert total == 331
