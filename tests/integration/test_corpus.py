"""Tests marked "corpus": they need data/raw/, which is in .gitignore under
D-002 and therefore does NOT exist in a clean checkout or on the GitHub Actions
runner (EF-7). tests/conftest.py skips them automatically when the directory is
missing, instead of failing for a reason that is not a bug.

When the corpus is present, they run and validate against the 331 real
articles, not only against the 5 fixtures.
"""

from __future__ import annotations

import re

import pytest

from ingest import download as dl
from ingest import hierarchy as h
from tests.conftest import MANIFEST_PATH, fixture_path, reconstruct

pytestmark = pytest.mark.corpus

_REF_RX = re.compile(r"Articles? \d+[a-z]?\(\d{1,2}\)")

# fixture name -> (act, article_id, family). The same mapping that was used to
# extract them (see tests/fixtures/PROVENANCE.md). The family also says which
# document the excerpt comes from: doue = the original text, consolidado = the
# consolidated one.
FIXTURES = [
    ("dora_art45", "DORA", "art_45", "doue"),
    ("dora_art60", "DORA", "art_60", "doue"),
    ("dora_art15", "DORA", "art_15", "doue"),
    ("dora_art36", "DORA", "art_36", "doue"),
    ("psd2_art69", "PSD2", "art_69", "consolidado"),
    ("psd2_art111", "PSD2", "art_111", "consolidado"),
    ("psd2_art9", "PSD2", "art_9", "consolidado"),
    ("psd2_art9_original", "PSD2", "art_9", "doue"),
]


def _docs():
    """Manifest already validated (EF-3). If an entry breaks the contract,
    load_manifest aborts and no test gets to parse anything."""
    return dl.load_manifest(MANIFEST_PATH)


def _source_and_family(act):
    source = dl.parsed_source_for(act, _docs())
    return source, ("consolidado" if source.role == "consolidated" else "doue")


def _document_for(act, family):
    """The document a fixture was cut from."""
    role = "original" if family == "doue" else "consolidated"
    return next(d for d in _docs() if d.act == act and d.role == role)


def _family_of(document):
    return "consolidado" if document.role == "consolidated" else "doue"


@pytest.mark.parametrize("fixture_name,act,article_id,family", FIXTURES)
def test_fixtures_match_corpus(fixture_name, act, article_id, family):
    """A fixture is a real excerpt, not an approximation, only if it keeps
    giving the same result as the full document. This is what stops the
    fixtures from silently ageing if the corpus is downloaded again."""
    fixture_article = h.parse(fixture_path(fixture_name), family)[0]

    source = _document_for(act, family)
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


@pytest.mark.parametrize("celex", [d.celex for d in dl.load_manifest(MANIFEST_PATH)])
def test_every_raw_document_parses_with_unique_paths(celex):
    """The five downloaded documents, the two originals that do not feed the
    processed layer included. Article rejects a repeated path, so parsing at all
    already proves it; the explicit check says what is being proved."""
    document = next(d for d in _docs() if d.celex == celex)
    for article in h.parse(document.local_path, _family_of(document)):
        paths = [n.path for n in article.nodes]
        assert len(paths) == len(set(paths)), article.article_id


def test_unmarked_items_are_only_the_psd2_art9_methods(monkeypatch):
    """The doue extension of 1.4.0 (an EMPTY first cell makes a table a
    transparent container) rests on 3 tables, all in PSD2 Art. 9(1) of the
    original text. If a new download or a new act brings more, this fails, so
    that the narrow rule is looked at again before it silently applies to them."""
    seen = []
    original = h._is_unmarked_item

    def spy(table):
        result = original(table)
        if result:
            seen.append(" ".join(h._text(table).split())[:8])
        return result

    monkeypatch.setattr(h, "_is_unmarked_item", spy)
    counts = {}
    for document in _docs():
        if document.role == "original":
            before = len(seen)
            h.parse(document.local_path, "doue")
            counts[document.act] = len(seen) - before
    assert counts == {"DORA": 0, "MiCA": 0, "PSD2": 3}
    assert seen == ["Method A", "Method B", "Method C"]


# Articles whose reconstruction is NOT identical to Article.text in parser 1.4.0,
# each with its explained reason (T2 phase 1):
#   own text of a POINT after its sub-points: inside a point nothing is split
#   (DORA 20.a, DORA 35.1.d, PSD2 3.l, PSD2 9.1.sub_7.a);
#   marker format, as in M-006: quoted markers written "'(a)" in amending
#   articles, and points written "(1)" where the text has "1.".
RECONSTRUCTION_EXCEPTIONS = {
    "DORA": {"art_20", "art_35", "art_59", "art_60", "art_61", "art_62", "art_63"},
    "MiCA": {"art_146", "art_147"},
    "PSD2": {"art_3", "art_9", "art_52", "art_113"},
}


def test_reconstruction_full_corpus():
    """The M-006 invariant on all 331 articles, not only on the fixtures: 318 of
    them rebuild Article.text exactly from their nodes (254 with parser 1.3.0).
    The 13 that do not are pinned, so that any change in either direction has to
    be looked at."""
    identical = 0
    exceptions: dict[str, set[str]] = {}
    for act in ("DORA", "MiCA", "PSD2"):
        source, family = _source_and_family(act)
        for article in h.parse(source.local_path, family):
            if reconstruct(article) == article.text:
                identical += 1
            else:
                exceptions.setdefault(act, set()).add(article.article_id)
    assert exceptions == RECONSTRUCTION_EXCEPTIONS
    assert identical == 318
