"""The PSD2 exclusion list (D-019): its contract, the two controls that guard its
derivation, the reading of EUR-Lex consolidation markers, and the versioned list.

All of it runs in any checkout. Rebuilding the list from the real documents needs
data/raw/ and data/reference/, and is the job of
`scripts/build_psd2_exclusions.py --check` (recorded in the measurement log).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ingest import download as dl
from ingest import hierarchy as h
from ingest import psd2_exclusions as px
from tests.conftest import MANIFEST_PATH

SYNTHETIC = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<table><tr><td><p class="arrow"><a title="32022L2556">►M1</a></p></td></tr></table>
<div class="eli-subdivision" id="art_1">
  <p class="modref"><a title="32015L2366">▼B</a></p><p>Base.</p></div>
<div class="eli-subdivision" id="art_2">
  <p class="modref"><a title="32022L2556: REPLACED">▼M1</a></p>
  <p>Replaced, and the block runs on.</p></div>
<div class="eli-subdivision" id="art_2a">
  <p>Inserted inside the M1 block, with no marker of its own.</p></div>
<div class="eli-subdivision" id="art_3">
  <p class="modref"><a title="32015L2366">▼B</a></p>
  <p>Opens with its own return to the base text, with an
  <a title="32024R0886">►M2</a> span ◄.</p></div>
<div class="eli-subdivision" id="art_4">
  <p class="modref"><a title="32015L2366R(05)">▼C1</a></p><p>Corrected.</p></div>
<div class="eli-subdivision" id="art_5"><p>Base, after the corrigendum block.</p></div>
</body></html>
"""


def test_markers_follow_the_eur_lex_block_semantics(tmp_path):
    path = tmp_path / "synthetic.xhtml"
    path.write_text(SYNTHETIC, encoding="utf-8")
    assert px.amendment_markers(path) == {
        "2": {"32022L2556"},
        "2a": {"32022L2556"},  # inherits the block it opens in
        "3": {"32024R0886"},  # its own inline span; its ▼B stops the M1 inheritance
    }  # 1 base, 4 a corrigendum, 5 base again; the header ►M1 belongs to no article


def _articles(texts: dict[str, str]) -> list[h.Article]:
    return [h.Article(article_id=f"art_{p}", path=p, title="T", text=t) for p, t in texts.items()]


def test_the_amended_articles_are_explained_by_their_markers():
    base = _articles({"1": "a", "2": "b", "10": "c"})
    reference = _articles({"1": "a", "2": "B", "10": "C", "2a": "new"})
    markers = {"2": {"32022L2556"}, "2a": {"32022L2556"}, "10": {"32024R0886"}}
    result = px.amended_articles(base, reference, markers)
    assert [(a.article, a.change, a.amended_by) for a in result] == [
        ("2", "modified", ["32022L2556"]),
        ("2a", "inserted", ["32022L2556"]),
        ("10", "modified", ["32024R0886"]),  # numeric order, not text order
    ]


@pytest.mark.parametrize(
    "reference,markers,message",
    [
        ({"1": "A", "2": "b"}, {}, r"changed without an amendment marker \['1'\]"),
        ({"1": "a", "2": "b"}, {"2": {"32022L2556"}}, r"with a marker but unchanged \['2'\]"),
        ({"1": "a"}, {}, r"missing from the reference: \['2'\]"),
    ],
)
def test_a_derivation_that_a_control_contradicts_is_refused(reference, markers, message):
    with pytest.raises(px.ExclusionsError, match=message):
        px.amended_articles(_articles({"1": "a", "2": "b"}), _articles(reference), markers)


def _list(**overrides) -> dict:
    record = {
        "schema_version": "1",
        "base": {"celex": "02015L2366-20151223", "sha256": "a" * 64},
        "reference": {"celex": "02015L2366-20250117", "sha256": "b" * 64},
        "parser_version": "1.4.0",
        "articles": [{"article": "96", "change": "modified", "amended_by": ["32022L2556"]}],
    }
    return record | overrides


@pytest.mark.parametrize(
    "overrides",
    [
        {"articles": [{"article": "96", "change": "modified", "amended_by": ["32022L2556"]}] * 2},
        {"articles": [{"article": "96", "change": "modified", "amended_by": []}]},
        {"articles": [{"article": "96", "change": "modified", "amended_by": ["2022/2556"]}]},
        {"articles": [{"article": "Art. 96", "change": "modified", "amended_by": ["32022L2556"]}]},
        {"articles": [{"article": "96", "change": "deleted", "amended_by": ["32022L2556"]}]},
        {"schema_version": "2"},
        {"extra": 1},
    ],
)
def test_the_list_contract_rejects(overrides):
    with pytest.raises(ValidationError):
        px.Psd2Exclusions.model_validate(_list(**overrides))


def test_a_missing_list_says_how_to_build_it(tmp_path):
    with pytest.raises(px.ExclusionsError, match="Run scripts/build_psd2_exclusions.py"):
        px.load_psd2_exclusions(tmp_path / "absent.json")


def test_the_versioned_list():
    """The nine articles that M-001 found changed after 2015, now derived and
    explained marker by marker: six by Directive 2022/2556 (the DORA package) and
    three by Regulation 2024/886 (instant payments), Article 35a among them."""
    exclusions = px.load_psd2_exclusions()
    by_act: dict[str, list[str]] = {}
    for a in exclusions.articles:
        for act in a.amended_by:
            by_act.setdefault(act, []).append(a.article)
    assert by_act == {
        "32022L2556": ["3", "5", "19", "95", "96", "98"],
        "32024R0886": ["10", "35", "35a"],
    }
    assert [a.article for a in exclusions.articles if a.change == "inserted"] == ["35a"]
    assert exclusions.reference.celex == px.REFERENCE_CELEX


def test_the_list_was_built_on_the_psd2_snapshot_fundamento_parses():
    base = dl.parsed_source_for("PSD2", dl.load_manifest(MANIFEST_PATH))
    exclusions = px.load_psd2_exclusions()
    assert (exclusions.base.celex, exclusions.base.sha256) == (base.celex, base.sha256)
