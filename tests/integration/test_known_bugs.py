"""Regression tests: each one comes from a real incident that already happened,
described in the measurement log under M-004 (text defects) and M-006 (a D-004
limitation).

These tests do not test an isolated function (that is already in
tests/unit/test_text_cleaning.py with minimal cases): they test that the
specific bug, with the complete real article that suffered it, stays gone.
"""

from __future__ import annotations

import re

import pytest

from ingest import hierarchy as h
from tests.conftest import fixture_path

_REF_RX = re.compile(r"Articles? \d+[a-z]?\(\d{1,2}\)")


def test_dora_art60_not_truncated():
    """DORA Art. 60 (Regulation (EU) No 648/2012 amended): with the M-004 bug,
    the footnote rule with an optional asterisk swallowed the first literal
    "(*1)" of the quoted replacement text and truncated EVERYTHING that came
    after it. The loss measured then: 594 of 677 words."""
    article = h.parse(fixture_path("dora_art60"), "doue")[0]
    words = article.text.split()
    assert len(words) == 677
    # the final sentence of the article, which the bug swallowed whole
    assert article.text.endswith(
        "with certainty and to complete settlement on the scheduled date;'."
    )


def test_psd2_art111_not_truncated():
    """PSD2 Art. 111 (Directive 2009/110/EC amended): same bug, loss measured
    then: 398 of 459 words."""
    article = h.parse(fixture_path("psd2_art111"), "consolidado")[0]
    words = article.text.split()
    assert len(words) == 459
    assert article.text.endswith(
        "July 2018 they shall be prohibited from issuing electronic money.'."
    )


@pytest.mark.parametrize(
    "name,family,anchor_id_pattern,surviving_legal_reference",
    [
        # <a id="ntc*2-L_2022333EN.01000101-E0041" href="#ntr*2-...">, doue family
        ("dora_art60", "doue", re.compile(r'id="ntc'), "Article 79(1)"),
        # <a id="src.E0010" href="#E0010">, consolidado family
        ("psd2_art111", "consolidado", re.compile(r'id="src\.E'), "Article 15(4)"),
    ],
)
def test_footnote_anchor_preserved_and_distinguished_from_legal_reference(
    name, family, anchor_id_pattern, surviving_legal_reference
):
    """Each of these two fixtures carries ONE real footnote (citing the Official
    Journal of the act being amended) in the SAME article that contains several
    "Article N(M)" legal references. It differs from
    test_references_survive_full_parse (which counts the total): here it checks
    that the legitimate footnote is recognised and survives exactly once,
    without swallowing a specific legal citation right next to it.

    The two markup families use different id conventions for the footnote
    anchor (see _FN_ANCHOR_ATTRS in ingest/hierarchy.py), so both are checked
    separately instead of just once."""
    raw_xhtml = open(fixture_path(name), encoding="utf-8").read()
    assert anchor_id_pattern.search(raw_xhtml), "the fixture lacks the expected real anchor"

    article = h.parse(fixture_path(name), family)[0]
    assert article.text.count("<FN>") == 1
    assert surviving_legal_reference in article.text


@pytest.mark.parametrize(
    "name,family,expected_refs",
    [
        ("dora_art60", "doue", 5),
        ("psd2_art111", "consolidado", 8),
    ],
)
def test_references_survive_full_parse(name, family, expected_refs):
    """Independent control (the same as in scripts/measure_nodes.py, but at
    fixture level): it counts "Article N(M)" in Article.text after the FULL
    parse, not only after _normalize in isolation. It is the control that would
    really have caught the M-004 bug at the time, because it does not depend on
    any cleaning function to know how many references "should" survive: it
    counts the final pattern."""
    article = h.parse(fixture_path(name), family)[0]
    assert len(_REF_RX.findall(article.text)) == expected_refs


def test_psd2_art9_duplicate_paths():
    """CHARACTERIZATION test, not a correctness one. PSD2 Art. 9(1) contains
    "Method A", "Method B" and "Method C": three sibling lists under the same
    paragraph that restart the lettering from (a). The materialized path of
    D-004 (article.paragraph.point) does not tell which method a point belongs
    to, so 9.1.a and 9.1.b exist twice with different content (M-006,
    finding 1).

    This test does NOT fix anything: it pins the current behaviour on purpose.
    If D-004 is reopened to disambiguate Method A/B/C, this test must START
    FAILING and has to be updated by hand, not left to pass by accident.
    See decision D-004, open point."""
    article = h.parse(fixture_path("psd2_art9"), "consolidado")[0]
    paths = [n.path for n in article.nodes]
    assert paths.count("9.1.a") == 2
    assert paths.count("9.1.b") == 2
    assert paths.count("9.1.c") == 1  # Method C does not reach (c): no collision here
