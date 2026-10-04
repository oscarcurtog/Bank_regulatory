"""Unit tests: text cleaning functions in isolation, without touching disk.

Every test in this section runs in microseconds and depends on no file. The
ones with "regression" in their name come from a real incident described in
the measurement log under M-004 and in the header comment of
ingest/hierarchy.py (1.3.0).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from ingest import hierarchy as h

# ---------------------------------------------------------------------------
# Bug 1 (M-004): a footnote rule with an OPTIONAL asterisk mistook
# "Article 4(1)" for a footnote reference "(*1)" and destroyed it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Article 4(1) applies.",
        "Article 35(6) where the representatives.",
        "Articles 7(2) and 9(3) apply.",
        "Article 110a(3) applies.",
        "Point (12) is replaced.",
    ],
)
def test_legal_reference_survives_normalization(text):
    """Regression: bug 1 of M-004. A legal citation without an asterisk is not
    a footnote and _normalize must not touch it: the string comes out exactly
    as it went in."""
    assert h._normalize(text) == text


def test_footnote_definition_with_asterisk_is_removed():
    """The rule MUST still remove a real footnote definition, which in the
    corpus always carries an asterisk (D-002: EUR-Lex marks them that way)."""
    text = "Article 4(1) applies. ( *1 ) OJ L 123, 1.1.2020, p. 1."
    out = h._normalize(text)
    assert out == "Article 4(1) applies."
    assert "OJ L 123" not in out


def test_asterisk_is_the_discriminator():
    """Pins the exact criterion that separates the two cases above: the
    asterisk is mandatory for something to count as a footnote definition."""
    with_ref_and_footnote = "Article 4(1) applies. ( *1 ) OJ L 1."
    out = h._normalize(with_ref_and_footnote)
    assert "Article 4(1)" in out
    assert "*1" not in out
    assert "OJ L 1" not in out


def test_normalize_converts_fn_sentinel_to_placeholder():
    """The sentinel that _preclean leaves where there was a real footnote
    reference (see _is_fn_anchor) becomes <FN>: it never disappears without a
    trace, nor is it mistaken for a legal citation."""
    text = f"shall comply ({h.SENT}) with this Article."
    out = h._normalize(text)
    assert "<FN>" in out
    assert h.SENT not in out


# ---------------------------------------------------------------------------
# Bug 2 (M-004): ET.tostring() serializes with an "html:" prefix. The tag
# regexes of _clean_text did not expect it and matched nothing since the port,
# which left footnote detection without ever running.
# ---------------------------------------------------------------------------


def _serialize_with_namespace(inner_xml: str) -> str:
    """Builds a tree WITH the XHTML namespace and serializes it the way
    ingest.hierarchy.parse() does, so that the html: prefix really appears and
    is not assumed by hand in the test."""
    root = ET.fromstring(f'<div xmlns="http://www.w3.org/1999/xhtml" id="art_1">{inner_xml}</div>')
    return ET.tostring(root, encoding="unicode")


@pytest.mark.parametrize(
    "inner_xml,expected_substring,forbidden_substring",
    [
        # a real footnote reference becomes <FN>, not garbage
        (
            "<p>Financial entities shall comply with Article 4(1) "
            '<a id="ntc1-x" href="#ntr1-x"><sup>(1)</sup></a> of this Regulation.</p>',
            "Article 4(1)",
            "ntc1-x",
        ),
        # a note block (class="note") is removed entirely, not twice
        (
            "<p>Main text continues here.</p>"
            '<p class="note">Definition of a note that must disappear.</p>',
            "Main text continues here.",
            "must disappear",
        ),
        # a loose <sup> (superscript note number without an anchor) leaves no trace
        (
            "<p>Shall comply with the requirement<sup>3</sup> set out above.</p>",
            "Shall comply with the requirement",
            "<sup",
        ),
    ],
)
def test_clean_text_with_namespaced_tags(inner_xml, expected_substring, forbidden_substring):
    """Regression: bug 2 of M-004. If _clean_text does not remove the html:
    prefix before applying its tag regexes, NONE of them matches and the
    cleaning fails silently (the text comes out with the tags inside)."""
    namespaced_xml = _serialize_with_namespace(inner_xml)
    assert "<html:" in namespaced_xml  # confirms the test really reproduces the bug
    out = h._clean_text(namespaced_xml)
    assert expected_substring in out
    assert forbidden_substring not in out
    assert "<html:" not in out
    assert "<p" not in out  # no tag may survive


# ---------------------------------------------------------------------------
# The 1.2.0 bug: 'level' came from a manual counter that did not match the path
# segments for 2-segment paths (DORA art_46, MiCA art_54).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected_level",
    [
        ("45", 1),
        ("45.1", 2),
        ("45.1.a", 3),
        ("45.1.a.i", 4),
        ("9.1.a.iii", 4),
    ],
)
def test_level_of_path(path, expected_level):
    """Invariant: level = number of path segments, always. The 1.2.0 bug gave
    level=3 to 2-segment paths through a separate counter."""
    assert h._level_of(path) == expected_level


# ---------------------------------------------------------------------------
# DORA Art. 60 has a "'c" marker (a quote plus a letter, because it is a point
# quoted inside an amending article). Without a whitelist, "'c" is not a valid
# ltree label (D-004) and breaks the index.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_marker,expected",
    [
        ("c", "c"),
        ("'c", "c"),
        ("(a)", "a"),
        ("i.", "i"),
        ("12", "12"),
        ("'12", "12"),
        ("iii", "iii"),
    ],
)
def test_marker_whitelist_strips_non_alphanumeric(raw_marker, expected):
    assert h._MARKER_KEEP.sub("", raw_marker) == expected
