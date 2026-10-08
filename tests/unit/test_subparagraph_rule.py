"""Unit tests: the subparagraph rule of parser 1.4.0 on minimal synthetic markup,
one structural situation per test.

The markup copies the shapes of the real documents: doue paragraphs are
<div id="NNN.NNN"> with <p class="oj-normal"> blocks and one-row <table> points;
consolidated paragraphs are a span.no-parag plus a div.norm.inline-element, with
grid-list points. The real cases are in tests/fixtures/ and
tests/integration/test_subparagraphs.py. Synthetic markup is used here mainly for
the shapes the corpus does not contain (an indent or a two-row table outside a
point, an unknown block carrier), which are the ones that must NOT trigger the
rule.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from pydantic import ValidationError

from ingest import hierarchy as h

XHTML = "http://www.w3.org/1999/xhtml"


def _article(body: str) -> ET.Element:
    return ET.fromstring(f'<div xmlns="{XHTML}" class="eli-subdivision" id="art_7">{body}</div>')


def _p(text: str, cls: str = "oj-normal") -> str:
    return f'<p class="{cls}">{text}</p>'


def _table(*cells: str, rows: int = 1) -> str:
    row = "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    return f"<table><tbody>{row * rows}</tbody></table>"


def _doue_point(marker: str, text: str) -> str:
    return _table(_p(marker), _p(text))


def _doue_paragraph(n: int, inner: str) -> str:
    return f'<div id="007.{n:03d}">{inner}</div>'


def _cons_point(marker: str, text: str) -> str:
    return (
        '<div class="grid-container grid-list"><div class="list grid-list-column-1">'
        f"<span>{marker} </span></div>"
        f'<div class="grid-list-column-2">{_p(text, "norm")}</div></div>'
    )


def _cons_paragraph(n: int, inner: str) -> str:
    return (
        f'<div class="norm"><span class="no-parag">{n}.  </span>'
        f'<div class="norm inline-element">{inner}</div></div>'
    )


def _doue(body: str) -> list[h.Node]:
    return h._nodes_doue(_article(body), "7")[0]


def _cons(body: str) -> list[h.Node]:
    return h._nodes_consolidado(_article(body), "7")[0]


def _paths(nodes: list[h.Node]) -> list[str]:
    return [n.path for n in nodes]


def _by(nodes: list[h.Node]) -> dict[str, h.Node]:
    return {n.path: n for n in nodes}


# ---------------------------------------------------------------------------
# doue family
# ---------------------------------------------------------------------------


def test_doue_one_block_and_its_list_keep_their_paths():
    """The common case: a lead-in and its points are subparagraph 1, which is
    implicit, so the paths are the ones of 1.3.0."""
    nodes, depth = h._nodes_doue(
        _article(
            _doue_paragraph(
                1,
                _p("1.   Member States shall:")
                + _doue_point("(a)", "one;")
                + _doue_point("(b)", "two."),
            )
        ),
        "7",
    )
    assert _paths(nodes) == ["7", "7.1", "7.1.a", "7.1.b"]
    assert _by(nodes)["7.1"].own_text == "Member States shall:"
    assert depth == 3


def test_doue_second_block_opens_a_subparagraph_that_owns_its_list():
    nodes, depth = h._nodes_doue(
        _article(
            _doue_paragraph(
                1,
                _p("1.   First subparagraph.")
                + _p("For the purposes of the first subparagraph:")
                + _doue_point("(a)", "one."),
            )
        ),
        "7",
    )
    assert _paths(nodes) == ["7", "7.1", "7.1.sub_2", "7.1.sub_2.a"]
    sub = _by(nodes)["7.1.sub_2"]
    assert (sub.kind, sub.marker, sub.level) == ("subparagraph", "2", 3)
    assert sub.own_text == "For the purposes of the first subparagraph:"
    assert _by(nodes)["7.1"].own_text == "First subparagraph."
    assert depth == 4


def test_doue_block_after_a_list_is_a_subparagraph_after_its_points():
    """Up to 1.3.0 this closing text joined the lead-in and came out before the
    points. Now it is the second subparagraph, in its place."""
    nodes = _doue(
        _doue_paragraph(
            1,
            _p("1.   Member States shall:")
            + _doue_point("(a)", "one.")
            + _p("Member States shall notify the Commission."),
        )
    )
    assert _paths(nodes) == ["7", "7.1", "7.1.a", "7.1.sub_2"]
    assert _by(nodes)["7.1"].own_text == "Member States shall:"
    assert _by(nodes)["7.1.sub_2"].own_text == "Member States shall notify the Commission."


def test_doue_points_with_no_block_before_them_are_in_subparagraph_1():
    nodes = _doue(_doue_paragraph(1, _doue_point("(a)", "one.") + _p("Closing subparagraph.")))
    assert _paths(nodes) == ["7", "7.1", "7.1.a", "7.1.sub_2"]


def test_doue_article_without_paragraphs_has_unnumbered_paragraphs():
    nodes = _doue(
        _p("This Regulation shall enter into force on the twentieth day.")
        + _p("It shall apply from 17 January 2025.")
    )
    assert _paths(nodes) == ["7", "7.unp_2"]
    unp = _by(nodes)["7.unp_2"]
    assert (unp.kind, unp.marker, unp.level) == ("unnumbered_paragraph", "2", 2)
    assert (
        _by(nodes)["7"].own_text == "This Regulation shall enter into force on the twentieth day."
    )


def test_doue_unmarked_item_is_a_transparent_container():
    """A one-row, two-cell table with an EMPTY first cell (the PSD2 Art. 9 Method
    blocks of the original text): its blocks and points are seen."""
    item = _table(
        _p(""), _p("Method B") + _p("The sum of the following:") + _doue_point("(a)", "one.")
    )
    nodes = _doue(_doue_paragraph(1, _p("1.   One of the following methods:") + item))
    assert _paths(nodes) == ["7", "7.1", "7.1.sub_2", "7.1.sub_3", "7.1.sub_3.a"]
    assert _by(nodes)["7.1.sub_2"].own_text == "Method B"
    assert _by(nodes)["7.1.sub_3"].own_text == "The sum of the following:"


@pytest.mark.parametrize(
    "table",
    [
        _table(_p("—"), _p("an indent;") + _p("Another block inside the indent.")),
        _table(_p(""), _p("Row one.") + _p("Another block inside the table."), rows=2),
        _table(_p(""), _p("Middle cell."), _p("Another block inside the table.")),
    ],
    ids=["indent-with-a-dash", "two-rows", "three-cells"],
)
def test_doue_other_tables_are_not_transparent(table):
    """Deliberately narrow: an indent (a dash in the first cell), a table with
    more than one row, or with more than two cells keeps its 1.3.0 handling. Its
    text belongs to the paragraph and nothing inside it opens a subparagraph."""
    nodes = _doue(_doue_paragraph(1, _p("1.   Lead-in.") + table))
    assert _paths(nodes) == ["7", "7.1"]
    assert "Another block inside the" in _by(nodes)["7.1"].own_text


def test_doue_blocks_inside_a_point_do_not_open_subparagraphs():
    """Inside a point the markup does not tell a subparagraph from a connector
    ("plus" in PSD2 Art. 9), so nothing is split there: a known limitation."""
    point = _table(_p("(a)"), _p("the indicator is the sum of:") + _p("Each element counts."))
    nodes = _doue(_doue_paragraph(1, _p("1.   Lead-in:") + point))
    assert _paths(nodes) == ["7", "7.1", "7.1.a"]
    assert _by(nodes)["7.1.a"].own_text == "the indicator is the sum of: Each element counts."


def test_doue_unknown_carrier_does_not_open_a_subparagraph():
    """Only the carriers measured on the real documents open subparagraphs."""
    nodes = _doue(
        _doue_paragraph(1, _p("1.   Lead-in.") + _p("Text in another class.", cls="oj-ti-grseq-1"))
    )
    assert _paths(nodes) == ["7", "7.1"]
    assert _by(nodes)["7.1"].own_text == "Lead-in. Text in another class."


# ---------------------------------------------------------------------------
# consolidado family
# ---------------------------------------------------------------------------


def test_consolidated_block_after_a_list_is_a_subparagraph():
    """A paragraph's text is attributed sequentially in this family, so a closing
    block after the paragraph container still belongs to that paragraph."""
    nodes = _cons(
        _cons_paragraph(
            1, _p("Member States shall:", "norm inline-element") + _cons_point("(a)", "one.")
        )
        + _p("Member States shall notify the Commission.", "norm")
    )
    assert _paths(nodes) == ["7", "7.1", "7.1.a", "7.1.sub_2"]
    assert _by(nodes)["7.1"].own_text == "Member States shall:"


def test_consolidated_two_lists_in_one_paragraph_get_distinct_paths():
    """The PSD2 Art. 9(1) shape: lists that restart the lettering, separated by
    blocks. Without the rule both (a) would be 7.1.a."""
    content = (
        _p("One of the following methods:", "norm inline-element")
        + '<div class="list">Method B</div>'
        + '<div class="list">The sum of:</div>'
        + '<div class="list">'
        + _cons_point("(a)", "one;")
        + _cons_point("(b)", "two.")
        + "</div>"
        + '<div class="list">Method C</div>'
        + '<div class="list">'
        + _cons_point("(a)", "three.")
        + "</div>"
    )
    nodes = _cons(_cons_paragraph(1, content))
    assert _paths(nodes) == [
        "7",
        "7.1",
        "7.1.sub_2",
        "7.1.sub_3",
        "7.1.sub_3.a",
        "7.1.sub_3.b",
        "7.1.sub_4",
        "7.1.sub_4.a",
    ]
    assert _by(nodes)["7.1.sub_3.a"].own_text == "one;"
    assert _by(nodes)["7.1.sub_4.a"].own_text == "three."


def test_consolidated_indent_does_not_open_a_subparagraph():
    nodes = _cons(
        _cons_paragraph(
            1, _p("Lead-in:", "norm inline-element") + _cons_point("—", "an indent with a block.")
        )
    )
    assert _paths(nodes) == ["7", "7.1"]
    assert "an indent with a block." in _by(nodes)["7.1"].own_text


def test_consolidated_unknown_carrier_does_not_open_a_subparagraph():
    nodes = _cons(
        _cons_paragraph(
            1, _p("Lead-in.", "norm inline-element") + '<div class="other">Unknown carrier.</div>'
        )
    )
    assert _paths(nodes) == ["7", "7.1"]
    assert _by(nodes)["7.1"].own_text == "Lead-in. Unknown carrier."


def test_consolidated_article_without_paragraphs_has_unnumbered_paragraphs():
    nodes = _cons(
        _p("For the purposes of this Directive:", "norm")
        + _cons_point("(1)", "a definition.")
        + _p("Closing paragraph.", "norm")
    )
    assert _paths(nodes) == ["7", "7.1", "7.unp_2"]
    assert _by(nodes)["7.unp_2"].kind == "unnumbered_paragraph"


# ---------------------------------------------------------------------------
# unique paths, end to end
# ---------------------------------------------------------------------------


def test_parse_fails_loudly_when_a_markup_still_repeats_a_path(tmp_path):
    """Two points with the same marker in the same subparagraph: no rule tells
    them apart, and parsing must fail instead of fusing their text."""
    body = _doue_paragraph(
        1, _p("1.   Lead-in:") + _doue_point("(a)", "one;") + _doue_point("(a)", "again.")
    )
    doc = tmp_path / "doc.xhtml"
    doc.write_text(
        f'<html xmlns="{XHTML}"><body><div class="eli-subdivision" id="art_7">{body}</div>'
        "</body></html>",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError) as excinfo:
        h.parse(str(doc), "doue")
    assert "duplicate node paths" in str(excinfo.value)
    assert "7.1.a" in str(excinfo.value)
