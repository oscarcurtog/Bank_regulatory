"""Extracts the legal hierarchy (article > paragraph > subparagraph > point >
sub-point...) of an EUR-Lex XHTML as structured metadata: every node with its
ltree-style path (D-004, e.g. "5.4.c.i" or "9.1.sub_5.a"), its level and its own
text, plus the full text of the article (D-001..D-003 decide which text version
gets here).

Two markup families, verified by manual inspection (M-001..M-003):

- "doue" (original Official Journal text, the only source for DORA under
  D-003): <div id="NNN.NNN"> is the paragraph and is the REAL parent of its
  points/sub-points. Each nested point/sub-point is a <table> with exactly one
  direct row (553/553 in DORA) inside the <td> of the previous level. DOM depth
  matches the legal level.

- "consolidado" (parsed layer of PSD2 and MiCA under D-003): the paragraph
  number lives in <span class="no-parag">, a SIBLING of the list of points, not
  its parent. Only point->sub-point really nests, via <div class="grid-container
  grid-list"> inside another one of the same kind. A paragraph's text is
  assigned SEQUENTIALLY: every sibling that is neither a list nor a new
  paragraph belongs to the open paragraph, both before and after its lists.

Own text (own_text) of a node: whatever does not belong to any citable child
node. The marker ("(a)", "4.") is NOT part of the own text; it lives in
`marker`. A dash or bullet without a citable marker is not a node: its text
belongs to the parent. The "Article N" heading and the title are not own text
of the article node: they live in Article.path and Article.title.

Subparagraphs (1.4.0, see the section further down): a numbered paragraph is
divided into unnumbered subparagraphs, and an article without numbered
paragraphs into unnumbered paragraphs. The first one stays implicit; from the
second on each one is a node with its own text and its own points. That is what
gives the two lists of PSD2 Art. 9(1) distinct paths.

Two INDEPENDENT code paths on purpose: Article.text comes from string cleaning
(_clean_text) and each own_text comes from attribution over the pre-cleaned
tree (_preclean + walk). If the reconstruction from nodes reproduces
Article.text, it is because two different implementations agree, not because
one derives from the other.

PARSER_VERSION is stored with every record (D-002).
"""

from __future__ import annotations

import copy
import html
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

# 1.1.0: a dash/bullet with no citable marker is not a level (DORA Art. 35, manual review).
# 1.2.0: 'level' is derived from the path segments, not from a separate counter.
# 1.3.0: (a) FIX in _clean_text: a rule with an optional asterisk treated any "(1)".."(99)"
#            as a footnote and destroyed every "Article N(M)" reference in the corpus.
#            The bug had already been found and fixed during the research (M-001) and was
#            reintroduced when the code was ported to the repo. (b) FIX: ET.tostring
#            serializes with an html: prefix, and the tag regexes of _clean_text had not
#            matched since the port; bug (a) hid it. (c) own_text per node, and a level-1 node.
# 1.4.0: subparagraph structure (sub_<k> under a numbered paragraph, unp_<k> under an
#        article without them), unmarked table items of the doue family walked as
#        transparent containers, and node paths unique within an article (validated).
#        PSD2 Art. 9(1) no longer has duplicate paths (T2 phase 1 measurement).
PARSER_VERSION = "1.4.0"

NS = "{http://www.w3.org/1999/xhtml}"
SENT = "\x00FN\x00"  # sentinel for footnote references


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _text(el: ET.Element) -> str:
    return "".join(el.itertext())


# Materialized path (D-004), dot-separated. The first label is the article number;
# every other label is either a marker (alphanumeric only, see _MARKER_KEEP) or, since
# 1.4.0, a subparagraph label "sub_<k>" / "unp_<k>". Nothing else may contain "_", so a
# subparagraph label can never collide with a real marker. Every label is a valid ltree
# label (letters, digits, underscore).
LTREE_PATH_PATTERN = r"^[0-9A-Za-z]+(\.([0-9A-Za-z]+|(sub|unp)_[0-9]+))*$"


class Node(BaseModel):
    """A citable node of the legal tree. It IS both the object the parser
    produces and the record persisted in data/processed/: a single model, like
    FetchResult.

    What is deliberately NOT validated here, with the number that refutes it:
    - non-empty own_text: 267 of 3,379 nodes legitimately have it empty (the
      article is almost always a pure container, see M-006).
    - correspondence between kind and level: 248 nodes would break it. PSD2
      Art. 52 is a numbered list of points at level 2 and DORA Art. 60 has
      quoted paragraphs at level 4. They are different axes.
    - level == path segments: it is a derived relation and lives in
      tests/integration/test_parse_end_to_end.py, not in the schema.
    Path uniqueness IS validated, one level up: it is a property of an Article,
    not of a single node (Article._node_paths_are_unique, 1.4.0).
    """

    path: str = Field(pattern=LTREE_PATH_PATTERN)
    level: int = Field(ge=1)  # = number of path segments
    # "5", "4", "c", "i"; for a subparagraph, its ordinal ("2" for sub_2). Never
    # inside own_text.
    marker: str = Field(min_length=1)
    # Legacy serialized schema codes, kept on purpose. They are written into every
    # node of data/processed/*.json, so renaming them would be a schema migration,
    # not a translation (see the note on ProcessedDocument.family). Meaning:
    #   "articulo"  the article itself (level 1)
    #   "apartado"  a numbered paragraph ("1.", "2.")
    #   "punto"     a point with a citable marker ("(a)", "(i)", "(1)")
    # Added in 1.4.0, in English (the legacy codes above are not renamed):
    #   "subparagraph"          the 2nd, 3rd... subparagraph of a numbered
    #                           paragraph, path <paragraph>.sub_<k>
    #   "unnumbered_paragraph"  the 2nd, 3rd... unnumbered paragraph of an article
    #                           without numbered paragraphs, path <article>.unp_<k>
    kind: Literal["articulo", "apartado", "punto", "subparagraph", "unnumbered_paragraph"]
    own_text: str = ""  # own text, without the marker, without the children's text


class Article(BaseModel):
    article_id: str = Field(pattern=r"^art_")
    path: str = Field(pattern=LTREE_PATH_PATTERN)
    # title has NO minimum on purpose: _article_title returns "" when an article
    # has no title element. It does not happen today in the 331, but it is a
    # legitimate branch of the producer, and requiring non-empty would turn a
    # correct parse into a failure.
    title: str
    text: str = Field(min_length=1)
    # Declaration order IS serialization order in model_dump(), so max_depth goes
    # before nodes, so that data/processed/*.json comes out identical to the file
    # the dict literal in run_t1.py used to write.
    max_depth: int = Field(default=1, ge=1)
    nodes: list[Node] = Field(default_factory=list)

    @model_validator(mode="after")
    def _node_paths_are_unique(self) -> Self:
        """A path IS the identity of a node (D-004): its text, its citation and
        its descendants are all keyed by it. Up to 1.3.0, PSD2 Art. 9 had two
        nodes for 9.1.a and two for 9.1.b, and their text came out fused without
        anything failing. A future act or an unknown markup must fail here,
        loudly, instead of merging or duplicating identity in silence."""
        seen: set[str] = set()
        duplicated: set[str] = set()
        for node in self.nodes:
            if node.path in seen:
                duplicated.add(node.path)
            seen.add(node.path)
        if duplicated:
            raise ValueError(
                f"{self.article_id}: duplicate node paths {sorted(duplicated)}. A path "
                f"must identify exactly one node of the article."
            )
        return self


def _article_divs(root: ET.Element) -> Iterator[tuple[str, ET.Element]]:
    for el in root.iter(NS + "div"):
        idv = el.get("id", "")
        cls = el.get("class", "") or ""
        if idv.startswith("art_") and "eli-subdivision" in cls and "#" not in idv:
            yield idv[len("art_") :], el


def _article_title(article_el: ET.Element) -> str:
    for ch in article_el:
        if "eli-title" in (ch.get("class", "") or ""):
            return _normalize(_text(ch))
    return ""


def _level_of(path: str) -> int:
    return path.count(".") + 1


_MARKER_KEEP = re.compile(r"[^0-9A-Za-z]")  # ltree label: alphanumeric only
_PARA_ID = re.compile(r"^\d{2,3}[A-Z]?\.(\d{2,3})$")
_PARA_NUM = re.compile(r"^\d{1,3}\.$")
_LEADING_PARA_NUM = re.compile(r"^\s*\d{1,3}[A-Za-z]?\.\s*")
_FN_ANCHOR_ATTRS = (re.compile(r"^(ntc|src\.E)"), re.compile(r"^#(ntr|E\d)"))


# ---------------------------------------------------------------------------
# TREE pre-cleaning: the equivalent of the tag rules of _clean_text
# ---------------------------------------------------------------------------


def _is_fn_anchor(el: ET.Element) -> bool:
    if _local(el.tag) != "a":
        return False
    idv = el.get("id", "") or ""
    href = el.get("href", "") or ""
    return bool(_FN_ANCHOR_ATTRS[0].match(idv) or _FN_ANCHOR_ATTRS[1].match(href))


def _preclean(article_el: ET.Element) -> ET.Element:
    """Copy of the article without consolidation marks, footnotes or <sup>,
    with footnote references replaced by the sentinel. Keeps the tails."""
    el = copy.deepcopy(article_el)
    parent = {c: p for p in el.iter() for c in p}
    doomed = []
    for e in el.iter():
        tag = _local(e.tag)
        cls = e.get("class", "") or ""
        if tag == "p" and re.search(r"\b(modref|arrow)\b", cls):
            doomed.append(e)
        elif tag == "hr" and "note" in cls:
            doomed.append(e)
        elif tag in ("p", "div", "span", "td") and "note" in cls:
            doomed.append(e)
        elif tag == "sup":
            doomed.append(e)
        elif _is_fn_anchor(e):
            for c in list(e):
                e.remove(c)
            e.text = SENT
    for e in doomed:
        p = parent.get(e)
        if p is None:
            continue
        kids = list(p)
        i = kids.index(e)
        tail = e.tail or ""
        if i > 0:
            kids[i - 1].tail = (kids[i - 1].tail or "") + " " + tail
        else:
            p.text = (p.text or "") + " " + tail
        p.remove(e)
    return el


# ---------------------------------------------------------------------------
# TEXT normalization: identical for Article.text and for own_text
# ---------------------------------------------------------------------------

_TRANS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\xa0": " "})


def _normalize(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[▼►][MCB]\d*", " ", t).replace("◄", " ")
    t = re.sub(r"\(\s*" + re.escape(SENT) + r"\s*\)", " <FN> ", t)
    t = t.replace(SENT, " <FN> ")
    # footnote definitions with an asterisk ("( *3 ) Directive ..."): the asterisk
    # is MANDATORY. A legal reference "Article 4(1)" never carries one.
    t = re.sub(r"\(\s*\*\s*\d{1,2}\s*\).*?(?=\(\s*\*\s*\d{1,2}\s*\)|$)", " ", t, flags=re.S)
    t = t.translate(_TRANS)
    t = re.sub(r"\s+([;,.:)])", r"\1", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


_FN_ANCHOR_RX = re.compile(
    r'<a\b[^>]*(?:id="(?:ntc|src\.E)[^"]*"|href="#(?:ntr|E\d)[^"]*")[^>]*>.*?</a\s*>', re.S | re.I
)


def _clean_text(chunk_xml: str) -> str:
    """STRING cleaning of the whole article. A code path independent of _preclean."""
    # ET.tostring serializes with a prefix ("<html:p", "<html:a"). Without removing
    # it, NONE of the tag regexes below matches, and the cleaning fails silently.
    t = re.sub(r"<(/?)html:", r"<\1", chunk_xml)
    t = re.sub(
        r'<p[^>]*class="[^"]*\b(?:modref|arrow)\b[^"]*".*?</p\s*>', " ", t, flags=re.S | re.I
    )
    t = _FN_ANCHOR_RX.sub(SENT, t)
    t = re.sub(r'<hr\b[^>]*class="[^"]*note[^"]*"[^>]*/?>', " ", t, flags=re.I)
    t = re.sub(
        r'<(p|div|span|td)\b[^>]*class="[^"]*note[^"]*".*?</\1\s*>', " ", t, flags=re.S | re.I
    )
    t = re.sub(r"<sup\b.*?</sup\s*>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    return _normalize(t)


# ---------------------------------------------------------------------------
# text attribution: shared helpers
# ---------------------------------------------------------------------------


class _Buf:
    def __init__(self) -> None:
        self.parts: dict[str, list[str]] = {}

    def open(self, path: str) -> None:
        self.parts.setdefault(path, [])

    def add(self, path: str, s: str | None) -> None:
        if s and s.strip():
            self.parts[path].append(s)

    def text(self, path: str) -> str:
        return _normalize(" ".join(self.parts.get(path, [])))


def _is_heading(el: ET.Element) -> bool:
    tag = _local(el.tag)
    cls = el.get("class", "") or ""
    return (tag == "p" and ("oj-ti-art" in cls or "title-article-norm" in cls)) or (
        tag == "div" and "eli-title" in cls
    )


# ---------------------------------------------------------------------------
# subparagraphs (1.4.0)
# ---------------------------------------------------------------------------
#
# EU legislation divides a numbered paragraph into unnumbered subparagraphs, and an
# article without numbered paragraphs into unnumbered paragraphs; points belong to one
# of them. The EUR-Lex specification of subdivision identifiers (ELI) states it for
# exactly the PSD2 Art. 9(1) case: "2 lists inside the same numbered paragraph shall
# be in 2 separate subparagraphs". Up to 1.3.0 the parser hung every point directly
# from its paragraph, so the two lists of Art. 9(1) collided in 9.1.a and 9.1.b.
#
# The rule is structural and deliberately narrow: it never reads words, and it only
# knows the block carriers measured on the five raw documents (T2 phase 1).
# - A BLOCK is a carrier with text of its own at paragraph or article level. Never
#   inside a point, an indent (a list item without a citable marker), a table, or
#   another block.
# - Every block opens a new subparagraph. Points belong to the subparagraph of the
#   last block before them; a list with no block before it is in subparagraph 1.
# - Subparagraph 1 stays implicit, as in an ELI user reference ("Article 36(1),
#   point (a)"): its text is the parent's own_text and its points hang from the
#   parent, so the common case keeps its 1.3.0 path. From the second on, a node:
#   <paragraph>.sub_<k> ("subparagraph") or <article>.unp_<k> ("unnumbered_paragraph"),
#   with its own text and its own points.
#
# Known limitations, accepted: a heading block ("Method A" in PSD2 Art. 9(1)) and a
# quoted text block after a colon become subparagraphs of their own; and inside a
# point nothing is split, because there the markup does not tell a subparagraph from
# a connector such as "plus".

# Inline elements: their text belongs to the block that contains them.
_INLINE = frozenset(
    {"span", "a", "b", "i", "em", "strong", "sub", "sup", "br", "u", "small", "q", "abbr", "font"}
)

# Block carriers, as (tag, exact set of classes): every owner-level text of the five raw
# documents sits in one of these (T2 phase 1). An element outside the set never opens a
# subparagraph; its text stays in the subparagraph that is open.
_DOUE_BLOCKS = frozenset({("p", frozenset({"oj-normal"}))})
_CONSOLIDATED_BLOCKS = frozenset(
    {
        ("div", frozenset({"norm", "inline-element"})),
        ("p", frozenset({"norm"})),
        ("p", frozenset({"norm", "inline-element"})),
        ("div", frozenset({"list"})),
    }
)


def _own_direct_text(el: ET.Element) -> str:
    """The text el carries at its own level: its .text, its inline children and the
    tails of all its children. The consolidated paragraph number (span.no-parag) is
    the paragraph's marker, not text."""
    out = [el.text or ""]
    for ch in el:
        tag = _local(ch.tag)
        if tag in _INLINE and not (tag == "span" and "no-parag" in (ch.get("class", "") or "")):
            out.append(_text(ch))
        out.append(ch.tail or "")
    return "".join(out)


def _is_block(el: ET.Element, carriers: frozenset[tuple[str, frozenset[str]]]) -> bool:
    key = (_local(el.tag), frozenset((el.get("class", "") or "").split()))
    return key in carriers and bool(_own_direct_text(el).strip())


def _is_unmarked_item(table: ET.Element) -> bool:
    """doue only: a table with a single row of two cells whose FIRST cell is empty
    (no marker, not even a dash) is an unmarked item. The parser walks its content as
    if it were not in a table, so the blocks and points inside it are seen.

    Measured: 3 tables in the five raw documents, the three "Method" blocks of PSD2
    Art. 9(1) in the original text. Walked as transparent containers, that article
    comes out identical to the consolidated version. Deliberately narrow: an indent (a
    dash in the first cell) and any other table shape keep their 1.3.0 handling."""
    rows = [r for r in table if _local(r.tag) == "tr"]
    for section in table:
        if _local(section.tag) in ("thead", "tbody", "tfoot"):
            rows += [r for r in section if _local(r.tag) == "tr"]
    if len(rows) != 1:
        return False
    cells = [c for c in rows[0] if _local(c.tag) == "td"]
    return len(cells) == 2 and not _text(cells[0]).strip()


class _Subdivisions:
    """Subparagraph counter for one article: per owner (a numbered paragraph, or the
    article itself), the ordinal of the subparagraph that is open right now."""

    def __init__(self, article_path: str, nodes: list[Node], buf: _Buf) -> None:
        self.article_path = article_path
        self.nodes = nodes
        self.buf = buf
        self.open_k: dict[str, int] = {}

    def _path(self, owner: str, k: int) -> str:
        return f"{owner}.{'unp' if owner == self.article_path else 'sub'}_{k}"

    def current(self, owner: str) -> str:
        """Where the loose text of owner goes right now."""
        k = self.open_k.get(owner, 0)
        return owner if k <= 1 else self._path(owner, k)

    def point_base(self, owner: str) -> str:
        """Parent path for a point of owner. A list with no block before it is in
        subparagraph 1."""
        self.open_k.setdefault(owner, 1)
        return self.current(owner)

    def block(self, owner: str) -> None:
        """A new block opens a new subparagraph; from the second on, it is a node."""
        k = self.open_k.get(owner, 0) + 1
        self.open_k[owner] = k
        if k >= 2:
            path = self._path(owner, k)
            kind: Literal["subparagraph", "unnumbered_paragraph"] = (
                "unnumbered_paragraph" if owner == self.article_path else "subparagraph"
            )
            self.nodes.append(Node(path=path, level=_level_of(path), marker=str(k), kind=kind))
            self.buf.open(path)


# ---------------------------------------------------------------------------
# "doue" family
# ---------------------------------------------------------------------------


def _nodes_doue(article_el: ET.Element, article_path: str) -> tuple[list[Node], int]:
    el0 = _preclean(article_el)
    nodes = [Node(path=article_path, level=1, marker=article_path, kind="articulo")]
    buf = _Buf()
    buf.open(article_path)
    subs = _Subdivisions(article_path, nodes, buf)
    # The owners whose blocks open subparagraphs: the article and its numbered
    # paragraphs. A paragraph quoted inside a point (DORA Art. 60) is not one of them.
    owners = {article_path}
    skip: set[ET.Element] = set()

    def loose(owner: str) -> str:  # where the loose text of owner goes right now
        return subs.current(owner) if owner in owners else owner

    def walk(
        el: ET.Element, owner: str, path: str, in_point: bool, in_other: bool, in_block: bool
    ) -> None:
        if _is_heading(el) or el in skip:
            return
        tag = _local(el.tag)
        m = _PARA_ID.match(el.get("id", "") or "")
        here_path, here_owner = path, owner
        here_point, here_other, here_block = in_point, in_other, in_block
        if tag == "div" and m:
            num = m.group(1).lstrip("0") or "0"
            here_path = f"{path}.{num}"
            here_owner = here_path
            nodes.append(
                Node(path=here_path, level=_level_of(here_path), marker=num, kind="apartado")
            )
            buf.open(here_path)
            if not in_point:
                owners.add(here_path)
        elif tag == "table":
            tr = el.find(f".//{NS}tr")
            marker, marker_td = "", None
            if tr is not None and len(list(tr)):
                marker_td = list(tr)[0]
                marker = _MARKER_KEEP.sub("", _text(marker_td).strip())
            # marker is only non-empty if marker_td exists; both are checked so as
            # not to rely on a correlation the type checker cannot see.
            if marker and marker_td is not None:
                base = path
                if not in_point and path == owner and owner in owners:
                    base = subs.point_base(owner)
                here_path = f"{base}.{marker}"
                here_owner = here_path
                nodes.append(
                    Node(path=here_path, level=_level_of(here_path), marker=marker, kind="punto")
                )
                buf.open(here_path)
                skip.add(marker_td)
                here_point = True
            elif not (in_point or in_other) and _is_unmarked_item(el):
                pass  # 1.4.0: a transparent container, see _is_unmarked_item
            else:
                # with no citable marker (a dash), the table is the parent's text,
                # and nothing inside it opens a subparagraph
                here_other = True
        elif not (in_point or in_other or in_block) and owner in owners:
            if _is_block(el, _DOUE_BLOCKS):
                subs.block(owner)
                here_block = True
        buf.add(loose(here_owner), el.text)
        for ch in el:
            walk(ch, here_owner, here_path, here_point, here_other, here_block)
            buf.add(loose(here_owner), ch.tail)

    walk(el0, article_path, article_path, False, False, False)
    for nd in nodes:
        t = buf.text(nd.path)
        if nd.kind == "apartado":  # in the OJ the "1." is inline in the text; it is the marker
            t = _LEADING_PARA_NUM.sub("", t, count=1)
        nd.own_text = t
    return nodes, max(nd.level for nd in nodes)


# ---------------------------------------------------------------------------
# "consolidado" family
# ---------------------------------------------------------------------------


def _is_list(el: ET.Element) -> bool:
    cls = el.get("class", "") or ""
    return "grid-container" in cls and "grid-list" in cls


def _list_marker(el: ET.Element) -> tuple[str, ET.Element | None]:
    for ch in el:
        if "grid-list-column-1" in (ch.get("class", "") or ""):
            return _MARKER_KEEP.sub("", _text(ch).strip()), ch
    return "", None


def _nodes_consolidado(article_el: ET.Element, article_path: str) -> tuple[list[Node], int]:
    el0 = _preclean(article_el)
    nodes = [Node(path=article_path, level=1, marker=article_path, kind="articulo")]
    buf = _Buf()
    buf.open(article_path)
    subs = _Subdivisions(article_path, nodes, buf)
    # list[str | None], not list[None]: the path of the open paragraph gets
    # written into it. mypy inferred list[None], and the assignment further
    # down was an error.
    current_para: list[str | None] = [None]
    skip: set[ET.Element] = set()

    def para() -> str:  # the open paragraph, or the article before the first one
        return current_para[0] or article_path

    def owner_of(point: str | None) -> str:  # owner of the loose text here
        return point or subs.current(para())

    def walk(
        el: ET.Element,
        point: str | None,
        path: str,
        list_depth: int,
        in_other: bool,
        in_block: bool,
    ) -> None:
        if _is_heading(el) or el in skip:
            return
        tag = _local(el.tag)
        cls = el.get("class", "") or ""
        if list_depth == 0 and tag == "span" and "no-parag" in cls:
            raw = (el.text or "").replace("\xa0", "").strip()
            if _PARA_NUM.match(raw):
                num = raw.rstrip(".")
                p2 = f"{article_path}.{num}"
                nodes.append(Node(path=p2, level=_level_of(p2), marker=num, kind="apartado"))
                buf.open(p2)
                current_para[0] = p2
                return  # the span IS the marker; nothing in it is own text
        here_path, here_point, new_ld = path, point, list_depth
        here_other, here_block = in_other, in_block
        if _is_list(el):
            marker, col1 = _list_marker(el)
            # same here: _list_marker returns ("", None) or (marker, element).
            if marker and col1 is not None:
                base = subs.point_base(para()) if list_depth == 0 else path
                here_path = f"{base}.{marker}"
                here_point = here_path
                new_ld = list_depth + 1
                nodes.append(
                    Node(path=here_path, level=_level_of(here_path), marker=marker, kind="punto")
                )
                buf.open(here_path)
                skip.add(col1)
            else:
                here_other = True  # an indent: a list item without a citable marker
        elif tag == "table":
            here_other = True
        elif list_depth == 0 and not (in_other or in_block):
            if _is_block(el, _CONSOLIDATED_BLOCKS):
                subs.block(para())
                here_block = True
        buf.add(owner_of(here_point), el.text)
        for ch in el:
            walk(ch, here_point, here_path, new_ld, here_other, here_block)
            buf.add(owner_of(here_point), ch.tail)

    walk(el0, None, article_path, 0, False, False)
    for nd in nodes:
        nd.own_text = buf.text(nd.path)
    return nodes, max(nd.level for nd in nodes)


# ---------------------------------------------------------------------------


def parse(xhtml_path: str, family: str) -> list[Article]:
    if family not in ("doue", "consolidado"):
        raise ValueError(f"unknown family: {family!r}")
    raw = open(xhtml_path, encoding="utf-8").read()
    root = ET.fromstring(raw)
    nodes_fn = _nodes_doue if family == "doue" else _nodes_consolidado
    out = []
    for path, el in _article_divs(root):
        nodes, depth = nodes_fn(el, path)
        out.append(
            Article(
                article_id=f"art_{path}",
                path=path,
                title=_article_title(el),
                text=_clean_text(ET.tostring(el, encoding="unicode")),
                nodes=nodes,
                max_depth=depth,
            )
        )
    return out
