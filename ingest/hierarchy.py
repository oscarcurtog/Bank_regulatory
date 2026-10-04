"""Extracts the legal hierarchy (article > paragraph > point > sub-point...) of
an EUR-Lex XHTML as structured metadata: every node with its ltree-style path
(D-004, e.g. "5.4.c.i"), its level and its own text, plus the full text of the
article (D-001..D-003 decide which text version gets here).

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
from typing import Literal

from pydantic import BaseModel, Field

# 1.1.0: a dash/bullet with no citable marker is not a level (DORA Art. 35, manual review).
# 1.2.0: 'level' is derived from the path segments, not from a separate counter.
# 1.3.0: (a) FIX in _clean_text: a rule with an optional asterisk treated any "(1)".."(99)"
#            as a footnote and destroyed every "Article N(M)" reference in the corpus.
#            The bug had already been found and fixed during the research (M-001) and was
#            reintroduced when the code was ported to the repo. (b) FIX: ET.tostring
#            serializes with an html: prefix, and the tag regexes of _clean_text had not
#            matched since the port; bug (a) hid it. (c) own_text per node, and a level-1 node.
PARSER_VERSION = "1.3.0"

NS = "{http://www.w3.org/1999/xhtml}"
SENT = "\x00FN\x00"  # sentinel for footnote references


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _text(el: ET.Element) -> str:
    return "".join(el.itertext())


# Materialized path label (D-004): alphanumeric only, dot-separated.
LTREE_PATH_PATTERN = r"^[0-9A-Za-z]+(\.[0-9A-Za-z]+)*$"


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
    - path uniqueness: PSD2 Art. 9 violates it today on purpose (D-004 open).
    - level == path segments: it is a derived relation and lives in
      tests/integration/test_parse_end_to_end.py, not in the schema.
    """

    path: str = Field(pattern=LTREE_PATH_PATTERN)
    level: int = Field(ge=1)  # = number of path segments
    marker: str = Field(min_length=1)  # "5", "4", "c", "i"; never inside own_text
    # Legacy serialized schema codes, kept on purpose. They are written into every
    # node of data/processed/*.json, so renaming them would be a schema migration,
    # not a translation (see the note on ProcessedDocument.family). Meaning:
    #   "articulo"  the article itself (level 1)
    #   "apartado"  a numbered paragraph ("1.", "2.")
    #   "punto"     a point with a citable marker ("(a)", "(i)", "(1)")
    kind: Literal["articulo", "apartado", "punto"]
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
# "doue" family
# ---------------------------------------------------------------------------


def _nodes_doue(article_el: ET.Element, article_path: str) -> tuple[list[Node], int]:
    el0 = _preclean(article_el)
    nodes = [Node(path=article_path, level=1, marker=article_path, kind="articulo")]
    buf = _Buf()
    buf.open(article_path)
    max_depth = [1]
    skip: set[ET.Element] = set()

    def walk(el: ET.Element, owner: str, path: str, depth: int) -> None:
        if _is_heading(el) or el in skip:
            return
        tag = _local(el.tag)
        m = _PARA_ID.match(el.get("id", "") or "")
        here_path, here_depth, here_owner = path, depth, owner
        if tag == "div" and m:
            num = m.group(1).lstrip("0") or "0"
            here_path = f"{path}.{num}"
            here_depth = _level_of(here_path)
            here_owner = here_path
            nodes.append(Node(path=here_path, level=here_depth, marker=num, kind="apartado"))
            buf.open(here_path)
        elif tag == "table":
            tr = el.find(f".//{NS}tr")
            marker, marker_td = "", None
            if tr is not None and len(list(tr)):
                marker_td = list(tr)[0]
                marker = _MARKER_KEEP.sub("", _text(marker_td).strip())
            # marker is only non-empty if marker_td exists; both are checked so as
            # not to rely on a correlation the type checker cannot see.
            if marker and marker_td is not None:  # with no citable marker (a dash),
                # the table is the parent's text
                here_path = f"{path}.{marker}"
                here_depth = _level_of(here_path)
                here_owner = here_path
                nodes.append(Node(path=here_path, level=here_depth, marker=marker, kind="punto"))
                buf.open(here_path)
                skip.add(marker_td)
        max_depth[0] = max(max_depth[0], here_depth)
        buf.add(here_owner, el.text)
        for ch in el:
            walk(ch, here_owner, here_path, here_depth)
            buf.add(here_owner, ch.tail)

    walk(el0, article_path, article_path, 1)
    for nd in nodes:
        t = buf.text(nd.path)
        if nd.kind == "apartado":  # in the OJ the "1." is inline in the text; it is the marker
            t = _LEADING_PARA_NUM.sub("", t, count=1)
        nd.own_text = t
    return nodes, max_depth[0]


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
    max_depth = [1]
    # list[str | None], not list[None]: the path of the open paragraph gets
    # written into it. mypy inferred list[None], and the assignment further
    # down was an error.
    current_para: list[str | None] = [None]
    skip: set[ET.Element] = set()

    def owner_of(point: str | None) -> str:  # owner of the loose text here
        return point or current_para[0] or article_path

    def walk(el: ET.Element, point: str | None, path: str, depth: int, list_depth: int) -> None:
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
                max_depth[0] = max(max_depth[0], 2)
                return  # the span IS the marker; nothing in it is own text
        here_path, here_depth, here_point, new_ld = path, depth, point, list_depth
        if _is_list(el):
            marker, col1 = _list_marker(el)
            # same here: _list_marker returns ("", None) or (marker, element).
            if marker and col1 is not None:
                base = (current_para[0] or article_path) if list_depth == 0 else path
                here_path = f"{base}.{marker}"
                here_depth = _level_of(here_path)
                here_point = here_path
                new_ld = list_depth + 1
                nodes.append(Node(path=here_path, level=here_depth, marker=marker, kind="punto"))
                buf.open(here_path)
                skip.add(col1)
                max_depth[0] = max(max_depth[0], here_depth)
        buf.add(owner_of(here_point), el.text)
        for ch in el:
            walk(ch, here_point, here_path, here_depth, new_ld)
            buf.add(owner_of(here_point), ch.tail)

    walk(el0, None, article_path, 1, 0)
    for nd in nodes:
        nd.own_text = buf.text(nd.path)
    return nodes, max_depth[0]


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
