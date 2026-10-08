"""Structural chunker v1 (T2): the processed layer cut into chunks ready to embed.

Strategy: the largest legal unit that fits. A unit (article, paragraph,
subparagraph, point, sub-point) is rendered with its inherited context, and its
tokens are counted on that final text. If it fits in max_tokens it becomes one
chunk and the walk does not go further down. If it does not fit and it has
children, its own text keeps its own chunk (":own") and each child is tried in
turn. A leaf that still does not fit is kept whole and flagged; there is no
textual fallback in this version.

Two separate notions, on purpose:
- body: the source text a chunk OWNS. Every non-empty own_text of the processed
  layer is in the body of exactly one chunk.
- context: inherited structure (the article heading and the own text of the
  ancestors that govern the chunk). It is repeated across chunks by design and
  takes no part in ownership.

text = render_text(context, body) is exactly what will be embedded, and the token
budget is measured on it. Changing anything that changes text (the template, the
context rule, the node rendering) requires a new CHUNKER_VERSION. max_tokens does
not: it is a run parameter, persisted in every ChunkedDocument.

The layer is derived and regenerable (D-002): data/chunks/{act}.json is written
from the processed layer only, and is not versioned.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal, Self

import tiktoken
from pydantic import BaseModel, Field, ValidationError, model_validator

from ingest import download as dl
from ingest import hierarchy as h
from ingest import processed as pr

# 0.1.0: first version (T2 phase 3).
CHUNKER_VERSION = "0.1.0"
Tokenizer = Literal["cl100k_base"]
TOKENIZER: Tokenizer = "cl100k_base"  # D-005, provisional until the embedding model is chosen
CHUNKS_DIR = Path("data/chunks")

# The template. text = heading, then the inherited context, then the body, as
# sections separated by a blank line; inside a section, one line per node.
SECTION_SEPARATOR = "\n\n"
LINE_SEPARATOR = "\n"

Coverage = Literal["subtree", "own"]
# leaf_over_budget: a node with no children whose text does not fit.
# own_over_budget:  the own text of a split node does not fit on its own. Both are
#                   units that the structure cannot divide any further.
BudgetException = Literal["leaf_over_budget", "own_over_budget"]
TokenCounter = Callable[[str], int]

SourcePath = Annotated[str, Field(pattern=h.LTREE_PATH_PATTERN)]
CHUNK_ID_PATTERN = r"^[^:\s]+:[0-9A-Za-z_.]+(:own)?$"
_SUBDIVISIONS = ("subparagraph", "unnumbered_paragraph")


# ---------------------------------------------------------------------------
# identity and rendering: pure functions, the only definition of each format
# ---------------------------------------------------------------------------


def chunk_id(act: str, root_path: str, coverage: Coverage) -> str:
    """Structural and deterministic: the act as namespace, the root path, and
    ":own" when the chunk holds only the root's own text. ":part<n>" is reserved
    for a future textual fallback and is not produced."""
    return f"{act}:{root_path}" + (":own" if coverage == "own" else "")


def article_heading(act: str, article: h.Article) -> str:
    heading = f"{act} Article {article.path}"
    return f"{heading}: {article.title}" if article.title else heading


def render_node(node: h.Node) -> str:
    """A node's line: its marker as the law prints it, then its own text. The
    article, a subparagraph and an unnumbered paragraph print no marker."""
    if node.kind == "apartado":
        return f"{node.marker}. {node.own_text}"
    if node.kind == "punto":
        return f"({node.marker}) {node.own_text}"
    return node.own_text


def render_context(heading: str, ancestors: list[h.Node]) -> str:
    lines = [render_node(n) for n in ancestors]
    if not lines:
        return heading
    return heading + SECTION_SEPARATOR + LINE_SEPARATOR.join(lines)


def render_body(sources: list[h.Node]) -> str:
    return LINE_SEPARATOR.join(render_node(n) for n in sources)


def render_text(context: str, body: str) -> str:
    """Exactly the string that will be embedded."""
    return context + SECTION_SEPARATOR + body


def governing_ancestors(article: h.Article, root_path: str) -> list[h.Node]:
    """The ancestors of root_path whose own text governs it, from the article down.

    In parser 1.4.0 the first subparagraph is implicit: the own text of a numbered
    paragraph IS its first subparagraph, and the own text of an article without
    numbered paragraphs IS its first unnumbered paragraph. So when the way down
    from an ancestor goes through an explicit sub_<k> or unp_<k>, that ancestor's
    own text is a SIBLING of the chunk, not a lead-in, and it is skipped. Decided
    on node kinds, never on words."""
    by_path = {n.path: n for n in article.nodes}
    labels = root_path.split(".")
    out = []
    for i in range(1, len(labels)):
        ancestor = by_path[".".join(labels[:i])]
        next_down = by_path[".".join(labels[: i + 1])]
        if next_down.kind in _SUBDIVISIONS:
            continue
        if ancestor.own_text:
            out.append(ancestor)
    return out


def cl100k_counter() -> TokenCounter:
    encoding = tiktoken.get_encoding(TOKENIZER)
    return lambda text: len(encoding.encode(text))


# ---------------------------------------------------------------------------
# contracts
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """One chunk. Declaration order is serialization order."""

    chunk_id: str = Field(pattern=CHUNK_ID_PATTERN)
    article_id: str = Field(pattern=r"^art_")
    root_path: str = Field(pattern=h.LTREE_PATH_PATTERN)
    coverage: Coverage
    # the nodes whose own text is in body, in document order
    source_paths: list[SourcePath] = Field(min_length=1)
    context: str = Field(min_length=1)
    body: str = Field(min_length=1)
    text: str = Field(min_length=1)
    token_count: int = Field(ge=1)
    budget_exception: BudgetException | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.text != render_text(self.context, self.body):
            raise ValueError(f"{self.chunk_id}: text is not render_text(context, body)")
        if self.article_id != f"art_{self.root_path.split('.')[0]}":
            raise ValueError(f"{self.chunk_id}: article_id does not match root_path")
        if len(set(self.source_paths)) != len(self.source_paths):
            raise ValueError(f"{self.chunk_id}: a source path appears twice")
        prefix = self.root_path + "."
        outside = [p for p in self.source_paths if p != self.root_path and not p.startswith(prefix)]
        if outside:
            raise ValueError(f"{self.chunk_id}: source paths outside the root: {outside}")
        if self.coverage == "own" and self.source_paths != [self.root_path]:
            raise ValueError(f"{self.chunk_id}: an 'own' chunk holds only its root's own text")
        if self.budget_exception == "leaf_over_budget" and self.coverage != "subtree":
            raise ValueError(f"{self.chunk_id}: leaf_over_budget needs coverage 'subtree'")
        if self.budget_exception == "own_over_budget" and self.coverage != "own":
            raise ValueError(f"{self.chunk_id}: own_over_budget needs coverage 'own'")
        return self


class ChunkedDocument(BaseModel):
    """A whole act, chunked. Provenance is stated once here, not in every chunk.

    Declaration order is serialization order."""

    act: str = Field(pattern=r"^[^:\s]+$")
    source_celex: str = Field(pattern=dl.CELEX_PATTERN)
    source_role: Literal["original", "consolidated"]
    source_sha256: str = Field(pattern=dl.SHA256_PATTERN)
    parser_version: str = Field(pattern=pr.VERSION_PATTERN)
    chunker_version: str = Field(pattern=pr.VERSION_PATTERN)
    tokenizer: Tokenizer
    max_tokens: int = Field(ge=1)
    chunks: list[Chunk] = Field(min_length=1)

    @model_validator(mode="after")
    def _chunks_agree_with_the_document(self) -> Self:
        seen_ids: set[str] = set()
        owned: set[str] = set()
        for c in self.chunks:
            if c.chunk_id in seen_ids:
                raise ValueError(f"duplicate chunk_id {c.chunk_id}")
            seen_ids.add(c.chunk_id)
            if c.chunk_id != chunk_id(self.act, c.root_path, c.coverage):
                raise ValueError(f"{c.chunk_id}: not the id of its act, root and coverage")
            if (c.token_count > self.max_tokens) != (c.budget_exception is not None):
                raise ValueError(
                    f"{c.chunk_id}: {c.token_count} tokens with max_tokens={self.max_tokens} "
                    f"and budget_exception={c.budget_exception!r}"
                )
            # paths include their article, so they are unique across the whole act
            twice = owned.intersection(c.source_paths)
            if twice:
                raise ValueError(f"{c.chunk_id}: source already owned by another chunk: {twice}")
            owned.update(c.source_paths)
        return self


# ---------------------------------------------------------------------------
# the chunker
# ---------------------------------------------------------------------------


def chunk_article(
    article: h.Article, act: str, max_tokens: int, count: TokenCounter
) -> list[Chunk]:
    """Pure and deterministic: the same article, act, budget and counter always
    give the same chunks, in document order."""
    children: dict[str, list[h.Node]] = {n.path: [] for n in article.nodes}
    for node in article.nodes[1:]:
        children[node.path.rsplit(".", 1)[0]].append(node)
    heading = article_heading(act, article)
    out: list[Chunk] = []

    def subtree_sources(root: h.Node) -> list[h.Node]:
        # node order is document order and every subtree is contiguous in it
        prefix = root.path + "."
        return [
            n
            for n in article.nodes
            if (n.path == root.path or n.path.startswith(prefix)) and n.own_text
        ]

    def build(
        root: h.Node, coverage: Coverage, sources: list[h.Node], over: BudgetException
    ) -> Chunk:
        context = render_context(heading, governing_ancestors(article, root.path))
        body = render_body(sources)
        text = render_text(context, body)
        tokens = count(text)
        return Chunk(
            chunk_id=chunk_id(act, root.path, coverage),
            article_id=article.article_id,
            root_path=root.path,
            coverage=coverage,
            source_paths=[n.path for n in sources],
            context=context,
            body=body,
            text=text,
            token_count=tokens,
            budget_exception=over if tokens > max_tokens else None,
        )

    def visit(node: h.Node) -> None:
        sources = subtree_sources(node)
        if not sources:
            return
        whole = build(node, "subtree", sources, "leaf_over_budget")
        if whole.token_count <= max_tokens or not children[node.path]:
            out.append(whole)
            return
        if node.own_text:
            out.append(build(node, "own", [node], "own_over_budget"))
        for child in children[node.path]:
            visit(child)

    visit(article.nodes[0])
    return out


def chunk_document(doc: pr.ProcessedDocument, act: str, max_tokens: int) -> ChunkedDocument:
    """A processed act, chunked with the official tokenizer."""
    count = cl100k_counter()
    chunks = [c for article in doc.articles for c in chunk_article(article, act, max_tokens, count)]
    return ChunkedDocument(
        act=act,
        source_celex=doc.source_celex,
        source_role=doc.source_role,
        source_sha256=doc.source_sha256,
        parser_version=doc.parser_version,
        chunker_version=CHUNKER_VERSION,
        tokenizer=TOKENIZER,
        max_tokens=max_tokens,
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# persistence: the same two boundaries as the processed layer (D-007, D-008)
# ---------------------------------------------------------------------------


class ChunksError(Exception):
    """The chunk layer exists but does not meet the contract."""


class StaleChunksError(ChunksError):
    """The chunk layer is valid but out of date: another chunker or parser wrote
    it, or it came from a corpus snapshot that is no longer in the manifest."""


def path_for(act: str, chunks_dir: Path = CHUNKS_DIR) -> Path:
    return chunks_dir / f"{act}.json"


def write_chunks(doc: ChunkedDocument, chunks_dir: Path = CHUNKS_DIR) -> None:
    """WRITE boundary: only a validated ChunkedDocument gets here."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    path_for(doc.act, chunks_dir).write_text(
        json.dumps(doc.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_chunks(
    act: str, manifest: list[dl.FetchResult], chunks_dir: Path = CHUNKS_DIR
) -> ChunkedDocument:
    """READ boundary. As in load_processed(), the versions are compared before the
    schema, so an older layer that breaks today's schema is reported as stale."""
    path = path_for(act, chunks_dir)
    if not path.exists():
        raise ChunksError(f"{path} does not exist. Run scripts/run_t2.py first.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ChunksError(f"{path} is not valid JSON: {e}") from e
    if isinstance(raw, dict):
        for field, current in (
            ("chunker_version", CHUNKER_VERSION),
            ("parser_version", h.PARSER_VERSION),
        ):
            version = raw.get(field)
            if (
                isinstance(version, str)
                and re.fullmatch(pr.VERSION_PATTERN, version)
                and version != current
            ):
                raise StaleChunksError(
                    f"{path} has {field} {version} and the installed one is {current}: "
                    f"rerun scripts/run_t2.py before using it."
                )
    try:
        doc = ChunkedDocument.model_validate(raw)
    except ValidationError as e:
        raise ChunksError(f"{path} does not meet the chunk layer contract.\n{e}") from e
    entry = next((d for d in manifest if d.celex == doc.source_celex), None)
    if entry is None or entry.sha256 != doc.source_sha256:
        raise StaleChunksError(
            f"{path} came from {doc.source_celex} at {doc.source_sha256[:12]}, which is not "
            f"the snapshot in the manifest: rerun scripts/run_t1.py and scripts/run_t2.py."
        )
    return doc
