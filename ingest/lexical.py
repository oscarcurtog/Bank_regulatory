"""Lexical retrieval: BM25 over the official chunk layer (config/retrieval/bm25.json).

The choices, each fixed in the contract before the first run:
- Scores: bm25s's "lucene" variant, the BM25 of Lucene 8 and later (and of Elasticsearch),
  as Kamphuis et al. (ECIR 2020) name it. For each query term,
  idf = ln(1 + (N - df + 0.5) / (df + 0.5)) times tf / (tf + k1 (1 - b + b dl / avgdl)),
  summed over the query's terms; a term repeated in the query counts each time, as in
  Lucene. k1 = 1.2 and b = 0.75 are Lucene's defaults. Scores are computed in float64, so
  that rounding does not create ties.
- Analyzer, an imitation of Lucene's EnglishAnalyzer: lowercase; the English possessive
  ("'s") removed; split on every character that is not a letter or a digit (hyphens,
  slashes and parentheses included); Lucene's 33 English stop words; the Porter stemmer
  (PyStemmer's "porter", the original algorithm, as Lucene's PorterStemFilter), applied
  only to words of three letters or more, as Lucene and Porter's reference implementation
  do. Without that rule the single letter "s" stemmed to an empty term (fixed after the
  first run of experiment 1; both runs are in the iteration's record).
- Documents are chunk.text: exactly the string the dense models embed, without their prefix.
- The same analyzer runs on the English and the Spanish queries. The corpus is English and
  BM25 has no way across languages: that is measured, not corrected.

bm25s and PyStemmer are pinned by the contract; a different installed version fails
explicitly instead of producing other scores.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ingest import retrieval as rt

CONFIG_DIR = Path("config/retrieval")
# Lucene's EnglishAnalyzer.ENGLISH_STOP_WORDS_SET.
LUCENE_ENGLISH_STOP_WORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their "
    "then there these they this to was will with".split()
)
_POSSESSIVE = re.compile(r"['’]s\b")


class LexicalError(Exception):
    """The lexical retriever cannot run as its contract says."""


class AnalyzerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lowercase: bool
    strip_possessive: bool
    token_pattern: str = Field(min_length=1)
    stopwords: Literal["lucene_english", "none"]
    stemmer: Literal["porter", "none"]
    # Porter's reference implementation and Lucene's PorterStemmer leave words of one or
    # two letters unstemmed (Porter's own "DEPARTURE" from the published algorithm).
    stem_min_length: int = Field(ge=1)
    stemmer_library: Literal["PyStemmer"]
    stemmer_library_version: str


class BM25Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"]
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    library: Literal["bm25s"]
    library_version: str
    method: Literal["lucene"]
    k1: float = Field(gt=0)
    b: float = Field(ge=0, le=1)
    text: Literal["chunk_text"]
    analyzer: AnalyzerConfig

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: Path) -> BM25Config:
    try:
        return BM25Config.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as e:
        raise LexicalError(f"{path} does not meet the BM25 contract.\n{e}") from e


def check_versions(config: BM25Config) -> dict[str, str]:
    """The installed bm25s and PyStemmer, which must be the contract's."""
    wanted = {
        "bm25s": config.library_version,
        "pystemmer": config.analyzer.stemmer_library_version,
    }
    found = {}
    for package in wanted:
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as e:
            raise LexicalError(f"{package} is not installed: install requirements.lock.txt") from e
    wrong = {p: (found[p], v) for p, v in wanted.items() if found[p] != v}
    if wrong:
        raise LexicalError(
            "the installed versions are not the contract's: "
            + ", ".join(f"{p} {have} (contract {want})" for p, (have, want) in wrong.items())
            + ". Install requirements.lock.txt."
        )
    return found


class Analyzer:
    """Text to index terms, the same way for documents and queries."""

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self._token = re.compile(config.token_pattern)
        self._stop = LUCENE_ENGLISH_STOP_WORDS if config.stopwords == "lucene_english" else None
        self._stemmer: Any = None
        if config.stemmer == "porter":
            import Stemmer

            self._stemmer = Stemmer.Stemmer("porter")

    def __call__(self, text: str) -> list[str]:
        if self.config.lowercase:
            text = text.lower()
        if self.config.strip_possessive:
            text = _POSSESSIVE.sub("", text)
        tokens = self._token.findall(text)
        if self._stop is not None:
            tokens = [t for t in tokens if t not in self._stop]
        if self._stemmer is not None:
            short = self.config.stem_min_length
            tokens = [t if len(t) < short else self._stemmer.stemWord(t) for t in tokens]
        return tokens


@dataclass(frozen=True)
class BM25Index:
    config: BM25Config
    doc_ids: tuple[str, ...]
    analyzer: Analyzer
    retriever: Any  # bm25s.BM25: untyped
    doc_lengths: tuple[int, ...]

    def scores(self, query: str) -> np.ndarray:
        """The BM25 score of every document for one query, in doc_ids order. A query with
        no term in the vocabulary scores 0 everywhere."""
        terms = [t for t in self.analyzer(query) if t in self.retriever.vocab_dict]
        if not terms:
            return np.zeros(len(self.doc_ids), dtype=np.float64)
        return np.asarray(self.retriever.get_scores(terms), dtype=np.float64)

    def rank(self, queries: Sequence[tuple[str, str]]) -> list[rt.Ranked]:
        return [rt.rank_by_scores(qid, self.doc_ids, self.scores(text)) for qid, text in queries]


def build_index(config: BM25Config, documents: Sequence[tuple[str, str]]) -> BM25Index:
    """The index of (doc_id, text) pairs: the analyzer on every text, then bm25s."""
    check_versions(config)
    import bm25s

    ids = tuple(doc_id for doc_id, _ in documents)
    if len(set(ids)) != len(ids):
        raise LexicalError("a document id appears twice")
    analyzer = Analyzer(config.analyzer)
    tokens = [analyzer(text) for _, text in documents]
    if any(not t for t in tokens):
        empty = [i for i, t in zip(ids, tokens, strict=True) if not t][:5]
        raise LexicalError(f"documents with no index term: {empty}")
    retriever = bm25s.BM25(k1=config.k1, b=config.b, method=config.method, dtype="float64")
    retriever.index(tokens, show_progress=False)
    return BM25Index(config, ids, analyzer, retriever, tuple(len(t) for t in tokens))


def reference_scores(index: BM25Index, documents: Sequence[str], query: str) -> np.ndarray:
    """The same Lucene BM25, computed directly in numpy from the analyzed texts: an
    independent check of the library's scores, never used to rank."""
    docs = [index.analyzer(text) for text in documents]
    terms = index.analyzer(query)
    n = len(docs)
    lengths = np.array([len(d) for d in docs], dtype=np.float64)
    avg = lengths.mean()
    k1, b = index.config.k1, index.config.b
    out = np.zeros(n, dtype=np.float64)
    for term in terms:
        tf = np.array([d.count(term) for d in docs], dtype=np.float64)
        df = int((tf > 0).sum())
        if df == 0:
            continue
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        out += idf * tf / (tf + k1 * (1 - b + b * lengths / avg))
    return out
