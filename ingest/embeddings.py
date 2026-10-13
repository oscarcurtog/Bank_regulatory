"""Dense retrieval baseline: the embedding contract, the explicit truncation of
documents, exact search and the cache of embeddings.

No model framework is imported here. The transformer that turns token ids into
vectors is in ingest/encoder.py (the optional "embed" extra); everything that can be
checked without it lives in this module, so the tests run without a model.

The contract of one embedding model is a versioned JSON file in config/embeddings/,
validated by EmbeddingConfig and identified by its SHA-256 (config_hash). It says
which model and revision, which tokenizer counts, the input limit, the prefixes, the
pooling, the normalization, the similarity and how the vectors are computed.

What is embedded is exactly the prefix plus chunk.text (a document) or the prefix plus
query.text (a query). Nothing else is added: no act, path, grade or other metadata.

Truncation is part of the model's contract, never a side effect:

- every input is tokenized whole first, with the special tokens, and counted;
- a document longer than max_input_tokens is cut at the tail, keeping the special
  tokens, and the cut is recorded (TokenRecord: original and encoded tokens, dropped
  tokens, and how many characters of the text the model still sees);
- a query is never cut: an over-limit query is an error;
- the encoder receives token ids, not text, so it cannot cut anything on its own, and
  check_inputs() refuses any input whose length differs from its record.

With L2-normalized vectors the dot product IS the cosine similarity, so the exact
search is one matrix product. Ties are broken by chunk id, so a ranking never depends
on the order in which the vectors happen to be stored.

Embeddings are derived data (D-002): they live in data/embeddings/, outside git, next
to a meta.json that records what produced them. A set is identified by everything that
can change its numbers (ArtifactIdentity):

- the model: the contract (model, revision, tokenizer revision, pooling, prefixes,
  normalization, similarity, input limit, truncation, dimension, dtype, device) and the
  SHA-256 of the tokenizer file that counted the tokens;
- the runtime: Python, torch, transformers, tokenizers and numpy versions, and whether
  sentence-transformers took part (it does not);
- the data: the chunk layer (corpus snapshot, parser, chunker, budget, the structure and
  the text of every chunk) or the queries (benchmark version, sealed file, text).

load_embedding_set() refuses a set whose identity differs in any of these, naming the
fields: StaleEmbeddingsError. The time of the run, the cache path and the hardware are
recorded for the report and never compared: they describe a run, they do not define it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import evaluation as ev

SCHEMA_VERSION: Literal["2"] = "2"
CONFIG_DIR = Path("config/embeddings")
EMBEDDINGS_DIR = Path("data/embeddings")
# Where Hugging Face keeps what it downloads (tokenizer and weights), unless HF_HOME
# names another directory. Relative to the working directory, like the tiktoken cache
# of ingest/chunks.py: one place, outside git (.cache/ is ignored), downloaded once.
HF_CACHE_DIR = Path(".cache/huggingface")
REVISION_PATTERN = r"^[0-9a-f]{40}$"
# How far from 1 the norm of a normalized float32 vector may be.
NORM_TOLERANCE = 1e-4
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"

Kind = Literal["query", "document"]
STRICT = ConfigDict(extra="forbid", frozen=True)


class EmbeddingError(Exception):
    """The embedding contract, an input or a set of vectors is not what it must be."""


class TruncationRefusedError(EmbeddingError):
    """An input is longer than the model accepts and its kind may not be truncated."""


class ModelUnavailableError(EmbeddingError):
    """A file of the model (tokenizer or weights) or its framework cannot be loaded."""


class EmbeddingsNotFoundError(EmbeddingError):
    """No embedding set has been written in that directory yet."""


class StaleEmbeddingsError(EmbeddingError):
    """An embedding set exists but was made from something that has changed since."""


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


class EmbeddingConfig(BaseModel):
    """One embedding model, exactly as it is run. Every field takes part in the hash,
    so changing any of them makes the embeddings computed before stale."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    schema_version: Literal["2"]
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    model_id: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    revision: str = Field(pattern=REVISION_PATTERN)
    # None: the tokenizer comes from the model's own revision.
    tokenizer_revision: str | None = Field(default=None, pattern=REVISION_PATTERN)
    tokenizer_file: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    # The whole input: prefix, text and special tokens.
    max_input_tokens: int = Field(ge=3)
    # What goes before the text of a query and of a document. Empty for a model that
    # takes both as they are.
    query_prefix: str
    document_prefix: str
    # mean: the mean of the token vectors over the attention mask; cls: the vector of the
    # first token.
    pooling: Literal["mean", "cls"]
    normalize: bool
    similarity: Literal["dot"]
    document_truncation: Literal["explicit_tail"]
    query_truncation: Literal["refuse"]
    batch_size: int = Field(ge=1)
    device: Literal["cpu"]
    dtype: Literal["float32"]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if bool(self.query_prefix) != bool(self.document_prefix):
            raise ValueError("a model prefixes both queries and documents, or neither")
        if self.query_prefix and self.query_prefix == self.document_prefix:
            raise ValueError("the query and document prefixes must differ")
        if self.similarity == "dot" and not self.normalize:
            raise ValueError("dot product is the cosine only on normalized vectors")
        return self

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def prefix(self, kind: Kind) -> str:
        return self.query_prefix if kind == "query" else self.document_prefix

    @property
    def tokenizer_commit(self) -> str:
        return self.tokenizer_revision or self.revision


def load_config(path: Path) -> EmbeddingConfig:
    try:
        return EmbeddingConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise EmbeddingError(f"{path} does not exist") from e
    except ValidationError as e:
        raise EmbeddingError(f"{path} does not meet the embedding contract.\n{e}") from e


def model_input(config: EmbeddingConfig, kind: Kind, text: str) -> str:
    """The exact string that is tokenized: the prefix of its kind, then the text
    unchanged."""
    return config.prefix(kind) + text


def environment_description() -> dict[str, str]:
    """The machine, for the report: never part of an identity."""
    cpu = platform.processor()
    if platform.system() == "Darwin":
        import subprocess

        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        cpu = brand.stdout.strip() or cpu
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "cpu_count": str(os.cpu_count()),
        "hf_home": cache_location(),
    }


def cache_location() -> str:
    """Where the Hugging Face cache is, said without an absolute path: a report is
    versioned and must not carry the local user or folder names."""
    home = hf_home().resolve()
    try:
        return str(home.relative_to(Path.cwd().resolve()))
    except ValueError:
        return "outside the working directory (HF_HOME)"


def hf_home() -> Path:
    """The Hugging Face cache, set once for every library that reads HF_HOME."""
    return Path(os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR.absolute())))


@dataclass(frozen=True)
class LoadedTokenizer:
    """The tokenizer that counts, and the SHA-256 of the file it was read from."""

    tokenizer: TokenizerLike
    sha256: str


def load_tokenizer(config: EmbeddingConfig) -> LoadedTokenizer:
    """The model's own tokenizer, from its file at the pinned revision, with
    truncation and padding switched off: it must count, never cut."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    cache = hf_home() / "hub"
    revision = config.tokenizer_commit
    try:
        path = hf_hub_download(
            config.model_id, config.tokenizer_file, revision=revision, cache_dir=cache
        )
    except Exception as e:
        raise ModelUnavailableError(
            f"{config.model_id}@{revision[:12]}: cannot load {config.tokenizer_file} "
            f"from the cache {cache} or the network. Connect once to download it, or point "
            f"HF_HOME at a cache that holds it."
        ) from e
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return LoadedTokenizer(tokenizer, hashlib.sha256(Path(path).read_bytes()).hexdigest())


# ---------------------------------------------------------------------------
# counting and explicit truncation
# ---------------------------------------------------------------------------


class EncodingLike(Protocol):
    @property
    def ids(self) -> list[int]: ...

    @property
    def special_tokens_mask(self) -> list[int]: ...

    @property
    def offsets(self) -> list[tuple[int, int]]: ...


class TokenizerLike(Protocol):
    def encode(self, sequence: str) -> EncodingLike: ...


class TokenRecord(BaseModel):
    """What the model receives of one input, counted before encoding."""

    model_config = STRICT

    item_id: str = Field(min_length=1)
    original_model_tokens: int = Field(ge=1)
    encoded_model_tokens: int = Field(ge=1)
    truncated: bool
    dropped_tokens: int = Field(ge=0)
    # characters of the text, without the prefix, that the model still sees
    kept_text_chars: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.dropped_tokens != self.original_model_tokens - self.encoded_model_tokens:
            raise ValueError(f"{self.item_id}: dropped is not original minus encoded")
        if self.truncated != (self.dropped_tokens > 0):
            raise ValueError(f"{self.item_id}: truncated must say whether tokens were dropped")
        return self


@dataclass(frozen=True)
class ModelInput:
    item_id: str
    input_ids: tuple[int, ...]
    record: TokenRecord


def _special_edges(item_id: str, mask: Sequence[int]) -> tuple[int, int]:
    """How many special tokens open and close the sequence. A special token in the
    middle would mean a template this module does not know how to cut."""
    lead = 0
    while lead < len(mask) and mask[lead]:
        lead += 1
    trail = 0
    while trail < len(mask) - lead and mask[len(mask) - 1 - trail]:
        trail += 1
    if any(mask[lead : len(mask) - trail]):
        raise EmbeddingError(f"{item_id}: a special token inside the text")
    return lead, trail


def prepare_inputs(
    items: Iterable[tuple[str, str]],
    kind: Kind,
    tokenizer: TokenizerLike,
    config: EmbeddingConfig,
) -> list[ModelInput]:
    """Token ids and their record for each (item_id, text), counted on the whole
    input first. A document over the limit is cut at the tail, inside the special
    tokens; a query over the limit raises TruncationRefusedError."""
    limit = config.max_input_tokens
    prefix = config.prefix(kind)
    out: list[ModelInput] = []
    for item_id, text in items:
        encoding = tokenizer.encode(model_input(config, kind, text))
        ids = list(encoding.ids)
        lead, trail = _special_edges(item_id, encoding.special_tokens_mask)
        if len(ids) <= limit:
            kept, kept_chars = ids, len(text)
        elif kind == "query" or config.document_truncation != "explicit_tail":
            raise TruncationRefusedError(
                f"{item_id}: {len(ids)} model tokens, over the limit of {limit}, and a {kind} "
                f"is never truncated"
            )
        else:
            room = limit - lead - trail
            if room < 1:
                raise EmbeddingError(f"{item_id}: the limit leaves no room for text")
            kept = ids[: lead + room] + ids[len(ids) - trail :]
            kept_chars = max(0, encoding.offsets[lead + room - 1][1] - len(prefix))
        record = TokenRecord(
            item_id=item_id,
            original_model_tokens=len(ids),
            encoded_model_tokens=len(kept),
            truncated=len(kept) < len(ids),
            dropped_tokens=len(ids) - len(kept),
            kept_text_chars=kept_chars,
        )
        out.append(ModelInput(item_id, tuple(kept), record))
    ids_seen = [i.item_id for i in out]
    if len(set(ids_seen)) != len(ids_seen):
        raise EmbeddingError("an item id appears twice")
    return out


def check_inputs(inputs: Sequence[ModelInput], config: EmbeddingConfig) -> None:
    """The guard before the model runs: every input is exactly what its record says,
    and within the limit. Anything else would be a truncation nobody recorded."""
    for item in inputs:
        n = len(item.input_ids)
        if n != item.record.encoded_model_tokens:
            raise EmbeddingError(
                f"{item.item_id}: {n} token ids, but the record says "
                f"{item.record.encoded_model_tokens}: an unrecorded change of the input"
            )
        if n > config.max_input_tokens:
            raise EmbeddingError(
                f"{item.item_id}: {n} token ids over the limit of {config.max_input_tokens}"
            )


# ---------------------------------------------------------------------------
# vectors and exact search
# ---------------------------------------------------------------------------


def check_vectors(vectors: np.ndarray, count: int, config: EmbeddingConfig) -> None:
    """Shape, type, finite values and, when normalized, unit norm."""
    if vectors.shape != (count, config.dimension):
        raise EmbeddingError(
            f"vectors of shape {vectors.shape}, expected ({count}, {config.dimension})"
        )
    if vectors.dtype != np.float32:
        raise EmbeddingError(f"vectors of type {vectors.dtype}, expected float32")
    if not np.isfinite(vectors).all():
        raise EmbeddingError("vectors with NaN or infinite values")
    if config.normalize and count:
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        worst = float(np.max(np.abs(norms - 1.0)))
        if worst > NORM_TOLERANCE:
            raise EmbeddingError(f"vectors not normalized: a norm is {worst:.2e} away from 1")


@dataclass(frozen=True)
class Ranked:
    query_id: str
    chunk_ids: tuple[str, ...]
    scores: tuple[float, ...]


def rank_exact(
    query_ids: Sequence[str],
    query_vectors: np.ndarray,
    doc_ids: Sequence[str],
    doc_vectors: np.ndarray,
) -> list[Ranked]:
    """Every query against every document: one dot product each, no index and no
    approximation. Ordered by score, best first, and by chunk id on a tie."""
    if len(set(doc_ids)) != len(doc_ids):
        raise EmbeddingError("a document id appears twice")
    if query_vectors.shape[0] != len(query_ids) or doc_vectors.shape[0] != len(doc_ids):
        raise EmbeddingError("ids and vectors do not line up")
    scores = query_vectors @ doc_vectors.T
    id_order = np.empty(len(doc_ids), dtype=np.int64)  # position of each id in sorted order
    id_order[sorted(range(len(doc_ids)), key=doc_ids.__getitem__)] = np.arange(len(doc_ids))
    out = []
    for i, query_id in enumerate(query_ids):
        row = scores[i]
        order = np.lexsort((id_order, -row))
        out.append(
            Ranked(
                query_id,
                tuple(doc_ids[j] for j in order),
                tuple(float(row[j]) for j in order),
            )
        )
    return out


def to_run(ranked: Sequence[Ranked]) -> ev.Run:
    """The evaluator's run: the whole ranking of every query, so the depth is the
    number of documents."""
    depths = {len(r.chunk_ids) for r in ranked}
    if len(depths) != 1:
        raise EmbeddingError("an exact search ranks every document for every query")
    return ev.Run(depth=depths.pop(), rankings={r.query_id: list(r.chunk_ids) for r in ranked})


# ---------------------------------------------------------------------------
# what a set of embeddings was made from
# ---------------------------------------------------------------------------


def _sha256_lines(pairs: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for item_id, text in pairs:
        digest.update(f"{item_id}\t{text}\n".encode())
    return digest.hexdigest()


class LayerSource(BaseModel):
    """The chunk layer that was embedded."""

    model_config = STRICT

    max_tokens: int
    chunker_version: str
    parser_version: str
    chunk_tokenizer: str
    sources: dict[str, str]  # act -> source_sha256
    structure_sha256: str  # chunk ids and the nodes each one owns, in layer order
    content_sha256: str  # chunk ids and texts, in layer order


class QuerySource(BaseModel):
    """The benchmark queries that were embedded."""

    model_config = STRICT

    benchmark_version: str
    queries_sha256: str  # the sealed queries.jsonl
    content_sha256: str  # query ids and texts, in file order


def layer_source(docs: Sequence[ch.ChunkedDocument]) -> LayerSource:
    one = {(d.max_tokens, d.chunker_version, d.parser_version, d.tokenizer) for d in docs}
    if len(one) != 1:
        raise EmbeddingError(f"one layer needs one budget, chunker, parser and tokenizer: {one}")
    max_tokens, chunker, parser, tokenizer = one.pop()
    return LayerSource(
        max_tokens=max_tokens,
        chunker_version=chunker,
        parser_version=parser,
        chunk_tokenizer=tokenizer,
        sources={d.act: d.source_sha256 for d in docs},
        structure_sha256=_sha256_lines(
            (c.chunk_id, " ".join(c.source_paths)) for d in docs for c in d.chunks
        ),
        content_sha256=_sha256_lines((c.chunk_id, c.text) for d in docs for c in d.chunks),
    )


def query_source(benchmark: bm.Benchmark) -> QuerySource:
    return QuerySource(
        benchmark_version=benchmark.manifest.benchmark_version,
        queries_sha256=benchmark.manifest.files["queries.jsonl"],
        content_sha256=_sha256_lines((q.query_id, q.text) for q in benchmark.queries),
    )


class RuntimeIdentity(BaseModel):
    """The software that turns token ids into vectors. A new version of any of these
    can move the last bits of a vector, so it makes a cached set stale."""

    model_config = STRICT

    python: str
    torch: str
    transformers: str
    tokenizers: str
    numpy: str
    sentence_transformers: str | None  # None: not used by this pipeline
    dtype: str
    device: str


def runtime_identity(config: EmbeddingConfig) -> RuntimeIdentity:
    """The runtime installed here. Without torch or transformers no set can be
    computed or checked, so their absence is an error, never a guess."""
    import importlib.metadata as md

    try:
        torch, transformers = md.version("torch"), md.version("transformers")
    except md.PackageNotFoundError as e:
        raise ModelUnavailableError(
            "torch and transformers are not installed: install requirements-embed.lock.txt "
            "to compute or check embeddings"
        ) from e
    return RuntimeIdentity(
        python=platform.python_version(),
        torch=torch,
        transformers=transformers,
        tokenizers=md.version("tokenizers"),
        numpy=md.version("numpy"),
        sentence_transformers=None,
        dtype=config.dtype,
        device=config.device,
    )


class ArtifactIdentity(BaseModel):
    """Everything that defines a set of embeddings. Two sets with the same identity
    are the same computation; any difference makes the stored one stale."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    config: EmbeddingConfig
    config_hash: str
    tokenizer_sha256: str
    runtime: RuntimeIdentity
    layer: LayerSource | None = None
    queries: QuerySource | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.config_hash != self.config.config_hash():
            raise ValueError("config_hash is not the hash of config")
        if (self.layer is None) == (self.queries is None):
            raise ValueError("a set comes from a chunk layer or from queries, and only one")
        return self

    @property
    def kind(self) -> Kind:
        return "document" if self.layer is not None else "query"


def artifact_identity(
    config: EmbeddingConfig,
    tokenizer: LoadedTokenizer,
    runtime: RuntimeIdentity,
    source: LayerSource | QuerySource,
) -> ArtifactIdentity:
    return ArtifactIdentity(
        config=config,
        config_hash=config.config_hash(),
        tokenizer_sha256=tokenizer.sha256,
        runtime=runtime,
        layer=source if isinstance(source, LayerSource) else None,
        queries=source if isinstance(source, QuerySource) else None,
    )


def identity_changes(recorded: ArtifactIdentity, current: ArtifactIdentity) -> list[str]:
    """What differs between two identities, field by field, in words."""

    def fields(a: BaseModel | None, b: BaseModel | None) -> list[str]:
        if a is None or b is None:
            return [] if a is b else ["the kind of input"]
        da, db = a.model_dump(), b.model_dump()
        return [k for k in db if da.get(k) != db[k]]

    changed = []
    if recorded.config_hash != current.config_hash:
        changed.append(
            f"the embedding contract ({', '.join(fields(recorded.config, current.config))})"
        )
    if recorded.tokenizer_sha256 != current.tokenizer_sha256:
        changed.append("the tokenizer file (tokenizer_sha256)")
    if recorded.runtime != current.runtime:
        changed.append(f"the runtime ({', '.join(fields(recorded.runtime, current.runtime))})")
    if recorded.kind != current.kind:
        changed.append("the kind of input (documents against queries)")
    else:
        a, b = (
            (recorded.layer, current.layer)
            if current.layer
            else (recorded.queries, current.queries)
        )
        if a != b:
            what = "chunk layer" if current.layer else "queries"
            changed.append(f"the embedded {what} ({', '.join(fields(a, b))})")
    return changed


class EmbeddingSetMeta(BaseModel):
    """meta.json next to vectors.npy: row i of the matrix is item_ids[i]. Only
    identity decides staleness; created_at and environment describe the run."""

    model_config = STRICT

    schema_version: Literal["2"]
    identity: ArtifactIdentity
    item_ids: list[str]
    token_records: list[TokenRecord]
    vectors_sha256: str
    created_at: str  # informational
    environment: dict[str, str]  # informational: hardware, threads, cache path

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("an item id appears twice")
        if [r.item_id for r in self.token_records] != self.item_ids:
            raise ValueError("token_records must follow item_ids")
        return self


def matrix_sha256(vectors: np.ndarray) -> str:
    """The SHA-256 of a matrix's bytes: equal hashes, bit-identical vectors."""
    return hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest()


def write_embedding_set(
    directory: Path,
    identity: ArtifactIdentity,
    inputs: Sequence[ModelInput],
    vectors: np.ndarray,
    environment: Mapping[str, str],
) -> EmbeddingSetMeta:
    check_vectors(vectors, len(inputs), identity.config)
    meta = EmbeddingSetMeta(
        schema_version=SCHEMA_VERSION,
        identity=identity,
        item_ids=[i.item_id for i in inputs],
        token_records=[i.record for i in inputs],
        vectors_sha256=matrix_sha256(vectors),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        environment=dict(environment),
    )
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / VECTORS_FILE, vectors, allow_pickle=False)
    (directory / META_FILE).write_text(
        json.dumps(meta.model_dump(mode="json"), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return meta


def load_embedding_set(
    directory: Path, identity: ArtifactIdentity
) -> tuple[EmbeddingSetMeta, np.ndarray]:
    """A set made with exactly this identity, or an error that names what changed. A
    set written by another schema version is stale: its identity cannot be compared."""
    meta_path, vectors_path = directory / META_FILE, directory / VECTORS_FILE
    if not meta_path.exists() or not vectors_path.exists():
        raise EmbeddingsNotFoundError(f"{directory}: no embedding set")
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        version = raw.get("schema_version") if isinstance(raw, dict) else None
        raise StaleEmbeddingsError(
            f"{directory} is stale: written with meta schema {version}, the current one is "
            f"{SCHEMA_VERSION}"
        )
    try:
        meta = EmbeddingSetMeta.model_validate(raw)
    except ValidationError as e:
        raise EmbeddingError(f"{meta_path} does not meet the contract.\n{e}") from e
    changed = identity_changes(meta.identity, identity)
    if changed:
        raise StaleEmbeddingsError(f"{directory} is stale: changed since: {'; '.join(changed)}")
    vectors = np.load(vectors_path, allow_pickle=False)
    if matrix_sha256(vectors) != meta.vectors_sha256:
        raise EmbeddingError(f"{vectors_path} is not the matrix its meta.json records")
    check_vectors(vectors, len(meta.item_ids), identity.config)
    return meta, vectors


def embedding_dir(config: EmbeddingConfig, what: str, root: Path = EMBEDDINGS_DIR) -> Path:
    """One directory per model name and input set. The revision is NOT in the path,
    so a new revision finds the old set and reports it stale instead of ignoring it."""
    return root / config.name / what
