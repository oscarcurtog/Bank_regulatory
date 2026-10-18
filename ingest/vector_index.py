"""The pgvector index of the chunk layer (T3, D-052): PostgreSQL holds the chunks with
their provenance and one row per (embedding set, chunk) with its vector, and answers a
query vector with an exact nearest-neighbour search under the embedding contract.

Exact, on purpose: no HNSW or IVFFlat index is ever created on the vectors, so every
query is compared with every vector of its set, the computation of rank_exact() in
ingest/embeddings.py moved into the database. vector_indexes() lists any approximate
index, and the T3 script refuses to measure if one exists.

Similarity. The contracts in config/embeddings/ use "dot" on L2-normalized vectors.
pgvector's operator for that function is <#>, the NEGATIVE inner product
(vector_negative_inner_product): ordering by it ascending is ordering by the dot product
descending, and the score is its negation. <=> (cosine distance) and <-> (Euclidean
distance) give the same order on unit vectors in exact arithmetic, but they are other
functions with other roundings, and the contract names the dot product.

Ties. rank_exact() breaks them by chunk id in Python's code-point order. chunk_id is
declared COLLATE "C" (byte order: the same order for these ASCII ids), because the
default collation of the image, en_US.utf8, puts "DORA:10" before "DORA:1.2".

Schema, version 1:

- chunk_layer: one chunk layer (LayerSource): budget, chunker, parser, tokenizer and the
  two hashes that identify it.
- layer_document: the provenance of each act in a layer (ChunkedDocument's header),
  stated once and not in every chunk.
- chunk: the source content, one row per chunk, in layer order (ordinal).
- embedding_set: one set of vectors (ArtifactIdentity, D-028): the model contract, the
  tokenizer, the runtime and the layer, with the hash of the matrix it was computed as.
- chunk_embedding: one vector per (set, chunk), with the token record of what the model
  saw.

Integrity, enforced by the database and not only by this module:

- a vector has exactly 1024 dimensions (vector(1024)) and a norm within the contract's
  tolerance of 1 (CHECK), so the dot product is the cosine;
- a vector references its set and a chunk of that set's own layer (two composite foreign
  keys), so no vector is an orphan and none can point at a chunk of another layer;
- one vector per chunk and set (primary key), one set per identity (unique), and a chunk
  or a set that still has vectors cannot be deleted (foreign keys without cascade).

What the database cannot express, the loaders prove: the stored layer is rebuilt from its
rows into ChunkedDocument models (validated again at this boundary, D-007) and compared
with the layer on disk; the stored matrix is read back and its SHA-256 compared with the
one meta.json records. Vectors travel in pgvector's text form: a float32 written as the
shortest repr of its double and parsed by pgvector as float4 is the same float32, and the
hash check proves it on every load.

Connection: the standard libpq variables PGHOST, PGPORT, PGUSER, PGDATABASE and
PGPASSWORD (compose.yaml starts the local server). Nothing here guesses a default: an
unreachable server is an error that names the remedy.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from ingest import chunks as ch
from ingest import embeddings as em

SCHEMA_VERSION = "1"
DIMENSION = 1024
_HEX64 = "~ '^[0-9a-f]{64}$'"

SCHEMA_STATEMENTS = (
    "CREATE TABLE index_schema (version text PRIMARY KEY)",
    f"""
    CREATE TABLE chunk_layer (
        layer_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        max_tokens integer NOT NULL CHECK (max_tokens > 0),
        chunker_version text NOT NULL,
        parser_version text NOT NULL,
        chunk_tokenizer text NOT NULL,
        structure_sha256 text NOT NULL CHECK (structure_sha256 {_HEX64}),
        content_sha256 text NOT NULL CHECK (content_sha256 {_HEX64}),
        UNIQUE (structure_sha256, content_sha256)
    )""",
    f"""
    CREATE TABLE layer_document (
        layer_id integer NOT NULL REFERENCES chunk_layer,
        act text NOT NULL,
        ordinal integer NOT NULL CHECK (ordinal >= 0),
        source_celex text NOT NULL,
        source_role text NOT NULL CHECK (source_role IN ('original', 'consolidated')),
        source_sha256 text NOT NULL CHECK (source_sha256 {_HEX64}),
        PRIMARY KEY (layer_id, act),
        UNIQUE (layer_id, ordinal)
    )""",
    """
    CREATE TABLE chunk (
        layer_id integer NOT NULL,
        chunk_id text COLLATE "C" NOT NULL,
        act text NOT NULL,
        ordinal integer NOT NULL CHECK (ordinal >= 0),
        article_id text NOT NULL,
        root_path text NOT NULL,
        coverage text NOT NULL CHECK (coverage IN ('subtree', 'own')),
        source_paths text[] NOT NULL CHECK (cardinality(source_paths) > 0),
        context text NOT NULL,
        body text NOT NULL,
        text text NOT NULL,
        token_count integer NOT NULL CHECK (token_count > 0),
        budget_exception text,
        PRIMARY KEY (layer_id, chunk_id),
        UNIQUE (layer_id, ordinal),
        FOREIGN KEY (layer_id, act) REFERENCES layer_document (layer_id, act)
    )""",
    f"""
    CREATE TABLE embedding_set (
        set_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        layer_id integer NOT NULL REFERENCES chunk_layer,
        model_name text NOT NULL,
        model_id text NOT NULL,
        revision text NOT NULL CHECK (revision ~ '^[0-9a-f]{{40}}$'),
        config_hash text NOT NULL CHECK (config_hash {_HEX64}),
        tokenizer_sha256 text NOT NULL CHECK (tokenizer_sha256 {_HEX64}),
        dimension integer NOT NULL CHECK (dimension = {DIMENSION}),
        similarity text NOT NULL CHECK (similarity = 'dot'),
        normalized boolean NOT NULL CHECK (normalized),
        identity jsonb NOT NULL,
        identity_sha256 text NOT NULL UNIQUE CHECK (identity_sha256 {_HEX64}),
        vectors_sha256 text NOT NULL CHECK (vectors_sha256 {_HEX64}),
        computed_at text NOT NULL,
        UNIQUE (set_id, layer_id)
    )""",
    f"""
    CREATE TABLE chunk_embedding (
        set_id integer NOT NULL,
        layer_id integer NOT NULL,
        chunk_id text COLLATE "C" NOT NULL,
        embedding vector({DIMENSION}) NOT NULL
            CHECK (abs(vector_norm(embedding) - 1) <= {em.NORM_TOLERANCE}),
        original_model_tokens integer NOT NULL CHECK (original_model_tokens >= 1),
        encoded_model_tokens integer NOT NULL CHECK (encoded_model_tokens >= 1),
        dropped_tokens integer NOT NULL
            CHECK (dropped_tokens = original_model_tokens - encoded_model_tokens),
        kept_text_chars integer NOT NULL CHECK (kept_text_chars >= 0),
        PRIMARY KEY (set_id, chunk_id),
        FOREIGN KEY (set_id, layer_id) REFERENCES embedding_set (set_id, layer_id),
        FOREIGN KEY (layer_id, chunk_id) REFERENCES chunk (layer_id, chunk_id)
    )""",
)

SEARCH_SQL = """
SELECT chunk_id, embedding <#> %(query)s::vector AS negative_dot
FROM chunk_embedding
WHERE set_id = %(set_id)s
ORDER BY negative_dot, chunk_id
LIMIT %(k)s
"""

Connection = psycopg.Connection[Any]


class VectorIndexError(Exception):
    """The index, its schema or its contents do not meet the contract."""


class IndexUnavailableError(VectorIndexError):
    """No PostgreSQL server answers with the libpq settings of this environment."""


class StaleIndexError(VectorIndexError):
    """The database holds another schema version, or other data than the files."""


class UnresolvedChunkError(VectorIndexError):
    """A chunk id that the index cannot resolve to its content and provenance."""


def connect(conninfo: str = "") -> Connection:
    """A connection with the libpq settings of the environment (PGHOST and the rest)."""
    try:
        return psycopg.connect(conninfo)
    except psycopg.OperationalError as e:
        raise IndexUnavailableError(
            f"cannot reach PostgreSQL ({e}). Start it with `docker compose up -d --wait "
            f"postgres` and set PGHOST, PGPORT, PGUSER, PGDATABASE and PGPASSWORD."
        ) from e


def create_schema(conn: Connection) -> None:
    """The tables of schema version 1 in the CURRENT schema (the first of the search
    path), created in one transaction (DDL is transactional in PostgreSQL, so a failure
    leaves no half schema). A schema that already holds another version is refused, never
    migrated silently.

    The check looks at the current schema only. to_regclass() would also find the tables
    of a later schema in the search path: with the index in public and a test schema in
    front of it, the tables would not be created in the test schema, and every unqualified
    statement after it would land on the index in public."""
    with conn.transaction():
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        here = conn.execute(
            "SELECT count(*) FROM pg_tables "
            "WHERE schemaname = current_schema() AND tablename = 'index_schema'"
        ).fetchone()
        if here is None or here[0] == 0:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            conn.execute("INSERT INTO index_schema (version) VALUES (%s)", (SCHEMA_VERSION,))
            return
        versions = [r[0] for r in conn.execute("SELECT version FROM index_schema")]
        if versions != [SCHEMA_VERSION]:
            raise StaleIndexError(
                f"the database holds index schema {versions}, this code needs {SCHEMA_VERSION}"
            )


# ---------------------------------------------------------------------------
# the chunk layer
# ---------------------------------------------------------------------------


def load_layer(conn: Connection, docs: Sequence[ch.ChunkedDocument]) -> int:
    """The layer's id, inserting it if the database does not hold it yet. Either way,
    the stored layer is rebuilt from its rows and must equal `docs` field by field."""
    source = em.layer_source(docs)
    first = docs[0]
    with conn.transaction():
        row = conn.execute(
            "SELECT layer_id FROM chunk_layer WHERE structure_sha256 = %s AND content_sha256 = %s",
            (source.structure_sha256, source.content_sha256),
        ).fetchone()
        if row is None:
            inserted = conn.execute(
                "INSERT INTO chunk_layer (max_tokens, chunker_version, parser_version, "
                "chunk_tokenizer, structure_sha256, content_sha256) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING layer_id",
                (
                    first.max_tokens,
                    first.chunker_version,
                    first.parser_version,
                    first.tokenizer,
                    source.structure_sha256,
                    source.content_sha256,
                ),
            ).fetchone()
            assert inserted is not None
            layer_id = int(inserted[0])
            rows = []
            ordinal = 0
            for position, d in enumerate(docs):
                conn.execute(
                    "INSERT INTO layer_document (layer_id, act, ordinal, source_celex, "
                    "source_role, source_sha256) VALUES (%s, %s, %s, %s, %s, %s)",
                    (layer_id, d.act, position, d.source_celex, d.source_role, d.source_sha256),
                )
                for c in d.chunks:
                    rows.append(
                        (
                            layer_id,
                            c.chunk_id,
                            d.act,
                            ordinal,
                            c.article_id,
                            c.root_path,
                            c.coverage,
                            list(c.source_paths),
                            c.context,
                            c.body,
                            c.text,
                            c.token_count,
                            c.budget_exception,
                        )
                    )
                    ordinal += 1
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunk (layer_id, chunk_id, act, ordinal, article_id, root_path, "
                    "coverage, source_paths, context, body, text, token_count, budget_exception) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    rows,
                )
        else:
            layer_id = int(row[0])
    if read_layer(conn, layer_id) != tuple(docs):
        raise StaleIndexError(
            f"layer {layer_id} in the database is not the chunk layer on disk, although the "
            f"two hashes match: a row was changed in the database"
        )
    return layer_id


def read_layer(conn: Connection, layer_id: int) -> tuple[ch.ChunkedDocument, ...]:
    """The layer rebuilt from its rows, in layer order, and validated by the same models
    that validate the files."""
    layer = conn.execute(
        "SELECT max_tokens, chunker_version, parser_version, chunk_tokenizer "
        "FROM chunk_layer WHERE layer_id = %s",
        (layer_id,),
    ).fetchone()
    if layer is None:
        raise VectorIndexError(f"no chunk layer {layer_id}")
    max_tokens, chunker_version, parser_version, tokenizer = layer
    documents = conn.execute(
        "SELECT act, source_celex, source_role, source_sha256 FROM layer_document "
        "WHERE layer_id = %s ORDER BY ordinal",
        (layer_id,),
    ).fetchall()
    chunks: dict[str, list[dict[str, Any]]] = {act: [] for act, *_ in documents}
    for r in conn.execute(
        "SELECT act, chunk_id, article_id, root_path, coverage, source_paths, context, body, "
        "text, token_count, budget_exception FROM chunk WHERE layer_id = %s ORDER BY ordinal",
        (layer_id,),
    ):
        act, *fields = r
        names = (
            "chunk_id",
            "article_id",
            "root_path",
            "coverage",
            "source_paths",
            "context",
            "body",
            "text",
            "token_count",
            "budget_exception",
        )
        chunks[act].append(dict(zip(names, fields, strict=True)))
    return tuple(
        ch.ChunkedDocument.model_validate(
            {
                "act": act,
                "source_celex": celex,
                "source_role": role,
                "source_sha256": sha,
                "parser_version": parser_version,
                "chunker_version": chunker_version,
                "tokenizer": tokenizer,
                "max_tokens": max_tokens,
                "chunks": chunks[act],
            }
        )
        for act, celex, role, sha in documents
    )


# ---------------------------------------------------------------------------
# embedding sets
# ---------------------------------------------------------------------------


def identity_sha256(identity: em.ArtifactIdentity) -> str:
    canonical = json.dumps(identity.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def vector_literal(vector: np.ndarray) -> str:
    """pgvector's text form of one float32 vector: the shortest repr of each value's
    double, which pgvector parses back into the same float32."""
    if vector.dtype != np.float32 or vector.shape != (DIMENSION,):
        raise VectorIndexError(
            f"a vector must be float32 of shape ({DIMENSION},): {vector.dtype} {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise VectorIndexError("a vector has a value that is not finite")
    return "[" + ",".join(repr(x) for x in vector.tolist()) + "]"


def parse_vector(text: str) -> np.ndarray:
    """A vector in pgvector's text form, as float32."""
    if not (text.startswith("[") and text.endswith("]")):
        raise VectorIndexError(f"not a pgvector literal: {text[:40]!r}")
    return np.array([float(x) for x in text[1:-1].split(",")], dtype=np.float32)


def load_embedding_set(
    conn: Connection, layer_id: int, meta: em.EmbeddingSetMeta, vectors: np.ndarray
) -> int:
    """The set's id, inserting it if the database does not hold it yet. The set must
    have been computed on exactly this layer, and the matrix read back from the database
    must hash to the matrix that meta.json records, or nothing is returned."""
    identity = meta.identity
    if identity.layer is None:
        raise VectorIndexError("only document sets are indexed; query vectors are inputs")
    if identity.layer != em.layer_source(read_layer(conn, layer_id)):
        raise StaleIndexError(f"the set was computed on another chunk layer than {layer_id}")
    config = identity.config
    if config.dimension != DIMENSION:
        raise VectorIndexError(
            f"{config.name}: {config.dimension} dimensions, the index has {DIMENSION}"
        )
    if vectors.shape != (len(meta.item_ids), DIMENSION):
        raise VectorIndexError(f"{vectors.shape} vectors for {len(meta.item_ids)} items")
    id_sha = identity_sha256(identity)
    with conn.transaction():
        row = conn.execute(
            "SELECT set_id, layer_id, vectors_sha256 FROM embedding_set WHERE identity_sha256 = %s",
            (id_sha,),
        ).fetchone()
        if row is None:
            inserted = conn.execute(
                "INSERT INTO embedding_set (layer_id, model_name, model_id, revision, config_hash, "
                "tokenizer_sha256, dimension, similarity, normalized, identity, identity_sha256, "
                "vectors_sha256, computed_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING set_id",
                (
                    layer_id,
                    config.name,
                    config.model_id,
                    config.revision,
                    identity.config_hash,
                    identity.tokenizer_sha256,
                    config.dimension,
                    config.similarity,
                    config.normalize,
                    Jsonb(identity.model_dump(mode="json")),
                    id_sha,
                    meta.vectors_sha256,
                    meta.created_at,
                ),
            ).fetchone()
            assert inserted is not None
            set_id = int(inserted[0])
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunk_embedding (set_id, layer_id, chunk_id, embedding, "
                    "original_model_tokens, encoded_model_tokens, dropped_tokens, kept_text_chars) "
                    "VALUES (%s, %s, %s, %s::vector, %s, %s, %s, %s)",
                    [
                        (
                            set_id,
                            layer_id,
                            item_id,
                            vector_literal(vectors[i]),
                            record.original_model_tokens,
                            record.encoded_model_tokens,
                            record.dropped_tokens,
                            record.kept_text_chars,
                        )
                        for i, (item_id, record) in enumerate(
                            zip(meta.item_ids, meta.token_records, strict=True)
                        )
                    ],
                )
        else:
            set_id, stored_layer, stored_sha = int(row[0]), int(row[1]), row[2]
            if stored_layer != layer_id or stored_sha != meta.vectors_sha256:
                raise StaleIndexError(f"set {set_id} has the same identity and other contents")
    if em.matrix_sha256(read_matrix(conn, set_id, meta.item_ids)) != meta.vectors_sha256:
        raise StaleIndexError(f"set {set_id}: the vectors read back are not the matrix on disk")
    if read_token_records(conn, set_id, meta.item_ids) != meta.token_records:
        raise StaleIndexError(f"set {set_id}: the token records read back differ")
    return set_id


def read_matrix(conn: Connection, set_id: int, item_ids: Sequence[str]) -> np.ndarray:
    """The set's vectors in the order of `item_ids`, which must be exactly its chunks."""
    stored = dict(
        conn.execute(
            "SELECT chunk_id, embedding::text FROM chunk_embedding WHERE set_id = %s", (set_id,)
        ).fetchall()
    )
    if set(stored) != set(item_ids) or len(stored) != len(item_ids):
        raise StaleIndexError(
            f"set {set_id} holds {len(stored)} vectors, {len(set(stored) ^ set(item_ids))} of "
            f"them or of the {len(item_ids)} expected ids differ"
        )
    return np.stack([parse_vector(stored[i]) for i in item_ids])


def read_token_records(
    conn: Connection, set_id: int, item_ids: Sequence[str]
) -> list[em.TokenRecord]:
    rows = {
        r[0]: r[1:]
        for r in conn.execute(
            "SELECT chunk_id, original_model_tokens, encoded_model_tokens, dropped_tokens, "
            "kept_text_chars FROM chunk_embedding WHERE set_id = %s",
            (set_id,),
        )
    }
    return [
        em.TokenRecord(
            item_id=i,
            original_model_tokens=rows[i][0],
            encoded_model_tokens=rows[i][1],
            truncated=rows[i][2] > 0,
            dropped_tokens=rows[i][2],
            kept_text_chars=rows[i][3],
        )
        for i in item_ids
    ]


# ---------------------------------------------------------------------------
# search and resolution
# ---------------------------------------------------------------------------


def search(conn: Connection, set_id: int, query: np.ndarray, k: int) -> list[tuple[str, float]]:
    """The k chunks of the set nearest to `query` by dot product, best first and by chunk
    id on a tie, with their scores: an exact search, every vector of the set compared."""
    if k < 1:
        raise VectorIndexError(f"k must be at least 1, not {k}")
    rows = conn.execute(
        SEARCH_SQL, {"query": vector_literal(query), "set_id": set_id, "k": k}
    ).fetchall()
    return [(chunk_id, -float(negative_dot)) for chunk_id, negative_dot in rows]


def search_plan(conn: Connection, set_id: int, query: np.ndarray, k: int) -> dict[str, Any]:
    """PostgreSQL's plan for one search (EXPLAIN, JSON), to show what it does."""
    row = conn.execute(
        "EXPLAIN (FORMAT JSON) " + SEARCH_SQL,
        {"query": vector_literal(query), "set_id": set_id, "k": k},
    ).fetchone()
    assert row is not None
    plan: dict[str, Any] = row[0][0]["Plan"]
    return plan


def vector_indexes(conn: Connection) -> list[str]:
    """Approximate indexes on the vectors (HNSW, IVFFlat). An exact search needs none."""
    return [
        f"{name} ({method})"
        for name, method in conn.execute(
            "SELECT i.relname, am.amname FROM pg_index x "
            "JOIN pg_class i ON i.oid = x.indexrelid JOIN pg_am am ON am.oid = i.relam "
            "WHERE x.indrelid = 'chunk_embedding'::regclass AND am.amname IN ('hnsw', 'ivfflat')"
        )
    ]


@dataclass(frozen=True)
class Provenance:
    """Everything a citation needs about one retrieved chunk: what it says, where it sits
    in the act, and which source document and pipeline versions produced it."""

    chunk_id: str
    act: str
    source_celex: str
    source_role: str
    source_sha256: str
    parser_version: str
    chunker_version: str
    max_tokens: int
    article_id: str
    root_path: str
    coverage: str
    source_paths: tuple[str, ...]
    context: str
    text: str
    token_count: int


def resolve(conn: Connection, layer_id: int, chunk_ids: Sequence[str]) -> list[Provenance]:
    """The provenance of each chunk id, in the order given. An id the layer does not
    hold is an error, never a gap in the answer."""
    rows = conn.execute(
        "SELECT c.chunk_id, c.act, d.source_celex, d.source_role, d.source_sha256, "
        "l.parser_version, l.chunker_version, l.max_tokens, c.article_id, c.root_path, "
        "c.coverage, c.source_paths, c.context, c.text, c.token_count "
        "FROM chunk c JOIN layer_document d USING (layer_id, act) "
        "JOIN chunk_layer l USING (layer_id) "
        "WHERE c.layer_id = %s AND c.chunk_id = ANY(%s)",
        (layer_id, list(chunk_ids)),
    ).fetchall()
    found = {
        r[0]: Provenance(
            chunk_id=r[0],
            act=r[1],
            source_celex=r[2],
            source_role=r[3],
            source_sha256=r[4],
            parser_version=r[5],
            chunker_version=r[6],
            max_tokens=r[7],
            article_id=r[8],
            root_path=r[9],
            coverage=r[10],
            source_paths=tuple(r[11]),
            context=r[12],
            text=r[13],
            token_count=r[14],
        )
        for r in rows
    }
    missing = [c for c in chunk_ids if c not in found]
    if missing:
        raise UnresolvedChunkError(f"layer {layer_id} does not hold {missing}")
    return [found[c] for c in chunk_ids]
