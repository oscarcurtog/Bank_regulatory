"""Unit tests for the pgvector index (T3, D-052): the schema's integrity rules, the round
trip of the layer and the vectors, the exact search and the resolution of provenance.

Each "postgres" test runs against a real server (compose.yaml locally, a service
container in CI) inside a schema of its own, dropped afterwards, so it never touches the
index tables. Every integrity rule is tested by trying to break it: the database, not
only this module, must refuse.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence

import numpy as np
import psycopg
import pytest

from ingest import chunks as ch
from ingest import embeddings as em
from ingest import hierarchy as h
from ingest import vector_index as vi

postgres = pytest.mark.postgres

CONFIG = em.load_config(em.CONFIG_DIR / "multilingual-e5-large.json")
RUNTIME = em.RuntimeIdentity(
    python="3.12.14",
    torch="2.14.1",
    transformers="5.15.1",
    tokenizers="0.22.2",
    numpy="2.5.4",
    sentence_transformers=None,
    dtype="float32",
    device="cpu",
)
CELEX = {"DORA": "32022R2554", "MiCA": "32023R1114"}


def _chunk(act: str, root: str) -> ch.Chunk:
    context = f"{act} Article {root.split('.')[0]}: Title"
    body = f"{root}. Obligations of financial entities under {act} {root}."
    return ch.Chunk(
        chunk_id=ch.chunk_id(act, root, "subtree"),
        article_id=f"art_{root.split('.')[0]}",
        root_path=root,
        coverage="subtree",
        source_paths=[root],
        context=context,
        body=body,
        text=ch.render_text(context, body),
        token_count=12,
        budget_exception=None,
    )


def _doc(act: str, roots: Sequence[str], sha: str = "1") -> ch.ChunkedDocument:
    return ch.ChunkedDocument(
        act=act,
        source_celex=CELEX[act],
        source_role="original",
        source_sha256=sha * 64,
        parser_version=h.PARSER_VERSION,
        chunker_version=ch.CHUNKER_VERSION,
        tokenizer="cl100k_base",
        max_tokens=500,
        chunks=[_chunk(act, r) for r in roots],
    )


def _unit_vectors(n: int, seed: int = 7) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, vi.DIMENSION)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _set(docs: Sequence[ch.ChunkedDocument], vectors: np.ndarray) -> em.EmbeddingSetMeta:
    ids = [c.chunk_id for d in docs for c in d.chunks]
    identity = em.ArtifactIdentity(
        config=CONFIG,
        config_hash=CONFIG.config_hash(),
        tokenizer_sha256="a" * 64,
        runtime=RUNTIME,
        layer=em.layer_source(docs),
    )
    records = [
        em.TokenRecord(
            item_id=i,
            original_model_tokens=20,
            encoded_model_tokens=20,
            truncated=False,
            dropped_tokens=0,
            kept_text_chars=40,
        )
        for i in ids
    ]
    return em.EmbeddingSetMeta(
        schema_version="2",
        identity=identity,
        item_ids=ids,
        token_records=records,
        vectors_sha256=em.matrix_sha256(vectors),
        created_at="2026-10-18T00:00:00+00:00",
        environment={},
    )


@pytest.fixture
def conn() -> Iterator[vi.Connection]:
    schema = f"t3_test_{uuid.uuid4().hex[:12]}"
    with vi.connect() as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.execute(f'CREATE SCHEMA "{schema}"')
        c.execute(f'SET search_path TO "{schema}", public')
        c.commit()
        try:
            vi.create_schema(c)
            # isolation guard: every table of the index must live in this test's schema,
            # or the tests would run against whatever index the database already holds
            tables = c.execute(
                "SELECT count(*) FROM pg_tables WHERE schemaname = %s", (schema,)
            ).fetchone()
            assert tables == (len(vi.SCHEMA_STATEMENTS),), f"index tables outside {schema}"
            yield c
        finally:
            c.rollback()
            c.execute(f'DROP SCHEMA "{schema}" CASCADE')
            c.commit()


@pytest.fixture
def loaded(conn: vi.Connection) -> tuple[vi.Connection, int, int, list[str], np.ndarray]:
    docs = [_doc("DORA", ["1.2", "10", "2"]), _doc("MiCA", ["3", "4.1"], sha="2")]
    vectors = _unit_vectors(5)
    layer_id = vi.load_layer(conn, docs)
    set_id = vi.load_embedding_set(conn, layer_id, _set(docs, vectors), vectors)
    ids = [c.chunk_id for d in docs for c in d.chunks]
    return conn, layer_id, set_id, ids, vectors


# ---------------------------------------------------------------------------
# what needs no database
# ---------------------------------------------------------------------------


def test_vector_literal_round_trips_float32_bit_for_bit():
    v = _unit_vectors(3, seed=11)
    v[0, :4] = [np.float32(0.1), np.float32(-1e-30), np.float32(1 / 3), np.float32(5e-8)]
    for row in v:
        assert vi.parse_vector(vi.vector_literal(row)).tobytes() == row.tobytes()


def test_vector_literal_refuses_other_shapes_types_and_non_finite_values():
    with pytest.raises(vi.VectorIndexError, match="float32 of shape"):
        vi.vector_literal(np.zeros(3, dtype=np.float32))
    with pytest.raises(vi.VectorIndexError, match="float32 of shape"):
        vi.vector_literal(np.zeros(vi.DIMENSION, dtype=np.float64))
    bad = np.zeros(vi.DIMENSION, dtype=np.float32)
    bad[5] = np.nan
    with pytest.raises(vi.VectorIndexError, match="not finite"):
        vi.vector_literal(bad)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


@postgres
def test_schema_is_created_once_and_another_version_is_refused(conn):
    vi.create_schema(conn)  # a second time: nothing to do
    conn.execute("UPDATE index_schema SET version = '0'")
    conn.commit()
    with pytest.raises(vi.StaleIndexError, match="schema"):
        vi.create_schema(conn)


@postgres
def test_schema_goes_in_the_current_schema_even_when_a_later_one_holds_an_index(conn):
    """The defect this pins down: with the index in a schema later in the search path,
    create_schema() saw it, created nothing in front, and every later statement reached
    the other index (a test once set the real index's version to '0')."""
    row = conn.execute("SELECT current_schema()").fetchone()
    assert row is not None
    first, second = row[0], f"t3_test_{uuid.uuid4().hex[:12]}"
    conn.execute(f'CREATE SCHEMA "{second}"')
    conn.execute(f'SET search_path TO "{second}", "{first}", public')
    conn.commit()
    try:
        vi.create_schema(conn)
        tables = conn.execute(
            "SELECT count(*) FROM pg_tables WHERE schemaname = %s", (second,)
        ).fetchone()
        assert tables == (len(vi.SCHEMA_STATEMENTS),)
        conn.execute("UPDATE index_schema SET version = '0'")
        conn.commit()
        assert conn.execute(f'SELECT version FROM "{first}".index_schema').fetchone() == ("1",)
    finally:
        conn.rollback()
        conn.execute(f'DROP SCHEMA "{second}" CASCADE')
        conn.execute(f'SET search_path TO "{first}", public')
        conn.commit()


# ---------------------------------------------------------------------------
# the layer and the vectors make the round trip
# ---------------------------------------------------------------------------


@postgres
def test_layer_round_trips_field_by_field_and_loading_twice_changes_nothing(conn):
    docs = [_doc("DORA", ["1.2", "10", "2"]), _doc("MiCA", ["3"], sha="2")]
    layer_id = vi.load_layer(conn, docs)
    assert vi.read_layer(conn, layer_id) == tuple(docs)
    assert vi.load_layer(conn, docs) == layer_id
    assert conn.execute("SELECT count(*) FROM chunk").fetchone() == (4,)


@postgres
def test_a_row_changed_in_the_database_is_detected(conn):
    docs = [_doc("DORA", ["1.2", "10"])]
    layer_id = vi.load_layer(conn, docs)
    conn.execute("UPDATE chunk SET token_count = 13 WHERE chunk_id = 'DORA:10'")
    conn.commit()
    with pytest.raises(vi.StaleIndexError, match="not the chunk layer on disk"):
        vi.load_layer(conn, docs)
    assert layer_id


@postgres
def test_vectors_round_trip_bit_for_bit_and_loading_twice_changes_nothing(loaded):
    conn, layer_id, set_id, ids, vectors = loaded
    back = vi.read_matrix(conn, set_id, ids)
    assert back.tobytes() == vectors.tobytes()
    docs = vi.read_layer(conn, layer_id)
    assert vi.load_embedding_set(conn, layer_id, _set(docs, vectors), vectors) == set_id
    assert conn.execute("SELECT count(*) FROM chunk_embedding").fetchone() == (5,)


@postgres
def test_a_set_computed_on_another_layer_is_refused(conn):
    docs = [_doc("DORA", ["1.2", "10"])]
    other = [_doc("DORA", ["1.2", "11"])]
    layer_id = vi.load_layer(conn, docs)
    vectors = _unit_vectors(2)
    with pytest.raises(vi.StaleIndexError, match="another chunk layer"):
        vi.load_embedding_set(conn, layer_id, _set(other, vectors), vectors)


# ---------------------------------------------------------------------------
# the database itself refuses what breaks the contract
# ---------------------------------------------------------------------------


def _insert_vector(
    conn: vi.Connection, set_id: int, layer_id: int, chunk_id: str, literal: str
) -> None:
    conn.execute(
        "INSERT INTO chunk_embedding (set_id, layer_id, chunk_id, embedding, "
        "original_model_tokens, encoded_model_tokens, dropped_tokens, kept_text_chars) "
        "VALUES (%s, %s, %s, %s::vector, 20, 20, 0, 40)",
        (set_id, layer_id, chunk_id, literal),
    )


@postgres
def test_database_refuses_a_vector_of_another_dimension(loaded):
    conn, layer_id, set_id, _, _ = loaded
    conn.execute("DELETE FROM chunk_embedding WHERE chunk_id = 'DORA:2'")
    with pytest.raises(psycopg.errors.DataException, match="1024"):
        _insert_vector(conn, set_id, layer_id, "DORA:2", "[1,0,0]")


@postgres
def test_database_refuses_a_vector_that_is_not_normalized(loaded):
    conn, layer_id, set_id, _, vectors = loaded
    conn.execute("DELETE FROM chunk_embedding WHERE chunk_id = 'DORA:2'")
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_vector(conn, set_id, layer_id, "DORA:2", vi.vector_literal(vectors[0] * 2))


@postgres
def test_database_refuses_an_orphan_vector(loaded):
    conn, layer_id, set_id, _, vectors = loaded
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        _insert_vector(conn, set_id, layer_id, "DORA:99", vi.vector_literal(vectors[0]))


@postgres
def test_database_refuses_a_vector_for_a_chunk_of_another_layer(loaded):
    conn, _, set_id, _, vectors = loaded
    other_layer = vi.load_layer(conn, [_doc("DORA", ["7"], sha="3")])
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        _insert_vector(conn, set_id, other_layer, "DORA:7", vi.vector_literal(vectors[0]))


@postgres
def test_database_refuses_two_vectors_for_one_chunk_of_a_set(loaded):
    conn, layer_id, set_id, _, vectors = loaded
    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert_vector(conn, set_id, layer_id, "DORA:2", vi.vector_literal(vectors[0]))


@postgres
def test_a_chunk_that_has_vectors_cannot_be_deleted(loaded):
    conn, _, _, _, _ = loaded
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        conn.execute("DELETE FROM chunk WHERE chunk_id = 'DORA:2'")


# ---------------------------------------------------------------------------
# exact search and provenance
# ---------------------------------------------------------------------------


@postgres
def test_search_is_the_ranking_of_rank_exact(loaded):
    conn, _, set_id, ids, vectors = loaded
    queries = _unit_vectors(4, seed=3)
    expected = em.rank_exact(["q0", "q1", "q2", "q3"], queries, ids, vectors)
    for q, want in zip(queries, expected, strict=True):
        got = vi.search(conn, set_id, q, k=len(ids))
        assert tuple(c for c, _ in got) == want.chunk_ids
        assert [s for _, s in got] == pytest.approx(list(want.scores), abs=1e-6)
    assert vi.search(conn, set_id, queries[0], k=2) == vi.search(conn, set_id, queries[0], k=5)[:2]
    assert vi.vector_indexes(conn) == []


@postgres
def test_ties_break_by_chunk_id_in_code_point_order(conn):
    docs = [_doc("DORA", ["10", "2", "1.2"])]
    layer_id = vi.load_layer(conn, docs)
    same = np.tile(_unit_vectors(1), (3, 1))
    set_id = vi.load_embedding_set(conn, layer_id, _set(docs, same), same)
    got = [c for c, _ in vi.search(conn, set_id, same[0], k=3)]
    # the database's default collation (en_US.utf8) would put DORA:10 before DORA:1.2
    assert got == sorted(["DORA:10", "DORA:2", "DORA:1.2"]) == ["DORA:1.2", "DORA:10", "DORA:2"]


@postgres
def test_resolve_returns_the_provenance_of_each_chunk_in_order(loaded):
    conn, layer_id, _, _, _ = loaded
    found = vi.resolve(conn, layer_id, ["MiCA:4.1", "DORA:10"])
    assert [p.chunk_id for p in found] == ["MiCA:4.1", "DORA:10"]
    mica = found[0]
    assert (mica.act, mica.source_celex, mica.source_sha256) == ("MiCA", "32023R1114", "2" * 64)
    assert (mica.article_id, mica.root_path, mica.source_paths) == ("art_4", "4.1", ("4.1",))
    assert mica.text.startswith("MiCA Article 4: Title")


@postgres
def test_resolve_refuses_an_id_the_layer_does_not_hold(loaded):
    conn, layer_id, _, _, _ = loaded
    with pytest.raises(vi.UnresolvedChunkError, match="DORA:99"):
        vi.resolve(conn, layer_id, ["DORA:10", "DORA:99"])
