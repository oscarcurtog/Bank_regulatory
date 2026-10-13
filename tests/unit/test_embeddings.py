"""The embedding contract, explicit truncation, exact search and stale embeddings.

A word tokenizer stands in for the model's: one id per word, <s> and </s> around the
sequence, like the template of XLM-RoBERTa. Vectors are synthetic. Nothing here
downloads a tokenizer or a model; the real model has its own tests, marked
"embedding_model"."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import embeddings as em
from ingest import evaluation as ev
from ingest import hierarchy as h

CONFIG_PATH = Path(__file__).parents[2] / "config" / "embeddings" / "multilingual-e5-large.json"
BASE = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
BOS, EOS = 0, 2


def config(**overrides) -> em.EmbeddingConfig:
    return em.EmbeddingConfig.model_validate(BASE | overrides)


@dataclass(frozen=True)
class FakeEncoding:
    ids: list[int]
    special_tokens_mask: list[int]
    offsets: list[tuple[int, int]]


class WordTokenizer:
    """One token per run of non-space characters, wrapped in <s> ... </s>."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def encode(self, sequence: str) -> FakeEncoding:
        words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", sequence)]
        ids = [self.vocab.setdefault(w, len(self.vocab) + 10) for w, _, _ in words]
        return FakeEncoding(
            [BOS, *ids, EOS],
            [1, *([0] * len(ids)), 1],
            [(0, 0), *[(s, e) for _, s, e in words], (0, 0)],
        )


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


def test_the_versioned_config_is_the_contract_of_multilingual_e5_large():
    c = em.load_config(CONFIG_PATH)
    assert (c.model_id, c.revision) == (
        "intfloat/multilingual-e5-large",
        "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
    )
    assert (c.dimension, c.max_input_tokens, c.pooling, c.normalize, c.similarity) == (
        1024,
        512,
        "mean",
        True,
        "dot",
    )
    assert (c.query_prefix, c.document_prefix) == ("query: ", "passage: ")
    assert (c.document_truncation, c.query_truncation) == ("explicit_tail", "refuse")


def test_the_versioned_config_is_the_contract_of_bge_m3_dense():
    c = em.load_config(CONFIG_PATH.with_name("bge-m3.json"))
    assert (c.model_id, c.revision, c.tokenizer_commit) == (
        "BAAI/bge-m3",
        "5617a9f61b028005a4858fdac845db406aefb181",
        "5617a9f61b028005a4858fdac845db406aefb181",
    )
    assert (c.dimension, c.max_input_tokens, c.pooling, c.normalize, c.similarity) == (
        1024,
        8192,
        "cls",
        True,
        "dot",
    )
    assert (c.query_prefix, c.document_prefix) == ("", "")
    assert (c.document_truncation, c.query_truncation) == ("explicit_tail", "refuse")
    assert (c.device, c.dtype) == ("cpu", "float32")


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"query_prefix": "passage: "}, "must differ"),
        ({"query_prefix": ""}, "both queries and documents, or neither"),
        ({"normalize": False}, "cosine only on normalized"),
        ({"revision": "main"}, "revision"),
        ({"similarity": "cosine"}, "similarity"),
        ({"pooling": "max"}, "pooling"),
        ({"device": "mps"}, "device"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_a_config_that_breaks_the_contract_is_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        config(**overrides)


def test_the_config_hash_changes_with_any_field_and_only_then():
    assert config().config_hash() == config().config_hash()
    assert config(batch_size=8).config_hash() != config().config_hash()
    assert config(revision="0" * 40).config_hash() != config().config_hash()


def test_the_prefix_is_the_only_thing_added_to_the_text():
    c = config()
    assert em.model_input(c, "query", "Can ESMA ban it?") == "query: Can ESMA ban it?"
    assert em.model_input(c, "document", "MiCA Article 103") == "passage: MiCA Article 103"


def bge_like(**overrides) -> em.EmbeddingConfig:
    """A model with the shape of BGE-M3 dense: no prefixes, CLS pooling, 8192 tokens."""
    return config(
        name="no-prefix-cls",
        query_prefix="",
        document_prefix="",
        pooling="cls",
        max_input_tokens=8192,
        **overrides,
    )


def test_the_contract_holds_mean_pooling_with_prefixes_and_cls_pooling_without():
    assert (config().pooling, config().query_prefix) == ("mean", "query: ")
    c = bge_like()
    assert (c.pooling, c.query_prefix, c.document_prefix, c.max_input_tokens) == (
        "cls",
        "",
        "",
        8192,
    )
    assert c.config_hash() != config().config_hash()


def test_without_prefixes_the_text_is_embedded_as_it_is():
    c = bge_like()
    assert em.model_input(c, "query", "Can ESMA ban it?") == "Can ESMA ban it?"
    [q, d] = [
        em.prepare_inputs([("x", "three words here")], kind, WordTokenizer(), c)[0]
        for kind in ("query", "document")
    ]
    # <s> + three words + </s>: nothing else is counted or added
    assert q.input_ids == d.input_ids and q.record.original_model_tokens == 5


def test_a_long_input_is_measured_against_its_own_models_limit():
    c, tok = bge_like(), WordTokenizer()
    fits, over = em.prepare_inputs([("a", words(8190)), ("b", words(8191))], "document", tok, c)
    assert (fits.record.original_model_tokens, fits.record.truncated) == (8192, False)
    assert over.record.truncated and over.record.encoded_model_tokens == 8192
    assert over.input_ids[-1] == EOS
    with pytest.raises(em.TruncationRefusedError):
        em.prepare_inputs([("q", words(8191))], "query", tok, c)


# ---------------------------------------------------------------------------
# counting and explicit truncation
# ---------------------------------------------------------------------------


def test_prefix_and_special_tokens_are_counted():
    c = config()
    [item] = em.prepare_inputs([("q", "three words here")], "query", WordTokenizer(), c)
    # <s> + "query:" + three words + </s>
    assert item.record.original_model_tokens == 6 == len(item.input_ids)
    assert item.input_ids[0] == BOS and item.input_ids[-1] == EOS


def test_an_input_that_fits_is_passed_unchanged():
    c, tok = config(max_input_tokens=8), WordTokenizer()
    text = words(5)
    [item] = em.prepare_inputs([("d", text)], "document", tok, c)
    assert item.input_ids == tuple(tok.encode(em.model_input(c, "document", text)).ids)
    assert item.record == em.TokenRecord(
        item_id="d",
        original_model_tokens=8,
        encoded_model_tokens=8,
        truncated=False,
        dropped_tokens=0,
        kept_text_chars=len(text),
    )


def test_the_exact_limit_is_not_a_truncation_and_one_more_token_is():
    c, tok = config(max_input_tokens=8), WordTokenizer()
    # <s> + "passage:" + n words + </s> = n + 3 tokens
    at_limit, over = em.prepare_inputs([("a", words(5)), ("b", words(6))], "document", tok, c)
    assert (at_limit.record.original_model_tokens, at_limit.record.truncated) == (8, False)
    assert over.record.original_model_tokens == 9
    assert over.record.truncated and over.record.dropped_tokens == 1
    assert len(over.input_ids) == over.record.encoded_model_tokens == 8


def test_a_truncated_document_keeps_its_special_tokens_and_records_what_it_lost():
    c, tok = config(max_input_tokens=8), WordTokenizer()
    text = words(10)  # 13 tokens with the prefix and the special tokens
    [item] = em.prepare_inputs([("d", text)], "document", tok, c)
    full = tok.encode(em.model_input(c, "document", text)).ids
    assert item.input_ids == (*full[:7], EOS)
    r = item.record
    assert (r.original_model_tokens, r.encoded_model_tokens, r.dropped_tokens) == (13, 8, 5)
    # the model sees "w0 ... w4": the text up to the end of the fifth word
    assert text[: r.kept_text_chars] == words(5)


def test_a_query_over_the_limit_is_refused_never_cut():
    with pytest.raises(em.TruncationRefusedError, match="never truncated"):
        em.prepare_inputs([("q", words(10))], "query", WordTokenizer(), config(max_input_tokens=8))


def test_no_truncation_can_happen_without_its_record():
    c = config(max_input_tokens=8)
    [item] = em.prepare_inputs([("d", words(10))], "document", WordTokenizer(), c)
    em.check_inputs([item], c)
    tampered = em.ModelInput("d", item.input_ids[:-1], item.record)
    with pytest.raises(em.EmbeddingError, match="unrecorded change"):
        em.check_inputs([tampered], c)
    unrecorded = em.ModelInput(
        "d", tuple(range(9)), item.record.model_copy(update={"encoded_model_tokens": 9})
    )
    with pytest.raises(em.EmbeddingError, match="over the limit"):
        em.check_inputs([unrecorded], c)


def test_a_record_must_add_up():
    with pytest.raises(ValueError, match="dropped is not original minus encoded"):
        em.TokenRecord(
            item_id="d",
            original_model_tokens=10,
            encoded_model_tokens=8,
            truncated=True,
            dropped_tokens=1,
            kept_text_chars=3,
        )
    with pytest.raises(ValueError, match="truncated must say"):
        em.TokenRecord(
            item_id="d",
            original_model_tokens=8,
            encoded_model_tokens=8,
            truncated=True,
            dropped_tokens=0,
            kept_text_chars=3,
        )


# ---------------------------------------------------------------------------
# vectors and exact search
# ---------------------------------------------------------------------------


def unit(rows: list[list[float]]) -> np.ndarray:
    m = np.array(rows, dtype=np.float32)
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)


def test_vectors_must_have_the_shape_type_values_and_norm_of_the_contract():
    c = config(dimension=2)
    good = unit([[1, 0], [1, 1]])
    em.check_vectors(good, 2, c)
    cases = [
        (good[:1], "shape"),
        (good.astype(np.float64), "float32"),
        (np.array([[np.nan, 0], [1, 0]], dtype=np.float32), "NaN"),
        (np.array([[np.inf, 0], [1, 0]], dtype=np.float32), "NaN or infinite"),
        (np.array([[2, 0], [1, 0]], dtype=np.float32), "not normalized"),
    ]
    for vectors, message in cases:
        with pytest.raises(em.EmbeddingError, match=message):
            em.check_vectors(vectors, 2, c)


def test_exact_search_ranks_by_dot_product_which_is_the_cosine_on_unit_vectors():
    docs = unit([[1, 0], [0, 1], [1, 1]])
    query = unit([[1, 0.2]])
    [ranked] = em.rank_exact(["q"], query, ["c", "a", "b"], docs)
    cosine = [float(np.dot(query[0], d) / np.linalg.norm(d)) for d in docs]
    assert ranked.chunk_ids == ("c", "b", "a")
    assert np.allclose(ranked.scores, sorted(cosine, reverse=True), atol=1e-6)


def test_ties_are_broken_by_chunk_id_whatever_the_storage_order():
    docs = unit([[1, 0], [1, 0], [0, 1], [1, 0]])
    ids = ["X:3", "X:1", "X:9", "X:2"]
    [a] = em.rank_exact(["q"], unit([[1, 0]]), ids, docs)
    order = [3, 0, 2, 1]
    [b] = em.rank_exact(["q"], unit([[1, 0]]), [ids[i] for i in order], docs[order])
    assert a.chunk_ids == b.chunk_ids == ("X:1", "X:2", "X:3", "X:9")


def test_the_same_query_ranks_the_same_twice():
    rng = np.random.default_rng(0)
    docs = unit(rng.normal(size=(50, 8)).tolist())
    queries = unit(rng.normal(size=(3, 8)).tolist())
    ids = [f"X:{i}" for i in range(50)]
    assert em.rank_exact(["a", "b", "c"], queries, ids, docs) == em.rank_exact(
        ["a", "b", "c"], queries, ids, docs
    )


# ---------------------------------------------------------------------------
# into the evaluator
# ---------------------------------------------------------------------------


def _layer() -> ev.ChunkLayer:
    refs = [ev.ChunkRef(f"X:{i}", (("X", str(i)),), 100) for i in range(1, 121)]
    return ev.layer_from_refs(refs, max_tokens=500, parser_version=h.PARSER_VERSION, sources={})


def test_an_exact_run_meets_the_run_contract_and_uses_the_existing_projection():
    layer = _layer()
    ids = sorted(layer.chunks)
    rng = np.random.default_rng(1)
    docs = unit(rng.normal(size=(len(ids), 4)).tolist())
    ranked = em.rank_exact(["q"], docs[[ids.index("X:7")]], ids, docs)
    run = em.to_run(ranked)
    ev.check_run(run, layer, ["q"])
    assert run.depth == 120 and run.rankings["q"][0] == "X:7"
    grades = ev.project({("X", "7"): 3, ("X", "8"): 1}, layer)
    assert grades == {"X:7": 3, "X:8": 1}
    recall = ev.recall_at_tokens(run.rankings["q"], layer, frozenset({("X", "7")}), 1000, run.depth)
    assert recall.value == 1.0


@pytest.mark.evaluator
def test_metrics_of_an_exact_run_come_from_the_existing_adapter():
    from ingest import ir_metrics

    layer = _layer()
    ids = sorted(layer.chunks)
    docs = unit(np.eye(len(ids), 120, dtype=np.float32).tolist())
    query = docs[[ids.index("X:7")]] * 0.9 + docs[[ids.index("X:8")]] * 0.1
    query = unit(query.tolist())
    [ranked] = em.rank_exact(["q"], query, ids, docs)
    assert ranked.chunk_ids[:2] == ("X:7", "X:8")
    scores = ir_metrics.standard_scores({"q": {"X:7": 3, "X:8": 0}}, {"q": list(ranked.chunk_ids)})
    assert scores["q"]["Recall@10"] == 1.0
    assert scores["q"]["MRR@10"] == 1.0
    assert scores["q"]["nDCG@10"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# stale embeddings
# ---------------------------------------------------------------------------


def _chunked(text: str = "Competent authorities shall supervise.") -> ch.ChunkedDocument:
    article = h.Article(
        article_id="art_7",
        path="7",
        title="Title",
        text=text,
        max_depth=1,
        nodes=[h.Node(path="7", level=1, marker="7", kind="articulo", own_text=text)],
    )
    return ch.ChunkedDocument(
        act="DORA",
        source_celex="32022R2554",
        source_role="original",
        source_sha256="1" * 64,
        parser_version=h.PARSER_VERSION,
        chunker_version=ch.CHUNKER_VERSION,
        tokenizer="cl100k_base",
        max_tokens=500,
        chunks=ch.chunk_article(article, "DORA", 500, lambda t: len(t.split())),
    )


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


def _identity(c: em.EmbeddingConfig, source, runtime=RUNTIME, tokenizer_sha="a" * 64):
    return em.artifact_identity(
        c, em.LoadedTokenizer(WordTokenizer(), tokenizer_sha), runtime, source
    )


def _written(tmp_path: Path) -> tuple[em.ArtifactIdentity, Path]:
    c = config(dimension=2)
    doc = _chunked()
    inputs = em.prepare_inputs(
        [(x.chunk_id, x.text) for x in doc.chunks], "document", WordTokenizer(), c
    )
    identity = _identity(c, em.layer_source([doc]))
    directory = tmp_path / "set"
    em.write_embedding_set(directory, identity, inputs, unit([[1, 0]]), {"cpu": "M1"})
    return identity, directory


def test_a_set_reads_back_with_the_same_identity(tmp_path):
    identity, directory = _written(tmp_path)
    meta, vectors = em.load_embedding_set(directory, identity)
    assert meta.item_ids == ["DORA:7"] and vectors.shape == (1, 2)
    assert meta.token_records[0].item_id == "DORA:7"
    assert meta.identity.runtime == RUNTIME


def _contract(identity: em.ArtifactIdentity, **changes) -> em.ArtifactIdentity:
    c = identity.config.model_copy(update=changes)
    return identity.model_copy(update={"config": c, "config_hash": c.config_hash()})


def _runtime(identity: em.ArtifactIdentity, **changes) -> em.ArtifactIdentity:
    return identity.model_copy(update={"runtime": identity.runtime.model_copy(update=changes)})


@pytest.mark.parametrize(
    "change,named",
    [
        (lambda i: _contract(i, revision="f" * 40), "revision"),
        (lambda i: _contract(i, tokenizer_revision="e" * 40), "tokenizer_revision"),
        (lambda i: _contract(i, pooling="cls"), "pooling"),
        (lambda i: _contract(i, normalize=False), "normalize"),
        (lambda i: _contract(i, query_prefix="search_query: "), "query_prefix"),
        (lambda i: _contract(i, document_prefix=""), "document_prefix"),
        (lambda i: _contract(i, max_input_tokens=8192), "max_input_tokens"),
        (lambda i: _contract(i, batch_size=1), "batch_size"),
        (lambda i: _runtime(i, transformers="5.16.0"), "transformers"),
        (lambda i: _runtime(i, torch="2.15.0"), "torch"),
        (lambda i: _runtime(i, tokenizers="0.23.0"), "tokenizers"),
        (lambda i: _runtime(i, numpy="2.6.0"), "numpy"),
        (lambda i: _runtime(i, python="3.13.1"), "python"),
        (lambda i: _runtime(i, dtype="float16"), "dtype"),
        (lambda i: _runtime(i, device="mps"), "device"),
        (lambda i: _runtime(i, sentence_transformers="5.0.0"), "sentence_transformers"),
        (lambda i: i.model_copy(update={"tokenizer_sha256": "b" * 64}), "tokenizer_sha256"),
    ],
)
def test_any_change_of_model_contract_or_runtime_makes_the_set_stale(tmp_path, change, named):
    identity, directory = _written(tmp_path)
    with pytest.raises(em.StaleEmbeddingsError, match=named):
        em.load_embedding_set(directory, change(identity))


def test_a_change_of_the_embedded_data_makes_the_set_stale(tmp_path):
    identity, directory = _written(tmp_path)
    assert identity.layer is not None
    text = em.layer_source([_chunked("Competent authorities shall not supervise.")])
    with pytest.raises(em.StaleEmbeddingsError, match="content_sha256"):
        em.load_embedding_set(directory, identity.model_copy(update={"layer": text}))
    parser = identity.layer.model_copy(update={"parser_version": "9.9.9"})
    with pytest.raises(em.StaleEmbeddingsError, match="parser_version"):
        em.load_embedding_set(directory, identity.model_copy(update={"layer": parser}))
    structure = identity.layer.model_copy(update={"structure_sha256": "c" * 64})
    with pytest.raises(em.StaleEmbeddingsError, match="structure_sha256"):
        em.load_embedding_set(directory, identity.model_copy(update={"layer": structure}))


def test_time_cache_path_and_hardware_never_make_a_set_stale(tmp_path, monkeypatch):
    identity, directory = _written(tmp_path)
    meta_path = directory / em.META_FILE
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw["created_at"] = "1999-01-01T00:00:00+00:00"
    raw["environment"] = {"cpu": "another machine", "hf_home": "/elsewhere", "torch_threads": "64"}
    meta_path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "another-cache"))
    meta, _ = em.load_embedding_set(directory, identity)
    assert meta.environment["cpu"] == "another machine"


def test_the_recorded_cache_location_is_never_an_absolute_path(tmp_path, monkeypatch):
    """Reports are versioned: they must not carry the local user or folder names."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / ".cache" / "huggingface"))
    assert em.environment_description()["hf_home"] == ".cache/huggingface"
    monkeypatch.setenv("HF_HOME", str(tmp_path.parent / "shared-cache"))
    assert em.cache_location() == "outside the working directory (HF_HOME)"


def test_a_set_written_by_an_older_meta_schema_is_stale(tmp_path):
    identity, directory = _written(tmp_path)
    meta_path = directory / em.META_FILE
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw["schema_version"] = "1"
    meta_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(em.StaleEmbeddingsError, match="meta schema 1"):
        em.load_embedding_set(directory, identity)


def test_vectors_that_do_not_match_their_record_are_refused(tmp_path):
    identity, directory = _written(tmp_path)
    np.save(directory / em.VECTORS_FILE, unit([[0, 1]]), allow_pickle=False)
    with pytest.raises(em.EmbeddingError, match="not the matrix"):
        em.load_embedding_set(directory, identity)


def test_the_runtime_identity_reads_the_installed_versions(monkeypatch):
    import importlib.metadata as md

    versions = {
        "torch": "2.14.1",
        "transformers": "5.15.1",
        "tokenizers": "0.22.2",
        "numpy": "2.5.4",
    }
    monkeypatch.setattr(md, "version", lambda name: versions[name])
    r = em.runtime_identity(config())
    assert (r.torch, r.transformers, r.tokenizers, r.numpy) == (
        "2.14.1",
        "5.15.1",
        "0.22.2",
        "2.5.4",
    )
    assert (r.dtype, r.device, r.sentence_transformers) == ("float32", "cpu", None)


def test_without_torch_the_runtime_cannot_be_identified(monkeypatch):
    import importlib.metadata as md

    def version(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", version)
    with pytest.raises(em.ModelUnavailableError, match="requirements-embed"):
        em.runtime_identity(config())


def test_a_missing_set_is_not_found_rather_than_stale(tmp_path):
    with pytest.raises(em.EmbeddingsNotFoundError):
        em.load_embedding_set(tmp_path / "nothing", _written(tmp_path)[0])


def test_query_sets_record_the_benchmark_they_came_from():
    fixtures = bm.load_benchmark(bm.FIXTURES_DIR)
    source = em.query_source(fixtures)
    assert source.benchmark_version == fixtures.manifest.benchmark_version
    assert source.queries_sha256 == fixtures.manifest.files["queries.jsonl"]
