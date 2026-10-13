"""The real BGE-M3, dense output only, at its pinned revision, against its own model card
and against the reference implementation of its dense vector (FlagEmbedding v1.4.2:
the hidden state of the first token, L2-normalized, no instruction).

Marked "embedding_model", like the tests of multilingual-e5-large: they need torch,
transformers and the weights, and are skipped only where the environment says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ingest import embeddings as em

pytestmark = pytest.mark.embedding_model

CONFIG = em.load_config(Path(__file__).parents[2] / "config" / "embeddings" / "bge-m3.json")
# The example of the model card at the pinned revision, and the scores it prints.
CARD_QUERIES = ["What is BGE M3?", "Defination of BM25"]
CARD_PASSAGES = [
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and "
    "multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the "
    "query terms appearing in each document",
]
CARD_SCORES = [[0.6265, 0.3477], [0.3499, 0.678]]


@pytest.fixture(scope="module")
def tokenizer():
    return em.load_tokenizer(CONFIG).tokenizer


@pytest.fixture(scope="module")
def official_tokenizer():
    import transformers

    return transformers.AutoTokenizer.from_pretrained(
        CONFIG.model_id, revision=CONFIG.revision, cache_dir=em.hf_home() / "hub"
    )


@pytest.fixture(scope="module")
def encoder():
    from ingest.encoder import Encoder

    return Encoder(CONFIG)


def test_the_counting_tokenizer_is_the_models_own_and_adds_no_prefix(tokenizer, official_tokenizer):
    for text in CARD_QUERIES + CARD_PASSAGES:
        for kind in ("query", "document"):
            [item] = em.prepare_inputs([("x", text)], kind, tokenizer, CONFIG)
            assert list(item.input_ids) == official_tokenizer(text)["input_ids"]


def test_dense_vectors_are_the_normalized_first_token_as_the_reference_computes_them(
    encoder, tokenizer, official_tokenizer
):
    import torch

    texts = CARD_QUERIES + CARD_PASSAGES
    ours = encoder.encode(
        em.prepare_inputs([(str(i), t) for i, t in enumerate(texts)], "document", tokenizer, CONFIG)
    )
    assert ours.shape == (4, 1024) and ours.dtype == np.float32
    batch = official_tokenizer(texts, padding=True, return_tensors="pt")
    with torch.inference_mode():
        hidden = encoder.model(**batch).last_hidden_state
    reference = torch.nn.functional.normalize(hidden[:, 0], p=2, dim=-1).numpy()
    assert np.abs(ours - reference).max() < 1e-5
    assert np.allclose(np.linalg.norm(ours, axis=1), 1.0, atol=1e-5)


def test_the_model_card_scores_are_reproduced(encoder, tokenizer):
    """The card printed its scores with use_fp16=True; this runs float32, so a small
    tolerance is expected, and nothing larger."""
    q = encoder.encode(
        em.prepare_inputs(
            [(str(i), t) for i, t in enumerate(CARD_QUERIES)], "query", tokenizer, CONFIG
        )
    )
    p = encoder.encode(
        em.prepare_inputs(
            [(str(i), t) for i, t in enumerate(CARD_PASSAGES)], "document", tokenizer, CONFIG
        )
    )
    assert np.abs(q @ p.T - np.array(CARD_SCORES)).max() < 0.005


def test_the_same_input_gives_the_same_vector(encoder, tokenizer):
    inputs = em.prepare_inputs([("q", CARD_QUERIES[0])], "query", tokenizer, CONFIG)
    assert np.array_equal(encoder.encode(inputs), encoder.encode(inputs))


def test_loading_never_fetches_a_revision_that_was_not_pinned(monkeypatch):
    """The pinned revision has pytorch_model.bin and no safetensors. Plain transformers
    loads it and starts a background download of the safetensors of the Hub's conversion
    pull request; the Encoder turns that off."""
    import transformers.modeling_utils as mu

    from ingest.encoder import Encoder

    calls: list[object] = []
    monkeypatch.setattr(mu, "auto_conversion", lambda *a, **k: calls.append(a))
    Encoder(CONFIG)
    assert calls == []
