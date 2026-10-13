"""The real Qwen3-Embedding-0.6B at its pinned revision, against its own model card: the
reference implementation there pads on the left and pools the last token; the encoder pads on
the right and pools the last real token, which must give the same vector.

Marked "embedding_model", like the tests of the other models: they need torch, transformers
and the weights, and are skipped only where the environment says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ingest import embeddings as em

pytestmark = pytest.mark.embedding_model

CONFIG = em.load_config(
    Path(__file__).parents[2] / "config" / "embeddings" / "qwen3-embedding-0.6b.json"
)
# The example of the model card at the pinned revision, and the scores it prints.
CARD_QUERIES = ["What is the capital of China?", "Explain gravity"]
CARD_DOCUMENTS = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to "
    "physical objects and is responsible for the movement of planets around the sun.",
]
CARD_SCORES = [[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]]


@pytest.fixture(scope="module")
def tokenizer():
    return em.load_tokenizer(CONFIG).tokenizer


@pytest.fixture(scope="module")
def official_tokenizer():
    import transformers

    return transformers.AutoTokenizer.from_pretrained(
        CONFIG.model_id,
        revision=CONFIG.revision,
        cache_dir=em.hf_home() / "hub",
        padding_side="left",
    )


@pytest.fixture(scope="module")
def encoder():
    from ingest.encoder import Encoder

    return Encoder(CONFIG)


def _inputs(texts, kind, tokenizer):
    return em.prepare_inputs([(str(i), t) for i, t in enumerate(texts)], kind, tokenizer, CONFIG)


def test_the_counting_tokenizer_is_the_models_own_with_its_end_token(tokenizer, official_tokenizer):
    for text in CARD_QUERIES + CARD_DOCUMENTS:
        for kind in ("query", "document"):
            [item] = _inputs([text], kind, tokenizer)
            expected = official_tokenizer(em.model_input(CONFIG, kind, text))["input_ids"]
            assert list(item.input_ids) == expected
            assert item.input_ids[-1] == official_tokenizer.convert_tokens_to_ids("<|endoftext|>")


def test_vectors_equal_the_model_cards_left_padded_last_token_pool(
    encoder, tokenizer, official_tokenizer
):
    import torch

    texts = [em.model_input(CONFIG, "query", q) for q in CARD_QUERIES] + CARD_DOCUMENTS
    ours = np.vstack(
        [
            encoder.encode(_inputs(CARD_QUERIES, "query", tokenizer)),
            encoder.encode(_inputs(CARD_DOCUMENTS, "document", tokenizer)),
        ]
    )
    assert ours.shape == (4, 1024) and ours.dtype == np.float32
    batch = official_tokenizer(texts, padding=True, return_tensors="pt")
    with torch.inference_mode():
        hidden = encoder.model(**batch).last_hidden_state
    reference = torch.nn.functional.normalize(hidden[:, -1], p=2, dim=-1).numpy()
    assert np.abs(ours - reference).max() < 1e-4
    assert np.allclose(np.linalg.norm(ours, axis=1), 1.0, atol=1e-5)


def test_the_model_card_scores_are_reproduced(encoder, tokenizer):
    q = encoder.encode(_inputs(CARD_QUERIES, "query", tokenizer))
    d = encoder.encode(_inputs(CARD_DOCUMENTS, "document", tokenizer))
    assert np.abs(q @ d.T - np.array(CARD_SCORES)).max() < 0.005


def test_the_same_input_gives_the_same_vector(encoder, tokenizer):
    inputs = _inputs(CARD_QUERIES[:1], "query", tokenizer)
    assert np.array_equal(encoder.encode(inputs), encoder.encode(inputs))
