"""The real multilingual-e5-large at its pinned revision, against its own model card.

These tests need torch, transformers and the weights (requirements-embed.lock.txt and
one download into the Hugging Face cache). They are marked "embedding_model" and skipped
ONLY where the environment says so (FUNDAMENTO_EMBEDDING_MODEL=unsupported), as CI and
the dev image do. Anywhere else a missing framework or model fails them, with the remedy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ingest import benchmark as bm
from ingest import embeddings as em

pytestmark = pytest.mark.embedding_model

CONFIG = em.load_config(
    Path(__file__).parents[2] / "config" / "embeddings" / "multilingual-e5-large.json"
)
# A Spanish query is read from the benchmark, the one place where Spanish text lives.
SPANISH = next(q.text for q in bm.load_benchmark(bm.DEV_DIR).queries if q.query_id == "N025-es-1")
TEXTS = [
    "Does the white paper need the regulator's approval before the offer goes ahead?",
    SPANISH,
    "MiCA Article 8: Notification of the crypto-asset white paper and of the marketing "
    "communications\n\nCompetent authorities shall not require prior approval of crypto-asset "
    "white papers, nor of any marketing communications relating thereto, before their "
    "respective publication.",
]


@pytest.fixture(scope="module")
def tokenizer():
    return em.load_tokenizer(CONFIG)


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


@pytest.mark.parametrize("kind", ["query", "document"])
def test_the_counting_tokenizer_is_the_models_own(tokenizer, official_tokenizer, kind):
    for text in TEXTS:
        [item] = em.prepare_inputs([("x", text)], kind, tokenizer, CONFIG)
        expected = official_tokenizer(em.model_input(CONFIG, kind, text))["input_ids"]
        assert list(item.input_ids) == expected


def test_explicit_truncation_equals_the_model_cards_and_is_recorded(tokenizer, official_tokenizer):
    text = " ".join(["The competent authority shall notify the provider."] * 80)
    [item] = em.prepare_inputs([("long", text)], "document", tokenizer, CONFIG)
    card = official_tokenizer(
        em.model_input(CONFIG, "document", text), truncation=True, max_length=512
    )["input_ids"]
    assert list(item.input_ids) == card
    r = item.record
    assert r.truncated and r.encoded_model_tokens == 512
    assert r.original_model_tokens == len(
        official_tokenizer(em.model_input(CONFIG, "document", text))["input_ids"]
    )
    assert 0 < r.kept_text_chars < len(text)


def test_vectors_follow_the_contract_and_match_the_model_card(encoder, official_tokenizer):
    import torch

    inputs = em.prepare_inputs(
        [(str(i), t) for i, t in enumerate(TEXTS)], "document", em.load_tokenizer(CONFIG), CONFIG
    )
    ours = encoder.encode(inputs)
    assert ours.shape == (3, 1024) and ours.dtype == np.float32
    batch = official_tokenizer(
        [em.model_input(CONFIG, "document", t) for t in TEXTS],
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        hidden = encoder.model(**batch).last_hidden_state
        masked = hidden.masked_fill(~batch["attention_mask"][..., None].bool(), 0.0)
        card = masked.sum(dim=1) / batch["attention_mask"].sum(dim=1)[..., None]
        card = torch.nn.functional.normalize(card, p=2, dim=1).numpy()
    assert np.abs(ours - card).max() < 1e-5


def test_the_same_inputs_give_the_same_vectors_and_prefixes_matter(encoder, tokenizer):
    queries = em.prepare_inputs([("q", TEXTS[0])], "query", tokenizer, CONFIG)
    passages = em.prepare_inputs([("q", TEXTS[0])], "document", tokenizer, CONFIG)
    first, second = encoder.encode(queries), encoder.encode(queries)
    assert np.array_equal(first, second)
    assert not np.allclose(first, encoder.encode(passages))


def test_a_question_is_closer_to_its_answer_than_to_another_text(encoder, tokenizer):
    [q] = em.prepare_inputs([("q", TEXTS[0])], "query", tokenizer, CONFIG)
    docs = em.prepare_inputs(
        [("answer", TEXTS[2]), ("other", "The payee shall be credited on the same day.")],
        "document",
        tokenizer,
        CONFIG,
    )
    [ranked] = em.rank_exact(["q"], encoder.encode([q]), ["answer", "other"], encoder.encode(docs))
    assert ranked.chunk_ids[0] == "answer"
