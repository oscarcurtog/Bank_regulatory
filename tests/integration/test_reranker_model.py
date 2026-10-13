"""The real BAAI/bge-reranker-v2-m3 at its pinned revision, against its own model card.

Marked "embedding_model": it needs torch, transformers and the weights, and is skipped only
where the environment says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ingest import embeddings as em
from ingest import reranker as rk

pytestmark = pytest.mark.embedding_model

CONFIG = rk.load_config(
    Path(__file__).parents[2] / "config" / "retrieval" / "bge-reranker-v2-m3.json"
)
CARD_QUERY = "what is panda?"
CARD_PASSAGES = [
    "hi",
    "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, "
    "is a bear species endemic to China.",
]


@pytest.fixture(scope="module")
def reranker():
    return rk.Reranker(CONFIG)


def test_a_pair_is_the_models_own_template(reranker):
    pair = reranker.pair(CARD_QUERY, CARD_PASSAGES[1])
    tok = reranker.tokenizer
    q = tok.encode(CARD_QUERY, add_special_tokens=False).ids
    p = tok.encode(CARD_PASSAGES[1], add_special_tokens=False).ids
    bos, eos = tok.token_to_id("<s>"), tok.token_to_id("</s>")
    assert list(pair.ids) == [bos, *q, eos, eos, *p, eos]
    assert not pair.truncated


def test_over_the_limit_the_passage_tail_is_cut_and_recorded(reranker):
    small = rk.Reranker.__new__(rk.Reranker)
    small.__dict__.update(reranker.__dict__)
    small.config = CONFIG.model_copy(update={"max_input_tokens": 20})
    pair = small.pair(CARD_QUERY, CARD_PASSAGES[1])
    full = reranker.pair(CARD_QUERY, CARD_PASSAGES[1])
    assert pair.truncated and len(pair.ids) == 20 and pair.original_tokens == len(full.ids)
    assert pair.ids[:-1] == full.ids[:19] and pair.ids[-1] == full.ids[-1]
    with pytest.raises(em.TruncationRefusedError):
        small.pair(" ".join(["panda"] * 40), "hi")


def test_the_model_card_example_ranks_the_answer_first(reranker):
    scores = reranker.score([reranker.pair(CARD_QUERY, p) for p in CARD_PASSAGES])
    assert scores[1] > scores[0]


def test_scores_equal_the_model_cards_code(reranker):
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        CONFIG.model_id, revision=CONFIG.revision, cache_dir=em.hf_home() / "hub"
    )
    pairs = [[CARD_QUERY, p] for p in CARD_PASSAGES]
    inputs = tokenizer(pairs, padding=True, return_tensors="pt")
    with torch.inference_mode():
        reference = reranker.model(**inputs, return_dict=True).logits.view(-1).float().numpy()
    ours = reranker.score([reranker.pair(q, p) for q, p in pairs])
    assert np.abs(ours - reference).max() < 1e-4


def test_the_same_pairs_score_the_same_twice(reranker):
    pairs = [reranker.pair(CARD_QUERY, p) for p in CARD_PASSAGES]
    assert np.array_equal(reranker.score(pairs), reranker.score(pairs))
