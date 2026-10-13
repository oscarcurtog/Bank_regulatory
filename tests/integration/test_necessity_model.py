"""The real Qwen/Qwen3-Reranker-0.6B at its pinned revision, against its own model card: the
prompt the card builds, the probability of "yes" its code computes, and the same scores on the
contract's device as on the CPU the card's code runs on.

Marked "embedding_model", like the tests of the other models: they need torch, transformers
and the weights, and are skipped only where the environment says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ingest import embeddings as em
from ingest import necessity as nc

pytestmark = pytest.mark.embedding_model

CONFIG = nc.load_config(
    Path(__file__).parents[2] / "config" / "retrieval" / "qwen3-reranker-0.6b.json"
)
# The model card at the pinned revision: its prompt and its example.
CARD_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
CARD_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CARD_TASK = "Given a web search query, retrieve relevant passages that answer the query"
CARD_QUERIES = ["What is the capital of China?", "Explain gravity"]
CARD_DOCUMENTS = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to "
    "physical objects and is responsible for the movement of planets around the sun.",
]


@pytest.fixture(scope="module")
def cpu():
    return nc.NecessityReranker(CONFIG.model_copy(update={"device": "cpu"}))


def _card_body(query: str, document: str) -> str:
    return f"<Instruct>: {CARD_TASK}\n<Query>: {query}\n<Document>: {document}"


def test_the_contract_holds_the_cards_prompt():
    assert CONFIG.prefix == CARD_PREFIX and CONFIG.suffix == CARD_SUFFIX
    assert CONFIG.body.format(instruction="I", query="Q", document="D") == (
        "<Instruct>: I\n<Query>: Q\n<Document>: D"
    )


def test_a_prompt_is_the_cards_token_sequence(cpu):
    tok = cpu.tokenizer
    for q, d in zip(CARD_QUERIES, CARD_DOCUMENTS, strict=True):
        card = (
            tok.encode(CARD_PREFIX, add_special_tokens=False)
            + tok([_card_body(q, d)])["input_ids"][0]
            + tok.encode(CARD_SUFFIX, add_special_tokens=False)
        )
        prompt = cpu.prompt(CARD_TASK, q, d)
        assert list(prompt.ids) == card and not prompt.truncated


def test_scores_equal_the_model_cards_code(cpu):
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        CONFIG.model_id,
        revision=CONFIG.revision,
        cache_dir=em.hf_home() / "hub",
        padding_side="left",
    )
    prefix = tokenizer.encode(CARD_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(CARD_SUFFIX, add_special_tokens=False)
    pairs = [_card_body(q, d) for q, d in zip(CARD_QUERIES, CARD_DOCUMENTS, strict=True)]
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=8192 - len(prefix) - len(suffix),
    )
    for i, ids in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix + ids + suffix
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=8192)
    yes, no = tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no")
    with torch.no_grad():
        last = cpu.model(**inputs).logits[:, -1, :]
    card = torch.nn.functional.log_softmax(torch.stack([last[:, no], last[:, yes]], 1), 1)
    reference = card[:, 1].exp().numpy()
    ours = cpu.score(
        [cpu.prompt(CARD_TASK, q, d) for q, d in zip(CARD_QUERIES, CARD_DOCUMENTS, strict=True)]
    )
    assert np.abs(np.array([nc.p_yes(x) for x in ours]) - reference).max() < 1e-5


def test_the_card_example_ranks_the_answer_first(cpu):
    s = cpu.score([cpu.prompt(CARD_TASK, CARD_QUERIES[0], d) for d in CARD_DOCUMENTS])
    assert s[0] > 0 > s[1]


def test_left_padding_changes_no_score(cpu):
    short = cpu.prompt(CARD_TASK, CARD_QUERIES[0], CARD_DOCUMENTS[0])
    long = cpu.prompt(CARD_TASK, CARD_QUERIES[1], " ".join([CARD_DOCUMENTS[1]] * 5))
    alone = cpu.score([short])[0]
    together = cpu.score([long, short])[1]
    assert abs(alone - together) < 1e-4


def test_the_same_prompts_score_the_same_twice(cpu):
    prompts = [cpu.prompt(CARD_TASK, CARD_QUERIES[0], d) for d in CARD_DOCUMENTS]
    assert np.array_equal(cpu.score(prompts), cpu.score(prompts))


def test_over_the_limit_the_document_tail_is_cut_and_recorded(cpu):
    full = cpu.prompt(CARD_TASK, CARD_QUERIES[1], CARD_DOCUMENTS[1])
    small = nc.NecessityReranker.__new__(nc.NecessityReranker)
    small.__dict__.update(cpu.__dict__)
    small.config = CONFIG.model_copy(update={"max_input_tokens": len(full.ids) - 5})
    cut = small.prompt(CARD_TASK, CARD_QUERIES[1], CARD_DOCUMENTS[1])
    assert cut.truncated and len(cut.ids) == len(full.ids) - 5
    assert cut.original_tokens == len(full.ids)
    suffix = len(cpu.tokenizer.encode(CARD_SUFFIX, add_special_tokens=False))
    assert cut.ids[:-suffix] == full.ids[: len(cut.ids) - suffix]
    assert cut.ids[-suffix:] == full.ids[-suffix:]
    tiny = nc.NecessityReranker.__new__(nc.NecessityReranker)
    tiny.__dict__.update(cpu.__dict__)
    tiny.config = CONFIG.model_copy(update={"max_input_tokens": 70})
    with pytest.raises(em.TruncationRefusedError):
        tiny.prompt(CARD_TASK, " ".join(["question"] * 100), CARD_DOCUMENTS[0])


def test_the_contracts_device_scores_as_the_cpu_does(cpu):
    device = nc.NecessityReranker(CONFIG)
    prompts = [cpu.prompt(CARD_TASK, q, d) for q in CARD_QUERIES for d in CARD_DOCUMENTS]
    assert np.abs(device.score(prompts) - cpu.score(prompts)).max() < 1e-3
    group = _group(cpu)
    assert np.abs(device.score_shared(group) - cpu.score(group)).max() < 1e-3


def _group(reranker):
    """One query and one anchor, several candidates: the prompts open with the same tokens."""
    query = "QUESTION: Which provision applies?\nANCHOR: DORA Article 6: framework text."
    documents = [*CARD_DOCUMENTS, "A third candidate, longer than the first by several words."]
    return [reranker.prompt(nc.INSTRUCTIONS["A2"], query, f"CANDIDATE: {d}") for d in documents]


def test_shared_prefix_scoring_gives_the_plain_scores(cpu):
    prompts = _group(cpu)
    assert np.abs(cpu.score_shared(prompts) - cpu.score(prompts)).max() < 1e-4
    assert np.array_equal(cpu.score_shared(prompts[:1]), cpu.score(prompts[:1]))
