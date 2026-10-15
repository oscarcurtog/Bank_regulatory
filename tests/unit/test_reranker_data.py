"""F's synthetic training data (ingest/reranker_data.py): the filters, the equivalents, the
negatives, the split by article and the ranking measures."""

from __future__ import annotations

import random

from ingest import reranker_data as rd

CONTEXT = "Article 19 Reporting of major ICT-related incidents. Financial entities shall report:"
BODY = (
    "(a) an initial notification, within four hours from the moment of classification of the "
    "incident as major, and no later than 24 hours from the moment of becoming aware of it"
)


def test_the_prompt_asks_for_two_questions_and_carries_the_passage():
    messages = rd.generation_prompt("some passage")
    assert "PASSAGE:\nsome passage" in messages[1]["content"]
    assert rd.TASK in messages[1]["content"] and rd.QUESTIONS == 2
    assert rd.questions("REQUIREMENTS:\n- One?\n- Two?\n- Three?") == ("One?", "Two?")


def test_a_good_question_passes_and_each_filter_names_its_reason():
    good = "Within how many hours must a financial entity send the initial incident notification?"
    assert rd.rejection(good, CONTEXT, BODY) is None
    assert rd.rejection("The time limit for the notification.", CONTEXT, BODY) == "not a question"
    leak = "What does Article 19 require for the initial notification?"
    assert rd.rejection(leak, CONTEXT, BODY) == "an article, paragraph, point or chunk reference"
    copied = "Is it within four hours from the moment of classification of the incident as major?"
    assert rd.rejection(copied, CONTEXT, BODY) == "a run of 8 or more words copied"
    inherited = "Which reporting of major ICT-related incidents applies to financial entities?"
    assert rd.rejection(inherited, CONTEXT, BODY) == "leans on the inherited context"


def test_the_longest_copied_run_counts_consecutive_words():
    assert rd.longest_copied_run("one two three four", "x one two three y") == 3
    assert rd.longest_copied_run("nothing shared here", "other words") == 0


def test_equivalents_are_chunks_sharing_half_of_their_shingles():
    base = "the competent authority shall publish the decision without undue delay on its website"
    bodies = {
        "a": base,
        "b": base + " and notify ESMA",  # most shingles shared
        "c": "an entirely different provision about the reserve of assets and its custody rules",
    }
    eq = rd.equivalents(bodies)
    assert eq["a"] == {"b"} and eq["b"] == {"a"} and eq["c"] == frozenset()


def _item() -> rd.Item:
    return rd.Item(
        "p#0", "q?", "p", "DORA:art_19", ("x", "p", "e", "v", "y", "z"), frozenset({"p", "e"})
    )


def test_train_negatives_skip_correct_and_validation_chunks_and_draw_one_from_the_rest():
    negatives = rd.train_negatives(_item(), frozenset({"v"}), random.Random(1))
    assert negatives[:2] == ["x", "y"] and negatives[2] == "z"
    again = rd.train_negatives(_item(), frozenset({"v"}), random.Random(1))
    assert negatives == again  # the draw is seeded


def test_select_candidates_put_the_positive_first_with_the_hardest_wrong_ones():
    assert rd.select_candidates(_item()) == ["p", "x", "v", "y", "z"]


def test_the_split_is_by_article_and_deterministic():
    keys = [f"MiCA:art_{i}" for i in range(400)]
    share = sum(rd.in_validation(k) for k in keys) / len(keys)
    assert 0.08 < share < 0.22
    assert [rd.in_validation(k) for k in keys] == [rd.in_validation(k) for k in keys]
    assert rd.article_key("MiCA:3.1.6", "art_3") == "MiCA:art_3"


def test_ranking_metrics_use_the_first_correct_chunk():
    ranked = {"a": ["x", "p"], "b": ["p", "x"], "c": ["x", "y"]}
    correct = {"a": frozenset({"p"}), "b": frozenset({"p"}), "c": frozenset({"p"})}
    m = rd.ranking_metrics(ranked, correct)
    assert m["P@1"] == 1 / 3 and abs(m["MRR"] - (0.5 + 1) / 3) < 1e-12
    assert m["Recall@3"] == 2 / 3
