"""D's decomposition (ingest/decomposition.py): the prompt, the reading of the model's answer, and
the fixed rule that merges the requirements' rankings with E5's."""

from __future__ import annotations

from ingest import decomposition as dc

E5 = [f"c{i}" for i in range(1, 21)]  # E5's ranking: c1 first


def test_the_prompt_carries_the_question_the_excerpts_and_the_task_and_no_label_word():
    messages = dc.requirements_prompt("Can she buy?", ["first text", "second text"])
    user = messages[1]["content"]
    assert user.startswith("QUESTION: Can she buy?")
    assert "[1] first text\n\n[2] second text" in user
    assert dc.NECESSARY in user and dc.TASK in user
    for m in messages:
        for banned in ("qrel", "grade", "gold", "relevant"):
            assert banned not in m["content"].lower()


def test_requirements_are_read_after_the_heading_once_each_and_at_most_four():
    out = (
        "Here they are.\n- not this one\nREQUIREMENTS:\n- The definition of a payment account.\n"
        "* the definition of a payment  account.\n1. The time limit for the refund.\n"
        "2) The exemption for small institutions.\n- <question>\n- The fifth.\n- The sixth."
    )
    r = dc.parse(out)
    assert r.items == (
        "The definition of a payment account.",
        "The time limit for the refund.",
        "The exemption for small institutions.",
        "The fifth.",
    )
    assert r.readable and r.dropped == 3  # a repeat, the placeholder and the sixth


def test_without_the_heading_every_listed_line_counts_and_one_item_is_not_readable():
    assert dc.parse("- One thing.\n- Another thing.").items == ("One thing.", "Another thing.")
    assert not dc.parse("REQUIREMENTS:\n- Only one.").readable
    assert not dc.parse("I cannot help with that.").readable


def test_with_no_requirement_the_ranking_is_e5s():
    a = dc.assemble(E5, [])
    assert a.ranking == tuple(E5) and a.added == () and a.displaced == ()


def test_the_first_six_stay_and_corroborated_tail_chunks_go_before_new_ones():
    # c9 is corroborated by the second requirement; c15, c16, c17 are new
    hits = [["c15", "c1", "c16"], ["c9", "c17"]]
    a = dc.assemble(E5, hits)
    assert a.ranking[:10] == ("c1", "c2", "c3", "c4", "c5", "c6", "c9", "c15", "c16", "c17")
    assert a.corroborated == ("c9",)
    assert a.added == ("c15", "c17", "c16")  # in turns: first, second, first again
    assert a.displaced == ("c7", "c8", "c10")
    assert a.ranking[10:13] == ("c7", "c8", "c10")  # the rest keeps E5's order
    assert sorted(a.ranking) == sorted(E5)


def test_new_chunks_are_taken_in_turns_skipping_repeats_and_only_from_the_first_k():
    hits = [["c11", "c12", "c13"], ["c11", "c14"], ["c1", "c2", "c3", "c4", "c5", "c20"]]
    a = dc.assemble(E5, hits)
    # turn 1: c11 (first), c14 (second, past its repeat c11), nothing new in the first 5 of the
    # third (c20 is its sixth and does not count); turn 2: c12; turn 3: c13
    assert a.added == ("c11", "c14", "c12", "c13")
    assert a.ranking[:10] == ("c1", "c2", "c3", "c4", "c5", "c6", "c11", "c12", "c13", "c14")
    assert a.displaced == ("c7", "c8", "c9", "c10")


def test_requirements_that_only_find_e5s_top_six_change_nothing():
    a = dc.assemble(E5, [["c1", "c2"], ["c3", "c6"]])
    assert a.ranking == tuple(E5)


def test_named_articles_are_found_in_lists_and_ranges():
    assert dc.articles_named("as Article 59 and Articles 11 and 12 say") == {"59", "11", "12"}
    assert dc.articles_named("Articles 3, 4 or 5a apply") == {"3", "4", "5a"}
    assert dc.articles_named("no article here") == set()
