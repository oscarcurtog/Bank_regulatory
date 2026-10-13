"""Round 4's closed-catalog legal reformulation (ingest/reformulation.py): the catalog, the
prompts, the literal check of every concept against the catalog, and the text E5 ranks."""

from __future__ import annotations

from types import SimpleNamespace

from ingest import dependencies as dp
from ingest import reformulation as rf

CORPUS = {
    "MiCA": SimpleNamespace(
        articles=[
            SimpleNamespace(title="Inside information"),
            SimpleNamespace(title="Prohibition of insider dealing"),
            SimpleNamespace(title="Scope"),
        ]
    ),
    "DORA": SimpleNamespace(articles=[SimpleNamespace(title="Scope")]),
}
DEFINITIONS = [dp.Definition("DORA", "3.1", "ICT risk"), dp.Definition("PSD2", "4.1", "scope")]


def test_the_catalog_holds_every_title_then_every_term_once():
    assert rf.catalog(CORPUS, DEFINITIONS) == [
        "Scope",
        "Inside information",
        "Prohibition of insider dealing",
        "ICT risk",
    ]


def test_the_prompts_carry_the_question_and_the_catalog_opens_the_concepts_prompt():
    facts = rf.facts_prompt("Can she buy?")
    assert facts[1]["content"].startswith("QUESTION: Can she buy?")
    concepts = rf.concepts_prompt("Can she buy?", ["Inside information", "ICT risk"])
    assert concepts[0]["content"].endswith(
        "CATALOG (article titles and defined terms of DORA, "
        "MiCA and PSD2):\nInside information\nICT risk"
    )
    for m in facts + concepts:
        for banned in ("qrel", "grade", "gold"):
            assert banned not in m["content"].lower()


def test_concepts_must_be_literally_in_the_catalog_and_are_capped():
    entries = rf.catalog(CORPUS, DEFINITIONS)
    out = (
        "FACTS:\n- An employee knew of a listing.\n- She bought tokens.\nCONCEPTS:\n"
        "- prohibition of insider dealing | bought some tokens\n"
        "- market abuse | rules\n- 'Inside information' | before the announcement\n"
        "- ICT risk | nothing\n- Scope | anything"
    )
    r = rf.parse(out, "She bought some tokens before the announcement.", entries)
    assert r.facts == ("An employee knew of a listing.", "She bought tokens.")
    assert [c.entry for c in r.concepts] == [
        "Prohibition of insider dealing",
        "Inside information",
        "ICT risk",
    ]
    assert [c.fragment_in_question for c in r.concepts] == [True, True, False]
    assert r.rejected == ("market abuse", "Scope")
    assert rf.retrieval_text(r) == (
        "An employee knew of a listing. She bought tokens. Legal concepts: Prohibition of "
        "insider dealing; Inside information; ICT risk."
    )


def test_an_answer_without_facts_or_concepts_gives_no_rewrite():
    r = rf.parse("I cannot help with that.", "q", ["Scope"])
    assert r.facts == () and r.concepts == () and rf.retrieval_text(r) == ""
    assert rf.parse("FACTS:\n- one fact", "q").concepts == ()
