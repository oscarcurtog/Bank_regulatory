"""Layer 0 of S7-S8 and the three abstention rules, on synthetic chunks."""

from __future__ import annotations

import json

from ingest import answer_format as af
from ingest import answer_verifier as av

TEXTS = {
    "K:1": "Financial entities shall notify the supervisory authority within 72 hours of becoming "
    "aware of a major incident.",
    "K:2": "The minimum capital shall be EUR 20 000 at the time of authorisation, and 0,5 % of the "
    "payment volume afterwards.",
    "K:3": "A  third-party provider\nmust be listed in the register of information.",
    "K:9": "Unrelated chunk about something else entirely and nothing more to say.",
}
CONTEXT = ["K:1", "K:2", "K:3"]


def _claim(text: str, *citations: tuple[str, str]) -> af.Claim:
    return af.Claim(text=text, citations=[af.Citation(chunk_id=c, quote=q) for c, q in citations])


def _verdict(claim: af.Claim, context: list[str] | None = None) -> av.ClaimVerdict:
    return av.verify_claim(claim, frozenset(context or CONTEXT), TEXTS)


GOOD_QUOTE = "notify the supervisory authority within 72 hours of becoming aware"


def test_normalisation_collapses_whitespace_and_keeps_everything_else() -> None:
    assert av.normalize("A  b c\n d") == "A b c d"
    assert av.normalize("Case Matters.") == "Case Matters."


def test_figures_are_read_in_a_canonical_form() -> None:
    assert av.figures("within 72 hours") == {"72"}
    assert av.figures("EUR 20 000 and EUR 20 000 and 20000") == {"20000"}
    assert av.figures("0,5 % and 0.5 and 12,75") == {"0.5", "12.75"}
    assert av.figures("Article 30(2)") == {"30", "2"}
    assert av.figures("no numbers here, only seventy-two") == set()


def test_a_claim_with_a_literal_quote_and_matching_figures_passes() -> None:
    verdict = _verdict(_claim("Entities must notify within 72 hours.", ("K:1", GOOD_QUOTE)))
    assert verdict.passed and verdict.all_failures() == ()


def test_the_quote_is_compared_after_whitespace_normalisation_only() -> None:
    spaced = _claim(
        "Providers are listed.", ("K:3", "third-party provider must be listed in the register")
    )
    assert _verdict(spaced).passed
    changed = _claim(
        "Entities must notify.",
        ("K:1", "notify the supervisory authority within 72 hours of knowing"),
    )
    assert av.QUOTE_NOT_LITERAL in _verdict(changed).all_failures()
    cased = _claim(
        "Entities must notify.",
        ("K:1", "Notify the supervisory authority within 72 hours of becoming"),
    )
    assert av.QUOTE_NOT_LITERAL in _verdict(cased).all_failures()


def test_every_failure_code_is_reachable() -> None:
    assert _verdict(_claim("No source.")).failures == (av.NO_CITATION,)
    unknown = _verdict(_claim("Invented.", ("K:404", "some quote with five words")))
    assert (
        av.UNKNOWN_CHUNK in unknown.all_failures()
        and av.CHUNK_NOT_IN_CONTEXT in unknown.all_failures()
    )
    outside = _verdict(
        _claim("Real chunk, not shown.", ("K:9", "Unrelated chunk about something else entirely"))
    )
    assert outside.all_failures() == (av.CHUNK_NOT_IN_CONTEXT,)
    short = _verdict(_claim("Short.", ("K:1", "within 72 hours")))
    assert av.QUOTE_TOO_SHORT in short.all_failures()
    long_quote = " ".join(["word"] * (av.MAX_QUOTE_WORDS + 1))
    assert av.QUOTE_TOO_LONG in _verdict(_claim("Long.", ("K:1", long_quote))).all_failures()
    invented = _verdict(_claim("Notify within 48 hours.", ("K:1", GOOD_QUOTE)))
    assert invented.failures == (av.FIGURE_NOT_IN_CITED_CHUNKS,)


def test_figures_may_come_from_any_cited_chunk_but_not_from_an_uncited_one() -> None:
    two = _claim(
        "Notify within 72 hours; capital is EUR 20 000.",
        ("K:1", GOOD_QUOTE),
        ("K:2", "The minimum capital shall be EUR 20 000 at the time of authorisation"),
    )
    assert _verdict(two).passed
    one = _claim("Notify within 72 hours; capital is EUR 20 000.", ("K:1", GOOD_QUOTE))
    assert _verdict(one).failures == (av.FIGURE_NOT_IN_CITED_CHUNKS,)


def test_a_claim_fails_if_any_of_its_citations_fails() -> None:
    claim = _claim(
        "Entities must notify.", ("K:1", GOOD_QUOTE), ("K:3", "this was never written there at all")
    )
    verdict = _verdict(claim)
    assert not verdict.passed
    assert [c.passed for c in verdict.citations] == [True, False]


def test_lexical_overlap_is_a_diagnostic_between_zero_and_one() -> None:
    same = av.lexical_overlap("Notify the supervisory authority", [GOOD_QUOTE])
    other = av.lexical_overlap("Banks must hold diversified collateral portfolios", [GOOD_QUOTE])
    assert same == 1.0 and other == 0.0
    assert av.lexical_overlap("72 and the", [GOOD_QUOTE]) == 1.0  # nothing left to compare


def _parsed(sufficiency: bool, claims: list[af.Claim]) -> af.ParsedOutput:
    raw = json.dumps(
        {
            "sufficiency": sufficiency,
            "claims": [c.model_dump() for c in claims],
            "missing": "" if sufficiency else "something is missing",
        }
    )
    return af.parse_output(raw)


def _decisions(sufficiency: bool, claims: list[af.Claim]) -> av.Decisions:
    parsed = _parsed(sufficiency, claims)
    assert parsed.answer is not None
    return av.decide(parsed, av.verify_answer(parsed.answer, CONTEXT, TEXTS))


def test_the_three_rules_on_every_combination() -> None:
    good = _claim("Entities must notify within 72 hours.", ("K:1", GOOD_QUOTE))
    bad = _claim("Entities must notify within 48 hours.", ("K:1", GOOD_QUOTE))
    # sufficient, all claims pass: everyone answers
    assert _decisions(True, [good]) == av.Decisions(True, True, True)
    # sufficient, one claim fails: A answers; B and C abstain, even with a good claim beside it
    assert _decisions(True, [good, bad]) == av.Decisions(True, False, False)
    # the model says it is not sufficient but gave claims (a soft violation): only B answers
    assert _decisions(False, [good]) == av.Decisions(False, True, False)
    # sufficient with no claims: A answers an empty answer; B and C abstain
    assert _decisions(True, []) == av.Decisions(True, False, False)
    # a clean abstention
    assert _decisions(False, []) == av.Decisions(False, False, False)


def test_an_unusable_output_makes_every_rule_abstain() -> None:
    parsed = af.parse_output("not json at all")
    assert av.decide(parsed, None) == av.Decisions(False, False, False)


def test_a_quote_that_differs_only_in_case_or_glyphs_is_a_failure_flagged_as_near_literal() -> None:
    texts = dict(TEXTS)
    texts["K:5"] = "The payee\u2019s provider shall cooperate in those efforts by communicating."
    shown = frozenset([*CONTEXT, "K:5"])
    lowered = av.verify_claim(
        _claim(
            "The provider cooperates.",
            ("K:5", "the payee\u2019s provider shall cooperate in those"),
        ),
        shown,
        texts,
    )
    assert av.QUOTE_NOT_LITERAL in lowered.all_failures() and lowered.citations[0].near_literal
    plain = av.verify_claim(
        _claim(
            "The provider cooperates.", ("K:5", "The payee's provider shall cooperate in those")
        ),
        shown,
        texts,
    )
    assert av.QUOTE_NOT_LITERAL in plain.all_failures() and plain.citations[0].near_literal
    invented = av.verify_claim(
        _claim("The provider cooperates.", ("K:5", "The payee must always pay everything at once")),
        shown,
        texts,
    )
    assert not invented.citations[0].near_literal
    assert (
        not _verdict(_claim("Entities must notify within 72 hours.", ("K:1", GOOD_QUOTE)))
        .citations[0]
        .near_literal
    )
