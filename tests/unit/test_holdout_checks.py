"""The classification of every check of TEST's construction, fixed before authoring (D-055)."""

from __future__ import annotations

import re
import typing
from pathlib import Path

from ingest import holdout_checks as hk
from ingest import holdout_ledger as hl
from ingest import holdout_pipeline as hpl


def test_every_check_has_a_kind_and_a_consequence_of_that_kind():
    outcomes = set(typing.get_args(hl.Outcome))
    reasons = set(typing.get_args(hpl.TechnicalReason))
    for name, check in hk.CHECKS.items():
        if check.kind == "methodological_eligibility":
            assert check.consequence in outcomes, name
            assert not check.retriable, name
        elif check.kind == "technical_integrity":
            if check.reads == "model_output":
                assert check.retriable and check.consequence in reasons, name
            else:
                assert not check.retriable and check.consequence == "stop", name
        else:
            assert check.kind == "diagnostic_only" and check.consequence == "recorded", name


def test_the_pipeline_names_only_classified_checks_and_decides_none_itself():
    source = Path(hpl.__file__).read_text(encoding="utf-8")
    named = set(re.findall(r'(?:technical|_ineligible|_admitted|_built)\(\s*"([a-z_]+)"', source))
    named |= set(re.findall(r'_failed\([^)]*?"([a-z_]+)"\)', source))
    named |= set(hpl.SHAPE_CHECK.values())
    assert named and named <= set(hk.CHECKS)
    # a methodological rejection comes from a role's verdict or from the table, never ad hoc
    literal = set(re.findall(r'Rejected\("([a-z_]+)"', source))
    roles = {v.outcome for v in hk.ROLE_VERDICTS}
    assert literal <= roles, literal - roles


def test_a_technical_failure_is_built_only_from_a_check_on_a_model_output():
    failure = hpl.technical("judge_answer_complete", "judge C left an item out")
    assert (failure.reason, failure.check) == ("malformed_session", "judge_answer_complete")
    try:
        hpl.technical("dataset_records", "x")
    except ValueError:
        pass
    else:
        raise AssertionError("a check on assembled records is never retried")
    rejection = hpl.Pipeline._ineligible("statement_cites_no_provision", "statement_validation")
    assert rejection.outcome == "statement_invalid"


def test_at_the_diagnostic_only_the_process_and_its_runtime_are_technical():
    """A greedy local model gives the same answer to the same prompt: what is wrong with the
    answer is diagnostic_admissible's, never retried, and so is a prompt over the frozen limit
    (P1 #6); only the role audit of the run, the runtime it ran under, its deadline and its
    preflight record are technical there (the first START, aborted on 2026-10-20)."""
    at = {n: c for n, c in hk.CHECKS.items() if "diagnostic" in c.stages}
    retried = {n for n, c in at.items() if c.retriable}
    assert retried == {
        "role_audit",
        "runtime_identity",
        "execution_deadline",
        "diagnostic_preflight",
    }
    # what the code builds before the model runs, and a process not verified gone, stop
    assert {n for n, c in at.items() if c.kind == "technical_integrity"} - retried == {
        "packet_construction",
        "process_cleanup",
    }
    assert "input_exceeds_context" in hk.check("diagnostic_admissible").rule
    admissible = hk.check("diagnostic_admissible")
    assert (admissible.kind, admissible.consequence) == ("diagnostic_only", "recorded")
    assert not admissible.retriable


def test_the_record_carries_every_check_with_its_retry_rule():
    record = hk.record()
    assert set(record["checks"]) == set(hk.CHECKS)
    assert record["checks"]["role_audit"]["retriable"] is True
    assert record["checks"]["final_qrels_unambiguous"]["kind"] == "methodological_eligibility"
    assert len(record["stop_rules"]) == 7
    assert any(
        "audit search" in r and "exposure pool" in r and ">= 1" in r for r in record["stop_rules"]
    )
    # each normal search's declared problem is the need's; the audit search's stops instead
    assert {v.role for v in hk.ROLE_VERDICTS if v.outcome == "judge_problem"} == {
        "searcher_1",
        "searcher_2",
        "searcher_3",
    }


def test_every_search_stage_has_the_searches_checks():
    for name in ("broker_protocol", "broker_ceiling", "judge_answer_complete"):
        assert set(hpl.SEARCHES) <= set(hk.check(name).stages), name
    timeout = hk.check("execution_deadline")
    assert (timeout.consequence, timeout.retriable) == ("timeout", True)
    cleanup = hk.check("process_cleanup")
    assert (cleanup.consequence, cleanup.retriable) == ("stop", False)
    unclassified = hk.check("unclassified_failure")
    assert (unclassified.consequence, unclassified.retriable) == ("stop", False)
