"""Every check of TEST's construction, classified before authoring (S3, D-055).

What a failing check leads to is fixed here, in a table the pre-authoring contract carries; the
orchestrator looks it up and never decides it at the moment (ingest/holdout_pipeline.py):

- technical_integrity: an output, a record or the state is not what the protocol allows (a
  malformed answer, an impossible schema, a broken invariant, a failed audit). The SAME need
  stays. On a model's output it is one failed technical execution, which resume() runs again
  in a new context with the same packet, up to three executions; on records the code itself
  assembled, or on the state, no execution can change it, and the construction stops for a
  person. Never a methodological rejection.
- methodological_eligibility: a rule written before TEST says that the candidate is not a
  valid TEST need. The attempt closes with the outcome named here, and the cell gets a new
  blind attempt (at most three).
- diagnostic_only: recorded and reported. It never rejects, retries or stops.

The blind roles' own verdicts (ROLE_VERDICTS) are methodological by the protocol, and the
rules that stop the whole construction (STOP_RULES) are listed apart: they reject no need.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Kind = Literal["technical_integrity", "methodological_eligibility", "diagnostic_only"]
# what a check reads: a model's output can come out differently in another execution; the
# records the code assembled and the state cannot
Reads = Literal["model_output", "assembled_records", "state"]


@dataclass(frozen=True)
class Check:
    name: str
    kind: Kind
    reads: Reads
    stages: tuple[str, ...]
    consequence: str  # a technical reason, "stop", a ledger outcome, or "recorded"
    rule: str  # the rule it enforces, and where it was written before TEST

    @property
    def retriable(self) -> bool:
        return self.kind == "technical_integrity" and self.reads == "model_output"


_MODEL_STAGES = (
    "concept",
    "statement",
    "overlap_gate",
    "query_formulation",
    "judge_a",
    "omission_search",
    "judge_b",
    "judge_c",
    "diagnostic",
    "judge_d",
    "metadata",
    "evidence_gate",
)
CHECKS: dict[str, Check] = {
    c.name: c
    for c in (
        Check(
            "role_audit",
            "technical_integrity",
            "model_output",
            _MODEL_STAGES,
            "isolation",
            "each run against its role contract: authorized input, orchestrator messages, "
            "runtime context (ingest/holdout_roles.py, Codex Gate 2 P1.1)",
        ),
        Check(
            "reply_is_json",
            "technical_integrity",
            "model_output",
            tuple(
                s for s in _MODEL_STAGES if s not in ("query_formulation", "judge_a", "diagnostic")
            ),
            "malformed_session",
            "every packet asks for one JSON object (the role templates); the diagnostic's own "
            "answer is diagnostic_admissible's",
        ),
        Check(
            "broker_protocol",
            "technical_integrity",
            "model_output",
            ("judge_a", "omission_search"),
            "protocol",
            "the third invalid reply ends the run (ingest/holdout_broker.py, D-054)",
        ),
        Check(
            "broker_ceiling",
            "technical_integrity",
            "model_output",
            ("judge_a", "omission_search"),
            "resource",
            "32 requests are the technical ceiling (ingest/holdout_broker.py, Codex Gate 2 P1.3)",
        ),
        Check(
            "answer_fields",
            "technical_integrity",
            "model_output",
            ("concept", "statement", "overlap_gate", "evidence_gate"),
            "malformed_session",
            "the fields the role's template asks for, a statement not empty, a verdict from "
            "the gatekeeper's list naming only a need it was shown",
        ),
        Check(
            "query_formulations",
            "technical_integrity",
            "model_output",
            ("query_formulation",),
            "malformed_session",
            "query_variants.v1: exactly the three formulations, one per line",
        ),
        Check(
            "statement_cites_no_provision",
            "methodological_eligibility",
            "model_output",
            ("statement_validation",),
            "statement_invalid",
            "PROTOCOL.md step 1: a statement names no article, paragraph, point, structural "
            "unit, node path or CELEX (ingest/benchmark.py render_query_prompt)",
        ),
        Check(
            "statement_no_dev_wording",
            "methodological_eligibility",
            "model_output",
            ("statement_validation",),
            "statement_invalid",
            "D-054: no legal wording and no run of eight words of DEV or the fixtures "
            "(ingest/holdout_packets.py author_leaks)",
        ),
        Check(
            "concept_no_dev_wording",
            "methodological_eligibility",
            "model_output",
            ("statement",),
            "concept_invalid",
            "D-054's frozen guard on an author's packet (ingest/holdout_packets.py author_leaks: "
            "no legal wording and no run of eight words of DEV or the fixtures), unchanged, on "
            "the statement writer's packet: when the concept's own words make it fire (the "
            "packet built without them passes), the concept is no valid TEST concept. The "
            "attempt closes, never retried, and the cell gets a new blind attempt that learns "
            "nothing of it (CONSTRUCTION_AMENDMENT_1)",
        ),
        Check(
            "packet_construction",
            "technical_integrity",
            "assembled_records",
            _MODEL_STAGES,
            "stop",
            "a packet the code builds from frozen inputs and admitted outputs before a model "
            "sees it: a frozen input that no longer holds (the protocol's sections and examples, "
            "the attempt frames, a template), a node the corpus lacks, or the anti-copy guard "
            "firing on the frozen parts alone. No execution can change it: the construction "
            "stops for a person (CONSTRUCTION_AMENDMENT_1)",
        ),
        Check(
            "record_assembly",
            "technical_integrity",
            "assembled_records",
            ("lock", "adjudication", "metadata", "evidence_gate", "ready"),
            "stop",
            "the records the code assembles from admitted outputs (the draft and final need and "
            "queries, the adjudications and qrels with their gap notes, the evidence pairs, the "
            "ledger's transitions) hold the dataset's schema and the ledger's sequence. No "
            "execution can change them: the construction stops for a person "
            "(CONSTRUCTION_AMENDMENT_1)",
        ),
        Check(
            "judge_answer_complete",
            "technical_integrity",
            "model_output",
            ("judge_a", "omission_search", "judge_b", "judge_c", "judge_d"),
            "malformed_session",
            "every candidate graded, once, and only the candidates, and every item answered, "
            "by the protocol's labels (ingest/holdout_judging.py judge_a_problems, "
            "disagreements, judge_d_outcome); the diagnostic's is diagnostic_admissible",
        ),
        Check(
            "model_identity",
            "technical_integrity",
            "model_output",
            tuple(s for s in _MODEL_STAGES if s != "diagnostic"),
            "runtime_identity",
            "the effective model the authenticated runtime reports in the session is, exactly, "
            "the one frozen from Gate 1 (ingest/holdout_runtime.py claude_problems)",
        ),
        Check(
            "runtime_identity",
            "technical_integrity",
            "model_output",
            ("diagnostic",),
            "runtime_identity",
            "the diagnostic was produced under the frozen MLX runtime, recorded in its spool "
            "before the run and in the output itself (ingest/holdout_runtime.py local_problems)",
        ),
        Check(
            "metadata_answer",
            "technical_integrity",
            "model_output",
            ("metadata",),
            "malformed_session",
            "a type, difficulty and drivers from the rubric (ingest/holdout_judging.py "
            "final_metadata)",
        ),
        Check(
            "final_qrels_unambiguous",
            "methodological_eligibility",
            "assembled_records",
            ("adjudication",),
            "ambiguous",
            "PROTOCOL.md v1.4 section 4: an ambiguous need leaves TEST and is replaced blind",
        ),
        Check(
            "dataset_records",
            "technical_integrity",
            "assembled_records",
            ("validators",),
            "stop",
            "the benchmark's own validators accept the need with every ready one "
            "(ingest/benchmark.py dataset_problems)",
        ),
        Check(
            "qrel_texts",
            "technical_integrity",
            "assembled_records",
            ("validators",),
            "stop",
            "every qrel names a node of the corpus with its text (ingest/benchmark.py "
            "qrel_problems)",
        ),
        Check(
            "state_integrity",
            "technical_integrity",
            "state",
            ("start", "resume", "checkpoint", "freeze"),
            "stop",
            "the durable state verifies and equals what the replay rebuilds "
            "(ingest/holdout_state.py, D-055)",
        ),
        Check(
            "material_discrepancy",
            "diagnostic_only",
            "model_output",
            ("diagnostic",),
            "recorded",
            "Qwen3-8B has no vote: a material discrepancy only sends the item to judge D "
            "(D-054 decision 7)",
        ),
        Check(
            "diagnostic_admissible",
            "diagnostic_only",
            "model_output",
            ("diagnostic",),
            "recorded",
            "an answer of Qwen3-8B, from a run that ended normally under the frozen runtime, is "
            "a diagnostic only if it is one JSON object grading every candidate once with the "
            "protocol's labels (ingest/holdout_judging.py diagnostic_defects); otherwise the "
            "diagnostic is unavailable: the answer stays with its run, never read again, with "
            "no vote and no label, and a blind judge D grades every candidate and the "
            "answerability by the rule of a discrepancy. Unavailable is not agreement. Greedy "
            "decoding gives the same answer again, so it is never executed again (the first "
            "START, aborted on 2026-10-20 with no ready need)",
        ),
        Check(
            "evidence_overlap_stratum",
            "diagnostic_only",
            "assembled_records",
            ("evidence_gate",),
            "recorded",
            "shared grade-3 evidence with DEV is a stratum for S4, not an exclusion; only "
            "the evidence gatekeeper's verdict decides (D-054)",
        ),
        Check(
            "plan_deviation",
            "diagnostic_only",
            "assembled_records",
            ("ready",),
            "recorded",
            "the grid is a target, not a quota: departures are recorded (D-054 decision 8)",
        ),
    )
}


@dataclass(frozen=True)
class RoleVerdict:
    role: str
    verdict: str
    outcome: str  # the ledger's methodological outcome
    rule: str


ROLE_VERDICTS = (
    RoleVerdict("concept_author", "fit is not 'fits'", "concept_unfit", "concept_author.v1"),
    RoleVerdict(
        "gatekeeper",
        "paraphrase, same_legal_question, same_rule_superficial_change or uncertain",
        "overlap_*",
        "overlap_gate.v1 and D-054 decision 2",
    ),
    RoleVerdict(
        "gatekeeper_evidence",
        "paraphrase, same_legal_question, same_rule_superficial_change or uncertain",
        "overlap_*",
        "overlap_gate_evidence.v1",
    ),
    RoleVerdict("judge_a", "a problem with the need", "judge_problem", "judge_a.v2"),
    RoleVerdict(
        "judge_d",
        "does not hold the final label, or finds the item ambiguous",
        "adjudication_uncertain",
        "judge_d.v1 and D-054 decision 7",
    ),
)
STOP_RULES = (
    "an omission found by the second search adjudicated grade >= 1 (Codex Gate 2 P1.4)",
    "fewer than 50 evaluable needs still reachable (D-054 decision 3)",
    "a problem found at an operational checkpoint (D-055)",
    "a technical stage failed three executions: pending_technical_unresolved (D-055)",
)


def check(name: str) -> Check:
    return CHECKS[name]


def record() -> dict[str, Any]:
    """The classification as data, for the pre-authoring contract."""
    return {
        "checks": {n: asdict(c) | {"retriable": c.retriable} for n, c in sorted(CHECKS.items())},
        "role_verdicts": [asdict(v) for v in ROLE_VERDICTS],
        "stop_rules": list(STOP_RULES),
    }
