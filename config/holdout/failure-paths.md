# Reachable failures of the TEST construction, and what each leads to

Frozen before authoring with the pre-authoring contract, as a construction input. Every failure
that the construction can reach, as implemented, falls into one family below. Each family has
a classification fixed in advance (`ingest/holdout_checks.py`, carried by the contract), a
durable transition of the state (`ingest/holdout_state.py`), its retry semantics, and whether a
person decides what follows. No reachable production failure ends the construction without a
classified durable outcome. The only exceptions are, by design, a failure of the state's own
integrity or medium, which is never written over, and a refusal before anything is built.

The unattended controller (`ingest/holdout_controller.py`, policy `holdout-controller.v1`)
retries only rows marked "controller" below, and only for a failure produced by the invocation
it has just run. Everything else stops it.

## A. Refusals before anything runs (no transition: the state is not touched)

| # | Failure | Classification | Durable transition | Retry | Person |
|---|---|---|---|---|---|
| A1 | The contract, a frozen file, the systems, the grid, order or omission cells, the role contracts, templates or checks, the audit module, the corpus, snapshot or chunk layer differ | `LaunchRefused` / `IncompatibleState` at preflight (`state_integrity`) | none | none | yes |
| A2 | The pinned binary, the Claude runtime, the MLX runtime or the host differ | `IncompatibleState` at preflight | none | none | yes |
| A3 | The private paths are not private, the history does not refuse pushes, or the salt is not 32 bytes | `IncompatibleState` at preflight | none | none | yes |
| A4 | START with a state, RESUME/freeze without one, a mistyped SHA-256, no public commit recorded | `LaunchRefused` | none | none | yes |

## B. The state itself (never written over)

| # | Failure | Classification | Durable transition | Retry | Person |
|---|---|---|---|---|---|
| B1 | The state does not verify: SQLite integrity, the chain of events, a projection, an artifact's hash, a consumed failed run | `IntegrityError` (`state_integrity`) | none: every later START/RESUME refuses too | none | yes |
| B2 | The identities or runtimes the state was created with, or its cells, differ | `IncompatibleState` | none, as B1 | none | yes |
| B3 | A replay decides differently, or a packet rendered now is not the one a run received | `ReplayDiverged` / `IncompatibleState` | none, as B1 | none | yes |
| B4 | Another process holds the state | `StateBusy` | none | none | yes |
| B5 | The medium fails (disk, SQLite, fsync, an artifact of the state that cannot be written or read) | `MediumError` / `sqlite3.Error` | the transaction rolls back whole; nothing half written | none | yes |

## C. One technical execution of a stage (the same need stays; at most three executions)

Each is recorded as `run_failed` with the cell `pending_technical`, or
`pending_technical_unresolved` after the third execution. Its output is never consumed.

| # | Failure | Classification | Durable transition | Retry | Person |
|---|---|---|---|---|---|
| C1 | The executor fails: the API, the transport, the process exit (live or recovered), the infrastructure (spool, directory, a session that cannot be redacted), a corpus operation of the broker that fails on a valid request | `model_api`, `transport`, `infrastructure`, `broker` | `run_failed` | RESUME; controller | after the 3rd |
| C2 | An execution reaches its deadline (a process: its group ended, reaped and verified gone first; a corpus operation of the broker in this process: 60 s) | `execution_deadline` (`timeout`) | `run_failed` | RESUME; controller | after the 3rd |
| C3 | The orchestrator dies or is stopped mid-execution (crash, SIGTERM, SIGKILL, power) | `interrupted`, on the next RESUME; a complete spooled response is recorded and audited instead | `run_failed` or the recovered response | RESUME; controller | after the 3rd |
| C4 | The run's isolation audit fails, or cannot read the run (fail closed) | `role_audit` (`isolation`) | `run_failed` with the audit | RESUME; controller only when its one finding lies inside authorized metadata | yes otherwise |
| C5 | The session ran on another effective model, or the pinned binary changed (checked before each of its processes starts) | `model_identity`, `runtime_identity` | `run_failed` | RESUME by a person | yes |
| C6 | The diagnostic would run, or ran, under another MLX runtime or host | `runtime_identity` | `run_failed` | RESUME by a person | yes |
| C7 | An answer breaks its role's contract: its shape, fields, types, text, grades, answerability, candidate set or labels (each built as the adjudication record builds it), the rubric's vocabulary, a reference it was not shown (a need, a node), an item id missing, unknown, repeated or not the one given, or an identity beside an id that names another node (never repaired, never bound by position), a pool judge's answer that leaves a node out or ungraded (never a grade 0), three formulations not distinct, a value that cannot be written as UTF-8 text | `answer_fields`, `query_formulations`, `reply_is_json`, `judge_answer_complete`, `metadata_answer` (`malformed_session`) | `run_failed` | RESUME; controller | after the 3rd |
| C7s | A well-formed answer violates a frozen semantic invariant tying its labels to the need's evidence or answerability: the metadata's type against the answerability, a gap driver on a complete need, `negation` without a grade 3, a driver without relevant support; a search's complete without a grade 3, none with relevant evidence, partial without relevant evidence | `metadata_answer`, `judge_answer_complete` (`malformed_session`), with the invariants' codes | `run_failed`, with its `violations` | RESUME; controller | see E8 (the three normal searches, metadata); C10 (the audit search) |
| C8 | The diagnostic's record has no valid token count, or a generation that contradicts it | `diagnostic_preflight` (`malformed_session`) | `run_failed` | RESUME; controller | after the 3rd |
| C9 | A search's broker run ends in a protocol failure or at the ceiling of 32 requests | `broker_protocol` (`protocol`), `broker_ceiling` (`resource`) | `run_failed` | RESUME; controller | after the 3rd |
| C10 | A stage's third execution fails, and the three are not a semantic admissibility failure of a normal search or the metadata (E8): a contract failure, different invariants, or the audit search | `pending_technical_unresolved` | `run_failed`, cell unresolved; RESUME refuses | none | yes |

## D. Diagnostics (recorded; never a retry, a rejection or a stop)

| # | Failure | Classification | Durable transition | Retry | Person |
|---|---|---|---|---|---|
| D1 | The diagnostic's answer is not one JSON object grading every candidate once (the candidates judge D receives: every listed node and the pool's contested or relevant ones) | `diagnostic_admissible` | the diagnostic stage admitted as unavailable; judge D grades every candidate and the answerability | none | no |
| D2 | The diagnostic's prompt does not fit the frozen limit (`input_exceeds_context`), found before any generation | `diagnostic_admissible` | as D1 | none | no |
| D3 | A material discrepancy of the diagnostic | `material_discrepancy` | judge D grades the item | none | no |
| D4 | A node new to the audit search adjudicated grade 0 or left unjudged; the audit search's label on a node the exposure pool held already differing from its final grade (an audit judgment discrepancy, either direction: never a label, a judge C, a rejection or a stop); shared grade-3 evidence with DEV; a departure from the grid | `evidence_overlap_stratum`, `plan_deviation`, the audit's grade-0 additions, `judgment_discrepancies` | recorded in the report | none | no |

## E. Methodological rejections of an attempt (a new blind attempt, at most three per cell)

| # | Failure | Classification | Durable transition | Retry | Person |
|---|---|---|---|---|---|
| E1 | The concept author finds the cell unfit; the concept copies DEV's wording | `concept_unfit`; `concept_no_dev_wording` (`concept_invalid`) | `attempt_closed` | next attempt | no |
| E2 | The statement cites a provision or copies DEV or the fixtures | `statement_cites_no_provision`, `statement_no_dev_wording` (`statement_invalid`) | `attempt_closed` | next attempt | no |
| E3 | The gatekeeper, on the statement or on shared grade-3 evidence, finds an overlap or cannot establish independence | `overlap_*` | `attempt_closed` or `attempt_failed_after_gate` | next attempt | no |
| E4 | A normal search (search_1, search_2 or search_3) declares a problem with the need, before any of its candidates is read | `judge_problem` | `attempt_failed_after_gate` | next attempt | no |
| E5 | A final qrel is ambiguous | `final_qrels_unambiguous` (`ambiguous`) | `attempt_failed_after_gate` | next attempt | no |
| E6 | Judge D does not hold the final label, or finds it ambiguous | `adjudication_uncertain` | `attempt_failed_after_gate` | next attempt | no |
| E7 | A cell fails three attempts | `unfilled_after_attempt_limit` | the cell is terminal | none | no |
| E8 | The three executions of a normal search or of the metadata adjudicator all violate the same semantic invariant (C7s): the check and its codes, never the text | `semantic_admissibility` (`semantic_inadmissible`) | the third `run_failed`, then `attempt_failed_after_gate` with the check and the invariants as its note; nothing inferred, repaired or substituted | next attempt | no |

## F. Stops for a person (`construction_stopped`; RESUME refuses until a person decides)

| # | Failure | Classification | Durable transition | Retry | Person |
|---|---|---|---|---|---|
| F1 | A node the audit search listed or received in full, outside the normal exposure pool, adjudicated grade >= 1 (an ambiguous final label counts by its grade) | `omission_audit` | `construction_stopped` | none | yes |
| F2 | The audit search declares a problem: the audit cannot be completed | `omission_audit_incomplete` | `construction_stopped` | none | yes |
| F3 | Fewer than 50 evaluable needs remain reachable | `minimum` | `construction_stopped` | none | yes |
| F4 | A packet cannot be built from frozen inputs and admitted outputs, including a node of the exposure pool that alone renders a pool packet over its token ceiling | `packet_construction` | `construction_stopped`, cell unresolved | none | yes |
| F5 | A record cannot be assembled from admitted outputs, or the ledger refuses a transition | `record_assembly` | `construction_stopped` | none | yes |
| F6 | The benchmark's validators refuse a need. This includes a final answerability that the final qrels do not support, such as a complete need left with no grade 3 after judge C decided a node on which A and B disagreed while they agreed on the answerability. It is reachable, and frozen as an integrity stop: classifying it otherwise would change the adjudication | `dataset_records`, `qrel_texts` | `construction_stopped`, cell unresolved | none | yes |
| F7 | A checkpoint finds a problem, its backup does not verify, or its report cannot be made | `checkpoint` | `construction_stopped` | none | yes |
| F8 | A process group cannot be verified gone after SIGKILL | `process_cleanup` | `run_failed` unresolved, then `construction_stopped` | none | yes |
| F9 | Any failure nothing above classifies (named by its kind and place, never its text) | `unclassified_failure` | `construction_stopped` | none | yes |

## G. The unattended controller (no transition of the state; it stops and logs a code)

| # | Failure | Code | Person |
|---|---|---|---|
| G1 | validate() finds a problem before an invocation | `validate` | yes |
| G2 | The invocation recorded no event | `no_progress` | yes |
| G3 | No failure of the waiting stage was produced by this invocation, or it was used before | `stale` | yes |
| G4 | No new event for longer than any execution's deadline (90 minutes), or the invocation's outer deadline passed (72 hours, 48 before 2026-10-22): its group ended and verified gone | `stalled`, `deadline` | yes |
| G5 | A process group of an open run cannot be verified gone | `CleanupFailed` | yes |
| G6 | Every cell terminal: the freeze is the person's | `complete` | yes |
| G7 | The invocation ended with no recorded outcome; another controller runs; the cycle limit | `no_recorded_outcome`, `busy`, `cycle_limit` | yes |

## H. Known residual limits, accepted

- A descendant of a model's process that calls `setsid` leaves its process group: no
  group-based check can see it.
- The pinned binary's version check (`ingest/holdout_isolation.py check_pinned`, byte for byte the
  module Gate 1 passed with) runs `--version` with no deadline of its own. The orchestrator's group
  is bounded by the controller's stall limit.
- `end_registered` trusts the group id a spool recorded, which the system could in principle have
  reused for another process of this user after the orchestrator died.
- Writing a message to a judge's standard input can block before its turn's deadline starts; only
  ending the orchestrator interrupts it.
