# Evidence discovery of TEST: searches A and B, and the omission audit's search C

Frozen before authoring with the pre-authoring contract, as a construction input. It defines,
for every TEST need that reaches evidence discovery, how its candidate evidence is found and
how the candidates reach the adjudication. The code that implements it is
`ingest/holdout_pipeline.py` (`_judge`), `ingest/holdout_judging.py` (`evidence_union`) and
`ingest/holdout_admission.py` (`search_problems`).

It replaces the procedure of the two earlier constructions, which were aborted (the contract's
`aborted_runs`): there, judge A alone searched every need, and a second search ran only in the
omission audit's cells. The second construction stopped on the omission audit: a node judge A
had missed, found by the second search, was adjudicated grade >= 1. That rule says to correct
the evidence search for every need alike (`config/holdout/s4-analysis-plan.md`, the omission
audit). This procedure is that correction. It was fixed from the procedure and the structure
of the runs (the same packet, the same broker, the request budget unused, a different path
through the corpus), never from the content of any TEST need.

## 1. Who searches

Every need that reaches evidence discovery gets two searches, whatever its planned
answerability (`none` included):

- **search A**: the role `judge_a`, stage `judge_a`;
- **search B**: the role `second_searcher`, stage `second_search`.

The 12 cells of the omission audit, fixed before authoring (`holdout-omission.v1`) and never
resampled, also get:

- **search C**: the role `omission_searcher`, stage `omission_search`.

All three are the same procedure. Each is a fresh isolated Claude context with no tool, which
reads the parsed corpus only through the evidence broker (`ingest/holdout_broker.py`: titles,
article, nodes and search; 16 requests in a normal run and a technical ceiling of 32; pages of
8,000 characters). Each receives the same frozen packet, `judge_a.v2`, built from the locked
need and its cell alone. Only the opaque reference in it differs. No search receives another
search's candidates, path, labels, logs or output, judge B's, C's or D's labels, DEV, the qrels,
or any system ranking or result. Each run's isolation is audited against its role contract
before its output is admitted (`ingest/holdout_roles.py`).

The judges B, C and D below are the existing graders. They are not the searches.

## 2. What each search must answer

A search ends with one JSON object, which is admitted only if its complete contract holds
(`search_problems`). Otherwise it is a failed technical execution of its stage, run again in a
new context up to three executions:

- either a declared `problem`, a non-empty text, and then nothing else is read;
- or every candidate it examined, each a node of the corpus with its own text (never an
  excluded PSD2 article), listed once. Each candidate has a grade 0 to 3, a true or false
  `ambiguous`, its `scope` and `scope_basis` as lists of existing nodes, a non-empty
  rationale, and a `scope_basis` for a grade 0. The answer also gives an answerability
  (complete, partial or none) with a non-empty reason, and a gap note for partial or none.
  The protocol's rules on the file as a whole hold too, as in DEV batch 2 (a complete need
  has a grade 3, a partial need a relevant node, a none need no relevant node).

## 3. A declared problem

The transition comes before any candidate is read:

- search A or search B: the need is not judgeable under the protocol. The attempt is
  rejected as `judge_problem`, and the cell gets a new blind attempt (at most three);
- search C: the omission audit cannot be completed. The construction stops for a person
  (`omission_audit_incomplete`). The need is not rejected, so the audit's cells get no
  rejection the others do not get.

## 4. The union

The candidates of the need are the nodes listed by search A or by search B, each once, by its
exact `ACT:path`. They are put in the corpus's document order, the order every judge sees.

Each candidate keeps **one first judgment**: the label the adjudication compares with judge
B's.

- A node search A listed takes search A's label (grade, ambiguity, scope basis, rationale) and
  search A's scope provisions. This holds whether or not search B listed it too.
- A node only search B listed takes search B's label and scope provisions.
- The answerability's first judgment is search A's: its answerability, its reason and its gap
  note.

These are exactly the rules the frozen procedure already applied to a second search's
additions in the omission audit's cells, now applied to every need. The two searches are
exchangeable: the same packet and procedure, in independent contexts, A run first. Search B's
own label on a node search A also listed, and search B's answerability, are recorded as
provenance and reported as counts (how often the two agree). They are never a vote and never
open a dispute. Every candidate of the union is graded again, blind, by judge B, whose
answerability is also its own on the whole union.

## 5. The adjudication

The adjudication is unchanged, on the union:

1. judge B (`judge_b.v1`, blind to every search's labels) grades every candidate and gives the
   answerability;
2. every difference between a candidate's first judgment and judge B's, of grade or ambiguity,
   and any difference of answerability, goes to judge C (`judge_c.v1`), who decides it
   against the text;
3. the final label of each candidate is the agreed one or judge C's. A need with an ambiguous
   final qrel leaves TEST and gets a new blind attempt (`final_qrels_unambiguous`);
4. the diagnostic judge, Qwen3-8B (local, another model family, no vote), grades the union
   with judge B's packet. Before any generation, its prompt is counted with the frozen
   tokenizer and chat template. A prompt over the frozen limit is never generated or cut, and
   the diagnostic is unavailable (`input_exceeds_context`). An answer that is not one JSON
   object grading every candidate once also makes it unavailable;
5. judge D (`judge_d.v1`, blind) grades the items where the diagnostic crosses a material line,
   or every candidate and the answerability when the diagnostic is unavailable. A label D does
   not hold makes the need `adjudication_uncertain`: it leaves TEST and gets a new blind
   attempt. Judge D running is not a cross-family adjudication: it is the same model family
   as A, B and C;
6. the metadata adjudicator, the evidence gate with DEV and the validators, as before.

## 6. The omission audit's search C

In the omission audit's 12 cells, search C runs after searches A and B, blind to both. The
**nodes new to search C** are those it lists that are not in the union of A and B. They join
the candidates of the same adjudication (section 5), each with search C's label as its first
judgment. Search C's labels on the union's nodes are provenance only.

- **Any node new to search C whose final adjudicated grade is >= 1 stops the construction
  before the freeze** (`omission_audit`), so that the evidence search is reviewed for every
  need alike. The threshold is the frozen one, >= 1, unchanged.
- A node new to search C adjudicated grade 0 is a diagnostic, by the frozen audit semantics:
  it is recorded and counted, and it stays in the adjudication record as a grade 0 qrel,
  which enters no metric (Recall at grade >= 2; nDCG with the grades as linear gains).
- The audit is bound to the attempt that becomes READY. The audit of an earlier, rejected
  attempt of the same cell is kept as provenance, and never stands for that attempt or
  against it.
- The frozen order of the checks is kept: an ambiguous final qrel rejects the need
  (`final_qrels_unambiguous`) before the audit's threshold is read.

## 7. What is reported

Counts only, in the freeze record: how many candidates each search listed, how many only
search B listed, and how many of those were adjudicated grade >= 1 and relevant; how often the
answerabilities of A and B agree; the audit's added nodes and grade-0 additions over the READY
attempts, and every attempt audited.

## 8. What this does not claim

DEV's qrels were built with one search per need. TEST's are built with two. A difference
between DEV and TEST metrics therefore holds a difference of qrel construction too, besides
any overfitting of the system to DEV, and cannot be attributed to the latter alone. The
comparison of systems within TEST, on one common set of qrels, is not affected
(`config/holdout/s4-analysis-plan.md`, known limitations).
