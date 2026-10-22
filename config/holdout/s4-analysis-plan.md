# S4 analysis plan

**Status: frozen before authoring** (Codex Gate 2, P1.5). Its SHA-256 is in the pre-authoring
contract, `config/holdout/pre-authoring-contract.json`, which also fixes the exact systems, code,
corpus and evaluation. Both are fixed before any TEST need is written, and nothing in them changes
after authoring begins: the freeze of TEST adds only derived properties of TEST (dataset hashes,
counts, realized distributions, rejection and overlap statistics, file identities).

## 1. What TEST measures

> TEST evaluates unseen information needs from the same target task, corpus contract, taxonomy
> and relevance protocol, but its information needs are generated independently from the
> system-development loop.

- **Same target task, corpus contract, taxonomy and relevance protocol:** the three acts at the
  snapshot of the DEV manifest (DORA `32022R2554`, MiCA `02023R1114-20240109`, PSD2
  `02015L2366-20151223`), parser 1.4.0, the PSD2 exclusion list, protocol v1.4 (grades 0 to 3,
  relevant at grade >= 2, answerability complete, partial or none), the four need types, and the
  query template `query_variants.v1`.
- **Generated independently:** the concept and the statement of every TEST need are written by
  contexts that have no access to the systems, to DEV, to their results or to their failures. A
  separate gatekeeper, who never writes a need, compares each proposal with DEV and drops what
  overlaps; a dropped proposal is replaced from scratch by another blind context.
- **Not independent, and declared:** the orchestration (grid, packets, validation) and the judges
  belong to the same project and the same model family as the development work (section 7).

TEST does **not** show: generalization to real user questions; legal correctness adjudicated by
experts; generalization to other authors beyond those of TEST; behaviour in production; or an
exact replication of how DEV was built.

## 2. Systems: one frozen run each

| System | Role | Identity |
|---|---|---|
| E | the dev candidate frozen by D-050 | `systems.E` of the contract: decomposer, its prompt and parser, E5 retrieval of each requirement (depth 50), reranker and instruction, D's assembler (K 5, keep 6, depth 10), its DEV run identity |
| E5 | the frozen dense baseline (`multilingual-e5-large`, layer-500) | `systems.E5`: config, tokenizer, prefixes, pooling, normalization, dimension, scoring, ranking and ties |
| BM25 | sanity baseline | `systems.BM25`: analyzer, stemming, k1, b, corpus unit, ranking and ties |

Before it runs, S4 checks that every code and config file has the SHA-256 the contract records;
any difference stops S4. The only change allowed is the interface: pointing the same code at the
TEST directory.

- One run per system on TEST 1.0.0, with the frozen configuration. No tuning, no threshold, no
  prompt and no parameter is chosen after a TEST query has been run.
- A technical failure (a crash, an out-of-memory error) may be run again unchanged, once, and the
  failure is reported. Any change to a system makes it another system, reported under another
  name, never as E.
- Every run returns at least `MIN_RUN_DEPTH` (100) results per query, as in DEV.

## 3. Primary comparison: E against E5

- **Metrics:** Recall@10 and nDCG@10 on the chunk ranking, as in DEV: binary relevance at
  grade >= 2 for Recall; the grades as linear gains for nDCG (`ingest/ir_metrics.py`).
- **Statistical unit: the information need.** Each metric is computed per query, averaged over the
  queries of a need, and then averaged over the evaluable needs (complete and partial). Needs of
  answerability none never enter the metrics.
- **Uncertainty:** the paired difference E minus E5, with a bootstrap over needs that resamples
  needs, never queries: 10,000 resamples, percentile 95% interval, seed 20261013, the procedure of
  `scripts/compare_embedding_runs.py`.
- **Verdict for each metric:** *holds* if the lower end of the interval is above 0, *reverses* if
  the upper end is below 0, *inconclusive* otherwise.
- **Reported for each metric, separately:** the number of needs, the mean of E5's and E's scores,
  the paired delta (the mean over needs of E minus E5), its interval, and the effect size
  d_z = mean(delta) / sd(delta), with the standard deviation over needs (n - 1). There is no
  composite score.
- **Multiplicity:** the general claim needs both co-primary metrics to hold, so testing two does
  not make it easier to reach. A claim on one metric is worded as specific to it (below). There
  are no p-values and no correction for the secondary analyses, which are exploratory by
  definition and never support a claim.
- **What may be claimed, fixed now:**

| Recall@10 | nDCG@10 | The claim |
|---|---|---|
| holds | holds | "E improves retrieval over E5." |
| holds | inconclusive | "E improves evidence coverage, but we do not have evidence of a general ranking improvement." |
| inconclusive | holds | "E improves ranking quality, but we do not have evidence that it covers more of the evidence." |
| inconclusive | inconclusive | "We do not have evidence that E improves on E5 on TEST." |
| either reverses | | "E is worse than E5 on" the metric that reverses, stated next to the verdict on the other one. |
- **Power, for the record:** in DEV the paired difference had a standard deviation of about 0.15
  per need (an inference from its interval). With 50 evaluable needs a true +0.06 is detected
  about 80% of the time; a true +0.03 would need about 200 needs. TEST can say whether an
  improvement of DEV's size survives, not certify a small one.

## 3b. Languages, incomplete systems and failures

- **Languages:** every need is scored on its three formulations (EN formal, EN natural, ES
  natural), averaged within the need; the primary comparison uses that mean. By language and by
  style are secondary strata.
- **Incomplete systems:** a system that fails technically runs again, unchanged, once. If it
  fails again it is reported as not completed, with its error, and no comparison that needs it is
  made or approximated: no variant, no subset of TEST, no other configuration stands in for it.
  If E does not complete, the primary comparison is reported as not made.
- **A query missing from a system's run** (for instance one its contract refuses, such as a
  query over E5's 512-token limit) is scored 0 on every metric and listed, as the evaluator does
  (`ingest/evaluation.py`, `missing_rankings`). It is never dropped from the mean.

## 4. Secondary analyses: diagnostic and exploratory

None of them can become the main claim after the results are known.

- BM25 against E5, and E against BM25, with the same procedure.
- Recall@5, Recall@20 and MRR@10. E keeps E5's top 6, so its Recall@5 equals E5's by
  construction: the difference is reported as a check that it is 0.
- Strata, each reported with its number of needs and an interval marked exploratory:
  - language and style of the query (EN formal, EN natural, ES natural);
  - type of need;
  - answerability (complete against partial);
  - act;
  - evidence overlap with DEV: needs with no grade-3 article in common with any DEV need,
    against needs with at least one (`overlap/evidence.json`). The stratum is a diagnostic: it is
    never used to modify TEST.

## 5. DEV against TEST, described before any drop is interpreted

A difference between DEV and TEST can hold three effects at once:

1. selection and model-design bias: E was designed while looking at DEV;
2. author and process shift: TEST's needs come from other authors and a different process
   (section 7);
3. qrel construction: DEV keeps its historical qrels, built from one evidence search per need
   and its judge's listing; TEST's come from three blind, independent searches per need and a
   mechanical exposure pool: every node a search received in full is judged, listed or not
   (`config/holdout/evidence-discovery.md`). TEST's relevant sets can be more complete than
   DEV's, and that alone can lower recall and move nDCG.

What TEST supports, then: the paired comparison of E against E5 and BM25 within TEST, on one
common set of qrels, and the claim that it is a holdout of needs unseen during development,
built by the declared procedure. What it does not support cleanly: attributing the absolute
difference between DEV's metrics and TEST's to overfitting or model-selection bias alone.

So before a drop is read as selection bias, the two sets are compared descriptively, with
`scripts/holdout.py describe` on each, on these features and only these:

- type, act and answerability;
- statement length and query length, in words, by language and style;
- relevant evidence per evaluable need: nodes and articles of grade >= 2;
- the distribution of grades;
- answers that are a figure or a date: planned in the grid, realized in TEST (the `numeric` or
  `date` driver on the final qrels), and in DEV;
- complexity: the share of `multi_provision` needs and of needs with relevant evidence in more
  than one article;
- lexical features: for each query, the share of its content words found in its evidence and the
  longest verbatim run (`ingest.benchmark.lexical_overlap`), and the number of `POSSIBLE COPY`
  flags.

This is description, not adjustment. No reweighting, no subsetting and no change to TEST follows
from it.

## 6. How TEST was built, where it matters for reading it

- **The grid is a target, not a quota.** A need keeps the type, the answerability and the drivers
  its evidence gives it; a departure from its cell is recorded (`type_deviation`,
  `planned_figure_date_not_realized`) and never corrected by choosing needs after judging them.
  The freeze reports the planned distribution, the realized one and every departure.
- **Final metadata:** the acts, the type `partial_out_of_scope` and the driver `cross_act` follow
  from the final qrels by rule; the type of a complete need, its difficulty and its other drivers
  come from a blind metadata adjudicator that sees the locked need, its final grades and the
  rubric, never the plan. Only afterwards is the plan compared with what was realized.
- **Figure or date, softly:** a cell that plans a figure or a date asks for one only if a natural
  question of the area has one.
- **The order of the cells** is a pseudo-random permutation fixed before authoring
  (`holdout-order.v1`, seed and hash in the contract). It never changes with a rejection, a
  result, the difficulty, the evidence or a system. Each cell is finished before the next starts.
- **At most three blind attempts per cell,** each in a new context with the same packet (one
  frame for every attempt) and an opaque id; no context can tell an attempt, a retry or the
  order. A cell that fails three times methodologically stays empty
  (`unfilled_after_attempt_limit`). If fewer than 50 evaluable needs remain reachable, the work
  stops before any other mechanism is invented.
- **The gatekeeper drops** paraphrases, the same legal question, the same rule with superficially
  different facts, and anything it cannot establish as independent (`overlap_uncertain`), against
  DEV, the fixtures and the TEST needs already `ready`. The drops are counted by reason, by what
  they repeated (`overlap_dev`, `overlap_test_existing`, `overlap_uncertain`) and by area.
- **Technical failures are never rejections.** A transport, broker, session, isolation, model,
  protocol or infrastructure failure, or a process that dies, is one failed execution of the stage
  before any admissible output: the same need stays `pending_technical` and the construction
  stops; it never produces a new question. Every execution has a deadline: at it, its process
  group is ended, reaped and verified gone before the failure is recorded, and a group that
  cannot be verified gone stops the construction. Each search's broker allows 16 requests as the
  normal run and goes on for the same need up to a technical ceiling of 32
  (`resource_escalation`); at 32 the execution ends as `pending_technical_resource`. The freeze
  refuses any pending need.
- **Every answer is admitted only whole.** Before an output is admitted, its complete contract is
  checked: shape, required fields and types, non-empty text, valid grades and answerabilities,
  the exact candidate set, the rationale every label needs, and every invariant its consumers
  rely on. An answer that breaks it is a failed technical execution, never a rejection, with one
  exception fixed before this construction: a well-formed answer that violates a frozen semantic
  invariant tying its labels to the need's evidence or answerability (the metadata's type,
  drivers and negation against the final grades; a search's answerability against its own
  grades), in all three executions of a normal search or of the metadata adjudicator, rejects the
  attempt (`semantic_inadmissible`), with nothing inferred or repaired. A failure the construction
  does not classify is recorded as a stop for a person.
- **Technical retries and resume.** A resume continues the same cell, the same need and the same
  stage from the same frozen inputs, after verifying again the pre-authoring contract, the
  binary and runtime, every packet, the corpus, snapshot and chunk layer and every admitted
  output. Admitted stages are never executed again. A stage gets at most three technical
  executions in all (the first and two retries), each in a new isolated context with the same
  packet, model and runtime; after the third failure the need is `pending_technical_unresolved`
  and the construction stays stopped, with no replacement and no change of protocol. A valid
  answer is never executed again to obtain a different one. The retry count is separate from the
  three methodological attempts. A change of anything frozen makes the partial construction
  incompatible: it stops, and is never migrated to another method.
- **No person reviews a statement.** The statement is checked by code and by the blind roles
  only; the person, who knows E, DEV and their failure modes, neither edits nor approves any TEST
  need before S4, and reviews only the procedure, the distributions, the audit reports, the
  rejection reasons and aggregate diagnostics.
- **Operational checkpoints.** After the 20th, 40th and 60th cell of the frozen order the state is
  verified, a verified backup is made and a report of counts is recorded. A checkpoint never
  recalibrates a prompt, changes the protocol, the grid or a stop rule, or balances types; a
  problem it finds stops the construction.
- **Evidence discovery, the same for every need** (`config/holdout/evidence-discovery.md`):
  three blind, independent searches (search_1, search_2, search_3) with the same frozen packet
  and broker, each in its own isolated context, none seeing another's candidates, path or
  labels. Their listed candidates are joined, each once, with the first judgment of the first
  search that listed it. Every node a search received in full enters the exposure pool
  mechanically, listed or not. A pool node no search listed is graded blind by judge B's pool
  variant (the same rubric, no answerability). A grade >= 1 or an ambiguity gets a second blind
  judgment, and judge C decides where the two differ. The pool variant must grade every node it
  is shown, explicitly: a node left out or ungraded is a failure of its contract, never a grade
  0. An explicit grade 0 that is not ambiguous is final, kept in the pool's record and the
  provenance and not as a qrel: an unjudged node and a judged 0 count alike in every metric of
  this plan. A search that declares a problem makes the need `judge_problem`.
  This applies to every need that reaches evidence discovery, whatever its planned
  answerability.
- **The omission audit:** 12 cells fixed before authoring (`holdout-omission.v1`, stratified by
  type and acts) get a fourth blind search, with the same broker, which sees nothing of the
  other three. Its coverage is its own exposure: a node it listed or received in full outside
  the normal pool is adjudicated by the pool's rules, and any of them with a final grade >= 1
  stops the construction before the freeze, so that the evidence procedure is reviewed for every
  need alike. Grade 0 additions are diagnostics. Its label on a node the pool held already is
  provenance only, counted by direction as an audit judgment discrepancy, never a label, a
  rejection or a stop. The audit that counts is the one of the attempt that becomes READY, and a
  declared problem of the audit search stops the construction, since the audit cannot be
  completed.
- **The diagnostic judge** (Qwen3-8B, local) grades, with judge B's packet, the candidates judge
  D receives: every listed node, every pool node with a final grade >= 1, and every pool node any
  blind judgment graded >= 1 or found ambiguous. It has no vote. Where it crosses the relevance line or the answerability of the final label, a blind
  judge D grades the item: if D holds the final label it stays and the discrepancy is recorded; if
  not, or if D finds the item ambiguous, the need is `adjudication_uncertain`, leaves the scored
  TEST and gets a new blind attempt. Its prompt is counted with the frozen tokenizer before any
  generation; a prompt over the frozen limit, or an answer that is not one complete JSON object,
  makes the diagnostic unavailable, and judge D then grades every candidate and the
  answerability. The diagnostic may be unavailable for many needs, or all: the freeze reports
  how many had it, and judge D running is never reported as a cross-family adjudication.

## 7. Rules

- DEV is historical context only: it is never pooled with TEST in any statistic.
- No analysis outside this plan is presented as confirmatory.
- A labelling error found during S4 is corrected in TEST 1.0.1, with a record of the correction;
  the numbers on 1.0.0 are kept and reported next to the corrected ones. There is no silent fix.

## 8. Known limitations

- **Three effects in one difference** (section 5), one of them the different construction of
  the qrels: one search and its listing per need in DEV; in TEST, three searches and every node
  they received in full judged.
- **Process differences from DEV:** in DEV the orchestrating session wrote the scenario of 20 of
  the 30 needs, narrowed 6 more after seeing their evidence, and was judge A of 10. In TEST no
  author has seen the system or DEV, replacements are written blind, and judge A is always a
  fresh context.
- **Ambiguity:** an ambiguous need is excluded from TEST and replaced (protocol v1.4, section 4),
  while DEV keeps its two ambiguous qrels. TEST may be "cleaner", and so easier, than DEV.
- **One model family:** authors, writers, gatekeeper and judges A, B, C and D are contexts of the
  same model family as the development work; the 60 authors are 60 independent contexts, not 60
  independent people. Their errors may be correlated. A local model of another family (Qwen3-8B)
  grades the same candidates as a diagnostic with no vote. Judge D is of the same family as A, B
  and C, because no local model of a third family is available.
- **What the gatekeeper drops also shapes TEST:** dropping whatever resembles DEV moves TEST away
  from DEV's topics. The drops by area are in the freeze.
- **Departures from the grid** (unfilled cells, relabelled types, figures not realized) may leave
  TEST's composition different from DEV's; they are reported, not corrected.
- **Order sensitivity:** when two needs are duplicates of each other, the one earlier in the
  frozen order is kept. The order is random and fixed in advance, but a different seed would keep
  a different one.
- **The omission audit covers 12 cells of 60.** It is a control against systematic omissions,
  not a proof of exhaustive evidence: a search of the same model family as the three it audits,
  it can test that the pool is complete relative to what such searches reach, never beyond it.
- **Judge D** gives independence of context, not of model family. Where the diagnostic is
  unavailable, no label of TEST had a judge of another family.
- **The orchestration** that builds the packets and validates the data has seen DEV's evidence and
  knows the systems. It writes no concept, statement, query or label.
- **No legal expert** adjudicates any label.
- **Evidence search is structural and lexical,** as in DEV, with no retriever: the same bias in
  both sets, which affects the absolute numbers.
- **Queries in Spanish** are written by a model and may not read like real users'.
- **No complete run with real models before TEST:** the isolation (Gate 1), the broker over
  stream-json, the audit of real packets and the orchestrator with synthetic replies were each
  tested; no pseudo-need was written to test them together. The first TEST cell is the first
  complete run, and a technical failure in it follows the rules above, never an adaptation of
  the method to its content.
