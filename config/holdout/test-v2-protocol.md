# TEST v2 protocol

**Status: frozen by the TEST v2 contract** (`config/holdout/test-v2-contract.json`, decision
D-067). The contract records this file's SHA-256, the code that implements it and every identity
it relies on. Nothing here changes after the contract is pinned; a change makes another protocol.

TEST v2 is a new, prospectively versioned protocol. It is not a repair of the earlier
construction (runs 1 to 7, `config/holdout/s4-analysis-plan.md`), which is kept as history.

## 1. Why TEST v2 exists

Runs 1 to 6 were development runs of the earlier protocol. Run 7 was the first confirmatory
attempt: it stopped under the frozen `minimum` feasibility rule (240 events, 0 READY, the first
five evaluable cells exhausted, at most 49 of the required 50 evaluable needs reachable). No
E-vs-E5 TEST result was ever observed. An external methodological review followed before any
TEST retrieval existed, and its reviewers converged: the qrel pool must include the compared
systems, E and E5; incompleteness must be measured rather than fatal; topics must be generated
in parallel and deduplicated globally; the 50/54 design was brittle. The adjudication against
the repository found no blocking contradiction and approved this design.

The earlier construction tried to prove that no relevant evidence existed outside three agentic
searches, with a zero-tolerance omission audit, while the documents E and E5 actually retrieve
were never guaranteed to be judged. A development calibration of that audit on 24 historical
READY needs found 359 audit-new nodes: 325 of grade 0, 34 of grade 1 and none of grade 2 or more,
so it could stop a construction without finding any evidence Recall@10 counts. It is kept as
development evidence and controls nothing here.

## 2. The question

> On an independent private sample of legal and regulatory information needs drawn from DEV's
> task taxonomy, does the frozen system E improve over the frozen E5 at top-10 retrieval?

The corpus is DEV's (DORA, MiCA and PSD2 at the snapshot of the DEV manifest, parser 1.4.0,
the PSD2 exclusion list, the layer-500 chunks). E and E5 are frozen. No TEST result may change
the topics, the qrel method, the metrics, the sample size, the pool, the judging, the reserve
or any rule of inclusion.

## 3. Sample size

**N = 200 evaluable information needs**, fixed. It is never recomputed from TEST, lowered for
runtime or raised. Its empirical basis, from DEV: E's paired Recall@10 difference over E5 was
+0.061 with a per-need standard deviation of about 0.154; a bootstrap of the selection among 25
complete DEV candidates estimated an optimism of about +0.031 Recall and +0.015 to +0.028 nDCG,
so the corrected Recall effect is about +0.03, for which about 206 needs give 80% power.

TEST v2 has roughly adequate power for an improvement of about +0.03 Recall@10, and about
+0.02 nDCG@10 where the true nDCG effect supports it. An inconclusive nDCG result is possible
and acceptable.

## 4. The population and the bank

- **Profiles.** Each answerable DEV need (answerability complete or partial) gives one profile,
  its slot: acts, type, answerability, difficulty and whether the answer is a figure or a date.
  The DEV need itself never reaches TEST v2. The `none` profiles are removed: they enter no
  metric and conflict with an anchored need.
- **The bank.** 260 candidates, each slot replicated `260 // slots` times and once more for the
  `260 % slots` slots first in a draw. Every candidate has a canonical id (`<slot>-<copy>`), its
  profile, an anchor per act of its profile and a priority: the hex SHA-256 of
  `'<seed>:<salt>:priority:<candidate id>'` (seed 20261025, the private salt). The priority
  orders the bank and fixes the need id (N401 for the first). It never depends on completion
  order, on any system's output, on judging or on any result.
- **Primary and reserve.** The earliest candidates in priority order that survive deduplication
  and become evaluable form the primary sample; the unused tail is the single reserve. There is
  no second generation and no second reserve. A candidate leaves only for an outcome-independent
  reason: a duplicate, an unusable or anchor-copying generation, an exhausted technical retry,
  mechanical non-evaluability. If the bank is exhausted before 200 evaluable needs, the N
  achieved is frozen and its reduced power reported.

## 5. Anchors

The sampling unit is the **article** (legal unit), never a raw chunk or node frequency, which
would overweight long or fragmented articles. An article is eligible if it lies in an area of
the protocol's section 10 (so not a final or transitional provision), is not a PSD2 article
amended after the snapshot, and has at least one provision of its own text owned by a chunk.

Per act of the profile, in priority order, the act's eligible articles are put in an order drawn
once (`'article'`, seed 20261026); the i-th demand takes article i, wrapping only when every
article has been drawn: uniform and without replacement while capacity permits, then a
deterministic, counted reuse. Inside the article, the provision starts at a draw over its
eligible nodes made once for the article, and its k-th use takes the k-th provision after it, so
that a reused article gives another provision while it has one left. A cross-act profile gets one anchor per act. The capacity and reuse per act
are recorded in the contract (DORA's 56 eligible articles cannot cover its demand: the reuse
is counted, never hidden).

## 6. Generation in three hops

- **A, the concept** (`v2_concept_author`, `concept_author.v2`): the profile and its anchors,
  each with its act, its article's title (never its number), the own text of its ancestors and
  the provision itself. Nothing of DEV, the fixtures, the historical exclusions, other
  candidates, systems, rankings, qrels or metrics. It writes a realistic standalone need
  grounded in the anchor, in plain words, without copying it.
- **B, the need** (`statement_writer`, `statement_writer.v1`): the concept and its facts only;
  never an anchor, an identifier or a location.
- **C, the queries** (`query_writer`, `query_variants.v1`): the statement only; three
  formulations (EN formal, EN natural, ES natural).

Every hop runs in parallel over the bank; no candidate waits for another.

**Checks.** The concept keeps concept_author.v1's shape and may hold no legal reference, no
legal wording or eight-word run of DEV or the fixtures, and no eight-word run of its anchor's
article. **The anchor leakage filter:** a run of eight consecutive words of an anchor's article
in the statement or in a query is near-copying, and the candidate is not used. Shorter shared
runs (terms of art such as "critical ICT third-party service provider") and shared vocabulary
are measured and reported, never a filter. The report gives the distribution of the longest
run and of the shared share, the rejection rate, and the drops by profile and by acts.

## 7. Global deduplication

One deterministic stage after the whole bank is generated. Each candidate is compared with
DEV, the fixtures, the 99 historical statements, the development topics and the earlier kept
candidates of the bank, never all at once:

- a bounded prefilter nominates its neighborhood: mandatory, every reference or earlier
  candidate with the same normalized statement or a content-word Jaccard of 0.5 or more, and
  every earlier candidate sharing an anchor article; bounded, the 6 most similar references
  and 6 most similar earlier candidates by BGE-M3 cosine, and the 3 most similar of either by
  Jaccard;
- the frozen gate `overlap_gate.v2` decides, every reference under an opaque id drawn for the
  candidate (DEV's and the fixtures' included) with its kind, in packets of at most 20, never
  cut: a candidate is kept only if every packet finds it independent;
- in the bank, priority wins: the earlier candidate stays.

The neighborhoods are recorded before the first gate call. The gate runs in waves of candidates
whose earlier neighbors are decided; the recorded decisions must equal the one-by-one
resolution in priority order, so input or completion order changes nothing.

**The 99 historical statements** (84 from runs 1 to 6, 15 from run 7) stay hidden references.
Their exclusion is **not** claimed to be strictly necessary for statistical validity: they are
part of D-066's documented history, keeping them avoids reinterpreting it, and corpus anchoring
makes the cost low. Authors never see them. There is no recursive blacklist: only designated
development topic-generation material (the pilots' statements) may become a development
reference, and a technical failure never does. A duplicate is the same substantive legal need,
never merely the same article, act or terminology.

## 8. The topic freeze

Before any retrieval, one canonical manifest freezes the bank's hash, the selected topics in
priority order with their profiles, anchors, statements, formulations and per-topic hashes, the
dropped candidates, the deduplication decisions, the references' identity, the seeds and the
templates' hashes. After it, no topic text changes: a change is an integrity stop.

## 9. Retrieval, the run freeze and the pool

Every frozen retriever runs on every frozen query, one local model at a time: E5, BGE-M3,
Qwen3-Embedding-0.6B, BM25 and E (D-4B's requirements from E5's first three results, each
requirement's first 50 E5 chunks reranked by Qwen3-Reranker-0.6B, D's rule). Each system ranks
the whole layer as in DEV; each frozen file keeps a query's first 100 chunks. Each run is recorded
when it completes and checked (every query ranked or refused by its contract, no chunk twice, no
chunk outside the layer); a system that fails technically runs again, unchanged, once, and is then
not completed (S4): the pool goes on without it, and if E or E5 is not completed the primary
comparison is reported as not made. A process interruption (a kill, a sleep, the orchestrator's
crash) is no failure of the system twice; from the third on, each counts as a failed attempt. The runs are hashed with their configs, model identities,
corpus identity and code. No metric is computed.

**The pool of a need**, the union over its three formulations of:

| Contributor | Depth |
|---|---|
| E | 20 |
| E5 | 20 |
| BM25 | 20 |
| BGE-M3 | 10 |
| Qwen3-Embedding-0.6B | 20 |
| the chunk that owns each anchor | always |

Each chunk once. The provenance (contributor, formulation, rank) is kept internally and never
shown to a judge. There is no one-hop expansion and no sixth retriever. BGE-M3 and
Qwen3-Embedding-0.6B are pool contributors only, frozen and hashed, never a confirmatory
endpoint.

**Judged nodes.** The repository's projection is kept: every text-bearing node has one owning
chunk, and a chunk's grade is the highest final grade among the nodes it owns. For every pooled
chunk, every node it owns with its own text (outside the excluded PSD2 articles) is judged; its
ancestors are shown as context. No anchor, ancestor, sibling or same-article node is ever
labelled relevant automatically.

**The decisive set** of a need: the union over its formulations of E top-10 minus E5 top-10,
plus E5 top-10 minus E top-10.

## 10. Judging

- **B1** (`v2_pool_judge_1`, `judge_b_pool.v3`): every judged node of the pool. Explicit grade
  0 labels persist. Opaque ids, exact coverage (a node left out, ungraded or unknown is a
  failure of the role's contract, never a 0). A chunk's nodes stay together in document order;
  the chunk groups come in an order drawn per need and stage (seed 20261028); a batch never
  splits a group.
- **B2** (`v2_pool_judge_2`): an independent blind judgment of every node B1 graded 1 or more,
  every node B1 found ambiguous, and **every node of the decisive set, whatever B1 said**. The
  decisive set is never sampled.
- **C** (`judge_c`, `judge_c.v2`): exactly the nodes where B1 and B2 differ, in grade or
  ambiguity; never a node they agree on.
- **Final label:** the agreed one or C's. A final ambiguity stays in the qrel artifact as
  ambiguous and never deletes the need; the primary evaluation treats it as unjudged (D-019) and
  sensitivity SA1 shows the alternative.
- **Retired:** judge D, the Qwen diagnostic (E itself contains Qwen components), the
  `adjudication_uncertain` and `final_qrels_unambiguous` rejections.

Needs are judged in waves in priority order, each as large as the evaluable needs still
missing, until 200 evaluable needs exist or the bank is exhausted. Each wave is recorded before it
is judged, and a resume judges an interrupted wave as it was planned.

## 11. After the qrels

- **Mechanical evaluability:** a need is evaluable when it has at least one final,
  non-ambiguous node of grade 2 or more: the evaluator's concept of a need with relevant
  evidence. Partial evidence, judge disagreement and uncertain metadata never delete a need.
- **The qrel freeze:** the first 200 evaluable needs in priority order and their qrels, with the
  attempted candidates, the duplicates, the technical failures, the non-evaluable candidates and
  the final N.
- **Answerability** (`v2_answerability`, `answerability.v1`), descriptive and after the freeze:
  the need, its final evidence of grade 1 or more, no provenance, no system, no rank. Complete,
  partial or none, and what is missing. A `none` despite grade-2 evidence is a contradiction:
  it is recorded and routed to a second blind judgment (`v2_answerability_2`), and to judge C
  where the two differ. It never filters the primary sample. An exhausted judgment leaves it
  `unknown`.
- **Metadata** (`metadata_adjudicator`), after the qrels, on the final grades of 1 or more. A
  failure gives `unknown`; it never deletes a need.

## 12. The completeness diagnostic

- **Selection, before judging:** per act, the candidates whose profile's first act it is, in an
  order drawn with the bank (seed 20261027); the diagnostic's needs are the first five of each
  chain that end up in the primary sample: 15 needs. No outcome enters the selection.
- **Screen:** for each, every chunk of the layer outside its pool that owns a node to judge (about
  1,354 minus the pool), by one blind screening judge (`v2_screen`, `completeness_screen.v1`),
  about 50 chunks per batch, aiming at high recall.
- **Follow-up:** the flagged chunks' nodes through B1, B2 and C. Only a final non-ambiguous grade
  2 or more counts as a verified residual relevant chunk.
- **Report:** chunks screened, flagged, verified residual chunks, relevant chunks inside the pool,
  pool recall with a bootstrap interval over the needs, the contribution by act, and the effect
  of the residual positives on Recall@10 and nDCG@10 (sensitivity SA2).

It never modifies the primary qrels, restarts or invalidates TEST. **The omission-audit stop is
retired**, with no hidden equivalent: new relevance outside the pool is a measurement.

## 13. The optional human audit

Recommended, never required to run TEST. A person grades, blind, the decisive-set chunks of 15
needs fixed before judging (the first primary needs of a chain drawn with the bank, seed
20261029), with three other pooled chunks of each need mixed in as controls. The packet hides the
system, the rank, the formulation and every judge's label. The person is not a legal expert:
it is a coarse diagnostic of judge bias. Its labels never correct or overwrite a primary label:
they are reported through SA7, and TEST v2 1.0.0 stays immutable (there is no TEST 1.0.1).

## 14. Stops, retries and execution

**Hard stops, integrity only:** a contract, corpus or chunk-layer mismatch; a frozen retrieval
configuration or model identity mismatch; a runtime or binary change; an isolation failure; a
process group that cannot be verified gone; a deterministic-state inconsistency; a topic mutated
after its freeze; a mandatory artifact corrupted or lost. TEST v2 never stops because new
relevance was found, pool recall is low, a qrel is ambiguous, judges disagree, a need is hard,
N is below 200 after the bank, E performs poorly, E5 performs better, the effect is small or the
interval crosses zero: those are results.

**Technical retries:** at most three executions per job, each the same packet in a new isolated
context; a valid answer is never run again; an answer that breaks its role's contract is a
failed execution, never a label. A single forbidden-content finding of the isolation audit that
disappears once metadata values are neutralized is retried; any other finding stops. After the
last execution the job is exhausted (recorded once, never decided again by a resume), its
candidate or need is a technical failure, and the next one in priority order comes in. **A call the provider
refuses for its capacity** (the CLI's final error result or its stderr naming a usage limit,
`rate_limit_error` or `overloaded_error`, the pinned binary's own words) is not a failure of its
job: it never counts toward the limit, not even when it is read back after a pause or a crash.
The engine waits (five minutes, doubling up to an hour, once per episode) and runs the job again;
after six hours without a response the construction pauses, with nothing decided, for a resume,
and every call still running then is recorded with its own outcome. Every other failure counts, whatever its kind, so a
job that fails the same way every time is exhausted after three executions. What counts is a
function of the recorded reasons alone, so a resume decides exactly as the run it continues.

**Execution:** bounded parallelism for every model job; one writer of the state; the local
models one at a time. Completion order never changes a scientific outcome. A development-only
dry run measures the throughput before the contract. **Operational knobs** it may set: the
workers, the batch ceilings within validated limits, the retry backoff, the scheduling, the
backup frequency and the local queue. It may not change N, the universe, the taxonomy, the
anchors, the pool, the metrics, the relevance threshold, B1, B2 and C, the completeness size,
the deduplication, the reserve or the stops. With fewer workers TEST only takes longer.

## 15. State, provenance and the public boundary

A new namespace and schema (`holdout-v2-state.v1`): an append-only, hash-chained event log with
idempotent keys, content-addressed kept and raw artifacts, one writer. Events record the bank and
its anchors, every model run and admission, the generation, the deduplication, the topic, run,
pool, qrel and completeness freezes, every judging stage, the answerability, the metadata, the
audit selection, the sensitivity outputs and the evaluation. A restart replays the log and never
changes a scientific decision.

Public: the code, this protocol, the analysis plan, the contract, the tests. Private (the TEST
directory, whose history refuses every push, and folders outside the repository): the bank, the
topics, the anchors, the rankings, the pools, the qrels, the raw model outputs, the state, the
salts, the historical exclusions, the development references, the authorizations and the backups.

## 16. Retired and preserved

**Retired from active use** (kept for reproducing runs 1 to 7): the 60-cell sequential
controller, the 54/50 minimum, three mandatory agentic searches, the exposure pool as primary
discovery, the omission-audit stop, three attempts per cell, the gate that depends on accepted
TEST needs, recursive blacklisting, judge D, the Qwen diagnostic, the `adjudication_uncertain`
and `final_qrels_unambiguous` rejections, the `none` cells.

**Preserved:** the corpus, parser, chunker and node-to-chunk projection; the frozen E, E5 and
BM25 and the validated BGE-M3 and Qwen3-Embedding configurations; the relevance rubric; the B and
C prompts where compatible; the query generator; opaque ids and exact coverage; provenance
hashes; the event and state concepts; restart and technical retries; the evaluator and S4's
bootstrap core; DEV's taxonomy; `overlap_gate.v2`; the historical exclusion artifact; the
certified Gate 1 isolation primitives, byte for byte.

## 17. Known limitations

- **One model family** writes and judges (Claude contexts), and E contains Qwen components: the
  double blind judgment of the decisive set and the optional human audit mitigate, never remove,
  the coupling.
- **Pool incompleteness** remains: the pool is five retrievers deep, not the corpus; the
  completeness diagnostic measures it on 15 needs and SA2 and SA4 show its effect.
- **Anchor reuse in DORA**: its eligible articles are fewer than its demand; reuse is counted and
  each reuse draws another provision while one is left.
- **Cross-act profiles** get one anchor per act of the profile: an implementation reading of "the
  profile defines the acts".
- **The concept author reads the law**: the anchor-copy filter measures and bounds copying, but
  topics are closer to the corpus's vocabulary than DEV's free authoring.
- **No legal expert** adjudicates any label.
