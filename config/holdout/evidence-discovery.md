# Evidence discovery of TEST: three searches, the exposure pool v2 and the audit search

Frozen before authoring with the pre-authoring contract, as a construction input. It defines,
for every TEST need that reaches evidence discovery, how its candidate evidence is found and
how it reaches the adjudication. The code that implements it is `ingest/holdout_pipeline.py`
(`_judge`, `_gate`, `_pool_round`, `_omission_audit`), `ingest/holdout_judging.py` (`exposure`,
`snippets`, `reach`, `exposure_pool`, `audit_nodes`, `diagnostic_set`, `review_metrics`,
`audit_review`), `ingest/holdout_admission.py` (`search_problems`, `pool_problems`) and
`ingest/holdout_exclusions.py`.

It replaces the procedures of five earlier constructions, all aborted (the contract's
`aborted_runs`). The last two had two blind searches per need and an audit search in 12 cells,
and the fifth stopped on that audit: a node outside the two searches' union, adjudicated grade 1.
A structural review of the protocol-development runs (counts, ids and grades only, never their
text) found that most omissions were nodes the search that missed them had read in full and not
listed. Each search read some 80 nodes in full and listed some 16, and the template suggested 6
to 16. Coverage depended on a discretionary, softly capped listing, and more searches would only
sample that listing again. This procedure makes coverage mechanical (the person's design A of
2026-10-22), without changing the rubric, the threshold or any other rule.

Exposure v2 (D-066) follows the sixth construction, kept as development evidence only. It
stopped on the omission audit, a valid application of the rule: three nodes outside the pool,
each adjudicated grade 1. A read-only structural review found no controller defect but a gap in
the definition of exposure. One node only a `search` page had named, and no search opened it.
Two were article lead-ins the judges read as the context of their children, never as candidates.
Exposure v2 closes those paths by structure alone, with the smallest closure that covers them:
snippets followed up, and ancestors with their own text as candidates (rule A). A wider rule,
which also took every node of an article a search listed, was implemented and evaluated before
any freeze, and not adopted: it assumes that a listing makes its whole article relevant, at more
than twice the cost (D-066). Exposure v2 was designed after that stop and is declared as such;
the three searches, the threshold >= 1, the rubrics and the judges are unchanged, and no fourth
search is added.

## 1. Who searches

Every need that reaches evidence discovery gets exactly three searches, whatever its planned
answerability (`none` included), in this order:

- `search_1` (role `searcher_1`), `search_2` (role `searcher_2`) and `search_3` (role
  `searcher_3`).

The 12 cells of the omission audit, fixed before authoring (`holdout-omission.v1`) and never
resampled, also get, after the three:

- `audit_search` (role `audit_searcher`).

All four are the same procedure. Each is a fresh isolated Claude context with no tool, which
reads the parsed corpus only through the evidence broker (`ingest/holdout_broker.py`: titles,
article, nodes and search; 32 requests at most, 16 the normal run; pages of 8,000 characters).
Each receives the same frozen packet, `judge_a.v3`, built from the locked need and its cell
alone; only the opaque reference in it differs. `judge_a.v3` is `judge_a.v2` without the
sentence "Usually 6 to 16 candidates.": it suggests no number of candidates, and keeps "Do not
list nodes that nobody would consider." A search's listing is its own judgment of what deserves
a grade; it no longer decides what is judged. No search receives another search's candidates,
path, labels, logs or output, any judge's labels, DEV, the qrels, or any system ranking or
result. Each run's isolation is audited against its role contract before its output is admitted
(`ingest/holdout_roles.py`).

## 2. What each search must answer

A search ends with one JSON object, which is admitted only if its complete contract holds
(`search_problems`). Otherwise it is a failed technical execution of its stage, run again in a
new context up to three executions:

- either a declared `problem`, a non-empty text, and then nothing else is read;
- or every candidate it lists, each a node of the corpus with its own text (never an excluded
  PSD2 article), listed once. Each candidate has a grade 0 to 3, a true or false `ambiguous`,
  its `scope` and `scope_basis` as lists of existing nodes, a non-empty rationale, and a
  `scope_basis` for a grade 0. The answer also gives an answerability (complete, partial or
  none) with a non-empty reason, and a gap note for partial or none. The protocol's rules on the
  file as a whole hold too, as in DEV batch 2.

A well-formed answer whose answerability its own grades do not support (complete without a
grade 3, none with relevant evidence, partial without relevant evidence) violates a semantic
invariant: one failed technical execution. When the three executions of a normal search all
violate the same invariant, the attempt is rejected (`semantic_inadmissible`) and the cell gets
a new blind attempt. The audit search's stay a technical exhaustion, which stops the
construction.

A normal search that declares a problem makes the need not judgeable under the protocol: the
attempt is rejected as `judge_problem`, before any candidate is read, and the cell gets a new
blind attempt (at most three). The audit search's problem stops the construction for a person
(`omission_audit_incomplete`): the audit cannot be completed, and the audit's cells get no
rejection the others do not get.

## 3. The exposure pool (exposure v2, rule A)

The admitted value of each search carries two lists read from the broker's own log of the run,
never from what the model wrote. Its **exposure** (`hj.exposure`) is every node whose id is on a
page the broker returned for an `article` or a `nodes` request; those pages hold each node's
whole own text. Its **snippets** (`hj.snippets`) are every node a `search` page named, with a
snippet of its text. `titles` names no node. The freeze rebuilds both from the recorded logs and
refuses any difference.

A node **can be judged** when it is a node of the corpus with its own, non-empty text (a
structural parent whose text is all in its children has nothing to judge), outside the excluded
PSD2 articles. Each search's **exposure v2** (`hj.reach`) is the deduplicated union of:

- **A. listed:** every node it listed;
- **B. full text:** every node of its exposure that can be judged;
- **C. snippet follow-up:** every node of its snippets that can be judged. The snippet only names
  the node and is never evidence: the node's judge reads its whole own text from the frozen
  corpus, and nothing of the snippet is a grade;
- **D. ancestor closure:** every ancestor that can be judged of any node reached by A, B or C (a
  node's ancestors are the prefixes of its path by whole segments, in its own act). It becomes an
  explicit candidate with its own grade and its whole own text, not only the context of its
  children.

No article is expanded: not because a search listed one of its nodes, received one in full or saw
one named by a snippet, and not because of an ancestor, a grade, a rationale or an audit. An
article's own node, listed or reached, is one node like any other. Exposure v2 depends on each
search's listing, its broker log and the frozen corpus alone: never on a grade, a rationale, an
audit, a qrel or another search.

The **exposure pool** of a need is the union of its three normal searches' exposure v2. Each
pool node keeps, for every search that reached it, the paths that brought it. Entering the pool
is mechanical: no model decides it.

- **A listed node** keeps the rules of the earlier procedure. Its first judgment is the label of
  the first search, in the order search_1, search_2, search_3, that listed it, with that search's
  scope provisions. A later search's label on it is provenance, never a vote. The answerability's
  first judgment is search_1's: its answerability, its reason and its gap note.
- **A pool node no search listed**, however it entered, has no first judgment: its initial state
  is an abstention, and no label is implied for it.

## 4. The adjudication

1. **Judge B** (`judge_b.v2`, blind to every search's labels) grades every listed node and gives
   the answerability, as before.
2. **Judge B's pool variant** (`judge_b_pool.v2`, role `pool_judge_1`) grades every node of the
   pool no search listed. It is judge B's packet and rubric, word for word, without the
   answerability: it grades only the nodes it is shown, each under an opaque item id, in fresh
   contexts, blind to every label. Its answer must cover exactly the nodes it was shown (the
   exact-coverage contract of 2026-10-22): each id once, each with an explicit grade 0 to 3 and
   an explicit true or false ambiguity. A node left out, ungraded (null), unknown, repeated or
   replaced is a failure of the role's contract (`judge_answer_complete`, `malformed_session`),
   retried as one, and never a grade 0 or a semantic rejection. The nodes go in document order, in batches (stages
   `pool_b1:<n>`). A batch closes before the node that would make it hold more than 34 nodes,
   the most judge B graded in one development packet. It also closes before the node that would
   make its packet longer than 9,773 tokens, the largest judge B packet the development runs
   admitted. Tokens are counted with the project's frozen tokenizer, `cl100k_base`. Both numbers
   are ceilings, not targets, and a node is never split. A node that alone would pass the token
   ceiling is no packet the protocol allows, and stops the construction (`packet_construction`).
   The same nodes always give the same batches.
3. **A pool node judge B's pool variant grades 0 explicitly, without an ambiguity**, is final:
   an explicit negative, judged not relevant by one blind judgment. It is kept in the pool's
   adjudication record, in the READY provenance and in the replayed state, with its grade 0, and
   never confused with an unjudged node, which the admission refuses. It is not a qrel: the
   record format of the benchmark (`ingest/benchmark.py`, `Adjudication`) holds two judgments
   for every qrel, and no metric of the S4 plan tells a judged 0 from an unjudged node (Judged@10,
   a diagnostic of the evaluation code outside the plan, would count it as unjudged; the pool's
   record keeps it).
4. **A pool node graded 1 or more, or found ambiguous**, gets a second blind judgment in a new
   context (role `pool_judge_2`, the same template and exact-coverage contract, stages
   `pool_b2:<n>`, the same batching), under the same item id, without the first judgment. If the two agree by the frozen rule (the
   same grade and the same ambiguity), that is the final label. If they differ, judge C decides.
5. **Judge C** (`judge_c.v2`) decides, in one context, every difference of the adjudication: a
   listed node's first judgment against judge B's, a pool node's two blind judgments, and the
   answerability. Each item's two positions appear in an order drawn for it.
6. The final label of each listed node, and of each pool node with two judgments, is the agreed
   one or judge C's. Each has its adjudication record and, when it has a grade, its qrel. A need
   with an ambiguous final qrel leaves TEST and gets a new blind attempt
   (`final_qrels_unambiguous`).
7. **Judge D and the diagnostic** receive exactly the same candidates (`hj.diagnostic_set`):
   - every node a search listed;
   - every pool node with a final grade of 1 or more;
   - every pool node any blind judgment graded 1 or more or found ambiguous;
   - and the answerability.

   A pool node nobody listed, graded 0 explicitly by the pool variant, with no positive or
   ambiguous label, reaches neither. The diagnostic (Qwen3-8B, local, another model family, no
   vote) grades those candidates with judge B's packet. Before any generation its prompt is
   counted with the frozen Qwen tokenizer and chat template, and a prompt over the frozen limit
   is never generated or cut: the diagnostic is unavailable (`input_exceeds_context`). An answer
   that is not one JSON object grading every candidate once also makes it unavailable. Judge D
   (`judge_d.v2`, blind) grades the items where the diagnostic crosses a material line, or every
   candidate and the answerability when the diagnostic is unavailable. A label D does not hold
   makes the need `adjudication_uncertain`. Judge D's rubric is unchanged; it is the same model
   family as the searches and judges B and C.
8. The metadata adjudicator, the evidence gate with DEV and the validators, as before.

## 5. The omission audit

In the omission audit's 12 cells, the audit search runs after the three normal searches, blind
to them. Its coverage is its own exposure v2, by exactly the rules of section 3 (A to D), not
only its listing:

- **A node is new for the omission audit** when it is in the audit search's exposure v2 and not
  in the normal exposure pool, closed under every rule of section 3. Each new node keeps the
  paths of the audit's exposure v2 that brought it.
- The new nodes are adjudicated by the rules of the pool, in contexts of their own (stages
  `audit_b1:<n>`, `audit_b2:<n>` and `audit_judge_c`). A new node the audit search listed takes
  its label as the first judgment, and the pool variant's as the second. A new node it only
  received in full gets the pool variant's judgment, and a second one for a grade of 1 or more
  or an ambiguity. The same exact-coverage contract holds: the audit can never turn a missing
  answer into a grade 0. Judge C decides any difference. None of these contexts sees the normal
  adjudication's labels, and the normal adjudication is the same in an audited cell and in any
  other.
- **Any new node with a final grade of 1 or more stops the whole construction before the freeze**
  (`omission_audit`), so that the evidence procedure is reviewed for every need alike: no local
  replacement, no new attempt of the cell and no next cell. The
  threshold is the frozen one, >= 1, unchanged, and an ambiguous final label counts by its
  grade. A new node with a final grade of 0, or left unjudged by judge C, is a diagnostic: it is
  recorded and counted in the audit's record, and is no qrel of the need.
- **An audit judgment discrepancy is not an omission.** The audit search's label on a node the
  pool already held is provenance only. This holds even when it grades 1 or more a node whose
  final grade is 0, and in the other direction. It never changes a qrel, never runs judge C
  again, never rejects the need and never stops the construction. Each discrepancy is kept in
  the audit's record and counted by direction (0 to 1 or more, 1 or more to 0). This keeps
  relabeling out of the 12 audit cells: they get no path the other cells do not get.
- The audit is bound to the attempt that becomes READY. The audit of an earlier, rejected
  attempt of the same cell is kept as provenance, and never stands for that attempt or against
  it. The frozen order of the checks is kept: an ambiguous final qrel rejects the need before
  the audit's threshold is read.

## 6. What is reported

Counts only, in the freeze record:
- how many candidates each search listed, and how many only one search listed;
- the pool's size and its unlisted nodes;
- how many of those needed a second judgment, and how many the adjudication made grade 1 or more
  and relevant;
- how often the searches' answerabilities agree;
- the audit's new nodes and their grade-0 diagnostics;
- the audit judgment discrepancies, by direction, over the READY attempts and over every attempt
  audited;
- the pool's nodes by the paths of exposure v2 that brought them, and how many only a path new in
  exposure v2 brought.

D-061's review triggers, defined operationally (D-066), are reported as measurements, never as a
gate: the one completeness gate is the omission audit's threshold.

- **`third_search_incremental_g2`:** the numerator counts the final needs where search_3 is the
  first normal search to reach a node of final grade >= 2 that neither search_1 nor search_2
  reached (reached: in that search's exposure v2). The denominator counts the final needs that
  completed the three normal searches and their adjudication. Both are reported with the rate. A
  rate above 7% (D-061's legacy figure) raises a review flag only: never a replacement of a need
  and never a change of the number of searches during a run.
- **`per_search_recall_g2`:** for each normal search and each final need with at least one node
  of final grade >= 2, the share of those nodes the search reached. Reported: the mean, the
  median, the minimum, how many values are below 0.90 (D-061's legacy figure, never a pass or
  fail) and the denominator, the number of pairs.
- **The omission audit's new nodes:** their total, and how many have a final grade 0, 1, >= 2 or
  none, by the first path of the audit's exposure v2 that reached them (listed, other full-text
  reach, snippet follow-up, ancestor closure), over the READY attempts and over every attempt
  audited. A new node with no recorded path is counted as unknown, with the path's name when it
  has one, and never folded into another category.

The S4 metrics are unchanged: Recall@10 counts grade >= 2 as relevant and nDCG@10 uses the grades
0 to 3 as gains (`config/holdout/s4-analysis-plan.md`). That a grade 1 does not count for Recall
is no reason to leave it unjudged.

## 7. Earlier constructions

The six earlier constructions are development evidence only (the contract's `aborted_runs`).
None of their needs, READY records, qrels, judgments or other outputs enters a later TEST: a new
construction starts from an empty state, and a state made under another contract never resumes
or freezes under this one. Every statement they admitted, before their overlap gate decided, is
in the private development need exclusions (`ingest/holdout_exclusions.py`): those that reached
the lock and those the gate then rejected. They are collected mechanically from the durable
states, never by hand and never by reading what a statement means. Each state is read under its
construction's own lock, taken shared, so that no construction process writes it meanwhile; it
is verified whole, between two reads of its event log that must end at the same head; and the
statements are taken from the tables that log alone builds. Every lock must be its attempt's
admitted statement and every READY need its locked one, or the collection stops. The file also
holds the inventory of the states it comes from (each run's contract, event count and head).

The overlap gate (`overlap_gate.v2`) compares every proposal with those statements as references
of a fourth kind, DEVELOPMENT, beside DEV, the fixtures and the ready TEST needs, under opaque ids
drawn again for every proposal; a proposal that repeats one is rejected as an overlap
(`overlap_development`), exactly as one that repeats any other reference. No author receives
them. The contract records their identity only (schema, SHA-256 and count), fixed before the
first author's call. Every use reads the file once, so that the entries used and the identity
checked come from the same bytes; the launcher refuses any other file, or one not collected from
exactly the contract's aborted runs, so the set cannot grow or change during a run. At the
freeze, a READY need whose statement or record repeats a development need's stops the freeze.

## 8. What this does not claim

DEV's qrels were built with one search per need and its judge's listing. TEST's are built from
three searches and a mechanical exposure pool v2. A difference between DEV and TEST metrics
therefore holds a difference of qrel construction too, besides any overfitting of the system to
DEV, and cannot be attributed to the latter alone. The comparison of systems within TEST, on one
common set of qrels, is not affected (`config/holdout/s4-analysis-plan.md`, known limitations).
The pool is complete only relative to what three searches of one model family reach and to the
ancestor closure of exposure v2: a node none of them reaches is unjudged, and the audit, a search
of the same family, can only test that claim, never prove completeness. Exposure v2 was designed
after the sixth construction's stop, and its counterfactual on that construction's searches is
development evidence, never a guarantee of completeness: its effect is not yet measured.
