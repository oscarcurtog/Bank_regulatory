# S4 analysis plan, version 2 (TEST v2)

**Status: frozen by the TEST v2 contract** (`config/holdout/test-v2-contract.json`, D-067),
before any TEST v2 topic exists. Its implementation is `ingest/holdout_v2_evaluation.py`; both
are bound by hash. The primary analysis is S4's (`config/holdout/s4-analysis-plan.md`, section
3), unchanged; what this version adds is fixed here before any result and none of it can become
the claim.

## 1. Systems

E (the dev candidate frozen by D-050), E5 (the frozen dense baseline) and BM25 (sanity), each one
frozen run on TEST v2's frozen queries (`config/holdout/test-v2-protocol.md`, section 9). A
system that fails technically runs again unchanged, once; if it fails again it is reported as not
completed, and no comparison that needs it is made or approximated. BGE-M3 and
Qwen3-Embedding-0.6B contribute to the pool and are never compared.

## 2. Primary comparison: E against E5

- **Metrics, co-primary:** Recall@10 (relevant at grade >= 2) and nDCG@10 (the grades as linear
  gains), on the chunk ranking, by `ingest/ir_metrics.py` on the chunk qrels projected from the
  final node labels (`ingest/evaluation.py` project). An ambiguous or unresolved final label
  counts as unjudged (D-019). No composite metric; the fixed-scale G@10 is a sensitivity only.
- **Unit: the information need.** Each metric per query; a need's value is the mean of its three
  formulations; then the mean over the primary needs.
- **Population:** the primary sample of the qrel freeze, every need mechanically evaluable
  (protocol, section 11). Answerability and metadata never filter it.
- **Uncertainty:** the paired difference E minus E5, with S4's bootstrap
  (`ingest/run_report.py` paired_bootstrap): needs resampled with replacement, 10,000 times,
  seed 20261013, the 95% percentile interval (sorted means at `int(0.025 * reps)` and
  `int(0.975 * reps) - 1`).
- **A query missing from a run** scores 0 on every metric and is never dropped.
- **Verdict per metric:** *holds* if the lower end is above 0, *reverses* if the upper end is
  below 0, *inconclusive* otherwise.
- **Reported per metric:** the number of needs, the means of E and E5, the paired delta and its
  interval, d_z = mean(delta) / sd(delta) (n - 1), and the needs better, equal and worse.
- **What may be claimed, fixed now:**

| Recall@10 | nDCG@10 | The claim |
|---|---|---|
| holds | holds | "E improves retrieval over E5." |
| holds | inconclusive | "E improves evidence coverage, but we do not have evidence of a general ranking improvement." |
| inconclusive | holds | "E improves ranking quality, but we do not have evidence that it covers more of the evidence." |
| inconclusive | inconclusive | "We do not have evidence that E improves on E5 on TEST." |
| either reverses | | "E is worse than E5 on" the metric that reverses, stated next to the verdict on the other one. |

- **Multiplicity:** the general claim needs both co-primary metrics to hold. There are no
  p-values for the primary analysis; the secondary and sensitivity analyses never support a
  claim.
- **Power:** N = 200 evaluable needs; roughly adequate power for an improvement of about +0.03
  Recall@10, and about +0.02 nDCG@10 where the true effect supports it. If the bank is exhausted
  first, the N achieved is analysed and its reduced power reported.

## 3. Secondary analyses, exploratory

- BM25 against E5, and E against BM25, by the same procedure.
- Recall@5, Recall@20 and MRR@10 of E against E5 (E keeps E5's top 6: Recall@5's difference is
  a check that it is 0).
- Strata, each with its number of needs, descriptive only under 5 needs: formulation (language
  and style), type, answerability, difficulty, the acts of the relevant evidence, and the
  profile's acts.

## 4. Sensitivity analyses

Each is computed on the frozen artifacts and reported beside the primary result; none can
replace it.

- **SA1, ambiguity:** every final ambiguous label with a grade counted at that grade instead of
  as unjudged; the co-primary comparison again. Reported with the number of ambiguous nodes,
  how many sit in E's or E5's top 10, and how many in the decisive set.
- **SA2, augmented qrels:** for the 15 completeness needs, the verified residual positives
  (final non-ambiguous grade >= 2 outside the pool) added; the co-primary comparison on those 15
  with the primary and the augmented qrels, and on the whole sample with them augmented. With
  the diagnostic's own report: screened, flagged, residual, relevant inside the pool, pool recall
  with its bootstrap interval, by act.
- **SA3, fixed-scale discounted gain:** G@10 = DCG@10 with the grades as linear gains divided by
  the DCG@10 of ten grade-3 chunks, so that nothing outside a top 10 moves it; the paired
  comparison by the same bootstrap.
- **SA4, residual relevance:** (a) the chunks of E's or E5's top 10 that are unjudged
  (ambiguous, unresolved, or owning no node to judge): a lower bound making E5's alone relevant
  (grade 3) and E's 0, an upper bound the reverse; (b) relevance outside the pool, in
  scenarios fixed now: such a chunk is in neither top 10 (the pool holds both top 20s), so it
  only enlarges both denominators: Recall@10 = hits / (R + r) and nDCG@10 with r more relevant
  chunks in the ideal ranking, in two scenarios: one chunk of grade 2 per need, and three
  chunks of grade 3 per need. The completeness diagnostic's own positives enter SA2 only;
  SA2 reports the residual it measured beside these scenarios.
- **SA5, nDCG gains:** nDCG@10 with grade 1 as 0, and with exponential gains (2^grade - 1), by
  the same evaluator on the mapped chunk grades.
- **SA6, inference:** for each co-primary metric, the paired t interval on the per-need deltas
  and a paired sign-flip permutation test (10,000 flips, seed 20261013, two-sided).
- **SA7, human audit** (optional, protocol section 13): agreement of the person's passage grades
  with the projected final grades (exact, and on the side of grade 2), and the co-primary
  comparison on the audited needs with the person's grades in place of the final ones. Never a
  correction: TEST v2 1.0.0 is immutable and no TEST 1.0.1 replaces it.
- **SA8, pool contributors:** for each contributor (E, E5, BM25, BGE-M3, Qwen3-Embedding-0.6B,
  the anchor), the chunks only it brought into a pool made unjudged, the needs still evaluable
  kept, and the co-primary comparison again; with the relevant chunks each one alone brought.

## 5. Pool diagnostics

1. Judged@10 of E; 2. Judged@10 of E5; 3. Judged@20 of E; 4. Judged@20 of E5 (the share of the
top-k chunks with a final judged label; for the frozen qrels they should be 1 where every node
can be judged, and the exceptions are counted); 5. the relevant chunks each pool contributor
alone brought; 6. the cumulative relevant discovery by depth, per contributor and in union;
7. the distribution of pool sizes (chunks and nodes per need); 8. leave-one-contributor-out
(SA8); 9. the decisive set's coverage (chunks and nodes per need, nodes with a second blind
judgment, nodes with a final resolved label); 10. the share of evaluable needs whose only grade
>= 2 evidence one compared system alone brought.

They are diagnostics: none can replace a topic or change the sample.

## 6. The construction record

Attempted candidates, drops by reason (generation, anchor copy, deduplication by the kind of
reference repeated, technical), needs judged, not evaluable, technical failures and not judged,
the final N, the answerability distribution and its paths (first judgment, contradiction
confirmed, judge C, unknown), the metadata distribution, and the anchor leakage report.

## 7. Rules

- DEV is history only: never pooled with TEST v2 in any statistic.
- No analysis outside this plan is presented as confirmatory.
- No result reopens the construction: a confirmed improvement, an inconclusive result, a
  reversal, a lower N, an incomplete pool or a judge sensitivity are all valid outcomes.
