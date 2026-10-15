# Retrieval iteration 5 on DEV 0.10.0: three assembly and coverage changes, fixed in advance

On the night of 2026-10-15 the person authorized autonomous iteration for about 12 hours if F
([protocol](reranker-finetuning.md)) did not reach the useful target. That authorization overrides
the stop rules of E and F (their protocols end retrieval work on DEV when they fail); it does not
override anything else. The rules and the menu below were fixed before F's DEV result was read. The
menu does come from DEV: from the aggregate, label-based diagnostics of D and E (where E5's missing
pairs go), never from F's result and never from a single query.

- at most 3 new DEV evaluations, one per experiment, in the order G1, G2, G3 (the script refuses
  another order);
- each experiment is published before it runs and audited by an independent reviewer agent; the
  audit of this page found five blocking or should-fix points, all fixed before F's DEV result was
  read (the second commit of this page);
- component choices come from synthetic validation, never from DEV: the reranker is F's chosen
  adapter, because F's training passed on VAL-REPORT (MRR 0.837 to 0.960, +0.123 [+0.067, +0.179]);
- DEV, its queries, labels and chunks are untouched;
- each experiment's ranking is written, and its SHA-256 recorded, before any label is read; a rerun
  after a crash must reproduce that ranking exactly;
- the iteration stops at the first variant that reaches the useful target, or after G3;
- nothing from this night is a final result: at best a candidate for validation on new needs.

## The working reference

The working reference starts as F's DEV system (D-4B's requirements, E5's 50 per requirement
reordered by F's adapter, D's rule). Before each experiment its recorded ranking is rebuilt exactly
from the carried scores, or the experiment does not run. A variant replaces it when its Recall@10 is
at least 0.02 above the reference's or its full-evidence at least 0.03 above, with nDCG@10 not below
the reference's by more than 0.01 (values rounded to 3 decimals): F's system signal, applied
mechanically and without an interval, which is recorded beside it. Each later experiment changes one
thing of the working reference of that moment. Every variant is also paired with E5 and with E, so
that a variant below E is visible.

## G1: the four places by the best score, not in turns

D's rule gives places 7 to 10, after the corroborated E5 ranks 7 to 10, to each requirement's best
new chunk in turns. With 3 or 4 requirements that is roughly one chunk each, whatever the reranker
thinks of them; 10 to 12 relevant pairs waited in a requirement's first 5 without a place (D, E).
G1 gives those places to the new chunks with the highest score over all requirements
(`ingest/decomposition.py`, `assemble_scored`); a chunk takes its best score among the requirements
that hold it in their first 5. Equal scores go first to the chunks D's turns would place, in that
order, then by E5's order. Everything else is D's rule.

**Scores across requirements.** Each requirement's scores are the softmax over its own candidates
(its first 50, or 100 in G3; `scripts/night_iteration.py`, `softmax`), not the raw logit differences.
F's adapter was trained with a listwise loss, which leaves each list's offset free: listwise-trained
rankers are reported uncalibrated across queries (Bai et al., CIKM 2023, on tabular learning to
rank), so raw scores from two requirements are not on one scale. A softmax within the list cancels
exactly that offset; it assumes about one answering chunk per requirement, which is how the
requirements are asked. No study was found that compares normalizations for choosing across
sub-query lists, so this is a reasoned choice, not a measured one (the literature review of the night
is in the record).

No new scoring: F's DEV scores are reused.

## G2: the requirements of both decomposers

D-4B's and D-8B's requirements together, per query: each decomposer's answer counts if it is
readable on its own; the two lists are interleaved (the 4B's first, the 8B's first, the 4B's second,
and so on), so that neither model's requirements always take their turns first; a text repeated,
ignoring case, counts once. 13 to 18 relevant pairs were reached by no requirement of a single model,
and the two models gained in partly different needs. Each decomposer's requirements are encoded
exactly as its D run encoded them. Their E5's first 50 are scored by F's adapter where no carried
score exists (about 17,400 new pairs), and the working reference's rule builds the ten.

One known side effect, stated before the run: the 8B's first 5 enlarge the set of chunks that
corroborate E5's ranks 7 to 10, so more of those places can stay with E5. G2 therefore measures more
requirements and more corroboration together.

## G3: the first 100 of each requirement, not 50

The reranker reorders E5's first 100 for each requirement of the working reference; 13 to 18
relevant pairs were outside the first 50. The pairs of places 51 to 100 are scored by F's adapter, the
others carried, and the working reference's rule builds the ten.

## Measures

The evaluator of every system, as in iteration 4: Recall@10, nDCG@10, MRR@10, full-evidence@10,
`multi_provision`, dependent evidence, `cross_reference`, `cross_act`, by language; each variant
paired by need with E5, with E and with the working reference (Recall@10, nDCG@10,
full-evidence@10). The useful target is unchanged: Recall@10 >= 0.85, `multi_provision` >= 0.60,
dependent evidence >= 0.45, full-evidence@10 >= 0.75.

## Not touched

DEV 0.10.0, its queries, qrels, grades and chunks, the parser, the evaluator; D's prompts, parser
and K; F's data, recipe and adapter.
