# Retrieval iteration 5 on DEV 0.10.0: three assembly and coverage changes, fixed in advance

On the night of 2026-10-15 the person authorized autonomous iteration for about 12 hours if F
([protocol](reranker-finetuning.md)) did not reach the useful target. The rules and the menu below
were fixed before F's DEV result was read, so that no choice of the night comes from reading DEV:

- at most 3 new DEV evaluations, one per experiment, in the order G1, G2, G3;
- each experiment is published before it runs and audited by an independent reviewer agent;
- component choices come from synthetic validation, never from DEV: the reranker is F's adapter,
  because F's training passed on VAL-REPORT (MRR 0.837 to 0.960, +0.123 [+0.067, +0.179]);
- DEV, its queries, labels and chunks are untouched, and no single query's failure is used to
  design anything;
- the iteration stops at the first variant that reaches the useful target, or after G3;
- nothing from this night is a final result: at best a candidate for validation on new needs.

Each experiment attacks a loss measured in D and E (`retrieval-iteration-4/log.md`).

## The reference and when a variant replaces it

The reference starts as F's DEV system (D-4B's requirements, E5's 50 per requirement reordered by
F's adapter, D's rule). A variant replaces the reference when its Recall@10 is at least 0.02 above
the reference's or its full-evidence at least 0.03 above, with nDCG@10 not below the reference's by
more than 0.01 (values rounded to 3 decimals): the system signal of F. Each later experiment builds
on the reference of that moment.

## G1: the four places by the best score, not in turns

D's rule gives places 7 to 10, after the corroborated E5 ranks 7 to 10, to each requirement's best
new chunk in turns. With 3 or 4 requirements that is roughly one chunk each, whatever the reranker
thinks of them; 10 to 12 relevant pairs waited in a requirement's first 5 without a place (D, E).
G1 gives those places to the new chunks with the highest score over all requirements
(`ingest/decomposition.py`, `assemble_scored`); a chunk takes its best score among the requirements
that hold it in their first 5; ties keep the turns' order. Everything else is D's rule.

**Scores across requirements.** Each requirement's scores are the softmax over its own 50 candidates
(`scripts/night_iteration.py`, `normalized`), not the raw logit differences. F's adapter was trained
with a listwise loss, which leaves each list's offset free: listwise-trained rankers are reported
uncalibrated across queries (Bai et al., CIKM 2023, on tabular learning to rank), so raw scores from
two requirements are not on one scale. A softmax within the list cancels exactly that offset; it
assumes about one answering chunk per requirement, which is how the requirements are asked. No study
was found that compares normalizations for choosing across sub-query lists, so this is a reasoned
choice, not a measured one (the literature review of the night is in the record). It was fixed
before F's DEV result was read.

No new scoring: F's DEV scores are reused.

## G2: the requirements of both decomposers

D-4B's and D-8B's requirements together, per query, in that order, an identical text once. 13 to 18
relevant pairs were reached by no requirement of a single model; the two models gained in partly
different needs. E5's first 50 for the 8B's requirements are scored by F's adapter (about 17,400
new pairs), and the reference's rule builds the ten.

## G3: the first 100 of each requirement, not 50

The reranker reorders E5's first 100 for each requirement of the reference; 13 to 18 relevant pairs
were outside the first 50. The pairs of places 51 to 100 are scored by F's adapter, and the
reference's rule builds the ten.

## Measures

The evaluator of every system, as in iteration 4: Recall@10, nDCG@10, MRR@10, full-evidence@10,
`multi_provision`, dependent evidence, `cross_reference`, `cross_act`, by language; each variant
paired with E5 and with the reference by need (Recall@10, nDCG@10, full-evidence@10). The useful
target is unchanged: Recall@10 >= 0.85, `multi_provision` >= 0.60, dependent evidence >= 0.45,
full-evidence@10 >= 0.75.

## Not touched

DEV 0.10.0, its queries, qrels, grades and chunks, the parser, the evaluator; D's prompts, parser
and K; F's data, recipe and adapter.
