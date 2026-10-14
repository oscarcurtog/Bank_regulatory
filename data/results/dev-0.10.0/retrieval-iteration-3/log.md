# Retrieval iteration 3 on DEV 0.10.0: the record

Every experiment of the iteration, in the order it ran, against the rules fixed before it
([protocol](../../../../config/retrieval/necessity-selection.md)); losers included. Global values
are macro means by need. Reference, E5 dense: global Recall@10 0.763, nDCG@10 0.528,
`multi_provision` Recall@10 0.481, dependent evidence Recall@10 0.300, full-evidence@10 0.642.

## A: an anchor-conditioned necessity reranker

`Qwen/Qwen3-Reranker-0.6B` scored 26,835 inputs: 10,545 (query, anchor, candidate) triples under A3
and A2, and 5,745 (query, candidate) pairs under A1. No input was truncated (the longest has 1,329
tokens). The run took 3.3 hours on the Apple GPU, interrupted once to free memory (resumed from its
saved scores). Rebuilt from the stored scores, the diagnosis is identical.

| Variant | Average precision | AUROC | P@1 | P@3 | Support recall@4 | Dependent-support recall@4 |
|---|---|---|---|---|---|---|
| A1: question + candidate | 0.238 | 0.662 | 0.264 | 0.185 | 0.120 | 0.011 |
| A2: + anchor | 0.238 | 0.719 | 0.372 | 0.277 | 0.205 | 0.146 |
| A3: + relation and citation context | 0.185 | 0.711 | 0.193 | 0.193 | 0.093 | 0.011 |

Paired over needs, average precision of A3 minus A1 is -0.014 [-0.118, +0.065] and A3 minus A2 is
-0.068 [-0.119, -0.013]. The hypothesis A3 > A2 > A1 does not hold: the anchor helps a little, the
relation and its clause hurt.

The probability of "yes" shows why. Medians by grade over the 5,204 evaluable (query, candidate)
pairs (36 of grade 2, 194 of grade 1):

| | grade 2 | grade 1 | grade 0 or unjudged |
|---|---|---|---|
| A1 | 0.30 | 0.60 | 0.03 |
| A2 | 0.87 | 0.81 | 0.13 |
| A3 | 0.99 | 0.97 | 0.87 |

Told that the anchor cites the candidate as a condition or defines one of its terms, the model
says "yes" to nearly every candidate: 65% of the unjudged and irrelevant ones pass 0.50 under A3.
No variant separates grade 2 from grade 1.

**Gate.** A3's average precision is not above A1's and A2's: B does not run, as registered. A3
does not improve clearly on A1: the iteration goes to C.

## B: evidence sets

Not run: its gate is closed.

## C: legal-function set selection over an expanded population

**Eligibility audit** (before any inference, population `8721a120`): 15,313 members over the 90
queries (160 at the median, 392 at most): hop 0 1,800, hop 1 6,809, hop 2 6,704. Oracle (the best
a selector could do choosing 10 chunks, labels read only here), whole population: Recall@10
0.954, full-evidence 0.926, `multi_provision` 0.848, dependent evidence 0.909; N022, N034 and
N035 reach 1.0 and N024 0.20 (a first-hop failure). The gate (0.93 and 0.85) passes. It removes
five limits of B: anchors 9 and 10 count; E5's ranks 1 to 20 compete; references are followed for
two hops; the terms of the question and of the supports are resolved to their definitions; no
relation to a kept anchor is required and no position is reserved for E5.

**Time rule.** `Qwen3-4B-Instruct-2507` at 4 bits (MLX 0.32.3) took 24.6 s per window on the
first timing sample, against 41.1 s for PyTorch on the Apple GPU in bfloat16 on one window. The
sample's windows are each query's first and longer than average, so the projection weighs seconds
per prompt token by every window's tokens: 9.6 hours for the whole population, 7.0 for its first
128 members in R1's order. The run used the first 128 (oracle Recall@10 0.948, full-evidence
0.914, gate passed): 1,356 windows, 6.7 hours of computing in the end (median 17.0 s), 2.6 million
prompt tokens, 0.4 million of them reused from a shared opening.

| | E5 | C | difference |
|---|---|---|---|
| nDCG@10 | 0.528 | 0.460 | -0.067 [-0.139, -0.003] |
| Recall@10 | 0.763 | 0.789 | +0.026 [-0.016, +0.067] |
| Recall@20 | 0.816 | 0.822 | +0.006 |
| MRR@10 | 0.746 | 0.576 | -0.171 |
| Recall@T = 1000 / 2000 | 0.680 / 0.782 | 0.572 / 0.794 | -0.108 / +0.013 |
| Full-evidence@10 | 0.642 | 0.667 | +0.025 [-0.025, +0.074] |
| `multi_provision` Recall@10 | 0.481 | 0.531 | +0.051 |
| `cross_reference` Recall@10 | 0.661 | 0.694 | +0.033 |
| `cross_act` Recall@10 (2 needs) | 0.292 | 0.417 | +0.125 |
| dependent evidence Recall@10 | 0.300 | 0.422 | +0.122 |

English / Spanish Recall@10: 0.776 / 0.815 (E5: 0.777 / 0.735). Queries better / equal / worse in
Recall@10: 9 / 69 / 3. The useful target (Recall@10 0.85) and the strong one are not reached; the
signal of the stop rule (Recall@10 +0.03, full-evidence +0.03, nDCG@10 >= 0.518) is not either.

**Why.** The model does not discriminate. It judges 97.2% of the relevant members necessary, but
also 51.8% of the others, so every window has several "necessary" members and the merge always
fills the 10 places (90 of 90 sets have 10): only 12.6% of the chosen chunks are relevant. The
oracle margin was 0.948 - 0.763 = 0.185 of Recall@10 and C took 0.026 of it (14%). 115 of the
10,762 expected answer lines were missing and one function was unreadable. The functions it
named most were "condition" (2,134 members) and "governing rule" (1,267).

In the four followed needs, full-evidence stays 0 for all 12 queries. N022 and N034/N035 improve
in some queries (N035 0.25 to 0.50 in two of three) and N034-en-1 falls from 0.50 to 0.25; the four
relevant chunks of N035 are in the population every time and the selector chooses one or two. N024
stays at 0.20, 0.20 and 0.00: none to two of its five relevant chunks are eligible.

## Round 4: closed-catalog legal reformulation

The model (`Qwen3-4B-Instruct-2507`, 4 bits, MLX) wrote 180 rewrites in 29 minutes (median 8.4 s):
R4-A, the facts of each question; R4-B, the facts and at most 3 entries copied from a catalog of
440 article titles and defined terms of the whole corpus (catalog fixed before any result). E5
ranked each rewrite, and that ranking was fused with E5's ranking of the original question by
RRF (k = 60, equal weights, not tuned). R4-A rewrote 85 of 90 questions (5 fell back to E5's
ranking, because the answer had no readable facts).

R4-B kept 222 concepts (0 for 1 query, 1 for 15, 2 for 15, 3 for 59) and rejected 34 that were not
literally in the catalog; in 143 of the 222 the words quoted from the question are in the
question. The most chosen were "payment service provider" (16), "Provision of crypto-asset
services by certain financial entities" (10) and "digital operational resilience" (9).

| | E5 | R4-A | R4-B |
|---|---|---|---|
| nDCG@10 | 0.528 | 0.520 | 0.503 |
| Recall@10 | 0.763 | 0.729 | 0.701 |
| Recall@20 | 0.816 | 0.814 | 0.799 |
| MRR@10 | 0.746 | 0.699 | 0.661 |
| Full-evidence@10 | 0.642 | 0.593 | 0.568 |
| `multi_provision` Recall@10 | 0.481 | 0.501 | 0.478 |
| `cross_reference` / `cross_act` Recall@10 | 0.661 / 0.292 | 0.652 / 0.292 | 0.629 / 0.250 |
| dependent evidence Recall@10 | 0.300 | 0.303 | 0.302 |

Paired with E5, Recall@10: R4-A -0.035 [-0.131, +0.038], R4-B -0.062 [-0.182, +0.036]; nDCG@10:
-0.008 [-0.040, +0.021] and -0.025 [-0.081, +0.028]. English / Spanish Recall@10: 0.713 / 0.760
and 0.705 / 0.693. Neither variant passes the gate (Recall@10 +0.03, nDCG@10 >= 0.518 and a loss
of full-evidence of at most 0.01): R4-A fails the first and the third, R4-B all three.

**First-hop failures move a little, and the rest pays for it.** Of the 20 relevant (query, chunk)
pairs in the 6 queries where E5's top 10 holds no grade-3 chunk, E5 finds 2, R4-A 5 and R4-B 6.
N024 goes from 0.13 to 0.20 in both variants: in N024-en-2, R4-B chose "Prohibition of insider
dealing" and moved MiCA 87 from rank 17 to 5 and MiCA 89.1 from 23 to 6, but in N024-en-1 it
chose "Scope", and in the Spanish query it chose "Prohibition of market manipulation" and "unlawful
disclosure of inside information". N026 goes from 0.17 to 0.33 and 0.50. But the pairs found in
the 69 queries where E5 already holds an answer chunk fall from 89 to 88 (R4-A) and 80 (R4-B).
The losses are concentrated: N027 falls from 1.00 to 0.00 in both variants (its single relevant
chunk, MiCA 143.3, goes from rank 4 to 15 once the rewrite's generic MiCA terms join the fusion),
N028 from 1.00 to 0.67 and 0.00 (R4-B's concepts "payment institution", "payment service
provider" and "payment services" pull headings and definitions above PSD2 7), and N023 falls to
0.17 in R4-B. A rewrite helps a question whose answer E5 cannot reach and dilutes one that E5
already ranks well, and equal-weight fusion cannot tell them apart.

## Outcome: the stop rule of the iteration

Neither C nor round 4 reaches the useful success (Recall@10 0.85, `multi_provision` 0.60,
dependent evidence 0.45, full-evidence 0.75), nor the registered minimum signal (Recall@10 +0.03
and full-evidence +0.03 with nDCG@10 >= 0.518): C gives +0.026, +0.025 and 0.460; R4-A -0.035,
-0.049 and 0.520; R4-B -0.062, -0.074 and 0.503. E5 stays the reference
([leaderboard](../retrieval-iteration/leaderboard.md)). The leaderboard orders C first by global
Recall@10, which is a mechanical ordering and not a decision: that gain is inside the noise (its
interval holds zero) and C worsens nDCG@10 by 0.067 with an interval that excludes zero.
