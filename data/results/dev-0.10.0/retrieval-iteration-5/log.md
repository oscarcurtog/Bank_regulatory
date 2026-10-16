# Retrieval iteration 5 on DEV 0.10.0: the record

Three changes fixed before F's DEV result was read ([protocol](../../../../config/retrieval/iteration-5.md),
`513d78e`, amended after an independent audit in `1743122` and `ab9e661`, all before any experiment
ran), each evaluated once on DEV in the order G1, G2, G3, under the person's authorization for an
unattended night. The reranker is F's adapter (chosen on VAL-REPORT, where its training passed).
F's own DEV result, read after the menu was fixed: Recall@10 0.805, below E's 0.824. Every variant
is paired with E5, with E and with the working reference, which starts as F.

## G1: the four places by the best normalized score, not in turns

No new scoring: the 15,350 pairs of F's DEV run were carried, each checked against its input, and
F's recorded ranking was rebuilt exactly from them before anything changed.

| | E5 | E | F (working reference) | G1 |
|---|---|---|---|---|
| Recall@10 | 0.763 | 0.824 | 0.805 | 0.815 |
| nDCG@10 | 0.528 | 0.561 | 0.564 | 0.564 |
| MRR@10 | 0.746 | 0.754 | 0.752 | 0.754 |
| Full-evidence@10 | 0.642 | 0.679 | 0.667 | 0.679 |
| `multi_provision` Recall@10 | 0.481 | 0.617 | 0.583 | 0.574 |
| dependent evidence Recall@10 | 0.300 | 0.409 | 0.367 | 0.391 |

Paired with F: Recall@10 +0.010 [-0.004, +0.031], nDCG@10 +0.000 [-0.007, +0.007], full-evidence
+0.012 [0.000, +0.037]. With E: Recall@10 -0.009 [-0.035, +0.015], full-evidence 0.000. With E5:
Recall@10 +0.052 [0.000, +0.118], nDCG@10 +0.036 [+0.018, +0.057]. English / Spanish Recall@10:
0.821 / 0.802.

**G1 does not replace the working reference** (it needed Recall@10 +0.02 or full-evidence +0.03):
filling the places by the best normalized score moves the ten in the expected direction, by less
than the rule asks, and stays below E.

## G2: the requirements of both decomposers, interleaved

D-8B's recorded ranking was rebuilt exactly from its requirements before anything ran, and F's
again. 17,250 new pairs (the 8B's requirements' first 50) were scored by F's adapter and 15,350
carried; 32,800 pairs ranked over the 90 queries.

| | E5 | E | F (working reference) | G2 |
|---|---|---|---|---|
| Recall@10 | 0.763 | 0.824 | 0.805 | 0.814 |
| nDCG@10 | 0.528 | 0.561 | 0.564 | 0.565 |
| MRR@10 | 0.746 | 0.754 | 0.752 | 0.751 |
| Recall@20 | 0.816 | 0.863 | 0.835 | 0.848 |
| Full-evidence@10 | 0.642 | 0.679 | 0.667 | 0.691 |
| `multi_provision` Recall@10 | 0.481 | 0.617 | 0.583 | 0.572 |
| dependent evidence Recall@10 | 0.300 | 0.409 | 0.367 | 0.427 |

Paired with F: Recall@10 +0.010 [-0.005, +0.030], nDCG@10 +0.001 [-0.007, +0.010], full-evidence
+0.025 [0.000, +0.062]. With E: Recall@10 -0.010 [-0.044, +0.025], full-evidence +0.012
[-0.025, +0.049]. With E5: Recall@10 +0.051 [+0.003, +0.114], nDCG@10 +0.038 [+0.020, +0.056],
full-evidence +0.049 [0.000, +0.111]. English / Spanish Recall@10: 0.815 / 0.813.

**G2 does not replace the working reference**: full-evidence rises by 0.024 at 3 decimals, short of
0.03, and Recall@10 by 0.010. More requirements raise dependent evidence (0.427, against F's 0.367
and E's 0.409) and full-evidence, and leave Recall@10 below E's.

## G3: each requirement's first 100 reranked, not 50

The working reference is still F; its ranking was rebuilt exactly from the carried scores before
anything ran. 15,350 new pairs (ranks 51 to 100 of each 4B requirement's E5 list) were scored by
F's adapter in 83 minutes and 15,350 carried; 30,800 pairs ranked over the 90 queries. Each
requirement's softmax covers its 100.

| | E5 | E | F (working reference) | G3 |
|---|---|---|---|---|
| Recall@10 | 0.763 | 0.824 | 0.805 | 0.801 |
| nDCG@10 | 0.528 | 0.561 | 0.564 | 0.566 |
| MRR@10 | 0.746 | 0.754 | 0.752 | 0.750 |
| Recall@20 | 0.816 | 0.863 | 0.835 | 0.838 |
| Full-evidence@10 | 0.642 | 0.679 | 0.667 | 0.667 |
| `multi_provision` Recall@10 | 0.481 | 0.617 | 0.583 | 0.597 |
| dependent evidence Recall@10 | 0.300 | 0.409 | 0.367 | 0.377 |

Paired with F: Recall@10 -0.003 [-0.019, +0.009] (1 query better, 1 worse, 79 equal), nDCG@10
+0.002 [-0.004, +0.008] (13 better, 6 worse), full-evidence 0.000 (no query changes). With E:
Recall@10 -0.023 [-0.058, +0.006], full-evidence -0.012 [-0.037, 0.000]. With E5: Recall@10 +0.038
[-0.010, +0.095], nDCG@10 +0.038 [+0.021, +0.058]. English / Spanish Recall@10: 0.816 / 0.773.

**G3 does not replace the working reference.** Reranking 100 candidates instead of 50 changes
Recall@10 for 2 of the 81 queries and nDCG@10 for 19: the relevant chunks beyond rank 50 of a
requirement's E5 list almost never reach the places the rule fills.

## Outcome of the night

The three evaluations are spent, so the iteration stops. No variant replaced the working reference,
and none reaches the useful target (Recall@10 0.85, `multi_provision` 0.60, dependent evidence
0.45 and full-evidence 0.75 at once).

| | E5 | E | F | G1 | G2 | G3 |
|---|---|---|---|---|---|---|
| Recall@10 | 0.763 | 0.824 | 0.805 | 0.815 | 0.814 | 0.801 |
| nDCG@10 | 0.528 | 0.561 | 0.564 | 0.564 | 0.565 | 0.566 |
| Full-evidence@10 | 0.642 | 0.679 | 0.667 | 0.679 | 0.691 | 0.667 |
| `multi_provision` Recall@10 | 0.481 | 0.617 | 0.583 | 0.574 | 0.572 | 0.597 |
| dependent evidence Recall@10 | 0.300 | 0.409 | 0.367 | 0.391 | 0.427 | 0.377 |

Read on DEV, and so as development evidence only:
- the three changes to the merge move Recall@10 by 0.010 or less against F: filling the places by
  score (G1), adding a second decomposer's questions (G2) and reranking deeper lists (G3);
- G2 raises full-evidence and dependent evidence the most, in the direction D-8B showed;
- no variant reaches E's Recall@10 (G1 -0.009, G2 -0.010, G3 -0.023 against E). The variants were
  built on F, chosen on synthetic validation before F's DEV result was read, and F is 0.020 below E;
- every difference among E, F, G1, G2 and G3 lies inside the noise of 27 needs.

With the night over, F's stop rule, which the authorization had suspended, applies again: E
(zero-shot) is the retrieval candidate and the project returns to T3 and T4, unless the person
decides otherwise. None of these numbers has been checked on new needs.
