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
