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
