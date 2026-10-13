# Retrieval iteration 2 on DEV 0.10.0: the record

Every experiment of the iteration, in the order it ran, against the rules fixed before it
([protocol](../../../../config/retrieval/dependency-retrieval.md)); losers included. Global
values are macro means by need. The frozen inputs, the evaluator and the measures are those of
iteration 1. Reference, E5: global Recall@10 0.763, nDCG@10 0.528, `multi_provision` Recall@10
0.481, dependent evidence Recall@10 0.300.

## R1: anchor-conditioned dependency retrieval

E5's top 10 are the anchors; 166 defined terms (DORA 65, MiCA 53, PSD2 48) and the reference graph
give each chunk 5 relations at the median. Every query got 3 bridges, such as "What is payment
institution?" or "What is account information service provider?". The candidate list keeps E5's
top 20, so its top 20 is E5's.

| Candidate Recall | E5 @50 | R1, k = 3, @50 | E5 @100 | R1, k = 3, @100 |
|---|---|---|---|---|
| global | 0.863 | 0.893 | 0.883 | 0.947 |
| English / Spanish | 0.866 / 0.857 | 0.902 / 0.877 | 0.893 / 0.864 | 0.955 / 0.931 |
| `multi_provision` | 0.633 | 0.742 | 0.725 | 0.872 |
| `cross_reference` | 0.755 | 0.844 | 0.805 | 0.915 |
| `cross_act` (2 needs) | 0.417 | 0.792 | 0.542 | 0.917 |
| dependent evidence | 0.582 | 0.774 | 0.640 | 0.909 |

With k = 5, Candidate Recall@50 is 0.884. The pool holds 86 candidates at the median (at most
243). Of the relevant chunks it brings into positions 21 to 50, 15 came by a bridge, 13 by a
definition, 4 by a reference and 2 by a definition by reference.

**Gate.** Candidate Recall@50 0.893, under 0.95: R1's reranker was not tuned, and the iteration
went to research, which retained three approaches (protocol file) and chose a dense-seeded
Personalized PageRank over the law's graph first.

## E2: dense-seeded Personalized PageRank over the legal structure

10,582 edges (references, defined terms, definitions by reference).

| Seeds, follow | Recall@10 | nDCG@10 | Recall@50 | `multi_provision` | dependent |
|---|---|---|---|---|---|
| 10, 0.3 | 0.763 | 0.477 | 0.868 | 0.481 | 0.300 |
| 10, 0.5 | 0.763 | 0.446 | 0.868 | 0.481 | 0.300 |
| 20, 0.3 | 0.737 | 0.421 | 0.889 | 0.513 | 0.349 |
| 20, 0.5 | 0.651 | 0.338 | 0.886 | 0.465 | 0.291 |

With 10 seeds the top 10 is E5's own, in a worse order; with 20 it loses Recall@10. A seed keeps
its reset mass, which outweighs what its dependencies receive. In HippoRAG 2 the mass travels
through phrase nodes built by a language model, which this graph does not have. Not kept.

## E3: bridge-conditioned cross-encoder reranking

`BAAI/bge-reranker-v2-m3` on the top 50 of R1's list: 4,669 pairs, none truncated, 1,154 s on CPU.

| Rule | Recall@10 | nDCG@10 | MRR@10 | `multi_provision` | dependent | English / Spanish Recall@10 |
|---|---|---|---|---|---|---|
| original query | 0.762 | 0.527 | 0.711 | 0.455 | 0.279 | 0.764 / 0.756 |
| maximum with the bridges | 0.629 | 0.250 | 0.180 | 0.506 | 0.213 | 0.652 / 0.583 |

With the original query the cross-encoder orders this benchmark as E5 does (paired nDCG@10
-0.000 [-0.037, +0.042]). With the bridges, "What is X?" scores the definition of X above every
anchor, and definitions take the top 10. Not kept.

## E4: a fixed quota of the top 10 for the best dependency candidates

| Places | Recall@10 | nDCG@10 | `multi_provision` | dependent | paired Recall@10 |
|---|---|---|---|---|---|
| 2 | 0.747 | 0.528 | 0.462 | 0.288 | -0.016 [-0.053, +0.017] |
| 3 | 0.732 | 0.523 | 0.453 | 0.282 | -0.031 [-0.066, -0.000] |

Not kept.

## Outcome: stop rule C

Three approaches after R1 (graph propagation, cross-encoder reranking, a quota) bring no
material improvement; E5 stays the best retriever measured
([leaderboard](../retrieval-iteration/leaderboard.md)).

The bottleneck, measured: the dependent evidence is gathered but not ordered. Of the 66 (query,
dependent chunk) pairs of the 11 needs with an anchor and dependent evidence, R1's list holds 16
in its top 10, 10 at 11 to 20, 22 at 21 to 50, 11 at 51 to 100 and 7 beyond. Past E5's top 20 the
pool holds 66 candidates per query at the median, and only 2% of its positions 21 to 30 are
relevant, against 3.3% of E5's own positions 8 to 10. No score tried tells a needed definition,
condition or provision from the other neighbours of the anchor: neither the E5 cosine to the
query or a bridge, nor the cross-encoder with the query or a bridge, nor the graph.
