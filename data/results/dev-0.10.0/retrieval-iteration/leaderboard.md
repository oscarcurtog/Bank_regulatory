# Retrieval leaderboard, DEV 0.10.0

Every system ranks all 1,354 chunks for the 81 scored queries; one code path measures them all (`ingest/run_report.py`). Global values are macro means by need over the 27 evaluable needs, mixing English and Spanish (descriptive under D-020). `cross_act` has 2 needs and is a diagnostic only. Dependent evidence: the grade-2 chunks of the 11 needs that also have a grade-3 chunk.

| System | nDCG@10 | Recall@10 | MRR@10 | Recall@20 | Recall@T=1000 | Recall@T=2000 | Recall@50 | multi_provision R@10 | cross_reference R@10 | cross_act R@10 | dependent R@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense, E5 (reference) | 0.528 | 0.763 | 0.746 | 0.816 | 0.680 | 0.782 | 0.863 | 0.481 | 0.661 | 0.292 | 0.300 |
| dense, BGE-M3 | 0.523 | 0.749 | 0.733 | 0.814 | 0.656 | 0.762 | 0.886 | 0.461 | 0.640 | 0.250 | 0.293 |
| BM25 | 0.229 | 0.366 | 0.277 | 0.437 | 0.303 | 0.390 | 0.521 | 0.222 | 0.278 | 0.083 | 0.067 |
| E5 + one-hop references, K = 5 | 0.493 | 0.653 | 0.688 | 0.693 | 0.612 | 0.664 | 0.801 | 0.548 | 0.617 | 0.500 | 0.380 |
| E5 + one-hop references, K = 10 | 0.492 | 0.640 | 0.687 | 0.680 | 0.612 | 0.664 | 0.783 | 0.548 | 0.617 | 0.500 | 0.380 |
| dense, Qwen3-Embedding-0.6B | 0.483 | 0.730 | 0.660 | 0.804 | 0.666 | 0.800 | 0.894 | 0.394 | 0.624 | 0.333 | 0.347 |
| R1: E5 top 20 + dependency pool (candidate list) | 0.528 | 0.763 | 0.746 | 0.816 | 0.680 | 0.782 | 0.893 | 0.481 | 0.661 | 0.292 | 0.300 |
| E2: dense-seeded PageRank, 10 seeds, follow 0.3 | 0.477 | 0.763 | 0.575 | 0.818 | 0.596 | 0.756 | 0.868 | 0.481 | 0.661 | 0.292 | 0.300 |
| E3: cross-encoder on R1's top 50, original query | 0.527 | 0.762 | 0.711 | 0.813 | 0.681 | 0.750 | 0.893 | 0.455 | 0.642 | 0.208 | 0.279 |
| E3: cross-encoder, bridge maximum | 0.250 | 0.629 | 0.180 | 0.820 | 0.458 | 0.733 | 0.893 | 0.506 | 0.498 | 0.458 | 0.213 |
| E4: 2 top-10 places for dependencies | 0.528 | 0.747 | 0.745 | 0.816 | 0.680 | 0.769 | 0.893 | 0.462 | 0.666 | 0.292 | 0.288 |

Per language (official, D-020), nDCG@10 and Recall@10:

| System | EN nDCG@10 | ES nDCG@10 | EN Recall@10 | ES Recall@10 |
|---|---|---|---|---|
| dense, E5 (reference) | 0.552 | 0.479 | 0.777 | 0.735 |
| dense, BGE-M3 | 0.536 | 0.498 | 0.761 | 0.723 |
| BM25 | 0.311 | 0.065 | 0.471 | 0.157 |
| E5 + one-hop references, K = 5 | 0.508 | 0.464 | 0.680 | 0.599 |
| E5 + one-hop references, K = 10 | 0.505 | 0.465 | 0.661 | 0.599 |
| dense, Qwen3-Embedding-0.6B | 0.519 | 0.412 | 0.757 | 0.674 |
| R1: E5 top 20 + dependency pool (candidate list) | 0.552 | 0.479 | 0.777 | 0.735 |
| E2: dense-seeded PageRank, 10 seeds, follow 0.3 | 0.511 | 0.410 | 0.777 | 0.735 |
| E3: cross-encoder on R1's top 50, original query | 0.545 | 0.492 | 0.764 | 0.756 |
| E3: cross-encoder, bridge maximum | 0.271 | 0.210 | 0.652 | 0.583 |
| E4: 2 top-10 places for dependencies | 0.557 | 0.471 | 0.771 | 0.698 |

Best by global Recall@10, then nDCG@10: dense, E5 (reference).

Each system against E5, paired by query (81 queries), with a 95% bootstrap interval over needs, and the protocol's four tests of an interesting configuration (A to D):

| System | Recall@10 delta | nDCG@10 delta | Queries better / equal / worse (nDCG@10) | Interesting |
|---|---|---|---|---|
| dense, BGE-M3 | -0.015 [-0.087, +0.054] | -0.004 [-0.084, +0.072] | 38 / 5 / 38 | none |
| BM25 | -0.397 [-0.472, -0.322] | -0.299 [-0.355, -0.243] | 7 / 6 / 68 | none |
| E5 + one-hop references, K = 5 | -0.110 [-0.206, -0.021] | -0.034 [-0.088, +0.021] | 30 / 10 / 41 | none |
| E5 + one-hop references, K = 10 | -0.123 [-0.235, -0.024] | -0.036 [-0.091, +0.020] | 30 / 10 / 41 | none |
| dense, Qwen3-Embedding-0.6B | -0.034 [-0.127, +0.050] | -0.044 [-0.106, +0.019] | 30 / 5 / 46 | none |
| R1: E5 top 20 + dependency pool (candidate list) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0 / 81 / 0 | none |
| E2: dense-seeded PageRank, 10 seeds, follow 0.3 | +0.000 [+0.000, +0.000] | -0.050 [-0.081, -0.021] | 21 / 20 / 40 | none |
| E3: cross-encoder on R1's top 50, original query | -0.002 [-0.031, +0.033] | -0.000 [-0.037, +0.042] | 41 / 8 / 32 | none |
| E3: cross-encoder, bridge maximum | -0.134 [-0.232, -0.040] | -0.277 [-0.354, -0.201] | 12 / 2 / 67 | none |
| E4: 2 top-10 places for dependencies | -0.016 [-0.053, +0.017] | +0.001 [-0.010, +0.011] | 18 / 49 / 14 | none |

Decision 5: stop testing dense embedding models: model swapping has diminishing returns (third model -0.033 global Recall@10 against E5; the rule asks for +0.05).

Reranker gate (Recall@50 >= 0.95): dense, E5 (reference) 0.863, dense, BGE-M3 0.886, BM25 0.521, E5 + one-hop references, K = 5 0.801, E5 + one-hop references, K = 10 0.783, dense, Qwen3-Embedding-0.6B 0.894, R1: E5 top 20 + dependency pool (candidate list) 0.893, E2: dense-seeded PageRank, 10 seeds, follow 0.3 0.868, E3: cross-encoder on R1's top 50, original query 0.893, E3: cross-encoder, bridge maximum 0.893, E4: 2 top-10 places for dependencies 0.893; met by none.
