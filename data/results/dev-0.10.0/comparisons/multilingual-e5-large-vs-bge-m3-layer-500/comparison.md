# Paired comparison: multilingual-e5-large against bge-m3, DEV 0.10.0, layer 500

Same benchmark, chunk layer, queries, qrels, projection and evaluator; only the model changes. Deltas are B minus A. Intervals: bootstrap over needs, 10,000 resamples, seed 20261013.

## Global

| Scope | Metric | multilingual-e5-large | bge-m3 | Delta | Relative |
|---|---|---|---|---|---|
| EN (official) | nDCG@10 | 0.552 | 0.536 | -0.016 | -2.9% |
| EN (official) | Recall@10 | 0.777 | 0.761 | -0.016 | -2.1% |
| EN (official) | MRR@10 | 0.772 | 0.756 | -0.016 | -2.1% |
| EN (official) | Recall@5 | 0.714 | 0.680 | -0.035 | -4.8% |
| EN (official) | Recall@20 | 0.820 | 0.823 | +0.003 | +0.4% |
| EN (official) | Recall@T=1000 | 0.742 | 0.673 | -0.069 | -9.3% |
| EN (official) | Recall@T=2000 | 0.798 | 0.777 | -0.021 | -2.6% |
| ES (official) | nDCG@10 | 0.479 | 0.498 | +0.019 | +4.0% |
| ES (official) | Recall@10 | 0.735 | 0.723 | -0.011 | -1.5% |
| ES (official) | MRR@10 | 0.694 | 0.687 | -0.007 | -1.1% |
| ES (official) | Recall@5 | 0.648 | 0.600 | -0.048 | -7.4% |
| ES (official) | Recall@20 | 0.809 | 0.798 | -0.011 | -1.4% |
| ES (official) | Recall@T=1000 | 0.554 | 0.622 | +0.068 | +12.2% |
| ES (official) | Recall@T=2000 | 0.748 | 0.732 | -0.017 | -2.2% |
| macro by need (descriptive) | nDCG@10 | 0.528 | 0.523 | -0.004 | -0.8% |
| macro by need (descriptive) | Recall@10 | 0.763 | 0.749 | -0.015 | -1.9% |
| macro by need (descriptive) | MRR@10 | 0.746 | 0.733 | -0.013 | -1.8% |
| macro by need (descriptive) | Recall@5 | 0.692 | 0.653 | -0.039 | -5.6% |
| macro by need (descriptive) | Recall@20 | 0.816 | 0.814 | -0.002 | -0.2% |
| macro by need (descriptive) | Recall@T=1000 | 0.680 | 0.656 | -0.023 | -3.4% |
| macro by need (descriptive) | Recall@T=2000 | 0.782 | 0.762 | -0.020 | -2.5% |

## By query style

| Style | Metric | multilingual-e5-large | bge-m3 | Delta |
|---|---|---|---|---|
| en formal | nDCG@10 | 0.596 | 0.587 | -0.009 |
| en formal | Recall@10 | 0.807 | 0.810 | +0.003 |
| en formal | MRR@10 | 0.826 | 0.831 | +0.004 |
| en formal | Recall@5 | 0.742 | 0.722 | -0.020 |
| en formal | Recall@20 | 0.850 | 0.850 | +0.000 |
| en formal | Recall@T=1000 | 0.799 | 0.766 | -0.033 |
| en formal | Recall@T=2000 | 0.841 | 0.847 | +0.006 |
| en natural | nDCG@10 | 0.508 | 0.484 | -0.024 |
| en natural | Recall@10 | 0.748 | 0.712 | -0.036 |
| en natural | MRR@10 | 0.719 | 0.682 | -0.037 |
| en natural | Recall@5 | 0.686 | 0.637 | -0.049 |
| en natural | Recall@20 | 0.790 | 0.796 | +0.006 |
| en natural | Recall@T=1000 | 0.686 | 0.581 | -0.105 |
| en natural | Recall@T=2000 | 0.755 | 0.708 | -0.048 |
| es natural | nDCG@10 | 0.479 | 0.498 | +0.019 |
| es natural | Recall@10 | 0.735 | 0.723 | -0.011 |
| es natural | MRR@10 | 0.694 | 0.687 | -0.007 |
| es natural | Recall@5 | 0.648 | 0.600 | -0.048 |
| es natural | Recall@20 | 0.809 | 0.798 | -0.011 |
| es natural | Recall@T=1000 | 0.554 | 0.622 | +0.068 |
| es natural | Recall@T=2000 | 0.748 | 0.732 | -0.017 |

## By type, difficulty, answerability and driver (macro by need)

| Group | Value | Needs | nDCG@10 multilingual-e5-large | nDCG@10 bge-m3 | Recall@10 multilingual-e5-large | Recall@10 bge-m3 |
|---|---|---|---|---|---|---|
| type | direct | 11 | 0.529 | 0.546 | 0.869 | 0.828 |
| type | exception_confusion | 7 | 0.590 | 0.538 | 0.833 | 0.810 |
| type | multi_provision | 6 | 0.502 | 0.477 | 0.481 | 0.461 |
| type | partial_out_of_scope (descriptive) | 3 | 0.428 | 0.498 | 0.778 | 0.889 |
| difficulty | easy | 7 | 0.506 | 0.521 | 0.706 | 0.730 |
| difficulty | hard | 6 | 0.485 | 0.429 | 0.582 | 0.517 |
| difficulty | medium | 14 | 0.556 | 0.565 | 0.869 | 0.857 |
| answerability | complete | 24 | 0.540 | 0.526 | 0.761 | 0.731 |
| answerability | partial (descriptive) | 3 | 0.428 | 0.498 | 0.778 | 0.889 |
| driver | (none) (descriptive) | 1 | 0.525 | 0.546 | 1.000 | 1.000 |
| driver | cross_act (descriptive) | 2 | 0.469 | 0.468 | 0.292 | 0.250 |
| driver | cross_reference | 11 | 0.528 | 0.553 | 0.661 | 0.640 |
| driver | date (descriptive) | 1 | 0.299 | 0.613 | 1.000 | 1.000 |
| driver | exception | 14 | 0.597 | 0.573 | 0.802 | 0.764 |
| driver | level2_gap (descriptive) | 1 | 0.535 | 0.588 | 1.000 | 1.000 |
| driver | lexical_twin | 17 | 0.479 | 0.533 | 0.714 | 0.732 |
| driver | negation | 7 | 0.477 | 0.448 | 0.638 | 0.633 |
| driver | numeric | 10 | 0.565 | 0.554 | 0.839 | 0.839 |

## Paired, per query

Mean of the per-query deltas. 'all' mixes languages and is descriptive; 'en' and 'es' follow D-020. Wins, ties and losses count queries (ties within 1e-9).

| Metric | Queries | Mean delta | 95% interval | bge-m3 wins | Ties | multilingual-e5-large wins |
|---|---|---|---|---|---|---|
| nDCG@10 | all | -0.004 | [-0.084, +0.072] | 38 | 5 | 38 |
| nDCG@10 | en | -0.016 | [-0.097, +0.065] | 21 | 4 | 29 |
| nDCG@10 | es | +0.019 | [-0.073, +0.107] | 17 | 1 | 9 |
| nDCG@10 | all, without N012, N026, N029 | +0.020 | [-0.040, +0.082] |  |  |  |
| Recall@10 | all | -0.015 | [-0.087, +0.054] | 8 | 64 | 9 |
| Recall@10 | en | -0.016 | [-0.069, +0.031] | 4 | 44 | 6 |
| Recall@10 | es | -0.011 | [-0.148, +0.126] | 4 | 20 | 3 |
| Recall@10 | all, without N029, N015, N018 | -0.016 | [-0.061, +0.029] |  |  |  |
| MRR@10 | all | -0.013 | [-0.149, +0.128] | 20 | 42 | 19 |
| MRR@10 | en | -0.016 | [-0.172, +0.145] | 12 | 29 | 13 |
| MRR@10 | es | -0.007 | [-0.160, +0.143] | 8 | 13 | 6 |
| MRR@10 | all, without N012, N026, N027 | -0.046 | [-0.159, +0.067] |  |  |  |

Per need (mean over its three queries), B minus A:

| Metric | Needs: bge-m3 better / equal / multilingual-e5-large better | Largest losses | Largest gains | Share of all gains from the top 3 needs |
|---|---|---|---|---|
| nDCG@10 | 15 / 0 / 12 | N012 -0.62, N029 -0.33, N030 -0.26, N021 -0.24 | N015 +0.24, N023 +0.30, N027 +0.31, N026 +0.35 | 0.481 |
| Recall@10 | 4 / 17 / 6 | N029 -0.67, N030 -0.33, N022 -0.17, N031 -0.17 | N024 +0.13, N015 +0.33, N018 +0.33, N026 +0.33 | 0.882 |

## Anchor found, dependent evidence missed

Needs whose relevant chunks include a grade-3 anchor and at least one grade-2 dependent chunk.

| Model | Needs | Queries | Anchor in top 10 | Dependent Recall@10 | Dependent median rank | Anchor found, a dependent missed |
|---|---|---|---|---|---|---|
| multilingual-e5-large | 11 | 33 | 0.848 | 0.300 | 40 | 0.606 |
| bge-m3 | 11 | 33 | 0.879 | 0.293 | 37 | 0.667 |

The groups of the core hypothesis (macro by need):

| Group | Needs | Metric | multilingual-e5-large | bge-m3 | Delta | 95% interval |
|---|---|---|---|---|---|---|
| type multi_provision | 6 | nDCG@10 | 0.502 | 0.477 | -0.025 | [-0.119, +0.050] |
| type multi_provision | 6 | Recall@10 | 0.481 | 0.461 | -0.019 | [-0.097, +0.053] |
| driver cross_act (descriptive) | 2 | nDCG@10 | 0.469 | 0.468 | -0.001 | [-0.019, +0.016] |
| driver cross_act (descriptive) | 2 | Recall@10 | 0.292 | 0.250 | -0.042 | [-0.083, +0.000] |
| driver cross_reference | 11 | nDCG@10 | 0.528 | 0.553 | +0.025 | [-0.051, +0.099] |
| driver cross_reference | 11 | Recall@10 | 0.661 | 0.640 | -0.021 | [-0.066, +0.024] |
| driver lexical_twin | 17 | nDCG@10 | 0.479 | 0.533 | +0.054 | [-0.028, +0.137] |
| driver lexical_twin | 17 | Recall@10 | 0.714 | 0.732 | +0.018 | [-0.034, +0.083] |

The needs under review (mean over their three queries):

| Need | Metric | multilingual-e5-large | bge-m3 | Delta |
|---|---|---|---|---|
| N022 | nDCG@10 | 0.420 | 0.189 | -0.231 |
| N022 | Recall@10 | 0.278 | 0.111 | -0.167 |
| N024 | nDCG@10 | 0.127 | 0.179 | +0.052 |
| N024 | Recall@10 | 0.133 | 0.267 | +0.133 |
| N026 | nDCG@10 | 0.048 | 0.398 | +0.350 |
| N026 | Recall@10 | 0.167 | 0.500 | +0.333 |
| N034 | nDCG@10 | 0.531 | 0.548 | +0.016 |
| N034 | Recall@10 | 0.333 | 0.250 | -0.083 |
| N035 | nDCG@10 | 0.407 | 0.388 | -0.019 |
| N035 | Recall@10 | 0.250 | 0.250 | +0.000 |

## Failure transitions

A query fails when some relevant chunk is outside its top 10 (Recall@10 under 1; no need has more than 6 relevant chunks). A fixed: fails under A, not under B. D worse (a new failure): fails under B only. When it fails under both: D worse if Recall@10 is lower, or equal with nDCG@10 down by more than 0.05; B improved but still failing if Recall@10 is higher, or equal with nDCG@10 up by more than 0.05; otherwise C same failure. The rule is symmetric between the models.

| Query | Transition | First relevant | Recall@10 | nDCG@10 | Gained in top 10 | Lost from top 10 |
|---|---|---|---|---|---|---|
| N026-es-1 | B improved but still failing | 40 / 2 | 0.000 / 0.500 | 0.000 / 0.275 | MiCA:83.1 | - |
| N024-es-1 | B improved but still failing | 29 / 4 | 0.000 / 0.200 | 0.000 / 0.172 | MiCA:89.1 | - |
| N015-en-2 | B improved but still failing | 21 / 7 | 0.000 / 0.500 | 0.000 / 0.194 | DORA:50.4 | - |
| N023-en-2 | B improved but still failing | 8 / 1 | 0.500 / 0.500 | 0.182 / 0.578 | - | - |
| N035-en-1 | C same failure | 1 / 1 | 0.250 / 0.250 | 0.432 / 0.423 | - | - |
| N035-en-2 | C same failure | 1 / 1 | 0.250 / 0.250 | 0.418 / 0.370 | - | - |
| N035-es-1 | C same failure | 1 / 1 | 0.250 / 0.250 | 0.370 / 0.370 | - | - |
| N034-en-1 | D worse | 2 / 1 | 0.500 / 0.250 | 0.555 / 0.597 | - | DORA:46.b |
| N034-en-2 | C same failure | 1 / 1 | 0.250 / 0.250 | 0.531 / 0.523 | - | - |
| N034-es-1 | C same failure | 1 / 1 | 0.250 / 0.250 | 0.508 / 0.523 | - | - |
| N022-en-1 | D worse | 1 / 1 | 0.333 / 0.167 | 0.547 / 0.396 | - | MiCA:46.4 |
| N022-en-2 | D worse | 2 / 34 | 0.333 / 0.000 | 0.329 / 0.000 | - | MiCA:39, MiCA:46.1 |
| N022-es-1 | D worse | 1 / 5 | 0.167 / 0.167 | 0.384 / 0.171 | - | - |

All 81 queries: A fixed by B 3, B improved but still failing 11, C same failure 6, D worse 9, D worse (a new failure) 5, not failing under either 47.

## Language effect, per model

Paired per need.

| Model | Metric | Difference | Mean | 95% interval |
|---|---|---|---|---|
| multilingual-e5-large | nDCG@10 | en formal minus en natural | +0.088 | [+0.046, +0.132] |
| multilingual-e5-large | nDCG@10 | es natural minus en natural | -0.029 | [-0.094, +0.033] |
| multilingual-e5-large | Recall@10 | en formal minus en natural | +0.059 | [+0.009, +0.120] |
| multilingual-e5-large | Recall@10 | es natural minus en natural | -0.014 | [-0.117, +0.079] |
| multilingual-e5-large | Recall@T=1000 | en formal minus en natural | +0.113 | [+0.025, +0.224] |
| multilingual-e5-large | Recall@T=1000 | es natural minus en natural | -0.132 | [-0.255, -0.029] |
| bge-m3 | nDCG@10 | en formal minus en natural | +0.103 | [+0.051, +0.162] |
| bge-m3 | nDCG@10 | es natural minus en natural | +0.014 | [-0.034, +0.063] |
| bge-m3 | Recall@10 | en formal minus en natural | +0.098 | [+0.006, +0.210] |
| bge-m3 | Recall@10 | es natural minus en natural | +0.011 | [-0.089, +0.098] |
| bge-m3 | Recall@T=1000 | en formal minus en natural | +0.185 | [+0.056, +0.333] |
| bge-m3 | Recall@T=1000 | es natural minus en natural | +0.041 | [+0.000, +0.107] |
| bge-m3 minus multilingual-e5-large | nDCG@10 | change of (ES natural - EN natural) | +0.043 | [-0.025, +0.110] |
| bge-m3 minus multilingual-e5-large | Recall@10 | change of (ES natural - EN natural) | +0.025 | [-0.099, +0.148] |
| bge-m3 minus multilingual-e5-large | Recall@T=1000 | change of (ES natural - EN natural) | +0.173 | [+0.047, +0.321] |

## Truncation as a control

Truncated chunks: multilingual-e5-large 26, bge-m3 0. Appearances of multilingual-e5-large's truncated chunks in a top 10: {'multilingual-e5-large': 61, 'bge-m3': 33}.

Every (query, relevant chunk) pair of the 81 queries, by whether A truncated the chunk:

| Pairs | Count | Better rank in bge-m3 | Same | Worse in bge-m3 | Median rank multilingual-e5-large | Median rank bge-m3 |
|---|---|---|---|---|---|---|
| relevant truncated in a | 12 | 0 | 4 | 8 | 1.0 | 4.5 |
| relevant not truncated | 147 | 53 | 40 | 54 | 5 | 6 |

Length of the chunks that fill the top 10 of the 81 queries (cl100k tokens of the official layer, the same for both models). Median: multilingual-e5-large 194.0, bge-m3 163.0.

| Tokens | Share of the corpus | Share of multilingual-e5-large's top 10 | Share of bge-m3's top 10 |
|---|---|---|---|
| 0-99 | 0.490 | 0.133 | 0.232 |
| 100-199 | 0.321 | 0.402 | 0.417 |
| 200-299 | 0.105 | 0.157 | 0.153 |
| 300-399 | 0.048 | 0.168 | 0.093 |
| 400-500 | 0.036 | 0.140 | 0.105 |

The needs where A truncated a relevant chunk:

| Need | Query | Relevant chunk | Rank multilingual-e5-large | Rank bge-m3 |
|---|---|---|---|---|
| N009 | N009-en-1 | MiCA:59 (truncated in A) | 1 | 6 |
| N009 | N009-en-1 | MiCA:61 | 2 | 1 |
| N009 | N009-en-2 | MiCA:59 (truncated in A) | 1 | 4 |
| N009 | N009-en-2 | MiCA:61 | 3 | 1 |
| N009 | N009-es-1 | MiCA:59 (truncated in A) | 1 | 3 |
| N009 | N009-es-1 | MiCA:61 | 4 | 1 |
| N022 | N022-en-1 | MiCA:3.1.6 | 79 | 36 |
| N022 | N022-en-1 | MiCA:3.1.7 | 264 | 25 |
| N022 | N022-en-1 | MiCA:39 (truncated in A) | 1 | 1 |
| N022 | N022-en-1 | MiCA:46.1 | 13 | 14 |
| N022 | N022-en-1 | MiCA:46.3 | 98 | 188 |
| N022 | N022-en-1 | MiCA:46.4 | 7 | 18 |
| N022 | N022-en-2 | MiCA:3.1.6 | 337 | 246 |
| N022 | N022-en-2 | MiCA:3.1.7 | 633 | 107 |
| N022 | N022-en-2 | MiCA:39 (truncated in A) | 2 | 34 |
| N022 | N022-en-2 | MiCA:46.1 | 8 | 119 |
| N022 | N022-en-2 | MiCA:46.3 | 105 | 518 |
| N022 | N022-en-2 | MiCA:46.4 | 11 | 78 |
| N022 | N022-es-1 | MiCA:3.1.6 | 345 | 155 |
| N022 | N022-es-1 | MiCA:3.1.7 | 462 | 56 |
| N022 | N022-es-1 | MiCA:39 (truncated in A) | 1 | 5 |
| N022 | N022-es-1 | MiCA:46.1 | 17 | 66 |
| N022 | N022-es-1 | MiCA:46.3 | 74 | 474 |
| N022 | N022-es-1 | MiCA:46.4 | 44 | 45 |
| N024 | N024-en-1 | MiCA:86 | 1 | 7 |
| N024 | N024-en-1 | MiCA:87 (truncated in A) | 63 | 93 |
| N024 | N024-en-1 | MiCA:89.1 | 94 | 38 |
| N024 | N024-en-1 | MiCA:89.2 | 138 | 146 |
| N024 | N024-en-1 | MiCA:89.5 | 407 | 232 |
| N024 | N024-en-2 | MiCA:86 | 4 | 10 |
| N024 | N024-en-2 | MiCA:87 (truncated in A) | 17 | 82 |
| N024 | N024-en-2 | MiCA:89.1 | 23 | 3 |
| N024 | N024-en-2 | MiCA:89.2 | 50 | 43 |
| N024 | N024-en-2 | MiCA:89.5 | 363 | 194 |
| N024 | N024-es-1 | MiCA:86 | 31 | 42 |
| N024 | N024-es-1 | MiCA:87 (truncated in A) | 29 | 60 |
| N024 | N024-es-1 | MiCA:89.1 | 107 | 4 |
| N024 | N024-es-1 | MiCA:89.2 | 150 | 85 |
| N024 | N024-es-1 | MiCA:89.5 | 490 | 309 |
| N031 | N031-en-1 | PSD2:71 | 19 | 29 |
| N031 | N031-en-1 | PSD2:74 (truncated in A) | 1 | 1 |
| N031 | N031-en-2 | PSD2:71 | 48 | 37 |
| N031 | N031-en-2 | PSD2:74 (truncated in A) | 1 | 1 |
| N031 | N031-es-1 | PSD2:71 | 9 | 23 |
| N031 | N031-es-1 | PSD2:74 (truncated in A) | 1 | 1 |

## Performance

Same machine, measured back to back in one session, one run per model: the variance between runs is not measured. Times exclude system sleep. Queries: the 90 queries in one batched pass, not a per-query latency. Peak RAM is the whole process.

| Model | Load (s) | Documents (s) | Queries (s) | Retrieval (s) | Dimension | Vectors (MB) | Peak RAM (GB) | Model cache (GB) |
|---|---|---|---|---|---|---|---|---|
| multilingual-e5-large | 3.1 | 191.0 | 5.4 | 0.030 | 1024 | 5.55 | 2.82 | 2.26 |
| bge-m3 | 3.6 | 200.9 | 5.3 | 0.034 | 1024 | 5.55 | 2.70 | 2.29 |
