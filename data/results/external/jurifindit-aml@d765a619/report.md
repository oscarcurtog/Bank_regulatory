# T4: the frozen retrieval on JuriFindIT's anti-money-laundering questions

Protocol: `config/external/jurifindit-aml.md`. Dataset `jurifindit/JuriFindIT` at `d765a61998a0`. One run per system, frozen before any qrel was read (`runs.json`). **validation-27 is the external result; pooled-108 is a secondary analysis, not an official validation score.** JuriFindIT's numbers are not comparable with Fundamento's: different corpus, questions, labels and units.

## validation-27 (27 questions)

Official metrics (JuriFindIT):

| System | Recall@5 | Recall@20 | Recall@60 | Recall@100 | nDCG@5 | nDCG@100 | MRR@5 | MRR@100 | mAP@5 | mAP@100 |
|---|---|---|---|---|---|---|---|---|---|---|
| BM25 (Italian) | 0.056 | 0.127 | 0.186 | 0.254 | 0.132 | 0.171 | 0.191 | 0.222 | 0.047 | 0.077 |
| E5 | 0.100 | 0.144 | 0.191 | 0.234 | 0.169 | 0.178 | 0.264 | 0.278 | 0.073 | 0.088 |

Our compatible metrics:

| System | Recall@10 | nDCG@10 | MRR@10 |
|---|---|---|---|
| BM25 (Italian) | 0.093 | 0.150 | 0.212 |
| E5 | 0.123 | 0.159 | 0.271 |

Paired differences (mean, 95% bootstrap interval over questions, better / equal / worse):

| Comparison | Metric | Mean | 95% interval | Better / equal / worse |
|---|---|---|---|---|
| E5 − BM25 (Italian) | Recall@10 (primary: inconclusive) | +0.030 | [-0.019, +0.087] | 7 / 12 / 8 |
| E5 − BM25 (Italian) | nDCG@10 (primary: inconclusive) | +0.009 | [-0.064, +0.085] | 8 / 10 / 9 |
| E5 − BM25 (Italian) | MRR@10 | +0.059 | [-0.084, +0.204] | 10 / 10 / 7 |
| E5 − BM25 (Italian) | Recall@20 | +0.017 | [-0.043, +0.087] | 7 / 12 / 8 |

By number of relevant articles (descriptive):

| Relevant | Questions | BM25 (Italian) R@10 / nDCG@10 | E5 R@10 / nDCG@10 |
|---|---|---|---|
| 1-4 | 13 | 0.115 / 0.091 | 0.218 / 0.193 |
| 5-15 | 3 | 0.042 / 0.036 | 0.037 / 0.034 |
| 16+ | 11 | 0.081 / 0.252 | 0.034 / 0.153 |

## pooled-108 (108 questions)

Official metrics (JuriFindIT):

| System | Recall@5 | Recall@20 | Recall@60 | Recall@100 | nDCG@5 | nDCG@100 | MRR@5 | MRR@100 | mAP@5 | mAP@100 |
|---|---|---|---|---|---|---|---|---|---|---|
| BM25 (Italian) | 0.057 | 0.151 | 0.245 | 0.303 | 0.152 | 0.176 | 0.218 | 0.241 | 0.043 | 0.075 |
| E5 | 0.079 | 0.150 | 0.216 | 0.264 | 0.136 | 0.165 | 0.224 | 0.247 | 0.059 | 0.076 |

Our compatible metrics:

| System | Recall@10 | nDCG@10 | MRR@10 |
|---|---|---|---|
| BM25 (Italian) | 0.108 | 0.149 | 0.232 |
| E5 | 0.121 | 0.137 | 0.237 |

Paired differences (mean, 95% bootstrap interval over questions, better / equal / worse):

| Comparison | Metric | Mean | 95% interval | Better / equal / worse |
|---|---|---|---|---|
| E5 − BM25 (Italian) | Recall@10 (primary: inconclusive) | +0.014 | [-0.016, +0.047] | 17 / 65 / 26 |
| E5 − BM25 (Italian) | nDCG@10 (primary: inconclusive) | -0.012 | [-0.049, +0.027] | 27 / 50 / 31 |
| E5 − BM25 (Italian) | MRR@10 | +0.004 | [-0.062, +0.073] | 27 / 57 / 24 |
| E5 − BM25 (Italian) | Recall@20 | -0.001 | [-0.047, +0.044] | 24 / 55 / 29 |

By number of relevant articles (descriptive):

| Relevant | Questions | BM25 (Italian) R@10 / nDCG@10 | E5 R@10 / nDCG@10 |
|---|---|---|---|
| 1-4 | 59 | 0.099 / 0.063 | 0.165 / 0.127 |
| 5-15 | 22 | 0.147 / 0.180 | 0.088 / 0.112 |
| 16+ | 27 | 0.094 / 0.313 | 0.052 / 0.180 |

## Performance (apart from quality)

Hardware: Apple M1, 8 cores, macOS-27.0.1-arm64-arm-64bit.

| Step | Measure |
|---|---|
| E5, corpus embedding | 23,458 articles in 74.3 min (5.3 per s), 3,335 truncated; model load 4.0 s; peak RSS 2.31 GiB |
| E5, one question (warm: encoding, exact search over 23,458, sort) | p50 127 ms, p95 150 ms (n = 108) |
| E5, all 108 from the stored vectors | 1.4 s |
| BM25 | index 3.0 s; 9 ms per question; peak RSS 0.49 GiB |
| E, runtime estimate (3 civil-law questions) | 79 s per question, 2.36 h projected for 108 |
| E, decomposition (108 questions, MLX) | 43.1 min, median 19.1 s per question, peak 4.36 GiB |

## Notes

- Relevant ids outside the 23,458 articles: 0 in 0 questions (kept in the qrels; they count in the denominators).
- Questions without any relevant article: 0.
- E passed its prompt gate (108 prompts within 16,384 tokens) and decomposed all 108 questions, then stopped on a technical failure: the reranker ran out of MPS memory in its group 2 of 383, after 50 of 19,150 pairs. Its frozen contract (batches of 8 on MPS, pairs of up to 32,768 tokens; 29 pairs here pass 8,192, against 662 at most in DEV) needs far more memory than this machine has. E is not completed; no rerun, no variant.
- Metrics computed by ir-measures 0.4.3 with pytrec-eval-terrier 0.5.10.
