# Retrieval iteration 4 on DEV 0.10.0: the record

Two experiments at zero additional cost, local open models only, each fixed before it read DEV:
D, question decomposition ([protocol](../../../../config/retrieval/decomposition.md), `6f0306a`),
and E, the reranking of D's candidates
([protocol](../../../../config/retrieval/requirement-reranking.md), `2e16281`). Global values are macro
means by need over the 27 evaluable needs; intervals pair each system with E5 by need (10,000
resamples), with the bootstrap averaging by need (`22c2294`). Reference, E5 dense: Recall@10
0.763, nDCG@10 0.528, full-evidence@10 0.642, `multi_provision` Recall@10 0.481, dependent
evidence Recall@10 0.300.

## Before DEV: the format check on the fixture queries

The 6 fixture queries (`data/benchmark/fixtures`, not DEV) checked the form of the answer before
it was frozen. The first form asked for "the content of one provision"; the 4B wrote statements of
the law with figures and article numbers, hypothetical legal text of the kind HyDE adds to a
search. The form was amended once: a requirement is a question, and states neither what the law
says nor an article number or a date. On the fixtures it then gave 6 readable answers, 3 or 4
questions each, none with an article number.

## D with Qwen3-4B-Instruct-2507 (4-bit, MLX)

90 answers in 21.5 minutes (14.3 s at the median, 1,076 prompt tokens, 128 new tokens), 3.0 GiB
of peak memory. 308 requirements (52 answers of 3, 38 of 4), none unreadable. Against the form:
17 are not questions, 11 write an article number (each one named by an excerpt) and 15 a year;
none has Spanish letters. The rule changed 85 of the 90 rankings: 217 chunks entered from
outside E5's top 10, 136 of E5's ranks 7 to 10 stayed as corroborated and 217 left.

| | E5 | D-4B | difference |
|---|---|---|---|
| Recall@10 | 0.763 | 0.803 | +0.040 [-0.008, +0.097] |
| nDCG@10 | 0.528 | 0.552 | +0.025 [+0.010, +0.041] |
| MRR@10 | 0.746 | 0.750 | +0.003 |
| Recall@20 | 0.816 | 0.838 | +0.022 |
| Recall@T = 1000 / 2000 | 0.680 / 0.782 | 0.680 / 0.805 | +0.000 / +0.024 |
| Full-evidence@10 | 0.642 | 0.667 | +0.025 [-0.025, +0.074] |
| `multi_provision` Recall@10 | 0.481 | 0.558 | +0.078 |
| `cross_reference` Recall@10 | 0.661 | 0.698 | +0.037 |
| `cross_act` Recall@10 (2 needs) | 0.292 | 0.458 | +0.167 |
| dependent evidence Recall@10 | 0.300 | 0.368 | +0.068 |

English / Spanish Recall@10: 0.801 / 0.807 (E5: 0.777 / 0.735). Queries better / equal / worse:
12 / 66 / 3 in Recall@10, 39 / 34 / 8 in nDCG@10. nDCG@10's interval is the first gain over E5 in
iterations 1 to 4 to exclude zero (the earlier intervals that excluded it were losses: BM25, E2,
E3 with the bridge, C). **The minimum signal is not reached**: Recall@10 (+0.040) and nDCG@10
(0.552) pass, full-evidence@10 rises by 0.025, 0.005 short of +0.03. The useful target is not
reached either (Recall@10 0.85, `multi_provision` 0.60, dependent evidence 0.45, full-evidence
0.75).

**Where E5's 63 missing relevant (query, chunk) pairs go** (labels read after the run): D brings
13 into the top 10 (5 of grade 3, 8 of grade 2) and loses 3 that E5 held. Of the other 50, 11 are
in a requirement's first 5 but found no place left among the four, 26 are in a requirement's
places 6 to 50, and 13 are in no requirement's first 50. The requirements reach 50 of the 63
somewhere in their first 50 (the pools' Recall: 0.823 at 5 per requirement, 0.948 at 50); what
limits D is mostly where those chunks fall in each requirement's ranking, and the four places.

By need, Recall@10 rises in N026 (0.17 to 0.67), N018 and N035 (+0.33), N024 (0.13 to 0.27), N022
and N025 (+0.11), and falls in N021 and N023 (-0.17) and N017 (-0.11). The 3 pairs lost are the
risk the protocol named: relevant chunks at E5's ranks 8 to 10 (N017-en-1 at 10, N021-es-1 and
N023-en-2 at 8) that no requirement had in its first 5, so each left the ten for a new chunk.

In N035 a definition of PSD2 (4(16)) rises from E5's ranks 724 and 855 to 10, and DORA 3(37) from
54 and 103 to 9 and 10; in N024, MiCA 89(1) from 23 and 107 to 8 and 9, a first-hop failure that
the search over the whole corpus reaches. N034 does not move.

## D with Qwen3-8B, without thinking (4-bit, MLX): the capacity control

**Pilot** on the fixture queries: 5.01 GiB of peak memory and 0.58 hours projected for DEV, inside
both limits (10 GiB, 3 hours); no answer held a thinking block. The pilot also showed the 8B
wrapping each question in the form's placeholder as a tag, which the first parser read as empty;
the parser was amended before the 8B's DEV run, and it reads the 4B's 90 answers exactly as before
(`ad55870`).

90 answers in 39.3 minutes (25.7 s at the median), 5.1 GiB of peak memory. 348 requirements (12
answers of 3, 78 of 4), none unreadable, all questions, 20 words at the median; 3 write an
article number (each named by an excerpt) and 4 a year. The rule changed 87 rankings: 218 chunks
entered and 130 of E5's ranks 7 to 10 stayed as corroborated.

| | E5 | D-4B | D-8B | D-8B difference |
|---|---|---|---|---|
| Recall@10 | 0.763 | 0.803 | 0.794 | +0.031 [-0.011, +0.083] |
| nDCG@10 | 0.528 | 0.552 | 0.548 | +0.020 [+0.007, +0.033] |
| MRR@10 | 0.746 | 0.750 | 0.745 | -0.001 |
| Recall@20 | 0.816 | 0.838 | 0.858 | +0.042 |
| Full-evidence@10 | 0.642 | 0.667 | 0.691 | +0.049 [0.000, +0.111] |
| `multi_provision` Recall@10 | 0.481 | 0.558 | 0.566 | +0.085 |
| `cross_reference` Recall@10 | 0.661 | 0.698 | 0.692 | +0.031 |
| `cross_act` Recall@10 (2 needs) | 0.292 | 0.458 | 0.375 | +0.083 |
| dependent evidence Recall@10 | 0.300 | 0.368 | 0.460 | +0.160 |

English / Spanish Recall@10: 0.807 / 0.769. Queries better / equal / worse: 10 / 68 / 3 in
Recall@10, 39 / 33 / 9 in nDCG@10, 4 / 77 / 0 in full-evidence. **The 8B passes the minimum
signal**: Recall@10 +0.031, full-evidence +0.049, nDCG@10 0.548. Dependent evidence (0.460) meets
its part of the useful target; Recall@10, `multi_provision` and full-evidence do not, so the
useful target is not reached.

Of E5's 63 missing pairs, the 8B brings 12 into the top 10 (2 of grade 3, 10 of grade 2) and loses
3 (N015-es-1, N023-en-2, N034-en-1); 12 wait in a requirement's first 5 with no place left, 21 are
in a requirement's places 6 to 50, and 18 in none of its first 50. By need, Recall@10 rises in N026
(0.17 to 0.67), N035 (+0.25), N022 and N031 (+0.17), N017 (+0.11) and N024 (+0.07), and falls in
N015 and N023 (-0.17) and N034 (-0.08). The two models gain in the same hard needs (N022, N024,
N026, N035) and differ elsewhere. The 8B is a model of another release, so the difference between
the two is not a measure of size.

## Outcome

`DECOMPOSITION SIGNAL FOUND: YES`, by the registered rule: the 8B passes the minimum signal, and
the 4B falls 0.005 short of it in full-evidence. Read with three limits:
- the signal sits on its thresholds: Recall@10 +0.031 against +0.03, with an interval that holds
  zero for both models, and full-evidence's interval starting at 0.000;
- two models were run and one passed: two tries raise the chance of crossing a threshold by noise;
- the most robust result is nDCG@10, whose interval excludes zero for both models: the rule leaves
  E5's first six in place and adds evidence below them.

The useful target is not reached, and these are development numbers chosen on DEV. By the
protocol, D is frozen as it is and DEV is not tuned further; no final TEST is consumed to decide
what comes next.

**What limits D** (both models): the requirements reach most of the missing evidence somewhere in
their first 50 results (50 and 45 of the 63 pairs) but less than half of it in their first 5 (24
and 24), and 11 and 12 pairs wait there with no place left among the four. The bottleneck is how
E5 ranks the candidates of each requirement and the four places, more than the decomposition;
13 and 18 pairs, all of grade 2, are not reached by any requirement.

## E: zero-shot reranking of each requirement's candidates (D-4B, Qwen3-Reranker-0.6B)

Fixed before any DEV score ([protocol](../../../../config/retrieval/requirement-reranking.md),
`2e16281`). Before scoring, D-4B's ranking was rebuilt exactly from its recorded requirements.
15,350 (requirement, chunk) pairs scored in 84.7 minutes (283 tokens at the median, 662 at most,
none truncated, 5.1 GB of peak memory); the check had projected 1.1 hours.

**The requirements' ranking**, each relevant chunk at its best rank over a query's requirements,
macro by need:

| | E5's order | reranked |
|---|---|---|
| Recall@1 | 0.610 | 0.663 |
| Recall@3 | 0.774 | 0.785 |
| Recall@5 | 0.823 | 0.825 |
| Recall@10 | 0.871 | 0.880 |
| Recall@20 | 0.923 | 0.928 |
| Recall@50 | 0.948 | 0.948 (by construction) |
| MRR | 0.836 | 0.915 |

Requirement Recall@5 changes by +0.001 [-0.058, +0.059], paired by need: **REQUIREMENT RERANKING
SIGNAL: NO** (the protocol asked for +0.03 with an interval above zero). Of E5's 63 missing pairs,
10 rise from places 6 to 50 into the first 5, 6 fall out of the first 5, 18 stay in them, 16 stay
in places 6 to 50 and 13 are outside the 50.

**The ten**:

| | E5 | D-4B | E | E against E5 | E against D-4B |
|---|---|---|---|---|---|
| Recall@10 | 0.763 | 0.803 | 0.824 | +0.061 [+0.009, +0.122] | +0.021 [-0.017, +0.062] |
| nDCG@10 | 0.528 | 0.552 | 0.561 | +0.034 [+0.013, +0.057] | +0.009 [-0.009, +0.027] |
| MRR@10 | 0.746 | 0.750 | 0.754 | +0.007 | +0.004 |
| Recall@20 | 0.816 | 0.838 | 0.863 | +0.047 | +0.025 |
| Full-evidence@10 | 0.642 | 0.667 | 0.679 | +0.037 [-0.012, +0.086] | +0.012 [-0.049, +0.086] |
| `multi_provision` Recall@10 | 0.481 | 0.558 | 0.617 | +0.136 | +0.058 |
| `cross_reference` Recall@10 | 0.661 | 0.698 | 0.735 | +0.074 | +0.037 |
| `cross_act` Recall@10 (2 needs) | 0.292 | 0.458 | 0.500 | +0.208 | +0.042 |
| dependent evidence Recall@10 | 0.300 | 0.368 | 0.409 | +0.109 | +0.041 |

English / Spanish Recall@10: 0.844 / 0.785. Against E5, 18 relevant pairs enter the ten (8 of
grade 3, 10 of grade 2) and 4 leave; queries better / equal / worse in Recall@10: 14 / 63 / 4. 10
relevant pairs wait in a requirement's first 5 without a place (D-4B: 11).

**Conclusion, as registered: `NO RERANKING SIGNAL`.** The reranker does not lift the requirements'
evidence into their first 5 as a whole. E passes D's minimum signal against E5 (Recall@10 +0.061,
full-evidence +0.037, nDCG@10 0.561). Its Recall@10 interval against E5 is the first gain in
Recall@10 of iterations 1 to 4 to exclude zero, and `multi_provision` reaches 0.617, above the
useful target's 0.60. The useful target as a whole is not reached (Recall@10 0.85, dependent
evidence 0.45, full-evidence 0.75). Against D-4B, the comparison that isolates the reranker,
every interval holds zero.

**After the fact: why the ten improve while Recall@5 does not.** The reranker raises the first
places (Recall@1 +0.053, MRR +0.079) and leaves place 5 where it was, while the rule takes each
requirement's best new chunk first, in turns. Over the requirements, the first new chunk of the
first 5 is relevant 22 times reranked against 18 in E5's order, and the second 10 times against 8.
The registered diagnostic measured depth 5; the rule mostly reads depth 1. This explains the
result and does not change the registered conclusion. These are development numbers: DEV has been
read by many systems, and E is the best of them on it.

## F: one LoRA recipe for the requirement reranker

Fixed before training ([protocol](../../../../config/retrieval/reranker-finetuning.md), `88f447c`;
code `00564d1`; a memory fix `821eeef`, after a first run swapped at 11 GB and was stopped before
its first evaluation).

**Data, from the corpus alone.** The 4B wrote 2 questions from each of the 1,233 chunks with a body
of 20 words or more. Of 2,466, the filters kept 1,256: 662 dropped for naming an article or
paragraph, 294 for copying 8 or more words, 232 for not being a question, 20 for leaning on the
inherited context, 1 for the round trip. The source chunk is E5's first for 993 of the 1,256
questions and within its first 5 for 1,200. Split by article: TRAIN 1,071 (231 articles), VAL-SELECT
100 and VAL-REPORT 85 (50 articles); no article on both sides, no validation chunk among the
training negatives.

**Training.** LoRA (rank 16, the last 8 of 28 layers, 2,883,584 trainable parameters), listwise
softmax over groups of 4, AdamW 1e-4. 140 minutes, 6.91 GB of peak memory. VAL-SELECT MRR: 0.880
before training, 0.921 at step 200, 0.937 at 400, 0.942 at 600, 0.933 at 800, 0.937 at 1,000;
stopped by patience, checkpoint of step 600.

**Component, on VAL-REPORT** (85 questions, E5's 50 each): P@1 0.812 / 0.753 / 0.929 and MRR
0.857 / 0.837 / 0.960 (E5's order / zero-shot / fine-tuned). The MRR gain is +0.123 [+0.067, +0.179]
over 25 source articles: **training success**.

**Parity.** The MLX scorer against E's PyTorch scores on 500 DEV pairs: largest difference 6e-05;
10 of 10 requirements ordered identically.

**DEV, once:**

| | E5 | D-4B | E (zero-shot) | F (fine-tuned) | F against E |
|---|---|---|---|---|---|
| Recall@10 | 0.763 | 0.803 | 0.824 | 0.805 | -0.020 [-0.047, +0.006] |
| nDCG@10 | 0.528 | 0.552 | 0.561 | 0.564 | +0.003 [-0.011, +0.016] |
| Full-evidence@10 | 0.642 | 0.667 | 0.679 | 0.667 | -0.012 [-0.037, 0.000] |
| `multi_provision` Recall@10 | 0.481 | 0.558 | 0.617 | 0.583 | -0.033 |
| dependent evidence Recall@10 | 0.300 | 0.368 | 0.409 | 0.367 | -0.042 |
| requirements' Recall@1 / MRR | | 0.610 / 0.836 | 0.663 / 0.915 | 0.629 / 0.871 | |

**`DOMAIN-ADAPTED RERANKER SUCCESS: NO`.** Training succeeds on the synthetic questions and the
adapter is worse than the zero-shot model on D-4B's real requirements, at the requirement level
(Recall@1 0.629 against 0.663, MRR 0.871 against 0.915) and in the ten. The synthetic gain does not
transfer. Hypotheses, not tested: the synthetic questions are easy (the source is E5's first for 79%
of them), so the adapter learns to agree with surface closeness; and the hardest E5 neighbours used
as negatives often answer too (siblings, parallel provisions), so the adapter learns to push down
exactly the related provisions that real requirements need. Both are failure modes the literature
reports for synthetic training and hard-negative mining.
