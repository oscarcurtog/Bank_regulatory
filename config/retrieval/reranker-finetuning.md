# Retrieval iteration 4 on DEV 0.10.0: F, one LoRA recipe for the requirement reranker

E ([protocol](requirement-reranking.md), [record](../../data/results/dev-0.10.0/retrieval-iteration-4/log.md))
gave no reranking signal by its registered diagnostic, yet a better ten (Recall@10 0.824 against
E5's 0.763), because the zero-shot reranker raises the first place of each requirement's list,
which D's rule takes first (Recall@1 0.610 to 0.663, MRR 0.836 to 0.915). On 2026-10-15 the person
chose a bounded block of fine-tuning, at zero cost: **one recipe**. If it fails, E is frozen as the
retrieval candidate and the project returns to T3 and T4. This page fixes the data, the filters,
the recipe and the measures before any training and before the one DEV run.

## The hypothesis

Adapting `Qwen3-Reranker-0.6B` to this corpus, with synthetic pairs (question, passage that
answers it) made from the corpus alone, puts the answering passage first more often for D-4B's
requirements, and the better first places give D's rule a better ten. What is trained is classic
relevance, requirement to passage, not necessity given an anchor (that formulation is set aside).

## Data, from the corpus alone

No DEV need, query, qrel, requirement, ranking or failure analysis is used to make or label the
data (`ingest/reranker_data.py`, `scripts/reranker_training_data.py`).

- **Source**: the chunks of the layer whose body has at least 20 words.
- **Generator**: `Qwen3-4B-Instruct-2507` (MLX, 4-bit, greedy, at most 160 new tokens), the
  decomposer of D-4B, so that the training questions look like the requirements the reranker
  will see. Prompt (`SYSTEM`, `TASK`): 2 questions that the passage answers, each about one
  point of law, self-contained, naming who and what, answerable mainly from the passage; no
  answer, no long copied phrase, no article, paragraph or point numbers, no added fact. The form
  was checked on the first 10 eligible chunks (corpus text, not DEV) and needed no change.
- **The source chunk is a provisional positive, not a perfect label.** Filters, fixed here:
  1. form: read as D reads requirements, and it must be a question;
  2. leak: no article, paragraph, point or chunk reference;
  3. copying: no run of 8 or more words of the chunk;
  4. inherited text: dropped when the question's content words meet the inherited context (the
     article heading and lead-in that siblings share) more than the chunk's own body;
  5. round trip: E5 must rank the source within its first 50 for the question.
- **Equivalents**: chunks whose bodies share at least half of their 5-word shingles (97 pairs,
  140 chunks in the corpus; E5's cosines are too compressed to tell). They are never negatives
  and count as correct when measuring.
- **Negatives**: E5's own first 50 for the question, the hardest first: for each training
  question the two highest-ranked candidates that are neither correct nor from a validation
  article, plus one drawn, with a seed, from the rest of the 50. No random negative from outside.
- **Split by article**, never by pair: an article is in VALIDATION when the SHA-256 of its key,
  modulo 100, is below 15. VALIDATION is ordered by a seeded hash and cut in two disjoint parts:
  VAL-SELECT, its first 100 questions, to choose the checkpoint; VAL-REPORT, the next 150, to
  report. No validation chunk is a training negative. The audit and the fingerprints of the
  three sets are recorded (`retrieval-iteration-4/ft-data.json`).

## The recipe, one

`config/retrieval/reranker-finetune.json`, in MLX 0.32.3 and mlx-lm 0.32.0 (`.cache/mlx-venv`):

- **Base**: `Qwen/Qwen3-Reranker-0.6B` at revision `e61197ed`, float32, the contract and
  instruction of E. The MLX score matches E's PyTorch score: on 64 pairs the token counts are equal,
  the largest difference is 0.0001 and the order of 50 candidates is the same.
- **LoRA**: rank 16, scale 2.0, dropout 0.05, on the attention and MLP projections of the last 8
  of the 28 layers (2.9 million trainable parameters). Measured before choosing: LoRA on all 28
  layers needs 28 GiB (the machine has 16) and 80 s per step; on the last 8, with groups of 4, 5.6
  GiB and about 2.5 s per group.
- **Objective**: the listwise softmax over each group (the positive and 3 negatives) of the
  inference score itself, logit("yes") minus logit("no") at the last position:
  `-log(exp(s_pos) / sum(exp(s_all)))`. It is implementable in MLX, so neither the pairwise nor
  the pointwise fallback is used. The loop is our own, not mlx-lm's causal trainer, which would
  not train this score.
- **Optimizer**: AdamW, learning rate 1e-4 after 50 steps of linear warmup, then constant; weight
  decay 0.01; gradient norm clipped at 1.0; 2 groups per step; at most 2 epochs; seed 20261015.
- **Checkpoints**: evaluated at step 0 and every 200 steps on VAL-SELECT (each question with its
  positive and 9 hardest wrong candidates: MRR and P@1); training stops after 2 evaluations without
  a better MRR; the best checkpoint by MRR is kept (ties: P@1, then the earlier step).
- No search over hyperparameters, and DEV chooses nothing: no epoch, rate, rank or checkpoint.

## Measures and gates

1. **Component, on VAL-REPORT** (150 questions, E5's 50 candidates each): P@1, MRR and Recall@3,
   @5 and @10 of the first correct chunk, in E5's order, for the zero-shot reranker and for the
   fine-tuned one. **Training success**: the fine-tuned MRR at least 0.03 above the zero-shot one,
   with a 95% interval (resampling source articles) above zero.
2. **Parity on DEV inputs, before the DEV run**: the zero-shot MLX scorer against E's stored
   PyTorch scores on the first 500 pairs of E's record (no label read). If the largest difference
   is at most 0.01, the fine-tuned run compares with E directly; otherwise a zero-shot MLX DEV run
   is added as the adapter's baseline.
3. **DEV, once**, with everything above frozen: D-4B's requirements, E5's 50 for each, the
   fine-tuned scores, E's rule unchanged, E5's final order. E5, D-4B, E and F with the evaluator
   of every system, F paired with E and with E5; and the requirements' P@1 and MRR on DEV.
   **System signal against E**: Recall@10 at least +0.02 or full-evidence at least +0.03, with
   nDCG@10 not below E's by more than 0.01. The useful target stays as it is (Recall@10 0.85,
   `multi_provision` 0.60, dependent evidence 0.45, full-evidence 0.75).
4. **DOMAIN-ADAPTED RERANKER SUCCESS: YES** when training success and the system signal both
   hold; otherwise `NO`.

## Stop rule

One recipe. If it fails, nothing is tuned on DEV: E (zero-shot) is frozen as the retrieval
candidate and the project returns to T3 and T4. If it succeeds, the architecture is frozen and the
next step is an independent final TEST, which stays blind until then.

## Time, estimated before running

Generation about 1.7 hours (1,233 eligible chunks), training at most about 2.5 hours with its
evaluations, the component measures about 1 hour, the DEV run about 1.1 hours: all on the M1, at
zero cost.

## Not touched

DEV 0.10.0, its queries, qrels, grades and chunks, the parser, the evaluator, D's prompt, parser,
K and rule, E's contract and instruction.
