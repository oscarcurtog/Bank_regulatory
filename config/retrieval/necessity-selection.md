# Retrieval iteration 3 on DEV 0.10.0: anchor-conditioned necessity selection

Committed before any score of this iteration. The frozen inputs (DEV 0.10.0 under protocol v1.4:
the 30 needs, their queries, qrels, grades and chunks), the evaluator, the measures and the chunk
layer at 500 tokens are those of [iteration 1](README.md) and
[iteration 2](dependency-retrieval.md). E5 is the anchor retriever and R1 (k = 3) the candidate
generator, both as recorded; nothing of the benchmark changes.

Reference, E5 dense: global Recall@10 0.763, nDCG@10 0.528, `multi_provision` Recall@10 0.481,
dependent evidence Recall@10 0.300. R1: Candidate Recall@50 0.893 and @100 0.947, dependent
evidence @100 0.909.

**The failure addressed.** The anchor is found and the evidence it depends on is in R1's pool,
but no score tried in iteration 2 tells a needed definition, condition, exception, reference or
provision of another act from the anchor's other neighbours.

**Success**, read on the selected configuration of B (and of C, if it runs): useful when global
Recall@10 >= 0.85 with `multi_provision` Recall@10 >= 0.60 and dependent evidence Recall@10 >=
0.45; strong when global Recall@10 >= 0.90 and nDCG@10 >= 0.65. A gain of 0.01 is not a success.

**Stop** when a success is reached, or when the anchor-conditioned reranker (A, B) and one
published set-wise method (C) have both failed. No training of any kind in this iteration.

## A: an anchor-conditioned necessity reranker

**Hypothesis.** A relevance reranker asks whether a candidate resembles the question, and a
needed definition or condition does not. Asked instead whether the candidate is necessary to apply
the anchor to the question, and shown the anchor, the relation and the clause that creates it, a
reranker separates the needed evidence from the anchor's other neighbours: A3 > A2 > A1 on the
dependent evidence.

**Model**: `Qwen/Qwen3-Reranker-0.6B`, under the contract of
[qwen3-reranker-0.6b.json](qwen3-reranker-0.6b.json), checked against the model card at revision
e61197ed45024b0ed8a2d74b80b4d909f1255473 (2026-04-16).
- Apache-2.0. A Qwen3 causal language model (28 layers, 40,960 positions, 32K tokens of context
  on the card) fine-tuned from Qwen3-0.6B-Base for reranking, in more than 100 languages.
- Instruction aware: the card recommends a task instruction, written in English, in the
  `<Instruct>` field.
- The prompt is the card's: a fixed system turn, then `<Instruct>: ...\n<Query>: ...\n<Document>:
  ...` as the user turn, then an assistant turn with an empty think block.
- The score is the logit of "yes" minus the logit of "no" at the last position. Its logistic is
  the card's probability of "yes", on which the thresholds of B are read.
- The card's example cuts at 8,192 tokens silently. Here the limit is the card's context of
  32,768 tokens, and a prompt over it loses the tail of the candidate and is recorded.
- Float32 on the Apple GPU (MPS). It is 1.8 times faster than the CPU here, and float16 was no
  faster.
- The prompts of one query and one anchor open with the same tokens. That shared prefix runs
  once and its keys and values serve each prompt's remaining tokens; tests check it gives the
  plain scores, within 1e-3 on MPS and 1e-4 on the CPU.

**Population**, built without a label:
- Queries: all 90, since a system cannot know which are answerable. Only the 81 evaluable ones
  are measured.
- Anchors: E5's first 8 chunks. No configuration of B keeps an anchor below rank 8.
- Candidates: every chunk of R1's pool (its recorded provenance, k = 3) outside E5's top 10 that
  the anchor reaches by one of R1's relations (a reference, a definition, a definition by
  reference) or by one of the query's bridges about a term the anchor uses.
- One (query, anchor, candidate) triple per pair, with every link that joins them, grouped by
  (query, anchor).

**Relation and citation context**, read deterministically from the own text of the nodes:
- Reference: the clause of the anchor (cut after '.', ';' or ':') that holds the reference.
  - Labelled *exception* if the 80 characters before it, within that clause, hold a derogation
    cue: by way of derogation from, derogating from, notwithstanding, except, save as, other
    than, exempt from.
  - Labelled *condition* if they hold a condition cue: subject to, in accordance with, pursuant
    to, provided that, in compliance with, comply with, fulfil, or conditions or requirements
    laid down, set out, referred to or specified in.
  - Labelled *reference* otherwise.
  - Marked *cross-act* when the candidate is in another act, *hierarchy* when it is in the
    anchor's own article.
- Definition: the term, and the clause of the anchor where it first appears.
- Definition by reference: that clause, and the clause of the definition that holds the
  reference; cross-act when the target is in another act.
- Bridge: the term, the clause of the anchor where it first appears, and the search that found
  the candidate.

At most 3 citation contexts per pair, each about 300 characters around the mention.

**Inputs**, in the model's Instruct, Query and Document fields:

| | Instruct | Query | Document |
|---|---|---|---|
| A1 | "Determine whether the candidate provision is necessary to correctly address the user's question." + verdict | QUESTION | CANDIDATE |
| A2 | "Determine whether the candidate provision is necessary to correctly apply the anchor provision to the user's question." + verdict | QUESTION, ANCHOR | CANDIDATE |
| A3 | as A2 | QUESTION, ANCHOR, RELATION, CITATION CONTEXT | CANDIDATE |

- The verdict reads: 'Reply "yes" if the candidate is necessary supporting evidence, and "no" if
  it is merely related context or unnecessary.'
- The anchor and the candidate are each chunk's text, with its act, article and title.
- A1 keeps the task and drops the anchor. A1 against A2 measures the anchor; A2 against A3, the
  relation and its context.
- No input names a label, a grade or the evidence sought.

**Measured once every score exists**, with the labels from here on, over the evaluable queries.
A candidate's score for a query is its best over the query's anchors (A1 has a single one).
- Primary: the average precision of each query's candidates (queries with a relevant candidate,
  grade >= 2), macro by need.
- AUROC over all (query, candidate) pairs.
- Precision at 1 and at 3 of each (query, anchor) group that holds a relevant candidate.
- Support recall at 4: the share of the relevant chunks outside E5's top 10 that are among the
  query's 4 best candidates (4 is the most supports B admits). Dependent-support recall: the
  same for the dependent evidence. Each comes with its ceiling, the share present in the
  population.
- The probability of "yes" for candidates of grade 3, 2, 1 and 0 or unjudged, with the share
  above 0.50, 0.70 and 0.85.

**Gates.**
- B runs if A3's average precision is above both A1's and A2's.
- A3 improves clearly on A1 if its average precision is higher by at least 0.05 and the 95%
  paired bootstrap interval over needs lies above zero.
- If A3 does not improve clearly on A1, or B does not run, or B reaches no success, the
  iteration goes to research and C.

## B: evidence sets

**Hypothesis.** Selected as a set, E5's anchors plus the few supports that are necessary, the top
10 carries the anchor and its dependent evidence together. A flat ranking of the same scores
would let one crowd the other out.

For each query, with A3's probabilities:
1. Keep E5's first n chunks, in order.
2. A candidate's score is its best probability under a kept anchor.
3. The supports are the candidates whose score reaches the threshold, best first (ties by the
   anchor's E5 rank, then chunk id), at most k.
4. The ranking is the n kept chunks, then the supports, then every other chunk in E5's order.

**Grid**: 9 configurations, (n, k) in {(6, 4), (7, 3), (8, 2)} times a threshold in {0.50, 0.70,
0.85}. Selection by global Recall@10, then nDCG@10, then grid order.

**Measured** as every system, paired with E5: nDCG@10, Recall@10, Recall@20, MRR@10,
Recall@T=1000 and 2000, the group recalls, the dependent evidence, and the full-evidence coverage
at 10, the share of queries whose every relevant chunk (grade >= 2) is in the top 10, macro by
need. N022, N024, N034 and N035 are followed one by one: E5, R1's pool and the selector.

## Research and C, if reached

- **Research**: anchor-conditioned and utility-based selection, set-wise and listwise selection
  for RAG (SetR among them), multi-hop supporting-evidence selection, and legal multi-passage
  reranking, in ACL, EMNLP, NAACL, arXiv, and code on GitHub or Hugging Face. At most two
  alternatives are kept.
- **C** runs once, if a set-wise method better documented than the reranker is found. A language
  model is given the question, the anchor and the candidate supports with their relations, and
  selects the minimum set of candidates necessary to answer with the anchor.
  - It returns candidate IDs only, and writes no answer.
  - It never sees a label.
  - A local model is preferred, and no API key is asked for.
- If no reasonable model can run here, C is reported as the recommended experiment, with the
  exact model or API.

## C, redesigned before it ran: expanded eligibility and legal-function set selection

Written while A was still scoring, before any result of A or B. An external review showed that
B's ceiling (0.884, section B) comes from B's eligibility rules, not from keeping E5's positions:
anchors stop at rank 8, a candidate needs a relation to a kept anchor, nothing from E5's top 10
competes, and R1 follows one hop. C removes those limits first and only then asks a model.

**Population**, built without a label (`ingest/functional_selection.py`):
- hop 0: E5's first 20 chunks; the first 10 are the anchors;
- hop 1: R1's recorded pool (the anchors' references, defined terms and bridge hits), and the
  definitions of the defined terms the question uses, in any of the three acts;
- hop 2: the anchors' definitions by reference, and the explicit references and defined terms of
  every hop-1 member reached by a reference, a definition or a term of the question;
- the definitions of the terms used by every member of hop 0 or 1, one hop further, never past 2;
- deduplicated by chunk, every route kept (source, relation, hop, clause), ordered by R1's list;
  each member also says whether its text already appears in the inherited context of another.

**Gate before any inference** (labels read only here, as an oracle): choosing 10 chunks from the
population at best must reach Recall@10 >= 0.93 and full-evidence coverage at 10 >= 0.85. Measured
(`c-eligibility-audit.json`, population `8721a1204c1d`): 15,313 members over the 90 queries (160
at the median, 392 at most), oracle Recall@10 0.954, full-evidence 0.926, `multi_provision` 0.848,
dependent evidence 0.909; N022, N034 and N035 reach 1.0; N024 stays at 0.20 (a first-hop failure,
round 4's target). The gate passes.

**Selector**: `Qwen/Qwen3-4B-Instruct-2507`, 4-bit MLX build `mlx-community/Qwen3-4B-Instruct-2507-4bit`
at `50d42775`, greedy decoding, in its own environment (D-040; [qwen3-4b-instruct-mlx.json](qwen3-4b-instruct-mlx.json)).
- Windows of at most 8 members, in the population's order. Each member is shown with its text
  (act, article, title), its shortest route (source, relation, hop, the clause shortened to about
  160 characters) and, if any, the members whose context already shows its text.
- The question asked of every member: "if this candidate were removed, what legally necessary
  part of the answer would no longer be supported?", and its legal function, one of: governing
  rule, scope or applicability, legal category or definition, condition, exception, addressee or
  authority, deadline or threshold, resolving cross-reference, merely related. One answer line per
  member: `[number] function | yes or no`. No label, grade or answer is ever shown or asked for.
- Merge, fixed now: the members judged necessary with a function other than merely related are
  taken one per function in turn, functions in the order above, each function in the
  population's order, up to 10. The ranking is that set, then E5's order. No position is reserved.

**Amended before C ran** (format only, no label and no result read): on the first timing sample
the 4-bit model copied the template's word "function" or wrote article titles as functions, so
no judgment could be read. The answer form now lists the nine names to copy exactly and shows the
form on other numbers (`[4] condition | yes`, `[5] merely related | no`); a member judged
necessary whose function is not one of the names counts in a last group of the merge, after the
eight functions. On the same six windows every function then parsed; a window may still give
seven lines for eight members (the unjudged member is not selected), and one window gave the
same function to all eight members. Both are the model's behaviour, measured as such.

**Time rule**: a fixed sample of windows (the first window of every tenth query) is timed first.
The whole population runs if it projects to 8 hours or less; otherwise the first 128 members of
each query in R1's order, if that projects to 8 hours or less (its oracle: Recall@10 0.948,
full-evidence 0.914, the gate passes); otherwise C does not run.

**Success**: useful when Recall@10 >= 0.85, `multi_provision` >= 0.60, dependent evidence >= 0.45
and full-evidence >= 0.75; strong when Recall@10 >= 0.90 and nDCG@10 >= 0.65. Always against E5.

## Round 4: closed-catalog legal reformulation (first-hop failures)

Separate from C. It attacks queries whose answer provision E5 does not reach because the question
describes facts and never names the legal concept (N024: "she learned at work before the
announcement and bought the token" is insider dealing).

- **Catalog**, built once from the whole corpus, never from E5's results: every article title
  and every defined term, each once (440 entries, `ingest/reformulation.py`).
- **R4-A**: the same local model lists the legally material facts of the question, in English,
  without naming laws or legal concepts and without adding facts.
- **R4-B**: the same facts, plus at most 3 entries copied from the catalog, each with the words
  of the question that motivate it. An entry not literally in the catalog is dropped and counted;
  whether its words are in the question is recorded.
- **Retrieval**: E5 ranks the rewrite (facts, then "Legal concepts: ..."), and that ranking is
  fused with E5's ranking of the original question by Reciprocal Rank Fusion, k = 60, equal
  weights. k and the weights are not tuned. A query whose rewrite is empty keeps E5's ranking.
- **This is not HyDE**: no hypothetical legal text is generated; every concept is an existing
  heading or defined term.
- **Kept** only if Recall@10 rises by at least 0.03 against E5, nDCG@10 >= 0.518 and full-evidence
  falls by at most 0.01. N024, N022, N034 and N035 are followed query by query.

## After C and round 4: stop

No further family of experiments. If neither C nor round 4 reaches the useful success, nor at
least Recall@10 +0.03 with full-evidence +0.03 and nDCG@10 >= 0.518, E5 is frozen provisionally
and DEV is not optimized further.
