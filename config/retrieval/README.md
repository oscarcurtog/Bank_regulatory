# Retrieval iteration on DEV 0.10.0: the protocol, fixed before any result

This file was committed before the first experiment of the iteration ran. It fixes what is
measured, how a configuration is called better, every grid, and the rule that picks the next
experiment from the result of the previous one. A grid is never widened after its results are
seen, every configuration that runs is recorded, and the losers are reported with the winners.

The iteration optimizes the retriever, never the benchmark. DEV serves development: a
configuration may be selected on it, so its DEV numbers are not a final quality claim. A
separate TEST set will measure the retriever that comes out of this iteration.

## Frozen

DEV 0.10.0 (protocol v1.4), parser 1.4.0, the official chunk layer (1,354 chunks of at most 500
`cl100k_base` tokens, chunker 0.1.0, the same chunk ids), the 90 queries (81 scored, of the 27
evaluable needs), the qrels and grades, the node-to-chunk projection, the evaluator
(`ingest/evaluation.py`, ir_measures 0.4.3 on pytrec-eval-terrier 0.5.10) and its metrics. No
need, query, qrel, grade or chunk changes, no label is reinterpreted because a system fails it,
and no case is removed.

Every system ranks all 1,354 chunks for all 90 queries; ties are broken by chunk id. The 9
queries of the `none` needs are ranked and stored, and never scored.

## Measures

- Official: nDCG@10, Recall@10, MRR@10, Recall@5, Recall@20, Recall@T=1000, Recall@T=2000, per
  language (D-020), and the breakdowns of `scripts/embed_baseline.py`.
- **Global**: the macro mean by need over the 27 evaluable needs (each need: the mean of its
  three queries; then the mean over needs). It mixes English and Spanish, so under D-020 it is
  descriptive; the thresholds below are stated on it because the reference values are. The
  per-language numbers are always reported next to it.
- Group recall: `multi_provision` (6 needs, by type), `cross_reference` (11), `lexical_twin`
  (17) and `cross_act` (2, diagnostic only, never a gate), macro by need.
- **Dependent evidence Recall@k**: in the 11 needs whose projected chunks include a grade-3
  chunk and at least one grade-2 chunk, the share of the grade-2 chunks found in the top k of
  each query, then the macro mean by need. Grade 2 is evidence the answer depends on without
  being the answer (PROTOCOL.md section 2): a definition, a condition, an exception, the
  provision a rule refers to.
- **Oracle union Recall@10 (A, B)**: per query, the share of relevant chunks in the union of
  the top 10 of A and the top 10 of B (up to 20 chunks), aggregated like Recall@10. It is a
  diagnostic of how much two retrievers complement each other, not a system.
- Recall@50 and Recall@100: the same evaluator with deeper cutoffs, to judge a candidate set.

## Reference

`multilingual-e5-large`, dense, exact search: global Recall@10 0.763, nDCG@10 0.528,
`multi_provision` Recall@10 0.481, `cross_act` 0.292, dependent evidence Recall@10 0.300.
BGE-M3 dense: 0.749, 0.523, 0.461, 0.250, 0.293.

## Thresholds

A configuration is **interesting** if at least one holds, against E5: global Recall@10 +0.05;
global nDCG@10 +0.05 with global Recall@10 down by no more than 0.02; `multi_provision`
Recall@10 +0.10; dependent evidence Recall@10 +0.10.

It is **strong** if it approaches global Recall@10 >= 0.85 (ideal 0.90), nDCG@10 >= 0.60 and
`multi_provision` Recall@10 >= 0.65 to 0.70. Global Recall@10 >= 0.90 with nDCG@10 >= 0.65 stops
the iteration with a strong candidate.

## The experiments and the rules between them

**1. BM25 alone**, configuration [`bm25.json`](bm25.json), one run, no tuning. Then the oracle
union with E5 and with BGE-M3.
- Oracle(E5, BM25) global Recall@10 at least E5 + 0.05: go to 2.
- Otherwise: skip 2 and go to 3.

**2. Hybrid E5 + BM25 by weighted Reciprocal Rank Fusion**, grid [`hybrid-grid.json`](hybrid-grid.json):
`score(c) = w_dense / (k + rank_dense(c)) + w_bm25 / (k + rank_bm25(c))`, ranks from 1 over the
full rankings, with k in {20, 60, 100} and (w_dense, w_bm25) in {(0.75, 0.25), (0.5, 0.5),
(0.25, 0.75)}: 9 configurations, no other. Raw scores are never mixed. Selection: highest global
Recall@10, then highest global nDCG@10, then the first in grid order.
- The best is **kept** if its global Recall@10 is at least 0.85 or E5 + 0.05. A kept candidate
  then has its Recall@50 and Recall@100 measured: Recall@50 >= 0.95 goes to 4; Recall@50 < 0.95
  with `multi_provision` or dependent evidence still low goes to 3.
- Under E5 + 0.03 it is not tuned any further, and the iteration goes to 3.

**3. One-hop explicit cross-reference expansion**, configuration
[`xref-expansion.json`](xref-expansion.json), anchor K in {5, 10}.
1. The base retriever ranks; its top K are the anchors.
2. From the own text of the nodes each anchor owns (not its inherited context), only explicit
   legal references are read, by a deterministic parser: "Article N", "Article N(p)", "Article
   N(p)(q)", "Article N(p), point (x)", "point (x) of Article N(p)", "Articles N and M", "Articles
   N to M", "paragraph N" of the same article. The act is the anchor's own unless the reference
   names one: "of this Regulation" or "this Directive" keeps it, Regulation (EU) 2022/2554 is DORA,
   Regulation (EU) 2023/1114 is MiCA, Directive (EU) 2015/2366 is PSD2, and any other act is
   outside the corpus and ignored. Not followed: "that Regulation" or "that Directive"
   (anaphora), a point or subparagraph without its article or paragraph, Titles, Chapters,
   Sections, Annexes and recitals.
3. A reference names a node; if its exact path does not exist, its nearest existing ancestor.
   The targets are the chunks that own that node or any node below it, in layer order.
4. Linearization: anchor 1, its targets in reference order, anchor 2, its targets, and so on to
   anchor K; a chunk keeps its first position; then the rest of the base ranking in its order.
   One hop only: a target's own references are never followed. No qrel decides anything.
5. Every added chunk keeps its trace: query, anchor, reference text, target node, chunk.

The base retriever is the candidate kept by experiment 2, or E5 when there is none. Expansion
precision is the share of the chunks that expansion placed (not already placed by an earlier
anchor or target) that are relevant to the query's need, overall and within the top 10.
- Kept if, against its base, global Recall@10 improves by at least 0.05 or `multi_provision`
  Recall@10 by at least 0.10. Kept with Recall@50 >= 0.95: go to 4; kept with Recall@50 under
  0.95: the iteration ends with it as the candidate. Not kept: no second hop, go to 5, which
  runs only if its own conditions hold; otherwise the iteration ends.

**4. Reranker**, only if some retriever reaches Recall@50 >= 0.95: one multilingual
cross-encoder, pinned to a revision with its license and contract recorded before it runs, no
training, its top 50 reranked into a top 10. Kept if global nDCG@10 improves by at least 0.05
over the same retriever without it. The iteration ends after it.

**5. A third dense model**, only if BM25 brought too little complementarity, the hybrid was not
justified and the expansion did not solve the problem: one model, materially different from
E5 and BGE-M3, its contract fixed before it runs, the same inputs and evaluator. Under E5 + 0.05,
dense model swapping stops.

### How the points the rules leave open are read

Fixed here, before any result:
- "Improves" compares global values, rounded to three decimals, with E5's (or the base's, for the
  expansion); per-language values are reported, never used to pick.
- A hybrid between E5 + 0.03 and E5 + 0.05 is not kept, and the iteration goes to 3 on E5.
- "Still low" means `multi_provision` Recall@10 under 0.65 or dependent evidence Recall@10 under
  E5's plus 0.10.
- Experiment 3 picks its K with the selection rule of experiment 2 (highest global Recall@10,
  then nDCG@10, then the smaller K), and rule 3 is applied to that one. (Added after experiment 1
  and before experiment 3 ran: the rules named no way to choose between the two values of K.)
- A transition the rules do not cover stops the iteration and is reported as such.

## Not in this iteration

Fine-tuning, LoRA, synthetic training queries, LLM query rewriting or expansion, GraphRAG, agents,
generation, pgvector, OpenSearch, approximate search, deployment.
