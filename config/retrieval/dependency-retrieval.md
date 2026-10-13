# Retrieval iteration 2 on DEV 0.10.0: anchor-conditioned dependency retrieval, and what follows

Committed before any result of this iteration. The frozen inputs, the evaluator and the measures
are those of [iteration 1](README.md); nothing of the benchmark changes. Each later experiment is
added to this file, with its hypothesis, grid and rule, before it runs.

Reference, E5 dense: global Recall@10 0.763, nDCG@10 0.528, `multi_provision` Recall@10 0.481,
dependent evidence Recall@10 0.300.

**Stop** when one holds: (A) global Recall@10 >= 0.90 and nDCG@10 >= 0.65; (B) global Recall@10
>= 0.85, `multi_provision` Recall@10 at least E5 + 0.10 and dependent evidence Recall@10 at least
E5 + 0.10; (C) three reasonably different approaches bring no material improvement. **Material**
means one of iteration 1's tests of an interesting configuration against E5: global Recall@10
+0.05; nDCG@10 +0.05 with Recall@10 down by no more than 0.02; `multi_provision` Recall@10 +0.10;
dependent evidence Recall@10 +0.10. Up to three experiments after R1.

## R1: anchor-conditioned dependency retrieval

**Hypothesis.** E5 finds the anchor; the evidence the answer depends on is reachable from the
anchor by the structure of the law (a reference, a defined term) or by a question about that
dependency, not by the original question. Collected as candidates, never inserted at the top,
it raises the recall of the candidate set.

1. **Anchors**: E5's top 10. **Base candidates**: E5's top 20.
2. **Relations**, read deterministically from the own text of the nodes each anchor owns:
   - `reference`: the explicit references of `ingest/references.py`, to their exact targets (the
     node named, or its nearest existing ancestor; an article reaches all its chunks).
   - `definition`: a term defined in the anchor's act (a node whose text opens with `'term'
     means`) that occurs in the anchor's own text, as a whole word, in any case, singular or
     plural; its target is the chunk that owns the definition.
   - `definition_reference`: the explicit references inside a definition target's own text (a
     definition by reference, such as "'payment institution' means a payment institution as
     defined in Article 4, point (4), of Directive (EU) 2015/2366"), one step, definitions only.
   No edge comes from a language model, and no qrel is read.
3. **Bridges**, at most 3 per query, deterministic and in English: among the terms found by
   `definition` in the query's anchors, the 3 whose text `What is <term>?` is closest to the
   original query (E5 cosine, both encoded as queries; ties by the rank of the first anchor that
   uses the term, then the term). No language model was needed to phrase them.
4. **Dependency retrieval**: each bridge queries E5 over the whole layer; its top k join the pool.
   Grid: k in {3, 5}.
5. **Candidate pool**: E5's top 20, every relation target and every bridge hit, deduplicated,
   each with its provenance (query, anchor, relation, candidate).
6. **Candidate list**, the order Candidate Recall@k reads and the reranker would take: E5's top 20
   in E5's order; then the rest of the pool by its highest E5 cosine against the original query
   or one of the query's bridges (ties by chunk id); then every other chunk in E5's order. Its
   top 20 is E5's by construction: it is a candidate generator, not a ranking.

**Measured before any reranking**: Candidate Recall@20, @50 and @100, globally, per language,
for `multi_provision`, `cross_reference`, `cross_act` and the dependent evidence.

**Gate.** The k with the higher global Candidate Recall@50 (then the smaller k). Candidate
Recall@50 >= 0.95 goes to reranking. Under 0.95 the reranker is not tuned and the iteration goes
to research.

**Reranking, if reached**: `BAAI/bge-reranker-v2-m3`, its contract verified and pinned before it
runs, the top 50 of the candidate list into a new order. A candidate from E5's top 20 is scored
with the original query; any other candidate with the maximum of its score with the original
query and its scores with the bridges that produced it (by retrieval, or as the definition of
the bridge's term). The rest of the list stays below, in its order.
