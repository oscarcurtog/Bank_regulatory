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

## After R1: research, and the experiments it chose

R1's Candidate Recall@50 was 0.893 (k = 3), under the gate: the reranker of R1 was not run and
the iteration went to research (2026-10-13). Three training-free approaches fit the failure
("the anchor is retrieved, the evidence it depends on is not") and this corpus:

- **Dense-seeded Personalized PageRank over a graph of passages**, as in HippoRAG 2 (Gutierrez
  et al., arXiv 2502.14802; code at github.com/OSU-NLP-Group/HippoRAG): passage nodes seeded with
  reset probabilities proportional to their embedding similarity, damping 0.5, passages ranked by
  their PageRank score; +5.0 and +13.9 recall@5 over NV-Embed-v2 on MuSiQue and 2Wiki. Its graph
  comes from an LLM; here the graph is the law's own: references and defined terms, read
  deterministically. Legal work argues for the same typed structural traversal (de Martim, arXiv
  2505.00039 and 2606.09724).
- **Bridge-conditioned scoring**, as in BridgeRAG (arXiv 2604.03384): a second-hop candidate is
  scored given the question and the bridge it came from, not the question alone. Its judge is a
  70B language model; here a multilingual cross-encoder.
- **Vector pseudo-relevance feedback** (Li et al., TOIS 2023, arXiv 2108.11044): the query vector
  moved towards its top passages, with the original query keeping most of the weight.

The most promising is the first: R1 showed that the structure reaches the dependent evidence
(Candidate Recall@100 0.947, dependent evidence @50 0.774 against E5's 0.582), and what fails is
its order; a PageRank ranks each dependency by how much dense-ranked mass reaches it, and dilutes
a reference to a whole article by its out-degree.

## E2: dense-seeded Personalized PageRank over the legal structure

**Hypothesis.** Ranked by Personalized PageRank from E5's top chunks over the graph of
references and defined terms, the dependent evidence enters the top 10 while the anchors stay.

- Graph: one node per chunk; a directed, unweighted edge from a chunk to each target of its R1
  relations (reference, definition, definition_reference), self-loops removed. The same graph for
  every query; no label.
- Seeds: E5's top N; reset probability proportional to the E5 cosine, summing to 1.
- PageRank: p = (1 - d) s + d W^T p, W the out-edges normalized per node, the mass of a node
  without out-edges returned to the seeds; iterated until no value moves by more than 1e-12.
- Ranking: by PageRank score, ties by E5 rank; chunks never reached follow, in E5's order.
- Grid: N in {10, 20} and d in {0.3, 0.5}: 4 configurations. Selection: global Recall@10, then
  nDCG@10, then grid order. Kept if material against E5.

## E3: bridge-conditioned cross-encoder reranking

**Hypothesis.** A multilingual cross-encoder that scores each dependency candidate against the
question and against the bridge that produced it puts the dependent evidence into the top 10.

- Candidate list: R1's selected list (k = 3), unless E2's selected ranking has a higher global
  Recall@50; its top 50 are reranked and the rest stays below in order.
- Reranker: `BAAI/bge-reranker-v2-m3`, contract (revision, license, languages, input limit,
  score) verified and pinned before it runs; a pair over the limit is cut at the passage's tail
  and recorded, never silently.
- Score of a candidate in E5's top 20: the reranker with the original query. Of any other: the
  maximum of that score and the reranker with each bridge that produced it in R1 (by retrieval,
  or as the definition of the bridge's term).
- Grid, 2 configurations: the original query only, and the maximum rule above. Selection and
  keep rule as E2.

## E4: a fixed quota of the top 10 for the best dependency candidates

Written after E2 and E3 failed and before E4 ran. E3 showed that one relevance scale does not
order the dependency candidates against the anchors: with the original query the cross-encoder
matches E5 (Recall@10 0.762), and with the bridge maximum the definitions of the bridges' terms
take over the top 10 (Recall@10 0.629).

**Hypothesis.** The dependent evidence that R1 gathers reaches the top 10 if it does not compete
on the anchors' scale: a small fixed number of top-10 positions go to the best dependency
candidates, the rest stays E5's.

- Ranking: E5's top 10 - s in its order; then the first s chunks of R1's selected candidate list
  after its position 20 (the pool's best by R1's probe score: the highest E5 cosine against the
  original query or one of its bridges); then E5's ranks 11 - s to 20; then the rest of R1's list.
- Grid: s in {2, 3}. Selection and keep rule as E2; the stop rules A and B are checked.
