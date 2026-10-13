# Retrieval iteration on DEV 0.10.0: the record

Every run of the iteration, in the order it ran, with the rule each decision applied
([protocol](../../../../config/retrieval/README.md)). Nothing is left out: runs replaced by a fix
stay here with their numbers. Global values are macro means by need (descriptive under D-020);
the per-language numbers are in each summary.

## Experiment 1: BM25 alone

**Run 1** (contract `23f702520655`, run `92bc618bfaaf`): global nDCG@10 0.233, Recall@10 0.382,
Recall@20 0.440, Recall@50 0.517. The oracle union with E5 reached global Recall@10 0.769 against
E5's 0.763 (+0.006).

**A fidelity bug, fixed before any further decision.** The analyzer is declared as an imitation
of Lucene's EnglishAnalyzer, whose Porter stemmer leaves words of one or two letters unstemmed
(as Porter's reference implementation does). PyStemmer stems them, so the single letter "s"
became an empty index term and "us" became "u". The contract now states the rule
(`stem_min_length` 3) and a test pins it. This is a fix towards the declared behaviour, not a
tuning: no parameter of BM25 changed, and the decision below is the same with either run.

**Run 2** (contract `e086bdb4511f`, run `49b9d78fd41c`), the one that counts: global nDCG@10 0.229,
Recall@10 0.366, Recall@20 0.437, Recall@50 0.521. English formal nDCG@10 0.494 and Recall@10
0.676; English natural 0.128 and 0.265; Spanish 0.065 and 0.157. The corpus is English: 85% of
the terms of an English query are in the index vocabulary, 12% of a Spanish one's.

**Oracle unions** (Recall@10 of the union of two top-10 lists; global; first system alone, then
the union):
- E5 + BM25: 0.763 to 0.769 (+0.006); English 0.777 to 0.787, Spanish 0.735 to 0.735; dependent
  evidence 0.300 to 0.336; 2 of 81 queries gain.
- BGE-M3 + BM25: 0.749 to 0.773 (+0.024); dependent evidence 0.293 to 0.329; 4 queries gain.
- E5 + BGE-M3, a diagnostic outside the rules: 0.763 to 0.817 (+0.054); Spanish 0.735 to 0.835;
  9 queries gain. The two dense models complement each other more than either complements BM25.

**Decision 1.** Rule: the oracle union of E5 and BM25 must reach E5 + 0.05 to justify a hybrid.
It reaches E5 + 0.006. The hybrid is not built (experiment 2 is skipped) and the iteration goes
to experiment 3 on E5.

## Experiment 3: one-hop explicit cross-reference expansion on E5

The base is E5 (experiment 2 did not run). The parser follows 2,031 references in the corpus
(812 to an article, 424 in article lists, 731 to a paragraph of the same article, 64 to points;
11 across acts; 32 resolved to the nearest existing ancestor; 8 unresolved, all in DORA's
articles that amend other acts; 241 to acts outside the corpus and 19 anaphoric ones not
followed). Half of the chunks (683) cite something; a reference reaches 1 chunk at the median
and 52 at most, when it names a whole article.

| | E5 (base) | K = 5 | K = 10 |
|---|---|---|---|
| global Recall@10 | 0.763 | 0.653 | 0.640 |
| global nDCG@10 | 0.528 | 0.493 | 0.492 |
| global Recall@20 | 0.816 | 0.693 | 0.680 |
| global Recall@T=1000 / 2000 | 0.680 / 0.782 | 0.612 / 0.664 | 0.612 / 0.664 |
| global Recall@50 / 100 | 0.863 / 0.883 | 0.801 / 0.877 | 0.783 / 0.884 |
| `multi_provision` Recall@10 | 0.481 | 0.548 | 0.548 |
| `cross_reference` Recall@10 | 0.661 | 0.617 | 0.617 |
| `cross_act` Recall@10 (2 needs, diagnostic) | 0.292 | 0.500 | 0.500 |
| dependent evidence Recall@10 | 0.300 | 0.380 | 0.380 |
| chunks added per query (median) | | 22 | 40 |
| expansion precision, all added / added into the top 10 | | 1.3% / 3.9% | 0.9% / 3.8% |

The expansion reaches real evidence (MiCA 39 citing "Article 46" brings 46(1), 46(3) and 46(4) of
N022; DORA 46(b) brings PSD2 33 of N035), but most references name whole articles whose chunks
are not relevant, and they push the base's own hits out of the top 10. Taken in any order, the
expanded set of K = 5 (26 chunks at the median) holds 0.773 of the relevant chunks, less than
E5's own top 20 (0.816).

**Decision 3.** K = 5 is selected (higher global Recall@10). Rule: kept if global Recall@10
improves by 0.05 or `multi_provision` Recall@10 by 0.10 against the base. Global Recall@10
falls by 0.110 and `multi_provision` rises by 0.067: not kept. No second hop. The iteration goes
to experiment 5, whose conditions all hold: BM25 brought too little complementarity (+0.006),
the hybrid was not built, and the expansion did not solve the problem.

## Experiment 5: a third dense model

The project's design asks for a Matryoshka model of at least 1024 dimensions and names no
candidate. Chosen before it ran: `Qwen/Qwen3-Embedding-0.6B` at `97b0c614`, Apache-2.0, a decoder
trained for embeddings in 2025, instruction-aware, Matryoshka up to 1024, against two XLM-RoBERTa
encoders ([contract](../../../../config/embeddings/README.md)). Its default query instruction is
used as published, not one written for this corpus.
