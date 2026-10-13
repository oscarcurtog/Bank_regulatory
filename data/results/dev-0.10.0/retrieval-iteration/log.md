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
