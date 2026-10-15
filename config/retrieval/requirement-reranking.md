# Retrieval iteration 4 on DEV 0.10.0: E, zero-shot reranking of each requirement's candidates

D ([protocol](decomposition.md), [record](../../data/results/dev-0.10.0/retrieval-iteration-4/log.md))
found the decomposition's signal and located what limits it: D-4B's requirements reach 50 of the
63 relevant (query, chunk) pairs that E5 misses somewhere in their first 50 results, but only 24
in their first 5, the only places D's rule reads. On 2026-10-15 the person chose to test that
alone, at zero additional cost, before any training. This page fixes E before any DEV score. No
other reranker, instruction or depth is tried after its result.

## The question

Does a cross-encoder, zero-shot, lift the evidence each requirement already finds (E5's places 6
to 50) into its first 5, and does D's unchanged rule turn that into a better top 10?

## Decomposer

D-4B's 90 recorded answers, read by D's parser, unchanged: 308 requirements. E checks that their
prompts are the ones D-4B's record names and rebuilds D-4B's ranking from them exactly before it
scores anything. D-4B is chosen over D-8B on DEV numbers, for this experiment only: its
requirements reach more of the missing evidence in their first 50 (50 of 63 against 45) and it
had the higher Recall@10 and nDCG@10. That says nothing general about the two models. D-8B stays
frozen and does not run again.

## Reranker contract

- `config/retrieval/qwen3-reranker-0.6b.json`, the contract of iteration 3's A:
  `Qwen/Qwen3-Reranker-0.6B` at revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`, its
  `tokenizer.json`, the model card's prompt (prefix, `<Instruct>`, `<Query>`, `<Document>`,
  suffix), the score as the logit of "yes" minus the logit of "no" at the last position, Apple
  GPU (MPS) in float32, batches of 8, at most 32,768 input tokens with the document's tail cut
  past that (none expected; the check's longest input has 645 tokens).
- **Instruction**, fixed here: "Given a question about EU financial regulation, retrieve the
  passages of the law that answer it". This is classic relevance, written in the form of the
  model card's own instruction. It is not A's necessity instruction.
- **Query**: the requirement, as D's parser read it. **Document**: the chunk's text (context and
  body), the text E5 embeds. One requirement and one chunk per input: never the original
  question, an anchor, another requirement, a qrel, a grade or an answer.
- No training.

## Input and order

For each requirement, E5's first 50 chunks over the 1,354, as D ranked them: 15,350 distinct
(requirement, chunk) pairs, each scored once and stored with the SHA-256 of its input. Each
requirement's 50 are reordered by descending score; equal scores keep E5's order. Nothing outside
the 50 can enter, so Recall@50 does not change, by construction.

## The primary diagnostic, before the ten

A query's requirements are read together: each relevant chunk takes its best rank over the
query's requirements. **Requirement Recall@k** is the share of the query's relevant chunks with a
best rank of at most k (1, 3, 5, 10, 20, 50), and **requirement MRR** is the reciprocal of the best
rank of any relevant chunk; both macro by need over the 81 evaluable queries, in E5's order and in
the reranked order. E5's 63 missing pairs are followed one by one: from places 1 to 5 or 6 to 50,
to places 1 to 5 or 6 to 50, or outside the 50.

**REQUIREMENT RERANKING SIGNAL: YES** when requirement Recall@5 rises by at least 0.03 against
E5's order of the same 50 (values rounded to 3 decimals) and its paired 95% interval over needs
lies above zero. Otherwise `NO`.

## The ten

D's rule, exactly (`ingest/decomposition.py`, `assemble`): E5's first 6 stay; places 7 to 10 to
E5's ranks 7 to 10 corroborated by a requirement's first 5, then each requirement's best chunk
outside E5's top 10 in turns, then the rest; the ten in E5's order. The only change: a
requirement's first 5 are the reranked ones. So E minus D-4B measures the reranker alone.

## Measures

E5, D-4B and E with the evaluator of every system: Recall@10, nDCG@10, MRR@10, full-evidence@10,
`multi_provision`, dependent evidence, `cross_reference`, `cross_act`, by language. E paired with
E5 and with D-4B by need (Recall@10, nDCG@10, full-evidence@10), the bootstrap averaging by need.
The places: the relevant pairs outside E5's top 10 that sit in a requirement's first 5 but are
left out of the ten (11 for D-4B).

## Conclusion, fixed in advance

- `NO RERANKING SIGNAL`: the signal above is `NO`.
- `RERANKER WORKS`: the signal is `YES` and E's Recall@10 is at least 0.03 above D-4B's.
- `RERANKER WORKS / ASSEMBLER BOTTLENECK`: the signal is `YES`, E's Recall@10 is less than 0.03
  above D-4B's, and more relevant pairs wait in the first 5 without a place than in D-4B. This is
  a finding about the rule, not a failure of the reranker.
- `RERANKER WORKS / NO FINAL GAIN, NOT THE PLACES`: the signal is `YES`, the final gain is under
  0.03 and the places are not the reason.

The minimum signal and the useful and strong targets of D stay as they are, against E5.

## After E

- **The useful target reached**: stop, freeze the architecture; the next phase is an
  independent final TEST; no fine-tuning.
- **The reranking signal without the target**: DEV experimentation stops. The next block designs
  training data for one objective, requirement and passage to a relevance score (not necessity
  given an anchor), with Qwen3-Reranker-0.6B as the first candidate. That design starts from
  InPars, Promptagator and GPL, and does not take a question generated from a passage as a
  perfect label. It needs quality filtering, filtering of ambiguous questions, mining of hard
  negatives, an independent validation set and no DEV contamination. Not executed here.
- **No reranking signal**: no other reranker. The person decides between synthetic domain
  adaptation with fine-tuning and freezing retrieval.
- **Either way**: the final TEST stays blind until the architecture, the model, any training
  recipe, the thresholds and the inference are frozen.

## Time

The check on 2 requirements of the fixture queries (not DEV): 100 inputs in 25.7 s, 0.257 s each,
none truncated; 15,350 DEV inputs, 1.1 hours projected. E runs only within 4 hours
(`retrieval-iteration-4/e-check.json`).

## Not touched

DEV 0.10.0, its queries, qrels, grades and chunks, the parser, the evaluator, D's prompt, parser,
K and rule. No label is read before every ranking exists.
