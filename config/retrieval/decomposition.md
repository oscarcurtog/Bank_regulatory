# Retrieval iteration 4 on DEV 0.10.0: D, question decomposition with a local model

Iteration 3 ended by its stop rule: E5 frozen provisionally, DEV not optimized further
([protocol](necessity-selection.md)). On 2026-10-15 the person reopened DEV for one more
experiment, D, with an additional budget of zero: no paid API, no billing, no paid GPU, only
local open models. This page fixes D before any DEV generation. It does not reopen tuning: no
prompt, parameter or rule changes after a DEV result, and one evaluation per model.

## The question

Does decomposing a question into information requirements, each searched by E5 over the whole
corpus, add evidence that E5 alone misses, without losing what E5 already finds? E5 holds an
answering chunk (grade 3) in its top 10 for most queries; what it misses is mostly the evidence
the answer depends on (dependent evidence Recall@10 0.300).

## Inputs, none of them a label

- the text of each DEV query (English and Spanish);
- E5's recorded ranking (`multilingual-e5-large`), and the text of its first 3 chunks (context
  and body) as excerpts;
- E5's cached document vectors, refused unless they reproduce E5's recorded run;
- a neutral statement of what a correct and complete answer can need, written from the grades 3
  and 2 of [PROTOCOL.md](../../data/benchmark/PROTOCOL.md) section 2, without its worked
  examples, without any article, label, answer or result of DEV:

> A legally correct and complete answer can need, besides the provision that states the rule,
> figure or deadline asked for:
> - a definition that decides whether a rule applies or how something is classified;
> - a condition that changes the answer;
> - an exception, exemption or derogation that could change it;
> - the introductory wording that gives a listed point its legal effect;
> - a specific provision that another provision refers to and that the answer needs;
> - a provision that would apply under a fact the question does not state but that could
>   reasonably hold.

It specifies the task; it is not ground truth.

## The model's task

The prompt is `ingest/decomposition.py` (`SYSTEM`, `NECESSARY`, `TASK`, `requirements_prompt`):
the question, the 3 excerpts ("the first results of a search for the question; they can be
incomplete or wrong"), the statement above, and the task:

> Write 2 to 4 questions that a search of the law must answer for the answer to the QUESTION to
> be correct and complete. Each asks about one point of law (a rule, a definition, a condition,
> an exception, a deadline or an amount, or a provision that another provision refers to), in
> English, in one self-contained sentence that names who it concerns and what. Do not answer, do
> not state what the law says, do not copy the excerpts, and write no article numbers and no
> dates.

Answer form: `REQUIREMENTS:` then one `- <question>` per line. Greedy decoding, at most 256 new
tokens, through the MLX worker of D-040 (`scripts/mlx_generate.py`).

**Format check on the 6 fixture queries (data/benchmark/fixtures, not DEV), and its single
amendment.** The first form asked for "the content of one provision" and allowed an article when
an excerpt named it. On the fixtures the 4B wrote statements of the law with figures and article
numbers ("DORA Article 19(4)(a): Financial entities shall submit the initial notification ...
within 24 hours ..."): hypothetical legal text, the failure of HyDE that `docs/fundamento.md`
rules out, joining the search. The form was amended once, to the one above: a requirement asks,
and never states what the law says, an article number or a date. On the fixtures the amended form
gave 6 readable answers of 3 or 4 questions, none not a question, none with an article number, one
with a date, 27 words at the median. No further change. Both checks are recorded
(`retrieval-iteration-4/d-check-qwen3-4b-first-form.json`, `d-check-qwen3-4b.json`).

## Models

- **qwen3-4b**: `config/retrieval/qwen3-4b-instruct-mlx.json`, the model and worker of C and
  round 4.
- **qwen3-8b**: `config/retrieval/qwen3-8b-mlx.json`, `mlx-community/Qwen3-8B-4bit` at revision
  `545dc425`, run without thinking (`chat_template_kwargs: {"enable_thinking": false}`; MLX-LM
  turns thinking on by default for a hybrid model). It is a local **capacity control**: a model of
  the first Qwen3 release, not the 4B-Instruct-2507 at a larger size, so no conclusion of the kind
  "more parameters, better requirements" follows from it alone. It runs only if its pilot on the
  fixture queries stays within both limits: peak MLX memory at most 10 GiB, and a projection of at
  most 3 hours (the slowest fixture prompt times the 90 DEV queries). Swap use is recorded before
  and after. Outside them it is documented as not viable and D goes on with the 4B alone.

## Reading the answer

The items after `REQUIREMENTS:` (or, without that heading, every line that opens with "-", "*",
a bullet or a number); a repeated item counts once; the form's placeholder is not an item; the
first 4 are kept. An answer with fewer than 2 items keeps E5's ranking, and is counted.

**Second amendment, to the reading only, before the 8B's DEV run.** The 8B's pilot on the
fixture queries wrapped each question in the placeholder, as if it were a tag
(`- <question>What is ...?</question>`), and the first parser dropped every item that held the
placeholder: 2 of its 6 fixture answers read as empty. The parser now removes `<question>` and
`</question>` and drops only what is left empty, the bare placeholder. It was applied after the
4B's DEV run, and it reads the 4B's 90 DEV answers exactly as the first parser did (the same items
and the same dropped counts for all 90), so the 4B's result does not change; the 8B's pilot,
read again with it, gives 6 readable answers of 6. Nothing else changed.

## Retrieval and the merge rule, without a model

E5 ranks each requirement over the 1,354 chunks, encoded as E5 encodes a query. **K = 5**, fixed,
with no grid. Then, for each query (`ingest/decomposition.py`, `assemble`):

1. E5's first 6 stay.
2. Places 7 to 10 go, in this order, to:
   1. E5's ranks 7 to 10 that some requirement also has in its first 5 (corroborated);
   2. each requirement's best chunk outside E5's top 10, in turns over the requirements in the
      model's order, each turn skipping a chunk already added;
   3. the rest of E5's ranks 7 to 10, in E5's order.
3. The 10 are ordered by their rank in E5; every other chunk follows in E5's order.

Corroboration answers a known risk: E5 holds the whole evidence of 52 of the 81 evaluable queries
in its top 10 but of 48 in its top 6, and a relevant chunk at rank 9 (N027-en-2) would leave the
10 for a requirement's chunk. It leaves now only if no requirement supports it; that lowers the
risk, it does not remove it. Selection and order stay separate: the order is E5's.

## Measures

The evaluator of every system: Recall@10, nDCG@10, MRR@10, Recall@20, Recall@T, full-evidence@10,
`multi_provision`, `cross_reference` and `cross_act` Recall@10, dependent evidence Recall@10, by
language; macro by need. Paired with E5 by need (Recall@10, nDCG@10, full-evidence@10), 10,000
resamples of needs, the bootstrap averaging by need (corrected on 2026-10-15; every interval of
iterations 1 to 3 except A's is unchanged by it).

Diagnostics:
- the requirements, without a label: per answer, unreadable answers, items dropped, length, not a
  question, an article number, an article no excerpt names, a year, Spanish letters;
- the merge: queries changed, chunks added, E5 ranks 7 to 10 corroborated and displaced;
- with the labels, once every ranking exists: relevant (query, chunk) pairs gained and lost
  against E5's top 10; Recall of the requirements' pools (their first 5, their first 50) and of
  E5's top 10 joined with the first-5 pool (the most the rule could find); N022, N024, N034 and
  N035 query by query, as analysis after the fact.

## Gates

- **Minimum signal**, all three: Recall@10 at least +0.03 over E5, full-evidence@10 at least
  +0.03, nDCG@10 at least 0.518 (values rounded to 3 decimals, as in iteration 3).
- **Useful success**, all four: Recall@10 >= 0.85, `multi_provision` >= 0.60, dependent evidence
  >= 0.45, full-evidence@10 >= 0.75. **Strong**: Recall@10 >= 0.90 and nDCG@10 >= 0.65.
- **Verdict**: `DECOMPOSITION SIGNAL FOUND: YES` if a model passes the minimum signal, otherwise
  `NO`.

## After D

- **No signal**: a diagnosis (the decomposer's capacity, the candidates' recall, or the rule),
  no further prompt round; the person decides whether the next problem deserves training data
  and fine-tuning, which requires a new documented decision.
- **Signal**: D is frozen as it is, with no further tuning on DEV.
- **Either way, no final TEST is consumed to decide.** A new set used to decide anything (how to
  train, for instance) is a holdout or validation set. A final TEST can be built and sealed
  before training, but it is read only once the recipe, the hyperparameters, the checkpoint, the
  inference and the thresholds are frozen.

## Not touched

DEV 0.10.0, its queries, qrels, grades and chunks, the parser and the evaluator. No DEV example in
a prompt; no label read before every ranking exists; K, the rule and the prompt do not change
after a DEV result.
