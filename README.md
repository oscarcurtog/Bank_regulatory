# Fundamento

Engineering foundation for a reproducible RAG system over EU financial regulation.

The goal is a system that answers which regulatory obligations apply to a case and cites the exact
article, verifying every citation before showing it. That system does not exist yet. This
repository contains its foundation: reproducible ingestion of the source texts, a parser for their
legal structure, validated data contracts, measurement tooling, tests, Docker images and CI.

## Status

- **Goal:** a RAG system that answers with verifiable citations to the exact article.
- **Implemented today:** ingestion and source handling, legal hierarchy parsing, structural
  chunking, a retrieval benchmark (contracts, validation, node-to-chunk projection, metrics and
  a DEV set of 30 judged questions), retrieval baselines measured on DEV (three dense embedding
  models with exact search, BM25, and a one-hop cross-reference expansion) under a protocol fixed
  before the results, Pydantic data contracts, stale-layer detection, measurement tooling, tests,
  Docker and CI.
- **Not implemented yet:** a vector index, hybrid search and reranking, generation, citation
  verification, API and deployment.

### Roadmap

- **Foundations:** implemented
- **Chunking:** implemented (structural, v1)
- **Retrieval benchmark:** implemented; DEV holds 30 judged questions, reviewed and sealed under
  protocol v1.4
- **Indexing and retrieval:** three retrieval iterations on DEV are done (three dense models,
  BM25, cross-reference expansion, dependency retrieval, graph propagation, two rerankers, a
  local language model as set selector, closed-catalog query reformulation); no configuration
  beat `multilingual-e5-large` by the protocols' margin
- **Generation, evaluation, API and deployment:** later

## What is implemented

The corpus covers three acts, 331 articles in total, downloaded on 2026-09-05:

| Act | Source document (CELEX) | Articles |
|---|---|---|
| DORA, Regulation (EU) 2022/2554 | text published in the Official Journal (`32022R2554`) | 64 |
| MiCA, Regulation (EU) 2023/1114 | consolidated version (`02023R1114-20240109`) | 150 |
| PSD2, Directive (EU) 2015/2366 | consolidated version (`02015L2366-20151223`) | 117 |

- **Ingestion** (`ingest/download.py`): downloads each act from CELLAR, the Publications Office
  repository behind EUR-Lex. A response only counts if it has status 200 and a realistic size, and
  each document is recorded in `data/manifest.json` with its URL, SHA-256, size and download date.
- **Legal hierarchy parsing** (`ingest/hierarchy.py`): turns EUR-Lex XHTML into articles and
  citable nodes (article, paragraph, subparagraph, point) with materialized paths such as `45.1.a`
  or `9.1.sub_5.a`. The first subparagraph of a paragraph stays implicit, so the usual paths keep
  their form; from the second on, a subparagraph is a node of its own (`sub_<k>`, or `unp_<k>`
  for the unnumbered paragraphs of an article). Every node keeps its own text, every article its
  full text, and a path identifies exactly one node of its article. It handles the two markup
  families EUR-Lex uses for original and consolidated texts.
- **Structural chunking** (`ingest/chunks.py`, `scripts/run_t2.py`): each article is cut into the
  largest legal unit that fits in a token budget, measured with `cl100k_base` on the exact text
  that will be embedded: the article heading, the text of the ancestors that govern the unit, and
  the unit itself. The budget is a run parameter (500 by default). A paragraph that does not fit
  keeps its own text in a chunk of its own, and every source text belongs to exactly one chunk.
  Chunk IDs are structural, such as `MiCA:3.1.5` or `MiCA:3.1:own`.
- **Retrieval benchmark** (`ingest/benchmark.py`, `ingest/evaluation.py`, `ingest/ir_metrics.py`,
  `scripts/benchmark.py`): information needs, their queries in English and Spanish, and relevance
  judgments graded 0-3 on legal nodes (`MiCA:50.2`), never on chunks, so the same judgments
  evaluate any chunking. See [Retrieval benchmark](#retrieval-benchmark).
- **Retrieval** (`ingest/embeddings.py`, `ingest/encoder.py`, `ingest/lexical.py`,
  `ingest/references.py`, `ingest/retrieval.py`, `ingest/run_report.py`): dense retrieval with
  exact search under a per-model contract, BM25, explicit cross-reference expansion, rank fusion,
  and one report that measures every retriever the same way. See
  [Retrieval iteration](#retrieval-iteration).
- **Data contracts** (Pydantic): the manifest, the processed layer (`data/processed/{act}.json`),
  the chunk layer (`data/chunks/{act}.json`) and the benchmark files are validated when they are
  written and when they are read. Loading aborts instead of returning partly checked data.
- **Stale-layer detection** (`ingest/processed.py`): the processed layer is rejected if another
  parser version wrote it, or if its source snapshot no longer matches the manifest.
- **Measurement tooling** (`scripts/`): node-level and article-level text and token profiles of the
  corpus.
- **Tests and static checks:** 764 tests (unit, integration, and regression tests for real defects
  found during development), Ruff and mypy.
- **Docker:** a multi-stage build with a non-root runtime image and a separate image for the tests.
- **CI:** GitHub Actions runs the checks and validates the runtime image on every push and pull
  request.

## Repository layout

```
ingest/              library: download, parser, processed and chunk layers, benchmark, retrieval
config/embeddings/   the contract of each embedding model, and the model comparison policy
config/retrieval/    the retrieval iterations' protocols and the contracts of BM25, fusion, rerankers
scripts/             pipeline and measurement entry points
tests/unit/          isolated tests, no files involved
tests/integration/   tests on real EUR-Lex excerpts (tests/fixtures/, see PROVENANCE.md)
ci/                  runtime image contract, run by CI
data/                manifest, measurement reports and the benchmark (the corpus is not versioned)
```

## Running the checks

Requirements: [uv](https://docs.astral.sh/uv/) and, for the images, Docker. The commands are for a
macOS or Linux shell.

```bash
git clone https://github.com/oscarcurtog/Bank_regulatory.git
cd Bank_regulatory
uv venv --python 3.12.14
uv pip install --require-hashes --no-deps -r requirements-dev.lock.txt
```

`requirements-dev.lock.txt` pins every package, with hashes, to the versions CI installs. The
dense retrieval baseline also needs the optional `embed` extra (torch and transformers), which
CI and the images do not install:

```bash
uv pip install --require-hashes --no-deps -r requirements-embed.lock.txt
```

```bash
.venv/bin/python -m pytest -rs
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy ingest scripts tests
```

Without the corpus, pytest skips the 38 tests marked `corpus`: they need `data/raw/`, which is
not in the repository (see [Data and reproducibility](#data-and-reproducibility)). The 31 tests
marked `embedding_model` need the `embed` extra and the weights of the three embedding models and
the two rerankers (1.2 to 2.3 GB each, downloaded once into `.cache/huggingface`); two of them also
need the 4-bit MLX build of Qwen3-4B-Instruct-2507 in its own environment, `.cache/mlx-venv`, on
Apple silicon (see [config/retrieval/necessity-selection.md](config/retrieval/necessity-selection.md)).
Without them they fail, unless the environment declares `FUNDAMENTO_EMBEDDING_MODEL=unsupported`, as
CI does. With that declaration and without the corpus, pytest reports `695 passed, 69 skipped`.

## Docker

```bash
docker build --target runtime -t fundamento .
docker run --rm -i fundamento python - < ci/check_runtime_image.py
```

The runtime image runs as a non-root user (UID 10001). Inside `/app` it can only write to `data/`
and `.cache/tiktoken`, and it contains no test tooling. The second command checks those properties
inside the image, as CI does. The project always passes `--target` explicitly: without it, Docker
builds whichever stage comes last in the Dockerfile.

The tests have their own image, which needs `data/` mounted because the manifest tests read
`data/manifest.json`:

```bash
docker build --target dev -t fundamento:dev .
docker run --rm -v "$PWD/data:/app/data" fundamento:dev
# the same, reusing the host's tokenizer file instead of downloading it (see Tokenizer file)
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/.cache/tiktoken:/app/.cache/tiktoken" fundamento:dev
```

The test image does not install the benchmark evaluator (`ir_measures`): its backend,
`pytrec-eval-terrier`, publishes no Linux ARM64 wheel. The image declares
`FUNDAMENTO_EVALUATOR=unsupported`, which is the only thing that skips the 17 tests marked
`evaluator`, and pytest prints the reason. Those tests run in the project venv and in CI. In the
same way, neither image installs torch or the embedding model, and the test image declares
`FUNDAMENTO_EMBEDDING_MODEL=unsupported` for the 31 tests marked `embedding_model`.

## Continuous integration

`.github/workflows/ci.yml` runs two independent jobs on every push and pull request:

- `quality`: installs `requirements-dev.lock.txt` with hash checking, then runs Ruff,
  `ruff format --check`, mypy and pytest. Before pytest it restores `.cache/tiktoken` from the
  GitHub Actions cache, keyed by runner OS, tiktoken version and encoding, and loads the
  tokenizer, so only a run with a cold cache downloads it.
- `runtime-image`: builds the runtime image and runs `ci/check_runtime_image.py` inside it.

The runner does not have the corpus, so the 38 `corpus` tests are skipped there. A green run does
not validate the full 331-article corpus. The `evaluator` tests do run there: CI checks the
benchmark machinery on synthetic data, while evaluation against the real corpus runs locally.

Reading a job log needs a signed-in GitHub account, so the test run also prints its summary as
workflow commands: the counts, each skip reason and the versions of the evaluator backend appear
as annotations on the check run, which the public API serves to anyone.

## Data and reproducibility

| Path | Versioned | What it is |
|---|---|---|
| `data/manifest.json` | yes | the exact source documents: URL, SHA-256, size and download date |
| `data/t1_report.json`, `data/t2_node_profile.json`, `data/t2_token_profile.json`, `data/t2_chunk_profile.json` | yes | measurement reports, kept as evidence |
| `data/raw/` | no | the downloaded EUR-Lex documents |
| `data/processed/` | no | the processed layer, derived from `data/raw/` |
| `data/t2_nodes.json` | no | per-node measurement output, derived from `data/raw/` |
| `data/chunks/` | no | the chunk layer, derived from `data/processed/` |
| `.cache/tiktoken/` | no | the tokenizer file, downloaded the first time (see [Tokenizer file](#tokenizer-file)) |
| `data/benchmark/` | yes | the retrieval benchmark: DEV, fixtures, protocol, templates, PSD2 exclusions |
| `data/reference/` | no | later versions of an act, downloaded only to derive the PSD2 exclusions |

Without the corpus you can run the test suite (695 tests pass and 69 are skipped with
`FUNDAMENTO_EMBEDDING_MODEL=unsupported`), the linters and type checks, and both Docker images.

With the source documents listed in `data/manifest.json` placed in `data/raw/`, the derived files
can be rebuilt from the repository root:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_t1.py          # data/processed/, data/t1_report.json
PYTHONPATH=. .venv/bin/python scripts/measure_nodes.py   # data/t2_nodes.json, data/t2_node_profile.json
PYTHONPATH=. .venv/bin/python scripts/profile_tokens.py  # data/t2_token_profile.json
PYTHONPATH=. .venv/bin/python scripts/run_t2.py          # data/chunks/, data/t2_chunk_profile.json
```

With documents that match the SHA-256 values in the manifest, the four versioned reports come out
identical byte for byte. The last three scripts need the tokenizer file (see
[Tokenizer file](#tokenizer-file)), and `profile_tokens.py` also downloads `bert-base-uncased` from
the Hugging Face Hub. The runtime image runs the first step by default:

```bash
docker run --rm -v "$PWD/data:/app/data" fundamento
```

There is no command to download the corpus yet. `ingest.download.fetch_all()` exists as a library
function, but a fresh download is not checked against the SHA-256 values recorded in the manifest.

### Tokenizer file

`run_t2.py`, `measure_nodes.py`, `profile_tokens.py` and the chunker tests (also without the
corpus) count tokens with tiktoken's `cl100k_base`. Its file (1.7 MB) does not ship with the
tiktoken package: it is downloaded the first time and kept in `.cache/tiktoken` under the working
directory, or in `TIKTOKEN_CACHE_DIR` if that is set, as it is in the images
(`/app/.cache/tiktoken`). tiktoken checks the file against a SHA-256 pinned in its own source,
both when it downloads it and whenever it reads it from the cache. Once the file is there, nothing
needs the network. Without the file and without network they stop with an explicit error: they
never count with another tokenizer or with an estimate.

### Retrieval benchmark

A benchmark directory holds `needs.jsonl` (what a user wants to know, the unit of the DEV/TEST
split), `queries.jsonl` (how it is asked: English, and natural Spanish), `qrels.jsonl` (which
legal nodes answer it, graded 0-3, each with a rationale and the SHA-256 of the judged text) and a
`manifest.json` that seals them against the parser, the corpus snapshot and the files. The
evaluation datasets under `data/benchmark/` contain Spanish queries on purpose, because
retrieval from Spanish questions over the English corpus is one of the things measured; code
and documentation stay in English.

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark.py validate data/benchmark/dev
PYTHONPATH=. .venv/bin/python scripts/benchmark.py seal data/benchmark/dev --role dev --version 0.2.0
PYTHONPATH=. .venv/bin/python scripts/benchmark.py prompt data/benchmark/dev N004
PYTHONPATH=. .venv/bin/python scripts/build_psd2_exclusions.py --check
```

`validate` checks the seal, the dataset rules and the corpus, and reports the lexical overlap of
every query with its evidence. `seal` is the only writer of a manifest and writes nothing if a
check fails. `prompt` prints the query-variant prompt of a question, built from its statement
alone. `data/benchmark/psd2_excluded_articles.json` lists the nine PSD2 articles amended after the
parsed snapshot (by Directive 2022/2556 and Regulation 2024/886); no question may rely on them. It
is derived by comparing the parsed version with the consolidated version of 2025-01-17, kept in
`data/reference/`, without changing which version Fundamento parses.

`data/benchmark/dev/` holds 30 DEV needs in two batches, built and judged with
`data/benchmark/PROTOCOL.md`. Every need and its queries were committed before any evidence was
searched. Every candidate was graded by two judges who did not see each other's grades, and every
disagreement by an independent third judge; `adjudications.jsonl` records the three judgments
next to the data. `data/benchmark/dev/CALIBRATION.md` records what the batch measured. Its
proposals became protocol v1.1, and the review of the batch against v1.1 led to v1.2: a
definition counts only when it is necessary, a hard negative needs a scope check, and the first
judge never settles its own disagreement. Under v1.2 two needs were replaced (N005 by N014, N007
by N015) and four were narrowed, each with new queries and its own lock. The methodological review
of the batch led to v1.3: a grade 3 needs the facts of the need to establish that the node
applies, ambiguity is kept for interpretation, and the drivers have a definition. Batch 2 (N016 to
N035) was planned, written, locked and judged under v1.3, and the same calibration record
covers it. A methodological review of the 30 needs followed, before any retrieval result existed.
It led to v1.4: answerability is judged against the legal snapshot that the manifest declares,
and `cross_reference` has a definition. The review also applied the necessity test to every
definition graded 2 again, and DEV was sealed at 0.10.0.

### Dense retrieval baseline

```bash
for model in multilingual-e5-large bge-m3; do
  PYTHONPATH=. .venv/bin/python scripts/embed_baseline.py \
      --config config/embeddings/$model.json --expect-version 0.10.0
  PYTHONPATH=. .venv/bin/python scripts/report_embedding_run.py \
      data/results/dev-0.10.0/$model-layer-500/summary.json
done
PYTHONPATH=. .venv/bin/python scripts/compare_embedding_runs.py \
    multilingual-e5-large bge-m3 --expect-version 0.10.0
```

The contract of each embedding model is a versioned file in
[`config/embeddings/`](config/embeddings/README.md), with the source of every value and the model
comparison policy: one fixed chunk layer, a tokenizer check per model, and explicit, recorded
truncation of a chunk that exceeds the model's input limit. `ingest/embeddings.py` counts every
input whole before encoding, cuts only documents and only at the tail, refuses to cut a query, and
ranks exactly (one dot product per query and chunk; with L2-normalized vectors that is the cosine).
`ingest/encoder.py` runs the model on token ids, never on text, so it cannot truncate by itself.

The embeddings (`data/embeddings/`) and the full run (`data/runs/`) are derived and not versioned.
Their `meta.json` records the full identity they were computed with: the model contract, the
tokenizer file, the runtime versions and the chunk layer or queries they came from. A set whose
identity differs in any field is refused as stale, naming the fields that changed. The summary and
the report of each run, and the paired comparison, are versioned in `data/results/`.

The comparison changes only the model: same DEV 0.10.0, chunk layer, queries, qrels, projection
and evaluator ([report](data/results/dev-0.10.0/comparisons/multilingual-e5-large-vs-bge-m3-layer-500/comparison.md)).
On DEV's 27 evaluable needs, no difference between BGE-M3 (dense output) and
`multilingual-e5-large` can be told apart from noise: in English nDCG@10 is 0.536 against 0.552
and Recall@10 0.761 against 0.777; in Spanish 0.498 against 0.479 and 0.723 against 0.735. Every
paired 95% interval contains zero. Counted by any change of nDCG@10, 38 queries improve, 38 get
worse and 5 are equal; counted by the report's failure transitions (a query fails when a relevant
chunk is outside its top 10), 14 move to a better state, 14 to a worse one and 53 keep theirs.
The main failure, where the chunk that answers ranks first and the evidence it
depends on (definitions, conditions, the other act) ranks far below, is the same for both models.
Both are reproduced bit for bit by a new process.

### Retrieval iteration

```bash
PYTHONPATH=. .venv/bin/python scripts/bm25_baseline.py --expect-version 0.10.0
PYTHONPATH=. .venv/bin/python scripts/retrieval_oracle.py --expect-version 0.10.0
PYTHONPATH=. .venv/bin/python scripts/xref_expansion.py --base multilingual-e5-large \
    --expect-version 0.10.0
PYTHONPATH=. .venv/bin/python scripts/embed_baseline.py \
    --config config/embeddings/qwen3-embedding-0.6b.json --expect-version 0.10.0
PYTHONPATH=. .venv/bin/python scripts/retrieval_leaderboard.py --expect-version 0.10.0
```

A sequence of controlled experiments on the frozen DEV benchmark, each one chosen by a rule
applied to the previous result. The rules, the grids and the measures were committed before the
first result ([`config/retrieval/README.md`](config/retrieval/README.md)), and every run is in
[the record](data/results/dev-0.10.0/retrieval-iteration/log.md), losers included.

- **BM25** (`ingest/lexical.py`, Lucene's variant through bm25s): global Recall@10 0.366 against
  E5's 0.763. Its union with E5's top 10 adds 0.006 of Recall@10, so a hybrid was not built.
- **One-hop cross-reference expansion** (`ingest/references.py`, a deterministic parser of
  explicit references): it reaches real dependent evidence (dependent Recall@10 0.300 to 0.380)
  but most references name whole articles, 1.3% of the added chunks are relevant, and global
  Recall@10 falls to 0.653.
- **A third dense model**, Qwen3-Embedding-0.6B: global Recall@10 0.730, under E5; dense model
  swapping stops there.
- **No reranker**: no retriever reaches the Recall@50 of 0.95 that would give it the evidence
  to reorder (best 0.894).

[The leaderboard](data/results/dev-0.10.0/retrieval-iteration/leaderboard.md) holds every
system, measured by one code path. E5 stays the best retriever measured. What it misses is
mostly evidence the answer depends on without being the answer: 54 of its 63 missed (query,
relevant chunk) pairs are grade 2, at a median rank of 103.

A second iteration ([protocol](config/retrieval/dependency-retrieval.md),
[record](data/results/dev-0.10.0/retrieval-iteration-2/log.md)) targeted that evidence.
Anchor-conditioned dependency retrieval (`ingest/dependencies.py`: references, defined terms and
definitions by reference read from E5's top 10, plus "What is <term>?" bridges) gathers it: the
candidate list's Recall@100 rises from 0.883 to 0.947 and its dependent evidence at 50 from 0.582
to 0.774. Ordering it failed three times: a dense-seeded Personalized PageRank over the law's
graph, the cross-encoder `bge-reranker-v2-m3` with the question or with the bridges
(`ingest/reranker.py`), and a fixed quota of the top 10. None changed Recall@10 by a material
amount; the bridge-scored reranking fell to 0.629.

A third iteration ([protocol](config/retrieval/necessity-selection.md),
[record](data/results/dev-0.10.0/retrieval-iteration-3/log.md)) asked a different question of
the same evidence: not "is this passage relevant?" but "is it necessary to apply the anchor to the
question?". `Qwen3-Reranker-0.6B`, shown the anchor, the relation and the clause that creates it,
said "yes" to nearly everything and ranked worse than without them (average precision 0.185
against 0.238), so its evidence-set step did not run. A local 4-bit `Qwen3-4B-Instruct-2507`
(MLX, in its own environment) then judged the legal function of each candidate of an expanded
population whose oracle reaches Recall@10 0.948. It used 14% of that margin (Recall@10 0.789
against E5's 0.763, inside the noise) and worsened nDCG@10 by 0.067, because it judges most
candidates necessary. Rewriting the question with the facts, or the facts and concepts copied from
a closed catalog of the corpus's own headings and defined terms, and fusing it with E5, lowered
Recall@10 to 0.729 and 0.701. E5 stays the reference and, by the iteration's registered stop
rule, optimization on DEV stops.

A fourth round, D ([protocol](config/retrieval/decomposition.md),
[record](data/results/dev-0.10.0/retrieval-iteration-4/log.md)), reopened DEV once at zero
additional cost. A local model reads the question and E5's first three results and writes two to
four questions about the points of law a complete answer needs. E5 searches each one over the
whole corpus, and a fixed rule keeps E5's first six and gives places 7 to 10 to what the questions
find. With Qwen3-4B, Recall@10 rose to 0.803 and nDCG@10 to 0.552. With Qwen3-8B, run without
thinking as a capacity control, they rose to 0.794 and 0.548, with dependent evidence at 0.460
(E5: 0.300) and full-evidence at 0.691. For both, nDCG@10's paired interval excludes zero: the
first gain over E5 of the iterations to do so. The 8B passes the registered minimum signal and the 4B misses it
by 0.005 of full-evidence; neither reaches the useful target (Recall@10 0.85). The questions reach
most of the missing evidence in their first 50 results but less than half of it in their first 5.

## Known limitations

- The corpus is not in the repository and there is no download command yet.
- CI skips the 38 `corpus` tests, so the full corpus is only validated where it is present.
- The DEV benchmark holds 30 judged questions in two batches (27 evaluable), so a difference
  between two retrievers needs to be large to be told apart from noise (about 0.08 of nDCG@10).
  The retrieval iterations selected on DEV, so their numbers are development numbers; an
  independent TEST set has not been built. No index or hybrid search has been measured, and
  neither the cross-encoder rerankers nor a local language model judging candidates improved on
  E5; the decomposition of round D improves nDCG@10 on DEV, and has not been tested on new needs. The benchmark evaluator is not in the Docker images (no Linux ARM64 wheel for its
  backend).
- Subparagraphs are detected from the markup alone, without reading the words. As a result, a
  heading line inside a paragraph (the "Method A/B/C" lines of PSD2 Article 9(1)) and a quoted text
  after a colon become subparagraphs of their own. Inside a point nothing is split, so in 4
  articles the text a point has after its sub-points stays joined to its lead-in.
- The node `kind` and markup `family` fields use five legacy serialized schema codes, documented in
  `ingest/processed.py`; the two node kinds added for subparagraphs are English. Renaming the
  legacy codes is a deliberate schema migration that has not been done.
- The test image installs pytest, Ruff and mypy without pinning their versions. CI uses the hashed
  lock.
- The images have been built and run with Docker Desktop on macOS, and CI builds and checks the
  runtime image on Linux. On native Linux, a bind-mounted `data/` or `.cache/tiktoken` keeps its
  host ownership, so it needs to be writable by UID 10001; this has not been tested.

## Engineering records

Code comments cite identifiers such as `D-004`, `M-006` or `EF-3`. They refer to internal
engineering decision and measurement records kept outside this public repository. They are not
needed to run the code or to understand its behaviour.

## License and sources

The code in this repository is licensed under the [MIT License](LICENSE).

The MIT License does not cover the regulatory texts. The source documents come from
[EUR-Lex](https://eur-lex.europa.eu), © European Union, 1998-2026. The repository contains parts
of them: the excerpts in `tests/fixtures/` (sources in
[PROVENANCE.md](tests/fixtures/PROVENANCE.md)), the article titles and text previews in
`data/t1_report.json`, and short quotations in the tests and in the rationales of the benchmark
(`data/benchmark/`).

- Legal documents published in EUR-Lex can be reused under the Commission's reuse policy
  (Decision 2011/833/EU).
- EUR-Lex consolidated texts, the source used here for MiCA and PSD2, are licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), which requires acknowledging the
  source and indicating changes. The changes made here are: extracting excerpts; in derived text,
  removing consolidation markers and footnote text (footnote references become `<FN>`) and
  normalizing whitespace, quotation marks and punctuation; and truncating the previews.

EUR-Lex marks each consolidated text as "meant purely as a documentation tool" with "no legal
effect", and its legal notice states: "Only European Union documents published in the Official
Journal of the European Union are deemed authentic." Nothing in this repository is legal advice.
