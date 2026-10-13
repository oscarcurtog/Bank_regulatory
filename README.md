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
  a DEV set of 30 judged questions), dense retrieval baselines with exact search for two
  embedding models and their paired comparison (explicit truncation, stale-embedding detection),
  Pydantic data contracts, stale-layer detection, measurement tooling, tests, Docker and CI.
- **Not implemented yet:** a vector index, hybrid search and reranking, generation, citation
  verification, API and deployment.

### Roadmap

- **Foundations:** implemented
- **Chunking:** implemented (structural, v1)
- **Retrieval benchmark:** implemented; DEV holds 30 judged questions, reviewed and sealed under
  protocol v1.4
- **Indexing and retrieval:** dense baselines with exact search implemented for
  `multilingual-e5-large` and BGE-M3 (dense output), compared on DEV; lexical, hybrid and
  reranked retrieval not measured yet
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
- **Data contracts** (Pydantic): the manifest, the processed layer (`data/processed/{act}.json`),
  the chunk layer (`data/chunks/{act}.json`) and the benchmark files are validated when they are
  written and when they are read. Loading aborts instead of returning partly checked data.
- **Stale-layer detection** (`ingest/processed.py`): the processed layer is rejected if another
  parser version wrote it, or if its source snapshot no longer matches the manifest.
- **Measurement tooling** (`scripts/`): node-level and article-level text and token profiles of the
  corpus.
- **Tests and static checks:** 709 tests (unit, integration, and regression tests for real defects
  found during development), Ruff and mypy.
- **Docker:** a multi-stage build with a non-root runtime image and a separate image for the tests.
- **CI:** GitHub Actions runs the checks and validates the runtime image on every push and pull
  request.

## Repository layout

```
ingest/              library: download, parser, processed and chunk layers, benchmark, embeddings
config/embeddings/   the contract of each embedding model, and the model comparison policy
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
not in the repository (see [Data and reproducibility](#data-and-reproducibility)). The 15 tests
marked `embedding_model` need the `embed` extra and the weights of the three embedding models
(1.2 to 2.3 GB each, downloaded once into `.cache/huggingface`); without them they fail, unless
the environment declares `FUNDAMENTO_EMBEDDING_MODEL=unsupported`, as CI does. With that
declaration and without the corpus, pytest reports `656 passed, 53 skipped`.

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
`FUNDAMENTO_EMBEDDING_MODEL=unsupported` for the 15 tests marked `embedding_model`.

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

Without the corpus you can run the test suite (656 tests pass and 53 are skipped with
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
`multilingual-e5-large` can be told apart from noise: in English nDCG@10 is 0.536 against 0.552 and Recall@10 0.761 against 0.777; in Spanish 0.498 against
0.479 and 0.723 against 0.735. Every paired 95% interval contains zero, and 14 queries improve
while 14 get worse. The main failure, where the chunk that answers ranks first and the evidence it
depends on (definitions, conditions, the other act) ranks far below, is the same for both models.
Both are reproduced bit for bit by a new process.

## Known limitations

- The corpus is not in the repository and there is no download command yet.
- CI skips the 38 `corpus` tests, so the full corpus is only validated where it is present.
- The DEV benchmark holds 30 judged questions in two batches (27 evaluable), so a difference
  between two retrievers needs to be large to be told apart from noise. Two dense retrievers
  (two embedding models, exact search) have been measured on it; no index, lexical or hybrid
  search or reranker has been measured yet. The benchmark evaluator is not in the Docker images
  (no Linux ARM64 wheel for its backend).
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
