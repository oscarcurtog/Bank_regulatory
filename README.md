# Fundamento

Engineering foundation for a reproducible RAG system over EU financial regulation.

The goal is a system that answers which regulatory obligations apply to a case and cites the exact
article, verifying every citation before showing it. That system does not exist yet. This
repository contains its foundation: reproducible ingestion of the source texts, a parser for their
legal structure, validated data contracts, measurement tooling, tests, Docker images and CI.

## Status

- **Goal:** a RAG system that answers with verifiable citations to the exact article.
- **Implemented today:** ingestion and source handling, legal hierarchy parsing, Pydantic data
  contracts, stale-layer detection, measurement tooling, tests, Docker and CI.
- **Not implemented yet:** chunking, embeddings, retrieval, generation, citation verification, API,
  evaluation and deployment.

### Roadmap

- **Foundations:** implemented
- **Chunking, indexing and retrieval:** next
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
- **Data contracts** (Pydantic): the manifest and the processed layer (`data/processed/{act}.json`)
  are validated when they are written and when they are read. Loading aborts instead of returning
  partly checked data.
- **Stale-layer detection** (`ingest/processed.py`): the processed layer is rejected if another
  parser version wrote it, or if its source snapshot no longer matches the manifest.
- **Measurement tooling** (`scripts/`): node-level and article-level text and token profiles of the
  corpus.
- **Tests and static checks:** 253 tests (unit, integration, and regression tests for real defects
  found during development), Ruff and mypy.
- **Docker:** a multi-stage build with a non-root runtime image and a separate image for the tests.
- **CI:** GitHub Actions runs the checks and validates the runtime image on every push and pull
  request.

## Repository layout

```
ingest/              library: download, legal hierarchy parser, processed-layer contract
scripts/             pipeline and measurement entry points
tests/unit/          isolated tests, no files involved
tests/integration/   tests on real EUR-Lex excerpts (tests/fixtures/, see PROVENANCE.md)
ci/                  runtime image contract, run by CI
data/                manifest and measurement reports (the corpus itself is not versioned)
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

`requirements-dev.lock.txt` pins every package, with hashes, to the versions CI installs.

```bash
.venv/bin/python -m pytest -rs
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy ingest scripts tests
```

Without the corpus, pytest reports `228 passed, 25 skipped`. The 25 skipped tests are marked
`corpus`: they need `data/raw/`, which is not in the repository (see
[Data and reproducibility](#data-and-reproducibility)).

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
```

## Continuous integration

`.github/workflows/ci.yml` runs two independent jobs on every push and pull request:

- `quality`: installs `requirements-dev.lock.txt` with hash checking, then runs Ruff,
  `ruff format --check`, mypy and pytest.
- `runtime-image`: builds the runtime image and runs `ci/check_runtime_image.py` inside it.

The runner does not have the corpus, so the 25 `corpus` tests are skipped there. A green run does
not validate the full 331-article corpus.

## Data and reproducibility

| Path | Versioned | What it is |
|---|---|---|
| `data/manifest.json` | yes | the exact source documents: URL, SHA-256, size and download date |
| `data/t1_report.json`, `data/t2_node_profile.json`, `data/t2_token_profile.json` | yes | measurement reports, kept as evidence |
| `data/raw/` | no | the downloaded EUR-Lex documents |
| `data/processed/` | no | the processed layer, derived from `data/raw/` |
| `data/t2_nodes.json` | no | per-node measurement output, derived from `data/raw/` |

Without the corpus you can run the test suite (228 tests pass, 25 are skipped), the linters and
type checks, and both Docker images.

With the source documents listed in `data/manifest.json` placed in `data/raw/`, the derived files
can be rebuilt from the repository root:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_t1.py          # data/processed/, data/t1_report.json
PYTHONPATH=. .venv/bin/python scripts/measure_nodes.py   # data/t2_nodes.json, data/t2_node_profile.json
PYTHONPATH=. .venv/bin/python scripts/profile_tokens.py  # data/t2_token_profile.json
```

With documents that match the SHA-256 values in the manifest, the three versioned reports come out
identical byte for byte. `measure_nodes.py` and `profile_tokens.py` download tokenizer files the
first time (`cl100k_base` for tiktoken, and `bert-base-uncased` from the Hugging Face Hub). The
runtime image runs the first step by default:

```bash
docker run --rm -v "$PWD/data:/app/data" fundamento
```

There is no command to download the corpus yet. `ingest.download.fetch_all()` exists as a library
function, but a fresh download is not checked against the SHA-256 values recorded in the manifest.

## Known limitations

- The corpus is not in the repository and there is no download command yet.
- CI skips the 25 `corpus` tests, so the full corpus is only validated where it is present.
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
  runtime image on Linux. On native Linux, a bind-mounted `data/` keeps its host ownership, so the
  pipeline needs it to be writable by UID 10001; this has not been tested.

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
`data/t1_report.json`, and short quotations in the tests.

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
