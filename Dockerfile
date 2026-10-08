# Fundamento: runtime image for the ingestion pipeline (EF-6).
#
# Two stages with different purposes:
#   runtime  (the one that runs)  only what is needed to run the pipeline
#   dev                           runtime + pytest, ruff and mypy, to run the suite
#
# Always built with an explicit --target (see D-011):
#   docker build --target runtime -t fundamento .
#   docker build --target dev -t fundamento:dev .
#
# NO data goes into the image. data/ is mounted at run time (see below).

# ---------------------------------------------------------------------------
# Base stage: what both images share
# ---------------------------------------------------------------------------
# Exact version, not "3.12": EF-1 pinned CPython 3.12.14 and M-007 showed that
# the runtime change does not alter any data. Leaving the version floating would
# give that up. The next level of rigour would be pinning the image digest too
# (python:3.12.14-slim@sha256:...): every build resolves it and prints it in its
# log, but it is not pinned here yet.
#
# slim (Debian, glibc) and not alpine (musl): checked on PyPI that pydantic-core,
# tiktoken and tokenizers DO publish musllinux wheels, so alpine would work
# without compiling. slim is chosen anyway because the saving is about 80 MB that
# changes nothing here, glibc is the most travelled path, and tokenizers
# publishes 3 manylinux aarch64 wheels against 1 musllinux: more margin if the
# architecture changes.
FROM python:3.12.14-slim AS base

# No .pyc in the image and no output buffering, so that the logs of a batch job
# come out in order.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# The ingest package is imported through PYTHONPATH instead of being installed.
# Installing it would require hatchling, which is NOT in requirements.lock.txt
# because it is a build dependency: it would be a download without a hash and
# would break exactly the reproducibility the lock exists to provide.
ENV PYTHONPATH=/app

# tiktoken downloads cl100k_base (1.7 MB) from openaipublic.blob.core.windows.net
# the first time and keeps it in TIKTOKEN_CACHE_DIR. This is the directory that
# ingest/chunks.py uses when the variable is unset (.cache/tiktoken under the
# working directory), written as an absolute path so that it does not depend on
# the working directory. The image starts with it empty; mounting the host's
# .cache/tiktoken on it reuses the file without the network.
ENV TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken

WORKDIR /app

# Dependencies first and ALONE. This layer is rebuilt only when the lock
# changes, not when the code changes: if the code were copied first, every
# one-line edit would invalidate the whole install.
#
# --require-hashes is the payoff of EF-1: the lock carries the hash of every
# artifact (tiktoken 64, pydantic-core 120), including the manylinux wheels for
# x86_64 and aarch64, so it installs the same way inside the container and a
# tampered package on the index does not go unnoticed.
COPY requirements.lock.txt ./
RUN pip install --require-hashes --no-deps -r requirements.lock.txt

# The code afterwards, since it is what changes often.
COPY ingest/ ./ingest/
COPY scripts/ ./scripts/

# Unprivileged user. A batch pipeline does not need root, and if the container
# is compromised the attacker does not start with the whole box.
# In runtime only the two folders the pipeline needs are writable. The code
# (ingest/, scripts/) stays owned by root: a chown -R over /app would have made
# it writable too.
RUN useradd --create-home --uid 10001 fundamento \
    && mkdir -p /app/data /app/.cache/tiktoken \
    && chown fundamento:fundamento /app/data /app/.cache/tiktoken
USER fundamento

# ---------------------------------------------------------------------------
# Dev stage: the suite inside the container
# ---------------------------------------------------------------------------
# It exists because condition 7 of the EF definition of done asks for the suite
# to run in a container, and that is incompatible with a clean runtime image.
# Two stages solve it: the one above carries neither pytest nor ruff nor mypy.
#
# Run it with data/ mounted. The manifest tests read data/manifest.json, which is
# not copied into the image, so without the mount 5 of them fail. Tests marked
# "corpus" also need data/raw/ and skip themselves when it is missing, as
# designed in EF-2:
#   docker run --rm -v "$PWD/data:/app/data" fundamento:dev
FROM base AS dev
USER root
RUN pip install --no-cache-dir "pytest>=8" "ruff>=0.16" "mypy>=2"
COPY pyproject.toml ./
COPY tests/ ./tests/
# Here the chown IS recursive, on purpose: pytest, ruff and mypy write their
# caches in /app. It is the price of a development image.
RUN chown -R fundamento:fundamento /app
USER fundamento
CMD ["python", "-m", "pytest", "-q"]

# ---------------------------------------------------------------------------
# Runtime stage: the image that runs
# ---------------------------------------------------------------------------
# It goes LAST on purpose. In a multi-stage Dockerfile, "docker build" without
# --target builds the last stage, so leaving "dev" at the end made the normal
# command produce the development image: 442 MB with pytest, ruff, mypy and the
# tests inside. Only a real build exposes it.
FROM base AS runtime
USER fundamento

# run_t1.py is the only pipeline step that does NOT need the network: it reads
# the local XHTML and writes the processed layer. That is why it is the default
# command. The other scripts are launched by overriding it, and they need the
# network or a mounted cache: run_t2, measure_nodes and profile_tokens need
# cl100k_base, and profile_tokens also bert-base-uncased from Hugging Face.
CMD ["python", "scripts/run_t1.py"]
