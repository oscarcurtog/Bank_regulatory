"""A dense model's vectors of the frozen layer and queries, for the experiments that build on a
dense ranking: read from data/embeddings/ only when their full identity matches (the same
check as scripts/embed_baseline.py), never recomputed here; and new query texts (bridges,
reformulations) encoded exactly as the benchmark's queries were.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ingest import embeddings as em
from ingest import retrieval as rt


@dataclass
class DenseSet:
    config: em.EmbeddingConfig
    tokenizer: em.LoadedTokenizer
    doc_ids: tuple[str, ...]
    doc_vectors: np.ndarray
    query_ids: tuple[str, ...]
    query_vectors: np.ndarray
    _encoder: list[Any] = field(default_factory=list)

    def rank(self) -> list[rt.Ranked]:
        return em.rank_exact(
            list(self.query_ids), self.query_vectors, list(self.doc_ids), self.doc_vectors
        )

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Unit vectors of new query texts, with the model's query prefix, as the encoder
        computes the benchmark's queries. Loads the model once, on first use."""
        if not texts:
            return np.zeros((0, self.config.dimension), dtype=np.float32)
        if not self._encoder:
            from ingest.encoder import Encoder

            self._encoder.append(Encoder(self.config))
        inputs = em.prepare_inputs(
            [(str(i), t) for i, t in enumerate(texts)],
            "query",
            self.tokenizer.tokenizer,
            self.config,
        )
        vectors: np.ndarray = self._encoder[0].encode(inputs)
        return vectors


def load_dense(config_path: Path, frozen: rt.FrozenInputs, max_tokens: int) -> DenseSet:
    """The cached document and query vectors of the model, refused unless their identity is
    exactly the current one."""
    config = em.load_config(config_path)
    loaded = em.load_tokenizer(config)
    runtime = em.runtime_identity(config)
    docs = list(frozen.docs)
    version = frozen.dev.manifest.benchmark_version
    doc_meta, doc_vectors = em.load_embedding_set(
        em.embedding_dir(config, f"layer-{max_tokens}"),
        em.artifact_identity(config, loaded, runtime, em.layer_source(docs)),
    )
    query_meta, query_vectors = em.load_embedding_set(
        em.embedding_dir(config, f"queries-{version}"),
        em.artifact_identity(config, loaded, runtime, em.query_source(frozen.dev)),
    )
    if list(doc_meta.item_ids) != list(frozen.chunks):
        raise rt.RetrievalError("the cached document vectors are not in the layer's order")
    return DenseSet(
        config,
        loaded,
        tuple(doc_meta.item_ids),
        doc_vectors,
        tuple(query_meta.item_ids),
        query_vectors,
    )
