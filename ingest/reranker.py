"""A cross-encoder reranker under an explicit contract (config/retrieval/bge-reranker-v2-m3.json).

The contract, read from the model's files at the pinned revision: BAAI/bge-reranker-v2-m3
(Apache-2.0, multilingual), an XLM-RoBERTa sequence classifier with one output, initialized from
BGE-M3; 8194 positions, so 8192 input tokens; the score is the raw logit (higher is more
relevant; the model card's sigmoid only maps it into [0, 1] and changes no order). The card's
example cuts pairs at 512 tokens silently; here the limit is the model's, and a pair over it is
cut at the passage's tail and recorded, never silently. A query is never cut.

A pair is the model's own template, <s> query </s></s> passage </s>, as the tokenizer builds it.
torch and transformers are the optional 'embed' extra, loaded only when a Reranker is built.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ingest import embeddings as em


class RerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    schema_version: Literal["1"]
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    model_id: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    revision: str = Field(pattern=em.REVISION_PATTERN)
    tokenizer_file: str
    max_input_tokens: int = Field(ge=8)
    pair_truncation: Literal["explicit_passage_tail"]
    score: Literal["logit"]
    batch_size: int = Field(ge=1)
    device: Literal["cpu"]
    dtype: Literal["float32"]

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: Path) -> RerankerConfig:
    return RerankerConfig.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Pair:
    ids: tuple[int, ...]
    original_tokens: int
    truncated: bool


class Reranker:
    def __init__(self, config: RerankerConfig) -> None:
        try:
            import torch
            import transformers
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as e:
            raise em.ModelUnavailableError(
                "torch and transformers are not installed: install requirements-embed.lock.txt"
            ) from e
        self.config = config
        self._torch = torch
        cache = em.hf_home() / "hub"
        os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
        try:
            path = hf_hub_download(
                config.model_id, config.tokenizer_file, revision=config.revision, cache_dir=cache
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                config.model_id, revision=config.revision, cache_dir=cache, dtype=torch.float32
            )
        except Exception as e:
            raise em.ModelUnavailableError(
                f"{config.model_id}@{config.revision[:12]}: cannot load the reranker from the "
                f"cache {cache} or the network"
            ) from e
        self.tokenizer: Any = Tokenizer.from_file(path)
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.tokenizer_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if model.config.num_labels != 1:
            raise em.EmbeddingError(f"the reranker has {model.config.num_labels} outputs, not 1")
        positions = model.config.max_position_embeddings - (model.config.pad_token_id or 0) - 1
        if positions < config.max_input_tokens:
            raise em.EmbeddingError(f"{positions} usable positions, the contract says more")
        self.pad_id = int(model.config.pad_token_id)
        self.model = model.to(config.device).eval()

    def pair(self, query: str, passage: str) -> Pair:
        """The model's own pair template; over the limit, the passage's tail is cut."""
        full = list(self.tokenizer.encode(query, passage).ids)
        if len(full) <= self.config.max_input_tokens:
            return Pair(tuple(full), len(full), False)
        q = list(self.tokenizer.encode(query, add_special_tokens=False).ids)
        p = list(self.tokenizer.encode(passage, add_special_tokens=False).ids)
        cls, sep = full[0], full[-1]
        room = self.config.max_input_tokens - len(q) - 4
        if room < 1:
            raise em.TruncationRefusedError("a query fills the reranker's input on its own")
        ids = [cls, *q, sep, sep, *p[:room], sep]
        return Pair(tuple(ids), len(full), True)

    def score(self, pairs: Sequence[Pair]) -> np.ndarray:
        """One logit per pair, in input order; batches go longest first."""
        torch = self._torch
        out = np.empty(len(pairs), dtype=np.float64)
        order = sorted(range(len(pairs)), key=lambda i: (-len(pairs[i].ids), i))
        with torch.inference_mode():
            for start in range(0, len(order), self.config.batch_size):
                batch = order[start : start + self.config.batch_size]
                width = max(len(pairs[i].ids) for i in batch)
                ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for row, i in enumerate(batch):
                    n = len(pairs[i].ids)
                    ids[row, :n] = torch.tensor(pairs[i].ids, dtype=torch.long)
                    mask[row, :n] = 1
                logits = self.model(input_ids=ids, attention_mask=mask).logits[:, 0]
                out[batch] = logits.to(torch.float64).numpy()
        return out
