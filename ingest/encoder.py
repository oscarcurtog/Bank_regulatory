"""The model side of the dense baseline: the transformer that turns token ids into
vectors, as the model card of the configured model does it.

It receives token ids already counted and, where the contract allows it, already
truncated by ingest/embeddings.py, never text: so it cannot cut anything by itself.
It checks every input against its record before running the model, pads each batch
with the model's pad token and masks the padding, takes the mean of the token vectors
over the mask, normalizes, and checks the vectors before returning them.

torch and transformers are the optional "embed" extra (requirements-embed.lock.txt):
not in CI and not in the Docker images. Importing this module never fails; building
an Encoder without them raises ModelUnavailableError with the remedy. The weights are
downloaded once into the Hugging Face cache of ingest/embeddings.py (HF_HOME,
.cache/huggingface by default) and loaded at the pinned revision only.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Sequence
from typing import Any

import numpy as np

from ingest import embeddings as em


def _framework() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as e:
        raise em.ModelUnavailableError(
            "torch and transformers are not installed: install requirements-embed.lock.txt "
            "(the optional 'embed' extra). They are deliberately absent from CI and the images."
        ) from e
    return torch, transformers


class Encoder:
    """One embedding model, loaded in evaluation mode, on the contract's device and type."""

    def __init__(self, config: em.EmbeddingConfig) -> None:
        torch, transformers = _framework()
        self.config = config
        self._torch = torch
        cache = em.hf_home() / "hub"
        try:
            model = transformers.AutoModel.from_pretrained(
                config.model_id,
                revision=config.revision,
                cache_dir=cache,
                dtype=getattr(torch, config.dtype),
            )
        except Exception as e:
            raise em.ModelUnavailableError(
                f"{config.model_id}@{config.revision[:12]}: cannot load the weights from the "
                f"cache {cache} or the network. Connect once to download them, or point HF_HOME "
                f"at a cache that holds them."
            ) from e
        hidden = model.config.hidden_size
        if hidden != config.dimension:
            raise em.EmbeddingError(
                f"the model returns {hidden} dimensions, the contract says {config.dimension}"
            )
        positions = model.config.max_position_embeddings - (model.config.pad_token_id or 0) - 1
        if positions < config.max_input_tokens:
            raise em.EmbeddingError(
                f"the model has {positions} usable positions, under the contract's "
                f"{config.max_input_tokens}"
            )
        self.pad_id = int(model.config.pad_token_id)
        self.model = model.to(config.device).eval()

    def backend(self) -> dict[str, str]:
        """What computed the vectors, for the record of every set."""
        return {
            "torch": self._torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "numpy": np.__version__,
            "device": self.config.device,
            "dtype": self.config.dtype,
            "torch_threads": str(self._torch.get_num_threads()),
        }

    def encode(self, inputs: Sequence[em.ModelInput]) -> np.ndarray:
        """One vector per input, in input order. Batches go longest first, so that
        each batch pads little; the order of the output does not depend on it."""
        em.check_inputs(inputs, self.config)
        torch = self._torch
        out = np.empty((len(inputs), self.config.dimension), dtype=np.float32)
        order = sorted(range(len(inputs)), key=lambda i: (-len(inputs[i].input_ids), i))
        size = self.config.batch_size
        with torch.inference_mode():
            for start in range(0, len(order), size):
                batch = order[start : start + size]
                width = max(len(inputs[i].input_ids) for i in batch)
                if width > self.config.max_input_tokens:
                    raise em.EmbeddingError(f"a batch of width {width} over the limit")
                ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for row, i in enumerate(batch):
                    n = len(inputs[i].input_ids)
                    ids[row, :n] = torch.tensor(inputs[i].input_ids, dtype=torch.long)
                    mask[row, :n] = 1
                ids, mask = ids.to(self.config.device), mask.to(self.config.device)
                hidden = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
                weights = mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1)
                if self.config.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out[batch] = pooled.to("cpu", torch.float32).numpy()
        em.check_vectors(out, len(inputs), self.config)
        return out
