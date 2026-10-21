"""The token count of each prompt, as scripts/mlx_generate.py renders it, with no generation
(S3, P1 #6 of the review before the clean restart).

    .cache/mlx-venv/bin/python scripts/mlx_prompt_tokens.py \\
        --config config/retrieval/qwen3-8b-mlx.json \\
        --prompts <prompts.jsonl> --out <tokens.jsonl> --max-new-tokens 2048

The diagnostic's preflight. It loads the model's own tokenizer from the same pinned snapshot,
with the function mlx_lm.load uses for it, and renders each prompt with the same chat template
and arguments as the generator, without loading the model's weights. It appends one line per
prompt (JSONL: id, prompt_sha256, prompt_tokens, max_prompt_tokens, and the versions the
generator would write), so that a prompt over the frozen limit is known before any generation:
it is never truncated, and never run three times to fail the same way.

Like the generator, it runs only in .cache/mlx-venv and imports nothing from `ingest`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface/hub"))
    args = parser.parse_args(argv)

    # MLX lives only in .cache/mlx-venv: the project's environment (and mypy) never has it
    import mlx.core as mx  # type: ignore[import-not-found]
    import mlx_lm  # type: ignore[import-not-found]
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer  # type: ignore[import-not-found]

    config = json.loads(args.config.read_text(encoding="utf-8"))
    path = Path(
        snapshot_download(
            config["model_id"], revision=config["revision"], cache_dir=str(args.cache_dir)
        )
    )
    model_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    # as mlx_lm.load builds it, without the model
    tokenizer = load_tokenizer(path, None, eos_token_ids=model_config.get("eos_token_id"))
    template_kwargs = config.get("chat_template_kwargs", {})
    versions = {
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "model_id": config["model_id"],
        "revision": config["revision"],
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
    }
    if template_kwargs:
        versions["chat_template_kwargs"] = template_kwargs
    rows = [json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines()]
    with args.out.open("a", encoding="utf-8") as out:
        for row in rows:
            ids = list(
                tokenizer.apply_chat_template(
                    row["messages"], add_generation_prompt=True, **template_kwargs
                )
            )
            line = {
                "id": row["id"],
                "prompt_sha256": sha(row["messages"]),
                "prompt_tokens": len(ids),
                "max_prompt_tokens": config["max_prompt_tokens"],
                "versions": versions,
            }
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
