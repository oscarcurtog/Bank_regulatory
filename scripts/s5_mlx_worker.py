"""A persistent MLX generation worker for the S5.3 cost harness.

It decodes exactly as scripts/mlx_generate.py does (greedy, the chat template, the shared opening
kept in a prompt cache per group of ids), but stays alive: it loads the model once, announces it,
then answers one JSON line per request, so that a caller can time a single generation with the
model warm. MLX lives only in .cache/mlx-venv, which this script is run with.

Protocol, one JSON object per line:
    stdout (first line)  {"ready": true, "load_seconds": ..., "versions": {...}}
    stdin                {"id": "d#<query id>", "messages": [...]}
    stdout               {"id": ..., "output": ..., "prompt_tokens": ...,
                          "reused_prefix_tokens": ..., "new_tokens": ..., "seconds": ...,
                          "peak_memory_bytes": ...}
The worker ends when stdin closes.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface/hub"))
    parser.add_argument("--group-separator", default="#")
    args = parser.parse_args(argv)

    import mlx.core as mx  # type: ignore[import-not-found]
    import mlx_lm  # type: ignore[import-not-found]
    from huggingface_hub import snapshot_download
    from mlx_lm.models.cache import (  # type: ignore[import-not-found]
        make_prompt_cache,
        trim_prompt_cache,
    )
    from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]

    config = json.loads(args.config.read_text(encoding="utf-8"))
    path = snapshot_download(
        config["model_id"], revision=config["revision"], cache_dir=str(args.cache_dir)
    )
    start = time.perf_counter()
    model, tokenizer = mlx_lm.load(path)
    load_seconds = time.perf_counter() - start
    template_kwargs = config.get("chat_template_kwargs", {})
    versions: dict[str, Any] = {
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "model_id": config["model_id"],
        "revision": config["revision"],
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
    }
    sampler = make_sampler(temp=0.0)
    print(
        json.dumps({"ready": True, "load_seconds": load_seconds, "versions": versions}), flush=True
    )

    cache: Any = None
    cached: list[int] = []
    group = None
    for line in sys.stdin:
        if not line.strip():
            continue
        row = json.loads(line)
        ids = list(
            tokenizer.apply_chat_template(
                row["messages"], add_generation_prompt=True, **template_kwargs
            )
        )
        if len(ids) > config["max_prompt_tokens"]:
            raise SystemExit(f"{row['id']}: {len(ids)} prompt tokens, over the limit")
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        this_group = row["id"].split(args.group_separator)[0]
        if this_group != group or cache is None:
            cache, cached, group = make_prompt_cache(model), [], this_group
        common = 0
        while common < min(len(cached), len(ids) - 1) and cached[common] == ids[common]:
            common += 1
        if cache[0].offset > common:
            trim_prompt_cache(cache, cache[0].offset - common)
        text, new_tokens = "", 0
        for response in mlx_lm.stream_generate(
            model,
            tokenizer,
            prompt=ids[common:],
            max_tokens=args.max_new_tokens,
            sampler=sampler,
            prompt_cache=cache,
        ):
            text += response.text
            new_tokens = response.generation_tokens
        if cache[0].offset > len(ids):
            trim_prompt_cache(cache, cache[0].offset - len(ids))
        cached = ids[: cache[0].offset]
        out = {
            "id": row["id"],
            "output": text,
            "prompt_tokens": len(ids),
            "reused_prefix_tokens": common,
            "new_tokens": new_tokens,
            "seconds": time.perf_counter() - t0,
            "peak_memory_bytes": mx.get_peak_memory(),
        }
        print(json.dumps(out, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
