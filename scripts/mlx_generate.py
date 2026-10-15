"""Greedy generation with an MLX model, in its own process and environment (D-040).

    .cache/mlx-venv/bin/python scripts/mlx_generate.py \\
        --config config/retrieval/qwen3-4b-instruct-mlx.json \\
        --prompts <prompts.jsonl> --out <outputs.jsonl> --max-new-tokens 160

MLX runs only on Apple silicon and lives in its own virtual environment (.cache/mlx-venv, not
versioned), so this script imports nothing from `ingest` and the project's environment, locks and
CI never see it. It reads prompts (JSONL: id, messages), renders each with the model's own chat
template, decodes greedily and appends one line per prompt to the outputs (JSONL: id,
prompt_sha256, prompt_tokens, output, new_tokens, seconds, peak_memory_bytes, and the versions
that produced it). A rerun generates only what is missing; an output whose prompt, model, limits
or template arguments changed is refused, never reused.

A config may set `chat_template_kwargs`, passed to the chat template as they are. MLX-LM turns a
model's thinking on by default when its template has one, so a hybrid model such as Qwen3-8B
runs without thinking only if its config says `{"enable_thinking": false}`.

Prompts that share an id prefix up to `--group-separator` (one query's windows) and open with
the same tokens are generated with that shared opening computed once: its key-value cache is
kept and trimmed back after each prompt. Greedy decoding gives the same text either way; the
`--no-prefix-cache` flag turns it off to check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
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
    parser.add_argument("--group-separator", default="#")
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=0, help="generate at most this many (a sample)"
    )
    args = parser.parse_args(argv)

    # MLX lives only in .cache/mlx-venv: the project's environment (and mypy) never has it
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
    done: dict[str, dict[str, Any]] = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    wanted = {row["id"]: sha(row["messages"]) for row in rows}
    for rid, row in done.items():
        if wanted.get(rid) != row["prompt_sha256"]:
            raise SystemExit(f"{args.out}: {rid} was generated from another prompt")
        keys = ("model_id", "revision", "max_new_tokens", "chat_template_kwargs")
        if any(row["versions"].get(k) != versions.get(k) for k in keys):
            raise SystemExit(f"{args.out}: {rid} was generated with another model or limit")
    sampler = make_sampler(temp=0.0)
    pending = [row for row in rows if row["id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"model loaded in {load_seconds:.1f} s; {len(pending)} prompts", flush=True)

    cache: Any = None
    cached: list[int] = []
    group = None
    started = time.perf_counter()
    for i, row in enumerate(pending, start=1):
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
        if args.no_prefix_cache or this_group != group or cache is None:
            cache, cached, group = make_prompt_cache(model), [], this_group
        # the shared opening: reuse what the cache holds while it matches this prompt
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
        # keep only this prompt's tokens in the cache: the generated ones are dropped
        if cache[0].offset > len(ids):
            trim_prompt_cache(cache, cache[0].offset - len(ids))
        cached = ids[: cache[0].offset]
        seconds = time.perf_counter() - t0
        out = {
            "id": row["id"],
            "prompt_sha256": wanted[row["id"]],
            "prompt_tokens": len(ids),
            "reused_prefix_tokens": common,
            "output": text,
            "new_tokens": new_tokens,
            "seconds": seconds,
            "peak_memory_bytes": mx.get_peak_memory(),
            "versions": versions,
        }
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        elapsed = time.perf_counter() - started
        if i % 20 == 0 or i == len(pending):
            print(
                f"{i}/{len(pending)} in {elapsed:.0f} s ({seconds:.1f} s for {row['id']}, "
                f"{len(ids)} tokens, {common} reused); about "
                f"{elapsed / i * (len(pending) - i) / 60:.0f} min left",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
