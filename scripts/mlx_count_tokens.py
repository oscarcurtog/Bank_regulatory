"""Counts the prompt tokens of a prompts file exactly as scripts/mlx_generate.py does before it
generates (T4's gate for E, D-053): the model's chat template with the generation prompt, the
same model snapshot, the same tokenizer. Nothing is generated. Runs in .cache/mlx-venv:

    .cache/mlx-venv/bin/python scripts/mlx_count_tokens.py --config <model.json> \\
        --prompts <prompts.jsonl> --out <counts.json>
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface/hub"))
    args = parser.parse_args(argv)

    import mlx_lm  # type: ignore[import-not-found]
    from huggingface_hub import snapshot_download

    config = json.loads(args.config.read_text(encoding="utf-8"))
    path = snapshot_download(
        config["model_id"], revision=config["revision"], cache_dir=str(args.cache_dir)
    )
    _, tokenizer = mlx_lm.load(path)
    template_kwargs = config.get("chat_template_kwargs", {})
    rows = [json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines()]
    counts = {
        row["id"]: len(
            list(
                tokenizer.apply_chat_template(
                    row["messages"], add_generation_prompt=True, **template_kwargs
                )
            )
        )
        for row in rows
    }
    record = {
        "model_id": config["model_id"],
        "revision": config["revision"],
        "max_prompt_tokens": config["max_prompt_tokens"],
        "mlx_lm": mlx_lm.__version__,
        "python": platform.python_version(),
        "counts": counts,
    }
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    worst = max(counts.values()) if counts else 0
    over = sum(n > config["max_prompt_tokens"] for n in counts.values())
    print(f"{len(counts)} prompts, {worst} tokens at most, {over} over the limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
