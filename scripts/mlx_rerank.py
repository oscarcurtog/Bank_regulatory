"""F: scores (query, document) inputs with Qwen3-Reranker-0.6B in MLX, the base or the base with a
LoRA adapter, as the reranker's contract scores them, in its own environment (D-040).

    .cache/mlx-venv/bin/python scripts/mlx_rerank.py \\
        --recipe config/retrieval/reranker-finetune.json \\
        --inputs <inputs.jsonl> --out <scores.jsonl> [--adapter <adapter.safetensors>]

Each input line holds an id, a query and a document; the instruction is the recipe's. The prompt
is the contract's (prefix, body, suffix, the same tokenizer), the score the logit of "yes" minus the
logit of "no" at the last position, computed in float32 in batches of similar length, padded on
the right (exact under the causal mask). One line per input is appended to the output: id,
input_sha256, tokens, logit_diff and the adapter's SHA-256 (or "base"). A rerun scores only what
is missing and refuses a stored score of another input or adapter. Imports nothing from `ingest`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

BATCH = 16
CACHE_LIMIT = 1 << 30  # bytes: MLX's buffer cache, which otherwise grows with every shape


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface/hub"))
    args = parser.parse_args(argv)

    # MLX lives only in .cache/mlx-venv: the project's environment (and mypy) never has it
    import mlx.core as mx  # type: ignore[import-not-found]
    import mlx_lm  # type: ignore[import-not-found]
    from huggingface_hub import snapshot_download
    from mlx_lm.tuner.utils import linear_to_lora_layers  # type: ignore[import-not-found]

    mx.set_cache_limit(CACHE_LIMIT)
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    contract = json.loads(Path(recipe["base"]["contract"]).read_text(encoding="utf-8"))
    base = recipe["base"]
    path = snapshot_download(
        base["model_id"], revision=base["revision"], cache_dir=str(args.cache_dir)
    )
    model, wrapper = mlx_lm.load(path)
    model.set_dtype(mx.float32)
    adapter_sha = "base"
    if args.adapter is not None:
        lora = recipe["lora"]
        model.freeze()
        linear_to_lora_layers(
            model,
            lora["layers"],
            {"rank": lora["rank"], "scale": lora["scale"], "dropout": 0.0, "keys": lora["keys"]},
        )
        model.load_weights(str(args.adapter), strict=False)
        adapter_sha = hashlib.sha256(args.adapter.read_bytes()).hexdigest()
    model.eval()
    tok = wrapper._tokenizer
    yes = tok.encode(contract["yes_token"], add_special_tokens=False)[0]
    no = tok.encode(contract["no_token"], add_special_tokens=False)[0]
    pad = tok.encode(contract["pad_token"], add_special_tokens=False)[0]
    prefix: list[int] = list(tok.encode(contract["prefix"], add_special_tokens=False))
    suffix: list[int] = list(tok.encode(contract["suffix"], add_special_tokens=False))
    weight = model.model.embed_tokens.weight
    direction = weight[yes] - weight[no]

    rows = [json.loads(line) for line in args.inputs.read_text(encoding="utf-8").splitlines()]

    def sha(row: dict[str, Any]) -> str:
        fields = [recipe["instruction"], row["query"], row["document"]]
        return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode("utf-8")).hexdigest()

    done: dict[str, dict[str, Any]] = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            stored = json.loads(line)
            done[stored["id"]] = stored
    wanted = {r["id"]: r for r in rows}
    for rid, stored in done.items():
        if rid not in wanted or stored["input_sha256"] != sha(wanted[rid]):
            raise SystemExit(f"{args.out}: {rid} was scored from another input")
        if stored["adapter"] != adapter_sha:
            raise SystemExit(f"{args.out}: {rid} was scored with another adapter")
    pending = [r for r in rows if r["id"] not in done]
    encoded = []
    for r in pending:
        body = contract["body"].format(
            instruction=recipe["instruction"], query=r["query"], document=r["document"]
        )
        middle: list[int] = list(tok(body)["input_ids"])
        encoded.append((r, prefix + middle + suffix))
    encoded.sort(key=lambda x: len(x[1]))
    print(f"adapter {adapter_sha[:12]}; {len(pending)} inputs to score", flush=True)
    started = time.perf_counter()
    for start in range(0, len(encoded), BATCH):
        part = encoded[start : start + BATCH]
        width = max(len(s) for _, s in part)
        arr = mx.array([s + [pad] * (width - len(s)) for _, s in part])
        last = mx.array([len(s) - 1 for _, s in part])
        hidden = model.model(arr)
        scores = hidden[mx.arange(len(part)), last] @ direction
        mx.eval(scores)
        with args.out.open("a", encoding="utf-8") as f:
            for (r, s), v in zip(part, scores.tolist(), strict=True):
                row = {
                    "id": r["id"],
                    "input_sha256": sha(r),
                    "tokens": len(s),
                    "logit_diff": v,
                    "adapter": adapter_sha,
                }
                f.write(json.dumps(row) + "\n")
        n = start + len(part)
        if (start // BATCH) % 100 == 0 or n == len(encoded):
            elapsed = time.perf_counter() - started
            print(
                f"{n}/{len(encoded)} in {elapsed:.0f} s, about "
                f"{elapsed / n * (len(encoded) - n) / 60:.0f} min left",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
