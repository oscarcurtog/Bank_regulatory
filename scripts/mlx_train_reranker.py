"""F: LoRA fine-tuning of Qwen3-Reranker-0.6B on synthetic (question, passage) groups, in its own
MLX environment (D-040), on the listwise softmax of the inference score
(config/retrieval/reranker-finetuning.md).

    .cache/mlx-venv/bin/python scripts/mlx_train_reranker.py \\
        --recipe config/retrieval/reranker-finetune.json --data <data dir> --out <out dir>

Reads train.jsonl (one group per line: the question, texts[0] the positive, texts[1:] the
negatives) and val_select.jsonl (the question, its candidates' texts, chunk ids and correct ids).
Each input is built as the reranker's contract builds it (prefix, body, suffix, the same
tokenizer) and scored as inference scores it: the last position's hidden state against the
embedding rows of "yes" minus "no". LoRA layers on the recipe's last layers are trained with AdamW
on -log(exp(s_pos) / sum(exp(s_group))), the recipe's groups per step accumulated. VAL-SELECT is
evaluated at step 0 and every `eval_every_steps`: each evaluated adapter is saved, the best by MRR
is kept, and training stops after `patience` evaluations without a better MRR or after
`max_epochs`. Writes training.json: the curve, the choice, the time, the memory, the versions. Not
resumable: a rerun starts over from the same seed. Imports nothing from `ingest`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

CACHE_LIMIT = 1 << 30  # bytes
LOG_EVERY = 50


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface/hub"))
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many (a test)")
    args = parser.parse_args(argv)

    # MLX lives only in .cache/mlx-venv: the project's environment (and mypy) never has it
    import mlx.core as mx  # type: ignore[import-not-found]
    import mlx.nn as nn  # type: ignore[import-not-found]
    import mlx.optimizers as optim  # type: ignore[import-not-found]
    import mlx_lm  # type: ignore[import-not-found]
    from huggingface_hub import snapshot_download
    from mlx.utils import tree_flatten, tree_map  # type: ignore[import-not-found]
    from mlx_lm.tuner.utils import linear_to_lora_layers  # type: ignore[import-not-found]

    # MLX keeps freed buffers for reuse; with a new shape at every step that cache grew to 11 GB
    # and the machine swapped (first run, stopped after 32 minutes). The limit changes no number.
    mx.set_cache_limit(CACHE_LIMIT)
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    contract = json.loads(Path(recipe["base"]["contract"]).read_text(encoding="utf-8"))
    base = recipe["base"]
    if (base["model_id"], base["revision"]) != (contract["model_id"], contract["revision"]):
        raise SystemExit("the recipe's base is not the reranker's contract")
    path = snapshot_download(
        base["model_id"], revision=base["revision"], cache_dir=str(args.cache_dir)
    )
    model, wrapper = mlx_lm.load(path)
    model.set_dtype(mx.float32)
    tok = wrapper._tokenizer
    ids_of_token = {}
    for t in (contract["yes_token"], contract["no_token"], contract["pad_token"]):
        encoded = tok.encode(t, add_special_tokens=False)
        if len(encoded) != 1:
            raise SystemExit(f"{t!r} is not one token of the reranker")
        ids_of_token[t] = encoded[0]
    yes, no = ids_of_token[contract["yes_token"]], ids_of_token[contract["no_token"]]
    pad = ids_of_token[contract["pad_token"]]
    prefix: list[int] = list(tok.encode(contract["prefix"], add_special_tokens=False))
    suffix: list[int] = list(tok.encode(contract["suffix"], add_special_tokens=False))

    def ids_of(query: str, document: str) -> list[int]:
        body = contract["body"].format(
            instruction=recipe["instruction"], query=query, document=document
        )
        middle: list[int] = list(tok(body)["input_ids"])
        return prefix + middle + suffix

    def read(name: str) -> tuple[list[dict[str, Any]], str]:
        data = (args.data / name).read_bytes()
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        return rows, hashlib.sha256(data).hexdigest()

    train_rows, train_sha = read("train.jsonl")
    select_rows, select_sha = read("val_select.jsonl")
    train = [[ids_of(r["question"], t) for t in r["texts"]] for r in train_rows]
    select = [[ids_of(r["question"], t) for t in r["texts"]] for r in select_rows]
    if len({len(g) for g in train}) != 1:
        raise SystemExit("every training group must have the same size")

    lora = recipe["lora"]
    model.freeze()
    linear_to_lora_layers(
        model,
        lora["layers"],
        {
            "rank": lora["rank"],
            "scale": lora["scale"],
            "dropout": lora["dropout"],
            "keys": lora["keys"],
        },
    )
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total = sum(v.size for _, v in tree_flatten(model.parameters()))
    mx.random.seed(recipe["seed"])

    def batch(seqs: list[list[int]]) -> tuple[Any, Any]:
        width = max(len(s) for s in seqs)
        arr = mx.array([s + [pad] * (width - len(s)) for s in seqs])
        return arr, mx.array([len(s) - 1 for s in seqs])

    def scores(m: Any, arr: Any, last: Any) -> Any:
        # right padding is exact under the causal mask: the last real token sees only its prefix
        hidden = m.model(arr)
        h = hidden[mx.arange(arr.shape[0]), last]
        w = m.model.embed_tokens.weight
        return h @ (w[yes] - w[no])

    def loss_fn(m: Any, arr: Any, last: Any) -> Any:
        s = scores(m, arr, last).reshape(1, -1)
        return mx.mean(mx.logsumexp(s, axis=1) - s[:, 0])

    def evaluate() -> dict[str, float]:
        model.eval()
        ranks = []
        for row, seqs in zip(select_rows, select, strict=True):
            arr, last = batch(seqs)
            s = scores(model, arr, last)
            mx.eval(s)
            values = s.tolist()
            order = sorted(range(len(values)), key=lambda i: (-values[i], i))
            correct = set(row["correct"])
            ranks.append(
                next((r for r, i in enumerate(order, 1) if row["chunks"][i] in correct), None)
            )
        model.train()
        n = len(ranks)
        return {
            "MRR": sum(1 / r for r in ranks if r) / n,
            "P@1": sum(r == 1 for r in ranks) / n,
        }

    opt_cfg = recipe["optimizer"]
    schedule = optim.linear_schedule(0.0, opt_cfg["learning_rate"], opt_cfg["warmup_steps"])
    opt = optim.AdamW(learning_rate=schedule, weight_decay=opt_cfg["weight_decay"])
    value_and_grad = nn.value_and_grad(model, loss_fn)
    per_step = recipe["groups_per_step"]
    steps_per_epoch = len(train) // per_step
    args.out.mkdir(parents=True, exist_ok=True)

    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    mx.reset_peak_memory()
    first = evaluate()
    curve.append({"step": 0, "epoch": 0, **first, "seconds": 0.0})
    print(f"step 0: val_select MRR {first['MRR']:.4f} P@1 {first['P@1']:.4f}", flush=True)
    best = {"step": 0, **first, "adapter": None}
    since_best, step, losses = 0, 0, []
    stopped = "max_epochs"
    model.train()
    for epoch in range(recipe["max_epochs"]):
        order = list(range(len(train)))
        random.Random(f"{recipe['seed']}:{epoch}").shuffle(order)
        for s in range(steps_per_epoch):
            accumulated = None
            loss_sum = 0.0
            for g in order[s * per_step : (s + 1) * per_step]:
                arr, last = batch(train[g])
                loss, grads = value_and_grad(model, arr, last)
                accumulated = (
                    grads
                    if accumulated is None
                    else tree_map(lambda a, b: a + b, accumulated, grads)
                )
                mx.eval(loss, accumulated)
                loss_sum += loss.item()
            grads = tree_map(lambda a: a / per_step, accumulated)
            grads, _ = optim.clip_grad_norm(grads, opt_cfg["grad_clip_norm"])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            step += 1
            losses.append(loss_sum / per_step)
            if step % LOG_EVERY == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  step {step}: loss {sum(losses[-LOG_EVERY:]) / LOG_EVERY:.4f}, "
                    f"{elapsed / step:.1f} s per step, active "
                    f"{mx.get_active_memory() / 2**30:.2f} GiB, cache "
                    f"{mx.get_cache_memory() / 2**30:.2f} GiB",
                    flush=True,
                )
            if args.max_steps and step >= args.max_steps:
                stopped = "max_steps"
                break
            if step % recipe["eval_every_steps"] == 0:
                m = evaluate()
                adapter = args.out / f"adapter-step{step:05d}.safetensors"
                mx.save_safetensors(str(adapter), dict(tree_flatten(model.trainable_parameters())))
                point = {
                    "step": step,
                    "epoch": step / max(1, steps_per_epoch),
                    **m,
                    "train_loss": sum(losses) / len(losses),
                    "seconds": time.perf_counter() - started,
                    "peak_memory_gib": mx.get_peak_memory() / 2**30,
                }
                curve.append(point)
                losses = []
                better = (m["MRR"], m["P@1"]) > (best["MRR"], best["P@1"])
                if better:
                    best = {"step": step, **m, "adapter": adapter.name}
                    since_best = 0
                else:
                    since_best += 1
                print(
                    f"step {step} (epoch {epoch + 1}): train loss {point['train_loss']:.4f}, "
                    f"val_select MRR {m['MRR']:.4f} P@1 {m['P@1']:.4f}, "
                    f"{point['seconds'] / 60:.1f} min, peak {point['peak_memory_gib']:.2f} GiB"
                    + (" (best)" if better else ""),
                    flush=True,
                )
                if since_best >= recipe["patience"]:
                    stopped = "patience"
                    break
        if stopped in ("patience", "max_steps"):
            break
    record = {
        "recipe": recipe,
        "data_sha256": {"train": train_sha, "val_select": select_sha},
        "groups": {"train": len(train), "val_select": len(select), "group_size": len(train[0])},
        "parameters": {"trainable": trainable, "total": total},
        "steps": step,
        "steps_per_epoch": steps_per_epoch,
        "stopped": stopped,
        "curve": curve,
        "best": best,
        "seconds": time.perf_counter() - started,
        "peak_memory_gib": mx.get_peak_memory() / 2**30,
        "versions": {
            "mlx": mx.__version__,
            "mlx_lm": mlx_lm.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
    }
    (args.out / "training.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"stopped by {stopped} at step {step}; best step {best['step']} MRR {best['MRR']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
