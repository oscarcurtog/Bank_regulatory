"""The MLX worker (scripts/mlx_generate.py) with the real Qwen3-4B-Instruct-2507 at 4 bits, in its
own environment (D-040): greedy decoding gives the same text twice, reusing a shared opening
gives the same text as computing it again, and an output of another prompt is never reused.

Marked "embedding_model", like the tests of the other models: it needs the MLX environment
(.cache/mlx-venv) and the weights, and is skipped only where the environment says so.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ingest import embeddings as em

pytestmark = pytest.mark.embedding_model

ROOT = Path(__file__).parents[2]
PYTHON = ROOT / ".cache" / "mlx-venv" / "bin" / "python"
CONFIG = ROOT / "config" / "retrieval" / "qwen3-4b-instruct-mlx.json"
ROWS = [
    {
        "id": f"group#{i}",
        "messages": [
            {"role": "system", "content": "Answer in a few words."},
            {"role": "user", "content": f"Name a colour, then write the number {i}."},
        ],
    }
    for i in range(3)
]


def _run(out: Path, rows: list[dict[str, Any]], *extra: str) -> list[dict[str, Any]]:
    if not PYTHON.exists():
        pytest.fail(f"{PYTHON} is missing: set up the MLX environment of D-040")
    prompts = out.with_suffix(".prompts.jsonl")
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    command = [str(PYTHON), str(ROOT / "scripts" / "mlx_generate.py"), "--config", str(CONFIG)]
    command += ["--prompts", str(prompts), "--out", str(out), "--max-new-tokens", "24", *extra]
    subprocess.run(command, check=True, cwd=ROOT, env=os.environ | {"HF_HOME": str(em.hf_home())})
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def test_greedy_outputs_repeat_and_the_shared_opening_changes_nothing(tmp_path):
    cached = _run(tmp_path / "cached.jsonl", ROWS)
    plain = _run(tmp_path / "plain.jsonl", ROWS, "--no-prefix-cache")
    again = _run(tmp_path / "again.jsonl", ROWS)
    texts = [[r["output"] for r in run] for run in (cached, plain, again)]
    assert texts[0] == texts[1] == texts[2]
    assert cached[1]["reused_prefix_tokens"] > 0 and plain[1]["reused_prefix_tokens"] == 0


def test_an_output_of_another_prompt_is_refused(tmp_path):
    out = tmp_path / "out.jsonl"
    _run(out, ROWS[:1])
    changed = [dict(ROWS[0], messages=[{"role": "user", "content": "Something else."}])]
    with pytest.raises(subprocess.CalledProcessError):
        _run(out, changed)
