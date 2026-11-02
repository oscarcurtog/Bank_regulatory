"""Generation harness of S7-S8: prompt development and the single evaluation run.

    PYTHONPATH=. .venv/bin/python scripts/s7_generate.py dev        # 12 prompt-development cases
    PYTHONPATH=. .venv/bin/python scripts/s7_generate.py evaluate   # the 96 preregistered cases

Both read the cases of `scripts/s7_cases.py materialize`, build each prompt with
ingest/answer_format.py and run `Qwen3-4B-Instruct-2507` (4 bit, MLX) greedily through the
persistent worker of scripts/s5_mlx_worker.py, one fresh prompt cache per case so that no output
depends on the order of the cases. Every raw output is appended to a JSONL file with the hash of
its prompt; nothing is ever regenerated to get a better answer. A model output that is cut or is
not valid JSON is a structural failure and stays as it is. Only a technical failure of the worker
that produced no output (a crash, a timeout) is resumed, idempotently and with a note in the run
log.

`evaluate` refuses to start unless the preregistration is pinned in a public commit, the code is
committed and pushed, the tree is clean, and the cases, the prompt and the verifier code are the
pinned ones. It never overwrites a finished evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ingest import answer_format as af
from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import cost_protocol as cp
from ingest import download as dl
from ingest import embeddings as em

MLX_PYTHON = Path(".cache/mlx-venv/bin/python")
WORKER = Path("scripts/s5_mlx_worker.py")
GENERATOR = Path("config/retrieval/qwen3-4b-instruct-mlx.json")
RESULTS = Path("data/results/dev-0.10.0/s7-s8")
CASES = RESULTS / "cases.json"
PIN = Path("config/generation/s7-s8-preregistration.json")
DOC = Path("docs/research/s7-s8-preregistro-v1.md")
CODE = (
    "scripts/s7_generate.py",
    "scripts/s7_cases.py",
    "scripts/s7_report.py",
    "ingest/answer_format.py",
    "ingest/answer_verifier.py",
    "ingest/answer_cases.py",
    "ingest/answer_metrics.py",
    "ingest/answer_report.py",
)
PINNED_CODE = (
    "ingest/answer_format.py",
    "ingest/answer_verifier.py",
    "ingest/answer_cases.py",
    "ingest/answer_metrics.py",
    "ingest/answer_report.py",
)
REQUEST_TIMEOUT_SECONDS = 900
MAX_TECHNICAL_RESTARTS = 3


def out_of(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def clean_tree() -> bool:
    return not out_of(["git", "status", "--porcelain", "--untracked-files=no"]).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Worker:
    """The persistent MLX process. A reader thread lets a request time out instead of hanging."""

    def __init__(self) -> None:
        if not MLX_PYTHON.exists():
            raise SystemExit(f"{MLX_PYTHON} is missing: the MLX environment is not set up")
        command = [str(MLX_PYTHON), str(WORKER), "--config", str(GENERATOR)]
        command += ["--max-new-tokens", str(af.MAX_NEW_TOKENS)]
        self.process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=os.environ | {"HF_HOME": str(em.hf_home())},
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        first = json.loads(self._next(REQUEST_TIMEOUT_SECONDS))
        if not first.get("ready"):
            raise RuntimeError("the MLX worker did not announce itself")
        self.load_seconds: float = first["load_seconds"]
        self.versions: dict[str, Any] = first["versions"]

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put("")  # end of stream

    def _next(self, timeout: float) -> str:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError(f"no answer from the worker in {timeout:.0f} s") from error
        if not line:
            raise RuntimeError("the MLX worker ended")
        return line

    def generate(self, request_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({"id": request_id, "messages": messages}) + "\n")
        self.process.stdin.flush()
        answer: dict[str, Any] = json.loads(self._next(REQUEST_TIMEOUT_SECONDS))
        return answer

    def close(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()


def load_texts() -> dict[str, str]:
    manifest = dl.load_manifest()
    docs = [ch.load_chunks(act, manifest) for act in bm.ACTS]
    return {c.chunk_id: c.text for d in docs for c in d.chunks}


def read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def log_event(folder: Path, event: str, **fields: Any) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    with (folder / "run-log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": stamp, "event": event, **fields}) + "\n")


def run_cases(cases: list[dict[str, Any]], folder: Path, allow_resume: bool) -> None:
    """Generate every case once, appending to folder/outputs.jsonl. Cases already there (same
    prompt hash) are skipped; a different prompt hash in the file is an error."""
    folder.mkdir(parents=True, exist_ok=True)
    outputs = folder / "outputs.jsonl"
    prompt_hash = af.prompt_sha256()
    done = {r["case_id"]: r for r in read_lines(outputs)}
    if any(r["prompt_sha256"] != prompt_hash for r in done.values()):
        raise SystemExit(f"{outputs} holds outputs of another prompt: use another folder")
    pending = [c for c in cases if c["case_id"] not in done]
    if done and not allow_resume:
        raise SystemExit(f"{outputs} already holds {len(done)} outputs: pass --resume to continue")
    if not pending:
        print(f"nothing to generate: {len(done)} outputs in {outputs}")
        return
    texts = load_texts()
    log_event(
        folder,
        "start",
        pending=len(pending),
        already=len(done),
        prompt_sha256=prompt_hash,
        commit=out_of(["git", "rev-parse", "HEAD"]).strip(),
        python=platform.python_version(),
    )
    restarts = 0
    worker = Worker()
    try:
        for index, case in enumerate(pending, start=1):
            messages = af.build_messages(
                case["question"], af.render_sources(case["chunk_ids"], texts)
            )
            started = datetime.now(UTC).isoformat(timespec="seconds")
            while True:
                try:
                    reply = worker.generate(case["case_id"], messages)
                    break
                except (TimeoutError, RuntimeError, json.JSONDecodeError) as error:
                    restarts += 1
                    log_event(
                        folder,
                        "technical_failure_no_output",
                        case_id=case["case_id"],
                        error=repr(error),
                        restart=restarts,
                    )
                    worker.close()
                    if restarts > MAX_TECHNICAL_RESTARTS:
                        raise SystemExit("too many technical failures of the worker") from error
                    worker = Worker()
            record = {
                "case_id": case["case_id"],
                "kind": case["kind"],
                "need_id": case["need_id"],
                "context_name": case["context_name"],
                "prompt_version": af.PROMPT_VERSION,
                "prompt_sha256": prompt_hash,
                "messages_sha256": sha256_text(json.dumps(messages, sort_keys=True)),
                "raw_output": reply["output"],
                "prompt_tokens": reply["prompt_tokens"],
                "new_tokens": reply["new_tokens"],
                "hit_token_limit": reply["new_tokens"] >= af.MAX_NEW_TOKENS,
                "seconds": reply["seconds"],
                "peak_memory_bytes": reply["peak_memory_bytes"],
                "started_utc": started,
                "worker_versions": worker.versions,
            }
            with outputs.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(pending)}] {case['case_id']}: {reply['new_tokens']} tokens, "
                f"{reply['seconds']:.1f} s",
                flush=True,
            )
    finally:
        worker.close()
    log_event(folder, "finished", outputs=len(read_lines(outputs)), restarts=restarts)


def finish(folder: Path, expected: int) -> None:
    outputs = folder / "outputs.jsonl"
    count = len(read_lines(outputs))
    if count != expected:
        raise SystemExit(f"{count} outputs, {expected} expected")
    (folder / "outputs.sha256").write_text(f"{cp.sha256_file(outputs)}  outputs.jsonl\n")
    outputs.chmod(0o444)


# ---------------------------------------------------------------------------
# guard of the evaluation
# ---------------------------------------------------------------------------


def guard() -> dict[str, Any]:
    for name in (*CODE, str(PIN)):
        done = subprocess.run(
            ["git", "ls-files", "--error-unmatch", name], capture_output=True, check=False
        )
        if done.returncode != 0:
            raise SystemExit(f"{name} is not committed: commit it first")
    if not clean_tree():
        raise SystemExit("tracked files changed since the commit: commit the code first")
    ahead = out_of(["git", "rev-list", "--count", "@{u}..HEAD"]).strip()
    if ahead != "0":
        raise SystemExit(f"the commit is not public yet ({ahead} commits ahead of origin)")
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    if cp.sha256_file(DOC) != pin["document_sha256"]:
        raise SystemExit(f"{DOC} does not hash to its public pin")
    inputs: dict[str, str] = pin["inputs"]
    text = DOC.read_text(encoding="utf-8")
    absent = [k for k, v in inputs.items() if v not in text]
    if absent:
        raise SystemExit(f"the preregistration text does not hold the pinned hashes of {absent}")
    if cp.sha256_file(CASES) != inputs["cases_json"]:
        raise SystemExit("cases.json is not the file the preregistration pins")
    if af.prompt_sha256() != inputs["prompt_sha256"]:
        raise SystemExit("the prompt is not the one the preregistration pins")
    for name in PINNED_CODE:
        if cp.sha256_file(Path(name)) != inputs[name]:
            raise SystemExit(f"{name} is not the code the preregistration pins")
    return {
        "preregistration_sha256": pin["document_sha256"],
        "commit": out_of(["git", "rev-parse", "HEAD"]).strip(),
        "prompt_sha256": af.prompt_sha256(),
        "cases_sha256": inputs["cases_json"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    dev = sub.add_parser("dev", help="generate the prompt-development cases")
    dev.add_argument("--resume", action="store_true")
    evaluate = sub.add_parser("evaluate", help="the single preregistered evaluation run")
    evaluate.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    payload = json.loads(CASES.read_text(encoding="utf-8"))
    if args.command == "dev":
        folder = RESULTS / f"prompt-dev-{af.prompt_sha256()[:12]}"
        run_cases(payload["cases"]["prompt_dev"], folder, args.resume)
        finish(folder, len(payload["cases"]["prompt_dev"]))
        print(f"prompt-development outputs: {folder}")
        return 0
    seal = guard()
    cases = payload["cases"]["evaluation_positive"] + payload["cases"]["evaluation_negative"]
    folder = RESULTS / "evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "seal.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8"))["seal"] != seal:
            raise SystemExit("the evaluation folder belongs to another seal")
    else:
        battery = out_of(["pmset", "-g", "batt"]).splitlines()
        note = {"seal": seal, "power": battery[:2], "started_unix": time.time()}
        marker.write_text(json.dumps(note, indent=1) + "\n")
        marker.chmod(0o444)
    run_cases(cases, folder, args.resume)
    finish(folder, len(cases))
    print(f"evaluation outputs: {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
