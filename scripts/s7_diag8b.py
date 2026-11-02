"""The 8B capability diagnostic of S7-S8: one run on GOLD_KNOWN and its comparison with the 4B.

    PYTHONPATH=. .venv/bin/python scripts/s7_diag8b.py run      # 18 generations with the 8B
    PYTHONPATH=. .venv/bin/python scripts/s7_diag8b.py analyse  # table, frozen rule, blind packet

`run` refuses to start unless the preregistration of the diagnostic is pinned in a public commit,
the code is committed and pushed, the tree is clean, and the cases, the prompt, the verifier and
the model configuration are the pinned ones. It reuses the generation harness of the 4B run
(scripts/s7_generate.py) unchanged and only points it at the 8B configuration. Nothing is ever
regenerated to get a better answer. `analyse` applies the frozen rule to the stored outputs; the
blind packet for the audit is written only if the rule is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import s7_generate as gen
import s7_report as sr

from ingest import answer_diag8b as d8
from ingest import answer_format as af
from ingest import answer_metrics as am
from ingest import answer_report as rep
from ingest import cost_protocol as cp

CONFIG_8B = Path("config/retrieval/qwen3-8b-mlx.json")
FOLDER = gen.RESULTS / "diag-8b"
BASELINE = gen.RESULTS / "evaluation"
PIN = Path("config/generation/s7-s8-8b-diagnostic-preregistration.json")
DOC = Path("docs/research/s7-s8-diagnostico-8b-v1.md")
CODE = ("ingest/answer_diag8b.py", "scripts/s7_diag8b.py")
HF_CACHE = Path(".cache/huggingface/hub")


def gold_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in payload["cases"]["evaluation_positive"] if c["context_name"] == d8.CONTEXT]


def weights_blob(revision: str) -> str:
    """The content hash that names the weights file of the cached snapshot."""
    snapshot = HF_CACHE / "models--mlx-community--Qwen3-8B-4bit" / "snapshots" / revision
    return (snapshot / "model.safetensors").resolve().name


def guard() -> dict[str, Any]:
    for name in (*gen.PINNED_CODE, *CODE, str(PIN), str(CONFIG_8B)):
        done = subprocess.run(
            ["git", "ls-files", "--error-unmatch", name], capture_output=True, check=False
        )
        if done.returncode != 0:
            raise SystemExit(f"{name} is not committed: commit it first")
    if not gen.clean_tree():
        raise SystemExit("tracked files changed since the commit: commit the code first")
    ahead = gen.out_of(["git", "rev-list", "--count", "@{u}..HEAD"]).strip()
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
    checks = {
        "cases_json": cp.sha256_file(gen.CASES),
        "prompt_sha256": af.prompt_sha256(),
        "generator_config_8b": cp.sha256_file(CONFIG_8B),
        "baseline_4b_outputs": cp.sha256_file(BASELINE / "outputs.jsonl"),
        **{name: cp.sha256_file(Path(name)) for name in (*gen.PINNED_CODE, *CODE)},
    }
    for name, got in checks.items():
        if got != inputs[name]:
            raise SystemExit(f"{name} is not the file the preregistration pins")
    config = json.loads(CONFIG_8B.read_text(encoding="utf-8"))
    if weights_blob(config["revision"]) != inputs["weights_sha256"]:
        raise SystemExit("the cached weights are not the pinned ones")
    return {
        "preregistration_sha256": pin["document_sha256"],
        "commit": gen.out_of(["git", "rev-parse", "HEAD"]).strip(),
        "prompt_sha256": af.prompt_sha256(),
        "model": config["model_id"],
        "revision": config["revision"],
    }


def run(resume: bool) -> int:
    payload = json.loads(gen.CASES.read_text(encoding="utf-8"))
    cases = gold_cases(payload)
    seal = guard()
    pinned = json.loads(PIN.read_text(encoding="utf-8"))["design"]["need_ids"]
    if sorted(c["need_id"] for c in cases) != sorted(pinned):
        raise SystemExit("the GOLD_KNOWN cases are not the 18 needs the preregistration pins")
    FOLDER.mkdir(parents=True, exist_ok=True)
    marker = FOLDER / "seal.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8"))["seal"] != seal:
            raise SystemExit("the diagnostic folder belongs to another seal")
    else:
        battery = gen.out_of(["pmset", "-g", "batt"]).splitlines()
        docker = gen.out_of(["pgrep", "-fl", "Docker"]).strip().splitlines()
        note = {
            "seal": seal,
            "power": battery[:2],
            "docker_processes": len(docker),
            "started_unix": time.time(),
        }
        marker.write_text(json.dumps(note, indent=1) + "\n")
        marker.chmod(0o444)
    gen.GENERATOR = CONFIG_8B  # the only change to the 4B harness: which model it loads
    gen.run_cases(cases, FOLDER, resume)
    gen.finish(FOLDER, len(cases))
    print(f"8B diagnostic outputs: {FOLDER}")
    return 0


def load_outcomes(
    cases: list[dict[str, Any]], folder: Path, texts: dict[str, str], model: str | None
) -> tuple[list[am.Outcome], dict[str, dict[str, Any]]]:
    """The outcomes of exactly these cases from a folder that may hold more; each stored input must
    be the one built now, and, for the 8B, each record must come from the pinned model."""
    records = {r["case_id"]: r for r in sr.read_lines(folder / "outputs.jsonl")}
    missing = [c["case_id"] for c in cases if c["case_id"] not in records]
    if missing:
        raise SystemExit(f"{folder}: missing outputs for {missing[:3]}")
    outcomes = []
    for d in cases:
        record = records[d["case_id"]]
        case = sr.to_case(d)
        messages = af.build_messages(case.question, af.render_sources(case.chunk_ids, texts))
        expected = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        if record["messages_sha256"] != expected or record["prompt_sha256"] != af.prompt_sha256():
            raise SystemExit(f"{case.case_id}: the stored input is not the preregistered one")
        if model and record["worker_versions"]["model_id"] != model:
            raise SystemExit(
                f"{case.case_id}: generated by {record['worker_versions']['model_id']}"
            )
        outcomes.append(am.make_outcome(case, record["raw_output"], texts))
    return outcomes, {c["case_id"]: records[c["case_id"]] for c in cases}


def analyse() -> int:
    payload = json.loads(gen.CASES.read_text(encoding="utf-8"))
    cases = gold_cases(payload)
    config = json.loads(CONFIG_8B.read_text(encoding="utf-8"))
    texts, grades = sr.load_world()
    four, four_records = load_outcomes(cases, BASELINE, texts, None)
    eight, eight_records = load_outcomes(cases, FOLDER, texts, config["model_id"])
    for r in eight_records.values():
        if r["worker_versions"]["revision"] != config["revision"]:
            raise SystemExit(f"{r['case_id']}: generated by another revision of the model")
    s4 = d8.summary(four, four_records, grades)
    s8 = d8.summary(eight, eight_records, grades)
    verdict = d8.judge(s4, s8)
    analysis = rep.clean({"4B": s4, "8B": s8, "decision": verdict})
    sr.write_once(FOLDER / "analysis.json", json.dumps(analysis, indent=1, sort_keys=True) + "\n")
    sr.write_once(FOLDER / "report.md", d8.render(s4, s8, verdict))
    for name in ("analysis.json", "report.md"):
        print(hashlib.sha256((FOLDER / name).read_bytes()).hexdigest(), FOLDER / name)
    if verdict["conclusion"] == d8.PASSED:
        info = d8.write_audit_inputs(eight, texts, FOLDER / "audit-input", FOLDER / "key")
        print("blind packet:", info)
    print(verdict["conclusion"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="the single preregistered 8B run")
    runner.add_argument("--resume", action="store_true")
    sub.add_parser("analyse", help="the frozen rule on the stored outputs")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args.resume)
    return analyse()


if __name__ == "__main__":
    sys.exit(main())
