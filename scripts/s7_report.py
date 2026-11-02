"""Report of S7-S8: from the generated outputs to the analysis, the blind human packets and,
after the human labels, the final scores.

    PYTHONPATH=. .venv/bin/python scripts/s7_report.py check    # prompt development: a quick look
    PYTHONPATH=. .venv/bin/python scripts/s7_report.py analyse  # the evaluation: analysis + packets
    PYTHONPATH=. .venv/bin/python scripts/s7_report.py score --claims C.csv --answers A.csv

It runs no model. `analyse` applies exactly the preregistered rules to the outputs of
`scripts/s7_generate.py evaluate`, writes the analysis, the report and the two blind packets (40
claims, and the answers) with their label templates, and keeps the keys apart. `score` reads the
labels the human wrote, scores them and recomputes the decisions that depend on them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from ingest import answer_cases as ac
from ingest import answer_format as af
from ingest import answer_metrics as am
from ingest import answer_report as rep
from ingest import benchmark as bm
from ingest import chunks as ch
from ingest import download as dl
from ingest import evaluation as ev

DEFAULT_RESULTS = Path("data/results/dev-0.10.0/s7-s8")


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class Paths:
    """Where a run lives. Only a folder under .cache/ may replace the default: it is for rehearsals
    with synthetic outputs, never for a result."""

    def __init__(self, results: Path) -> None:
        if results != DEFAULT_RESULTS and ".cache" not in results.parts:
            raise SystemExit("--results-dir must be under .cache/")
        self.results = results
        self.cases = results / "cases.json"
        self.evaluation = results / "evaluation"
        self.human = results / "human"
        self.keys = results / "human-key"


def write_once(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"{path} exists with other content: results are never overwritten")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o444)


def load_world() -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    dev = bm.load_benchmark(bm.DEV_DIR)
    manifest = dl.load_manifest()
    docs = [ch.load_chunks(act, manifest) for act in bm.ACTS]
    layer = ev.layer_from_documents(docs)
    ev.check_layer(dev.manifest, layer)
    texts = {c.chunk_id: c.text for d in docs for c in d.chunks}
    grades = {t.need.need_id: dict(t.chunk_grades) for t in ev.targets(dev, layer)}
    return texts, grades


def to_case(d: dict[str, Any]) -> ac.Case:
    return ac.Case(
        d["case_id"],
        d["kind"],
        d["need_id"],
        d["query_id"],
        d["question"],
        d["context_name"],
        tuple(d["chunk_ids"]),
        d["source_need_id"],
    )


def outcomes_of(
    cases: list[dict[str, Any]], folder: Path, texts: dict[str, str]
) -> tuple[list[am.Outcome], dict[str, dict[str, Any]]]:
    records = {r["case_id"]: r for r in read_lines(folder / "outputs.jsonl")}
    missing = [c["case_id"] for c in cases if c["case_id"] not in records]
    if missing or len(records) != len(cases):
        raise SystemExit(f"{len(records)} outputs for {len(cases)} cases; missing {missing[:3]}")
    outcomes = []
    for d in cases:
        record = records[d["case_id"]]
        case = to_case(d)
        messages = af.build_messages(case.question, af.render_sources(case.chunk_ids, texts))
        expected = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        if record["messages_sha256"] != expected or record["prompt_sha256"] != af.prompt_sha256():
            raise SystemExit(f"{case.case_id}: the stored input is not the preregistered one")
        outcomes.append(am.make_outcome(case, record["raw_output"], texts))
    return outcomes, records


def check(folder: Path, paths: Paths) -> int:
    """Prompt development: what came back, case by case, without any decision attached."""
    payload = json.loads(paths.cases.read_text(encoding="utf-8"))
    texts, _ = load_world()
    outcomes, records = outcomes_of(payload["cases"]["prompt_dev"], folder, texts)
    for o in outcomes:
        r = records[o.case.case_id]
        d = o.decisions
        print(
            f"\n== {o.case.case_id}  tokens {r['new_tokens']}  {r['seconds']:.0f}s  "
            f"A={d.a_self_only} B={d.b_verifier_only} C={d.c_combined}"
        )
        parse = o.parsed.error or "ok"
        print(f"   parse: {parse}  extra_text={o.parsed.extra_text}  soft={o.parsed.soft}")
        if o.verdict:
            for i, v in enumerate(o.verdict.claims):
                print(f"   claim {i}: {'PASS' if v.passed else 'FAIL ' + str(v.all_failures())}")
    return 0


def analyse(paths: Paths) -> int:
    payload = json.loads(paths.cases.read_text(encoding="utf-8"))
    cases = payload["cases"]["evaluation_positive"] + payload["cases"]["evaluation_negative"]
    texts, grades = load_world()
    outcomes, records = outcomes_of(cases, paths.evaluation, texts)
    analysis = rep.analyse(outcomes, grades, records)
    seal = json.loads((paths.evaluation / "seal.json").read_text(encoding="utf-8"))["seal"]
    header = {
        "preregistration": seal["preregistration_sha256"],
        "generation commit": seal["commit"],
        "prompt sha256": seal["prompt_sha256"],
        "cases sha256": seal["cases_sha256"],
        "outputs sha256": hashlib.sha256(
            (paths.evaluation / "outputs.jsonl").read_bytes()
        ).hexdigest(),
    }
    write_once(
        paths.results / "analysis.json", json.dumps(analysis, indent=1, sort_keys=True) + "\n"
    )
    write_once(paths.results / "report.md", rep.render_report(analysis, header))

    positives = [o for o in outcomes if o.case.kind == ac.POSITIVE]
    eligible = am.eligible_claims(positives)
    sample = am.sample_human_claims(eligible)
    claims_packet, claims_key = am.build_packet(sample, eligible, texts)
    by_case = {o.case.case_id: o for o in outcomes}
    items, abstained = am.select_answer_items(positives)
    answers_packet, answers_key = am.build_answer_packet(items, by_case, texts)
    write_once(paths.human / "claims-packet.md", rep.render_claims_packet(claims_packet))
    write_once(paths.human / "claims-labels.csv", rep.labels_template(claims_packet))
    write_once(paths.human / "answers-packet.md", rep.render_answers_packet(answers_packet))
    write_once(paths.human / "answers-labels.csv", rep.labels_template(answers_packet))
    write_once(paths.keys / "claims-key.json", json.dumps(claims_key, indent=1) + "\n")
    write_once(
        paths.keys / "answers-key.json",
        json.dumps({"items": answers_key, "abstained_needs_not_labelled": abstained}, indent=1)
        + "\n",
    )
    files = sorted(
        p
        for p in paths.results.rglob("*")
        if p.is_file() and p.name != "manifest.json" and "prompt-dev" not in str(p)
    )
    manifest = {
        str(p.relative_to(paths.results)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    write_once(
        paths.results / "manifest.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    )
    print(
        f"claims in packet: {len(claims_packet)}; answers in packet: {len(answers_packet)}; "
        f"abstained needs not labelled: {len(abstained)}"
    )
    for name in ("analysis.json", "report.md", "manifest.json"):
        print(hashlib.sha256((paths.results / name).read_bytes()).hexdigest(), paths.results / name)
    return 0


def score(claims: Path, answers: Path, paths: Paths) -> int:
    payload = json.loads(paths.cases.read_text(encoding="utf-8"))
    cases = payload["cases"]["evaluation_positive"] + payload["cases"]["evaluation_negative"]
    texts, grades = load_world()
    outcomes, records = outcomes_of(cases, paths.evaluation, texts)
    claims_key = json.loads((paths.keys / "claims-key.json").read_text(encoding="utf-8"))
    answers_key = json.loads((paths.keys / "answers-key.json").read_text(encoding="utf-8"))["items"]
    human_claims = am.score_human(rep.read_labels(claims.read_text(encoding="utf-8")), claims_key)
    human_answers = am.score_answers(
        rep.read_labels(answers.read_text(encoding="utf-8")), answers_key
    )
    analysis = rep.analyse(outcomes, grades, records, human_claims, human_answers)
    seal = json.loads((paths.evaluation / "seal.json").read_text(encoding="utf-8"))["seal"]
    header = {"preregistration": seal["preregistration_sha256"], "labels": "human, blind packets"}
    write_once(
        paths.results / "analysis-with-human.json",
        json.dumps(analysis, indent=1, sort_keys=True) + "\n",
    )
    write_once(paths.results / "report-with-human.md", rep.render_report(analysis, header))
    for name in ("analysis-with-human.json", "report-with-human.md"):
        print(hashlib.sha256((paths.results / name).read_bytes()).hexdigest(), paths.results / name)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    sub = parser.add_subparsers(dest="command", required=True)
    chk = sub.add_parser("check")
    chk.add_argument("--folder", type=Path, required=True)
    sub.add_parser("analyse")
    sc = sub.add_parser("score")
    sc.add_argument("--claims", type=Path, required=True)
    sc.add_argument("--answers", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = Paths(args.results_dir)
    if args.command == "check":
        return check(args.folder, paths)
    if args.command == "analyse":
        return analyse(paths)
    return score(args.claims, args.answers, paths)


if __name__ == "__main__":
    sys.exit(main())
