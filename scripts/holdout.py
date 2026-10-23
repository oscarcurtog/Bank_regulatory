"""Build, guard and freeze the internal TEST benchmark (S3, D-054).

    PYTHONPATH=. .venv/bin/python scripts/holdout.py init
    PYTHONPATH=. .venv/bin/python scripts/holdout.py grid
    PYTHONPATH=. .venv/bin/python scripts/holdout.py record "grid fixed before any concept"
    PYTHONPATH=. .venv/bin/python scripts/holdout.py isolation
    PYTHONPATH=. .venv/bin/python scripts/holdout.py describe data/benchmark/dev
    PYTHONPATH=. .venv/bin/python scripts/holdout.py overlap-statements
    PYTHONPATH=. .venv/bin/python scripts/holdout.py overlap-evidence
    PYTHONPATH=. .venv/bin/python scripts/holdout.py backup
    PYTHONPATH=. .venv/bin/python scripts/holdout.py state
    PYTHONPATH=. .venv/bin/python scripts/holdout.py restore <bundle> --into <new directory>
    PYTHONPATH=. .venv/bin/python scripts/holdout.py validate
    PYTHONPATH=. .venv/bin/python scripts/holdout.py record-public-commit <sha>
    PYTHONPATH=. .venv/bin/python scripts/holdout.py start --contract <pinned sha256>
    PYTHONPATH=. .venv/bin/python scripts/holdout.py resume --contract <pinned sha256>
    PYTHONPATH=. .venv/bin/python scripts/holdout.py freeze --contract <pinned sha256>
    PYTHONPATH=. .venv/bin/python scripts/holdout.py record-amendment-commit <sha>
    PYTHONPATH=. .venv/bin/python scripts/holdout.py amend --contract <amended sha256>
    PYTHONPATH=. .venv/bin/python scripts/holdout.py controller --contract <pinned sha256> \
        --policy holdout-controller.v1 --start | --resume
    PYTHONPATH=. .venv/bin/python scripts/holdout.py certification-init <certification id> \
        --claude-bin <binary> --binary-sha256 <its sha256> --account-evidence <raw transcript>
    PYTHONPATH=. .venv/bin/python scripts/holdout.py isolation-test --certification <id> \
        --claude-bin <binary>
    PYTHONPATH=. .venv/bin/python scripts/holdout.py broker-test --certification <id> \
        --claude-bin <binary>

init creates data/benchmark/test/, which git ignores, with a local git history of its own
that refuses to push. grid writes the coverage grid and refuses to overwrite a different
one. record commits the private history: that commit, before the step it precedes, is the
record of the order of the steps. isolation checks that no run, embedding or result holds
a TEST id. backup writes a git bundle of the private history to
~/.fundamento-private/holdout-backups/, verifies it and restores it into a temporary clone
that must match the private directory file by file; once the construction has a durable
state, the state is recorded first, its raw artifacts are copied beside the bundles, and
the clone's state must verify and equal it. state prints that state, verified and read-only,
as counts: never a statement, a query or a judgment. restore makes a new working copy from a
bundle and its raw artifacts, and refuses an existing directory.

The construction runs through ingest/holdout_launcher.py only. start and resume are the two
commands that can run a model on TEST, and each needs the pinned contract's full SHA-256
typed again; start also needs the public pre-authoring commit recorded (record-public-commit,
after the push). validate runs every check of the launcher and builds nothing. freeze replays
the state with every write refused, so it runs no model, then exports, seals and writes
freeze.json and the public commitment. isolation-test and broker-test run the pinned binary on
synthetic packets only, never on TEST, and only inside a Gate 1 certification: a namespace of
its own, never a construction's, made once by certification-init with its own runtime pin
(provenance/runtime.json), in the private history of certifications. Each run's raw folder is
reserved, with an exclusive mkdir, in the folder its raw evidence goes to, before anything
runs; every evidence file is written as a new file, so nothing is ever overwritten, and a
number is never reused, not even one whose run died. An amended contract
(CONSTRUCTION_AMENDMENT_1) has its public commit recorded with record-amendment-commit, and
amend records the amendment in the durable state at the point it names, after a replay under
the amended code that runs no model.
controller runs START or RESUME unattended and RESUME again only under its own policy
(ingest/holdout_controller.py), for the person who typed the pinned contract's SHA-256 and the
policy's id; it never freezes.

Nothing here runs a retriever, a reranker or an embedding model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_certification as hcert
from ingest import holdout_isolation as hi
from ingest import holdout_launcher as hla
from ingest import holdout_process as hproc
from ingest import holdout_state as hs

REPO = Path.cwd()
SUBDIRS = ("provenance", "overlap", "packets", "audits")
README = """# TEST (private)

The internal TEST benchmark of S3 (D-054). Never pushed: the public repository ignores this
directory, and this history refuses every push. Back it up with scripts/holdout.py backup.

- grid.json: the 60 cells, fixed before any concept is written
- needs.jsonl, queries.jsonl, qrels.jsonl, adjudications.jsonl, manifest.json: the benchmark
- state/: the construction's durable state (ingest/holdout_state.py) and its kept artifacts
- packets/: what every isolated context received, as written
- audits/: the isolation audit of every context, read from its transcript
- provenance/<need_id>.json: who wrote, gated, queried and judged each need, and the search
- overlap/: the gatekeeper's verdicts and the overlap diagnostics
- freeze.json: written once, at 1.0.0
"""


def git(directory: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(directory), *args], check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def ignored_by_repo(path: Path) -> bool:
    probe = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", str(path / "probe")], check=False
    )
    return probe.returncode == 0


def init(directory: Path) -> None:
    if not ignored_by_repo(directory):
        raise ho.HoldoutError(f"{directory} is not ignored by the public repository")
    if directory.exists() and any(directory.iterdir()):
        raise ho.HoldoutError(f"{directory} already exists and is not empty")
    directory.mkdir(parents=True, exist_ok=True)
    for name in SUBDIRS:
        (directory / name).mkdir()
        (directory / name / ".keep").write_text("", encoding="utf-8")
    (directory / "README.md").write_text(README, encoding="utf-8")
    git(directory, "init", "-q", "-b", "main")
    ho.guard_private_history(directory)
    git(directory, "add", "-A")
    git(directory, "commit", "-q", "-m", "test: private history begins")
    print(f"created {directory}, private history at {git(directory, 'rev-parse', 'HEAD')[:12]}")


def record(directory: Path, message: str) -> None:
    git(directory, "add", "-A")
    if not git(directory, "status", "--porcelain"):
        print("nothing to record")
        return
    git(directory, "commit", "-q", "-m", message)
    print(f"recorded {git(directory, 'rev-parse', 'HEAD')[:12]}: {message}")


def grid(directory: Path) -> None:
    dev = bm.load_benchmark(bm.DEV_DIR)
    areas = ho.load_areas()
    built = ho.build_grid(dev, areas)
    problems = ho.grid_problems(built, dev, areas)
    if problems:
        raise ho.HoldoutError("the grid breaks its rules:\n  " + "\n  ".join(problems))
    path = directory / ho.GRID_FILE
    if path.exists():
        if ho.load_grid(directory) != built:
            raise ho.HoldoutError(f"{path} exists and differs: a fixed grid is never rewritten")
        print(f"{path} is already this grid")
    else:
        ho.write_json(path, built.model_dump(mode="json"))
        print(f"wrote {path}")
    for b in range(1, ho.BATCHES + 1):
        cells = [c for c in built.cells if c.batch == b]
        print(
            f"  batch {b}: {len(cells)} cells, types {dict(Counter(c.type for c in cells))}, "
            f"figure or date {sum(c.number_or_date for c in cells)}"
        )


def test_ids(directory: Path) -> set[str]:
    ids: set[str] = set()
    for name, field in ((bm.NEEDS_FILE, "need_id"), (bm.QUERIES_FILE, "query_id")):
        path = directory / name
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ids.add(json.loads(line)[field])
    return ids


def isolation(directory: Path) -> list[str]:
    ids = test_ids(directory)
    problems = ho.isolation_problems(REPO, ids)
    checked = ", ".join(ho.SYSTEM_DIRS)
    print(f"isolation: {checked} checked against the N1xx range and {len(ids)} TEST ids")
    for p in problems:
        print(f"  {p}")
    print("  no sign of TEST" if not problems else f"  {len(problems)} problems")
    return problems


def describe(directory: Path) -> dict[str, Any]:
    benchmark = bm.load_benchmark(directory)
    docs = bm.load_corpus()
    return {
        "benchmark": f"{benchmark.manifest.role} {benchmark.manifest.benchmark_version}",
        "distributions": ho.distributions(benchmark, ho.load_areas()),
        "lexical": ho.lexical_profile(benchmark, docs),
    }


def _statements(directory: Path) -> dict[str, str]:
    return {n.need_id: n.statement for n in bm.read_jsonl(directory / bm.NEEDS_FILE, bm.Need)}


def overlap_statements(directory: Path) -> None:
    test = _statements(directory)
    reference = _statements(bm.DEV_DIR) | _statements(bm.FIXTURES_DIR)
    against_dev = ho.closest_statements(test, reference)
    within = ho.closest_statements(test, test)
    record = {
        need: {
            "closest_dev_or_fixture": [c._asdict() for c in against_dev[need]],
            "closest_test": [c._asdict() for c in within[need]],
        }
        for need in sorted(test)
    }
    ho.write_json(directory / "overlap" / "statements.json", record)
    flagged = [
        n
        for n, v in record.items()
        if max([c["jaccard"] for c in v["closest_dev_or_fixture"] + v["closest_test"]], default=0.0)
        >= ho.JACCARD_FLAG
    ]
    print(f"{len(test)} statements; Jaccard >= {ho.JACCARD_FLAG}: {len(flagged)} {flagged}")


def overlap_evidence(directory: Path) -> None:
    test = bm.read_jsonl(directory / bm.QRELS_FILE, bm.Qrel)
    dev = bm.load_benchmark(bm.DEV_DIR).qrels
    pairs = ho.evidence_overlap(test, dev)
    needs = {n.need_id for n in bm.read_jsonl(directory / bm.NEEDS_FILE, bm.Need)}
    strata = ho.evidence_strata(needs, pairs)
    ho.write_json(
        directory / "overlap" / "evidence.json",
        {"pairs": [p._asdict() for p in pairs], "strata": strata},
    )
    to_gate = sorted({p.test_need for p in pairs if p.shared_nodes})
    print(
        f"strata {dict(Counter(strata.values()))}; "
        f"shared grade-3 nodes, back to the gatekeeper: {to_gate}"
    )


def figure_density(directory: Path) -> None:
    from ingest import psd2_exclusions as px

    excluded = px.load_psd2_exclusions(px.PSD2_EXCLUSIONS_PATH).excluded()
    texts = bm.node_texts(bm.load_corpus())
    density = ho.figure_density(ho.load_areas(), texts, excluded)
    out = directory / "audits" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    ho.write_json(out / "figure_density.json", density)
    for area, d in density.items():
        print(f"  {d['share']:.3f} {d['with_figure_or_date']:>4}/{d['nodes']:<4} {area}")


HEADLESS_MODEL = "claude-opus-5-5"  # the model of DEV's contexts, recorded in their transcripts
# Claude Code's folder for this repository: its transcripts and its memory
PROJECT_STATE = Path.home() / ".claude" / "projects" / hi.encoded_project_dir(str(REPO.resolve()))
# a context of DEV batch 2 (its statement writer for N016): the real positive control
DEV_SUBAGENT = (
    PROJECT_STATE
    / "590cec27-c707-4df2-bf14-8991fc721a82"
    / "subagents"
    / "agent-aac37401cf9485f35.jsonl"
)


def isolation_terms() -> list[hi.Term]:
    return hla.project_terms(REPO)


RUNTIME_PIN = "runtime.json"
# Raw transcripts hold the account's email and organization id: they stay outside every git
# history, in a private folder; the TEST history keeps redacted copies and their hashes. This
# folder holds Gate 1's runs of 2026-10-19, which run 6 was built on: it is never written again.
RAW_DIR = Path.home() / ".fundamento-private" / "transcripts"
# A Gate 1 certification (ingest/holdout_certification.py): its provenance and its redacted
# audits in a private history of certifications, its raw transcripts beside it, outside every
# git history, under the same id
CERTIFICATIONS_README = """# Gate 1 certifications (private)

One folder per certification, named by its id: made once by scripts/holdout.py
certification-init, written by isolation-test and broker-test, sealed by certification-seal.
Never pushed: this history refuses every push.

- <id>/provenance/runtime.json: the pinned Claude Code binary, the requested model, the host
  and the organization hash the certification expects, with the evidence it was read from
- <id>/provenance/certification.json: the clean public commit it was made from, the code's
  hashes, and the execution substrate (VS Code, the extension, macOS's automatic installs)
- <id>/provenance/gate1.json, broker_test.json: written once, when the test passes
- <id>/audits/isolation|broker/run-N/: the redacted copies of a run's evidence and its report
- <id>/provenance/suites/, <id>/manifest.json: the suites' reports and the seal of a PASS

The raw transcripts hold the account's email and organization id: they live outside every git
history, in ~/.fundamento-private/certification-raw/<id>/isolation|broker/run-N/. A run-N there
holds reserved.json from the moment its number is taken and complete.json once its evidence is
whole; one without complete.json died, and its number is never used again.
"""


def next_run(root: Path) -> str:
    """The next run of a raw folder: one above the highest run-N it holds, so no number is
    reused, not even one whose run died before its evidence was whole."""
    numbers = [int(m.group(1)) for p in root.iterdir() if (m := hcert.RUN_NAME.fullmatch(p.name))]
    return f"run-{max(numbers, default=0) + 1}"


def reserve_run(root: Path) -> Path:
    """A run's raw folder, reserved before anything runs, in the folder its raw evidence goes
    to. The exclusive mkdir is the reservation: a name that exists, even one another process
    took an instant ago, is a stop, and nothing is overwritten, removed or reused."""
    run = root / next_run(root)
    try:
        run.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ho.HoldoutError(f"{run} exists: another run took it; nothing was written") from error
    reserved = {"run": run.name, "reserved_at": datetime.now(UTC).isoformat()}
    hcert.write_new_json(run / hcert.RESERVED, reserved)
    return run


def open_run(namespace: Path, raw: Path, kind: str) -> tuple[Path, Path]:
    """A certification's next run of a kind: its raw folder reserved first, then its audit
    folder by the same name, created as new. A stop after the reservation leaves the number
    taken."""
    reserved = reserve_run(raw / kind)
    out = namespace / "audits" / kind / reserved.name
    try:
        out.mkdir()
    except FileExistsError as error:
        raise ho.HoldoutError(f"{out} exists: {reserved.name} stays taken") from error
    return out, reserved


def keep_evidence(out: Path, raw: Path, texts: dict[str, str], values: list[str]) -> dict[str, str]:
    """The raw texts into the run's reserved raw folder, redacted copies into its audit folder,
    each a new file. Returns the SHA-256 of each raw file, which the history keeps instead of
    the values."""
    hashes = {}
    for name, text in texts.items():
        hcert.write_new(raw / name, text)
        hashes[name] = bm.sha256_file(raw / name)
        hcert.write_new(out / name, hi.redact(text, values))
    return hashes


def complete_run(raw: Path, verdict: str, hashes: dict[str, str]) -> None:
    hcert.write_new_json(
        raw / hcert.COMPLETE,
        {
            "run": raw.name,
            "verdict": verdict,
            "raw_evidence_sha256": hashes,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )


def overlaps(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a.is_relative_to(b) or b.is_relative_to(a)


def certification_problems(history: Path, raw_root: Path) -> list[str]:
    """Where certifications may not live: in or around the repository (TEST and its runs
    included) or a folder that holds a construction's or Gate 1's historical evidence."""
    from ingest import holdout_controller as hct

    historical = (RAW_DIR, hs.RAW_DIR, ho.BACKUP_DIR, hct.LOG_DIR)
    problems = []
    for root in (history, raw_root):
        if overlaps(root, REPO):
            problems.append(f"{root} overlaps the repository")
        problems += [f"{root} overlaps {h}" for h in historical if overlaps(root, h)]
    if overlaps(history, raw_root):
        problems.append(f"{raw_root} overlaps {history}: raw transcripts stay outside git")
    return problems


def certification_history(history: Path) -> None:
    """The private history of the certifications: made once, with the private author and the
    hook that refuses every push. An existing folder must be that history."""
    if not history.exists():
        ho.private_folder(history, REPO)
        git(history, "init", "-q", "-b", "main")
        ho.guard_private_history(history)
        hcert.write_new(history / "README.md", CERTIFICATIONS_README)
        git(history, "add", "-A")
        git(history, "commit", "-q", "-m", "certifications: private history begins")
        return
    hook = history / ".git" / "hooks" / "pre-push"
    if not hook.is_file() or hook.read_text(encoding="utf-8") != ho.NO_PUSH_HOOK:
        raise ho.HoldoutError(f"{history} exists and is not the private history of certifications")


def account_organization(transcript: Path) -> str:
    """The organization hash a certification expects, read from a raw transcript of the same
    login, its credential_org attachment, never typed: exactly one organization, or a stop."""
    found = {
        hi.sha256_text(str(body["organizationUuid"]))
        for r in hi.parse_records({"transcript": transcript.read_text(encoding="utf-8")})
        if isinstance(body := r.get("attachment"), dict)
        and body.get("type") == "credential_org"
        and body.get("organizationUuid")
    }
    if len(found) != 1:
        raise ho.HoldoutError(f"{transcript} holds {len(found)} organizations, not one")
    return found.pop()


def certification_init(
    history: Path,
    raw_root: Path,
    certification: str,
    binary: Path,
    binary_sha256: str,
    account_evidence: Path,
) -> Path:
    """A new Gate 1 certification: its namespace in the private history of certifications, its
    raw folder beside it, its own runtime pin, from the binary (its SHA-256 typed again), the
    host and the account evidence named, and what it is made from: the clean public commit
    checked out and the execution substrate, which must keep VS Code, the extension and macOS
    from changing anything. Everything is read and checked before anything is made; an id is
    used once: an existing namespace or raw folder is a stop, never reused or removed."""
    from ingest import holdout_runtime as hrt

    hcert.folders(history, raw_root, certification)  # the id's form
    problems = certification_problems(history, raw_root)
    if problems:
        raise ho.HoldoutError("; ".join(problems))
    code = hcert.code_identity(REPO)
    identity = hi.binary_identity(binary)
    if identity["sha256"] != binary_sha256:
        raise ho.HoldoutError(f"{binary} is {identity['sha256'][:12]}, not the SHA-256 typed")
    substrate = hcert.execution_substrate(binary)
    problems = hcert.substrate_problems(substrate)
    if problems:
        raise ho.HoldoutError("; ".join(problems))
    pin = identity | {
        "model": HEADLESS_MODEL,
        "organization_sha256": account_organization(account_evidence),
        "identity_source": {
            "logical": logical_source(account_evidence),
            "path": str(account_evidence),
            "sha256": bm.sha256_file(account_evidence),
            "attachment": "credential_org",
        },
        "host": hrt.host_identity(),
        "certification": certification,
        "pinned_at": datetime.now(UTC).isoformat(),
    }
    record = {
        "schema": hcert.SCHEMA,
        "certification": certification,
        "code": code,
        "substrate": substrate,
        "created_at": pin["pinned_at"],
    }
    certification_history(history)
    namespace, raw = history / certification, raw_root / certification
    ho.private_folder(raw_root, REPO)
    for folder in (namespace, raw):
        try:
            folder.mkdir(mode=0o700)
        except FileExistsError as error:
            raise ho.HoldoutError(f"{folder} exists: a certification id is used once") from error
    for kind in hcert.KINDS:
        (namespace / "audits" / kind).mkdir(parents=True)
        hcert.write_new(namespace / "audits" / kind / ".keep", "")
        (raw / kind).mkdir(mode=0o700)
    (namespace / hcert.PROVENANCE).mkdir()
    hcert.write_new_json(namespace / hcert.PROVENANCE / RUNTIME_PIN, pin)
    hcert.write_new_json(namespace / hcert.PROVENANCE / hcert.IDENTITY, record)
    git(history, "add", "-A")
    git(history, "commit", "-q", "-m", f"certification {certification}: namespace and runtime pin")
    return namespace


def logical_source(path: Path) -> str:
    """Where a piece of private evidence is, named from the private root when it is under it."""
    found, root = path.resolve(), hcert.PRIVATE_ROOT.resolve()
    return found.relative_to(root).as_posix() if found.is_relative_to(root) else str(found)


def certification_paths(certification: str) -> tuple[Path, Path]:
    """An existing certification's namespace and raw folder, by its id."""
    namespace, raw = hcert.folders(
        hcert.CERTIFICATIONS_DIR, hcert.CERTIFICATION_RAW_DIR, certification
    )
    if not namespace.is_dir() or not raw.is_dir():
        raise ho.HoldoutError(f"no certification {certification}: certification-init makes one")
    return namespace, raw


def pinned_runtime(directory: Path, binary: Path) -> dict[str, Any] | None:
    """The certification's pinned Claude Code binary (provenance/runtime.json), written by
    certification-init and the only one allowed. None means it is missing or changed: stop."""
    path = directory / "provenance" / RUNTIME_PIN
    if not path.exists():
        print(f"STOP: {path} does not exist: certification-init pins the runtime")
        return None
    current = hi.binary_identity(binary)
    pin = json.loads(path.read_text(encoding="utf-8"))
    if (current["version"], current["sha256"]) != (pin["version"], pin["sha256"]):
        print(
            f"STOP: the binary is {current['version']} {current['sha256'][:12]}, "
            f"pinned {pin['version']} {pin['sha256'][:12]}"
        )
        return None
    return dict(pin)


RECORDS = {"isolation": "gate1.json", "broker": "broker_test.json"}


def refused_run(namespace: Path, raw: Path, kind: str) -> str | None:
    """Why a certification cannot run a test of a kind now, or None. It must be OPEN (a FAILED
    one never runs again: a new attempt is a new certification), hold no final run of that kind
    nor its record (one each), and, for the broker test, have passed Gate 1. A run that died
    is no final run: the next takes another number."""
    found = hcert.status(namespace, raw)
    if found != "OPEN":
        return f"the certification is {found}: a new attempt needs a new certification"
    provenance = namespace / hcert.PROVENANCE
    if hcert.finals(raw, kind) or (provenance / RECORDS[kind]).exists():
        return f"this certification has its final {kind} run already: one of each"
    gate1 = (
        hcert.finals(raw, "isolation") == ["PASS"] and (provenance / RECORDS["isolation"]).exists()
    )
    if kind == "broker" and not gate1:
        return "the broker test runs only after Gate 1 passed in this certification"
    return None


def isolation_test(namespace: Path, raw: Path, binary: Path, extra_env: list[str]) -> int:
    """The safe-mode check D-054 requires before any need is written: positive controls
    first, then one dummy context through the exact mechanism the authors will use. PASS
    only if every check of the audit passes; any failure is a stop. It runs inside a
    certification (namespace and raw folder), which keeps one final Gate 1."""
    pin = pinned_runtime(namespace, binary)
    if pin is None:
        return 2
    refused = refused_run(namespace, raw, "isolation")
    if refused:
        print(f"STOP: {refused}")
        return 2
    terms = isolation_terms()
    failures = hi.positive_controls(terms, DEV_SUBAGENT)
    print(f"positive controls: {len(terms)} terms; failures {failures or 'none'}")
    if failures:
        return 1
    expected_org = pin.get("organization_sha256")
    if not expected_org:
        print("STOP: the pin has no expected organization hash")
        return 2
    out, reserved = open_run(namespace, raw, "isolation")
    name = reserved.name
    run = hi.run_headless(hi.DUMMY_PACKET, HEADLESS_MODEL, binary, pin, extra_env)
    texts = {
        "transcript.jsonl": run.transcript,
        "stream.jsonl": run.stream,
        "stderr.txt": run.stderr,
        "launcher-context.json": json.dumps(run.runtime_context, sort_keys=True),
    }
    report = hi.audit_run(
        {"transcript": run.transcript, "stream": run.stream, "stderr": run.stderr},
        terms,
        expected_cwd=run.workdir,
        expected_version=pin["version"],
        expected_org_sha256=expected_org,
        runtime_context=run.runtime_context,
    )
    if run.returncode != 0:
        report.problems.append(f"the run exited with {run.returncode}")
    if not run.transcript:
        report.problems.append("no transcript found: the context cannot be audited")
    try:
        answer = json.loads(hi_strip(run.answer or ""))
        missing = {"sentence", "context_report", "working_directory"} - set(answer)
        if missing:
            report.problems.append(f"the answer lacks {sorted(missing)}")
    except json.JSONDecodeError:
        answer = None
        report.problems.append("the answer is not the JSON the packet asks for")
    report.ok = not report.problems
    environment = hi.clean_environment(extra_env)
    values = hi.identity_values(
        hi.parse_records({"transcript": run.transcript, "stream": run.stream})
    )
    hcert.write_new(out / "dummy_packet.txt", hi.DUMMY_PACKET)
    raw_hashes = keep_evidence(out, reserved, texts, values)
    answer_text = hi.redact(json.dumps(answer, ensure_ascii=False), values) if answer else None
    hcert.write_new_json(
        out / "report.json",
        {
            "verdict": "PASS" if report.ok else "FAIL",
            "binary": pin,
            "mechanism": hi.headless_command(binary, HEADLESS_MODEL),
            "environment": {
                "inherited": [k for k in environment if k in (*hi.ENV_INHERITED, *extra_env)],
                "set": {k: v for k, v in environment.items() if k in hi.ENV_SET or k == "PATH"},
            },
            "workdir": run.workdir,
            "runtime_context": run.runtime_context,
            "session_id": run.session_id,
            "transcript_path": run.transcript_path,
            "terms": len(terms),
            "attachment_controls": len(hi.ATTACHMENT_CONTROLS),
            "provenance_controls": len(hi.PROVENANCE_CONTROLS),
            "positive_controls": "passed",
            "account_identity_metadata_present": report.account_identity_metadata_present,
            "raw_evidence_sha256": raw_hashes,
            "report": {k: v for k, v in report.__dict__.items()},
            "answer": json.loads(answer_text) if answer_text else None,
        },
    )
    print(f"run: exit {run.returncode}, cwd {report.cwd}, model {report.model}")
    print(f"tools {report.tools_offered}; attachments {report.attachments}")
    print(f"answer: {answer}")
    if report.ok:
        hcert.write_new_json(
            namespace / "provenance" / "gate1.json",
            {
                "gate_1": "PASS",
                "run": name,
                "binary": {k: pin[k] for k in ("path", "version", "sha256")},
                "model": HEADLESS_MODEL,
                "mechanism": hi.headless_command(binary, HEADLESS_MODEL),
                "environment_allowlist": {
                    "inherited": sorted([*hi.ENV_INHERITED, *extra_env]),
                    "path": hi.SYSTEM_PATH,
                    "set": hi.ENV_SET,
                },
                "plugin_allowlist": list(hi.BUILTIN_PLUGINS),
                "attachment_schemas": hi.schema_description(),
                "audit": hi.audit_identity(terms),
                "attachment_controls": len(hi.ATTACHMENT_CONTROLS),
                "provenance_controls": len(hi.PROVENANCE_CONTROLS),
                "account_identity_metadata_present": report.account_identity_metadata_present,
            },
        )
    complete_run(reserved, "PASS" if report.ok else "FAIL", raw_hashes)
    print("GATE 1 PASS" if report.ok else f"GATE 1 FAIL: {report.problems} {report.findings}")
    return 0 if report.ok else 1


def broker_test(namespace: Path, raw: Path, binary: Path, extra_env: list[str]) -> int:
    """The synthetic test of judge A's transport (D-054): one isolated judge in one
    stream-json process, a task on the corpus that is neither DEV nor TEST, the broker in
    between. PASS only if the whole session passes the Gate 1 audit, every user message was
    the broker's, the judge kept the state of the conversation, and the answer is right. It
    runs inside a certification (namespace and raw folder), after its Gate 1 passed, and the
    certification keeps one final broker test."""
    from ingest import holdout_broker as hb

    pin = pinned_runtime(namespace, binary)
    if pin is None:
        return 2
    refused = refused_run(namespace, raw, "broker")
    if refused:
        print(f"STOP: {refused}")
        return 2
    terms = isolation_terms()
    failures = hi.positive_controls(terms, DEV_SUBAGENT)
    print(f"positive controls: {len(terms)} terms; failures {failures or 'none'}")
    if failures:
        return 1
    out, reserved = open_run(namespace, raw, "broker")
    name = reserved.name
    conversation = hb.StreamJsonConversation(binary, HEADLESS_MODEL, pin, extra_env)
    problems: list[str] = []
    try:
        run = hb.run_judge_a(conversation, hb.SYNTHETIC_PACKET, hb.corpus_tools())
    except Exception as error:  # the transport itself failed: recorded, then a stop
        run = None
        problems.append(f"transport: {error}")
    session = conversation.close()
    report = hi.audit_run(
        {
            "transcript": session["transcript"],
            "stream": session["stream"],
            "stderr": session["stderr"],
        },
        terms,
        expected_cwd=session["_gate1_context"]["launcher_workdir"],
        expected_version=pin["version"],
        expected_org_sha256=pin["organization_sha256"],
        runtime_context=session["_gate1_context"],
    )
    problems += report.problems
    if session["returncode"] != 0:
        problems.append(f"the process exited with {session['returncode']}")
    if not session["transcript"]:
        problems.append("no transcript found: the session cannot be audited")
    if run is not None:
        foreign = hb.foreign_user_messages(session["transcript"], run.sent)
        if foreign:
            problems.append(f"{len(foreign)} user messages the broker did not send")
        if run.outcome != "final":
            problems.append(f"the run ended as {run.outcome}")
        if len(run.log) < 2:
            problems.append("fewer than two requests: the state was not exercised")
        elif run.log[1]["args"].get("article") != hb.SYNTHETIC_ANSWER["article"] and not any(
            hb.SYNTHETIC_ANSWER["article"] in json.dumps(e["args"]) for e in run.log[1:]
        ):
            problems.append("no later request used what the first result returned")
        final = run.final or {}
        if {k: str(final.get(k)) for k in hb.SYNTHETIC_ANSWER} != hb.SYNTHETIC_ANSWER:
            problems.append(f"wrong answer: {final}")
    values = hi.identity_values(
        hi.parse_records({"transcript": session["transcript"], "stream": session["stream"]})
    )
    texts = {
        "transcript.jsonl": session["transcript"],
        "stream.jsonl": session["stream"],
        "stderr.txt": session["stderr"],
        "launcher-context.json": json.dumps(session["_gate1_context"], sort_keys=True),
    }
    raw_hashes = keep_evidence(out, reserved, texts, values)
    verdict = "PASS" if not problems else "FAIL"
    record = {
        "verdict": verdict,
        "binary": {k: pin[k] for k in ("path", "version", "sha256")},
        "command": session["command"],
        "environment_inherited": sorted([*hi.ENV_INHERITED, *extra_env]),
        "audit": hi.audit_identity(terms),
        "account_identity_metadata_present": report.account_identity_metadata_present,
        "raw_evidence_sha256": raw_hashes,
        "runtime_context": session["_gate1_context"],
        "outcome": run.outcome if run else None,
        "log": run.log if run else [],
        "violations": run.violations if run else [],
        "final": run.final if run else None,
        "turns": len(run.sent) if run else 0,
        "problems": problems,
        "findings": report.findings,
    }
    hcert.write_new_json(out / "report.json", json.loads(hi.redact(json.dumps(record), values)))
    if verdict == "PASS":
        hcert.write_new_json(
            namespace / "provenance" / "broker_test.json",
            {"broker_test": "PASS", "run": name, **record},
        )
    complete_run(reserved, verdict, raw_hashes)
    print(f"requests: {[(e['op'], e['args']) for e in (run.log if run else [])]}")
    print(f"final: {run.final if run else None}; violations: {run.violations if run else None}")
    print(f"attachments: {report.attachments}")
    print(f"BROKER TEST {verdict}" + (f": {problems} {report.findings}" if problems else ""))
    return 0 if verdict == "PASS" else 1


def certified(certification: str, manifest_sha256: str) -> Path:
    """The sealed certification the person names for a contract, by its id and its manifest's
    SHA-256 typed again: verified (ingest/holdout_certification.py selected), with the pinned
    binary still the one it certified. Its namespace, or a stop."""
    roots = (hcert.CERTIFICATIONS_DIR, hcert.CERTIFICATION_RAW_DIR)
    namespace = hcert.selected(certification, manifest_sha256, REPO, roots)
    pin = hcert.read_json(namespace / hcert.PROVENANCE / RUNTIME_PIN)
    live = hi.binary_identity(Path(pin["path"]))
    if (live["version"], live["sha256"]) != (pin["version"], pin["sha256"]):
        raise hcert.CertificationError(f"the binary at {pin['path']} is not the certified one")
    return namespace


def pre_authoring_freeze(directory: Path, certification: str, manifest_sha256: str) -> int:
    """Fix everything decided before authoring (Codex Gate 2, P1.2, P1.4, P1.5): the order of
    the cells, the omission audit, the role contracts, the opaque-id salt, the templates, and
    the evaluation contract of E, E5, BM25 and S4. Written once; the contract's hash is pinned
    in the private provenance, and every later step checks it. Claude's runtime comes from the
    sealed certification the person names, checked before anything is written, and the
    contract records it; never from TEST's own provenance."""
    import secrets

    from ingest import holdout_contract as hc
    from ingest import holdout_roles as hr

    try:
        gate_1 = certified(certification, manifest_sha256)
    except hcert.CertificationError as error:
        print(f"STOP: {error}")
        return 1
    dev = bm.load_benchmark(bm.DEV_DIR)
    areas = ho.load_areas()
    grid_ = ho.load_grid(directory)
    if grid_ != ho.build_grid(dev, areas) or ho.grid_problems(grid_, dev, areas):
        print("STOP: the private grid is not the one the code builds, or breaks its rules")
        return 1
    provenance = directory / "provenance"
    plan = ho.plan_record(grid_)
    roles = hr.contracts_record()
    for name, data in (("plan.json", plan), ("role_contracts.json", roles)):
        path = provenance / name
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != data:
            print(f"STOP: {path} exists and differs")
            return 1
        ho.write_json(path, data)
    salt = provenance / "opaque_salt.hex"
    if not salt.exists():
        salt.write_text(secrets.token_hex(32) + "\n", encoding="utf-8")
    frozen = provenance / "frozen_templates.json"
    record = json.loads(frozen.read_text(encoding="utf-8"))
    record["files"] = {**record["files"], **hr.template_hashes()}
    record["templates_tree_sha256"] = ho.tree_sha256(bm.TEMPLATES_DIR / "holdout")
    record["change"] = "pre-authoring freeze: the role contracts' templates, frame A only (v2)"
    ho.write_json(frozen, record)
    from ingest import holdout_runtime as hrt

    claude = hrt.claude_runtime(gate_1 / hcert.PROVENANCE)
    mlx = hrt.mlx_runtime()
    reference = hcert.reference(gate_1)
    record = hc.construction(directory / ho.GRID_FILE, plan, roles, claude, mlx, reference)
    contract = hc.build(REPO, record)
    digest = hc.write(contract)
    ho.write_json(
        provenance / "contract_pin.json",
        {
            "path": hc.CONTRACT_PATH.as_posix(),
            "sha256": digest,
            "pinned_at": datetime.now(UTC).isoformat(),
        },
    )
    hc.verify(digest)
    print(f"contract {hc.CONTRACT_PATH}: {digest}")
    print(f"order {plan['order']['sha256'][:12]}, omission {plan['omission_audit']['sha256'][:12]}")
    return 0


def controller(paths: hla.Paths, contract: str, policy: str, first: str) -> int:
    """The unattended controller (ingest/holdout_controller.py): START or RESUME, then RESUME
    again only under its policy, until it stops. It never freezes. 0 when the construction is
    complete, 3 for any other stop; the code is printed and logged."""
    from ingest import holdout_controller as hct

    if policy != hct.POLICY:
        print(f"STOP: the controller's policy is {hct.POLICY}, typed again")
        return 3
    frozen, problems = hla.contract_problems(paths)
    if frozen is None or problems or frozen.contract_sha256 != contract:
        print("STOP: the pinned contract is not the one typed, or it does not verify")
        return 3

    def validate() -> list[str]:
        report = hla.validate(paths)
        found = [*report["problems"], *report.get("public_commit_problems", [])]
        return found if report["public_commit"] else [*found, "no public commit recorded"]

    run = hct.Controller(
        paths.test,
        paths.raw,
        contract,
        validate,
        hla.run_auditor(paths, frozen),
        hct.child_command(contract, REPO, paths),
        hct.LOG_DIR,
        cwd=REPO,
    )
    code = run.run(first)
    print(f"CONTROLLER STOPPED: {code}")
    return 0 if code.startswith("complete") else 3


def hi_strip(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
    return stripped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, default=ho.HOLDOUT_DIR, help="the TEST directory")
    parser.add_argument("--raw-dir", type=Path, default=hs.RAW_DIR, help="raw copies and spools")
    parser.add_argument("--backup-dir", type=Path, default=ho.BACKUP_DIR, help="the backups")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create the private directory and its history")
    sub.add_parser("grid", help="write the coverage grid, or check that it is unchanged")
    rec = sub.add_parser("record", help="commit the private history")
    rec.add_argument("message")
    sub.add_parser("isolation", help="check that no run, embedding or result holds TEST")
    cert = sub.add_parser(
        "certification-init", help="a new Gate 1 certification: its namespace and runtime pin"
    )
    cert.add_argument("certification", help="the new certification's id, used once")
    cert.add_argument("--claude-bin", type=Path, required=True, help="the Claude Code binary")
    cert.add_argument("--binary-sha256", required=True, help="the binary's SHA-256, typed again")
    cert.add_argument(
        "--account-evidence",
        type=Path,
        required=True,
        help="a raw transcript of the same login: its credential_org gives the organization hash",
    )
    iso = sub.add_parser("isolation-test", help="prove that a headless author receives no context")
    iso.add_argument("--certification", required=True, help="the certification's id")
    iso.add_argument("--claude-bin", type=Path, required=True, help="the Claude Code binary")
    iso.add_argument("--env-extra", action="append", default=[], help="a variable proved necessary")
    sub.add_parser("figure-density", help="diagnostic: share of figure or date nodes per area")
    paf = sub.add_parser(
        "pre-authoring-freeze", help="fix the order, roles and evaluation contract"
    )
    paf.add_argument("--certification", required=True, help="the sealed certification's id")
    paf.add_argument(
        "--certification-manifest", required=True, help="its manifest's SHA-256, typed again"
    )
    brk = sub.add_parser("broker-test", help="the synthetic stream-json test of judge A")
    brk.add_argument("--certification", required=True, help="the certification's id")
    brk.add_argument("--claude-bin", type=Path, required=True, help="the Claude Code binary")
    brk.add_argument("--env-extra", action="append", default=[], help="a variable proved necessary")
    seal = sub.add_parser("certification-seal", help="seal a PASS certification, once")
    seal.add_argument("certification", help="the certification's id")
    seal.add_argument(
        "--suite-report",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a pytest JUnit report of a suite run on the certified commit (repeatable)",
    )
    ver = sub.add_parser("certification-verify", help="a certification's status and its seal")
    ver.add_argument("certification", help="the certification's id")
    ver.add_argument("--manifest", help="the manifest's SHA-256 it must have")
    desc = sub.add_parser("describe", help="the descriptive profile S4 compares, as JSON")
    desc.add_argument("benchmark", type=Path)
    sub.add_parser("overlap-statements", help="Jaccard diagnostic of the statements")
    sub.add_parser("overlap-evidence", help="grade-3 evidence shared with DEV, and the strata")
    for name, text in (
        ("start", "START a new construction of TEST with real models"),
        ("resume", "RESUME the construction of TEST from its durable state"),
        ("freeze", "freeze TEST 1.0.0 from its state, running no model"),
        ("amend", "record the amended contract's amendment in the state, running no model"),
    ):
        action = sub.add_parser(name, help=text)
        action.add_argument("--contract", required=True, help="the pinned contract's SHA-256")
    sub.add_parser("validate", help="every check of the launcher, building nothing")
    ctl = sub.add_parser(
        "controller", help="START or RESUME unattended, under the controller's policy only"
    )
    ctl.add_argument("--contract", required=True, help="the pinned contract's SHA-256")
    ctl.add_argument("--policy", required=True, help="the controller policy's id, typed again")
    first = ctl.add_mutually_exclusive_group(required=True)
    first.add_argument("--start", action="store_true", help="begin with START (no state)")
    first.add_argument("--resume", action="store_true", help="begin with RESUME (a state)")
    pub = sub.add_parser("record-public-commit", help="the pushed pre-authoring commit's SHA")
    pub.add_argument("commit")
    apub = sub.add_parser("record-amendment-commit", help="the pushed amended contract's SHA")
    apub.add_argument("commit")
    bak = sub.add_parser("backup", help="a verified and restored bundle outside the repository")
    bak.add_argument("--folder", type=Path, default=ho.BACKUP_DIR)
    sub.add_parser("state", help="the construction's durable state, verified, as counts")
    rst = sub.add_parser("restore", help="a new working copy from a bundle and its raw artifacts")
    rst.add_argument("bundle", type=Path)
    rst.add_argument("--into", type=Path, required=True, help="a directory that does not exist")
    args = parser.parse_args(argv)

    directory: Path = args.dir
    history, raw_root = hcert.CERTIFICATIONS_DIR, hcert.CERTIFICATION_RAW_DIR
    paths = hla.Paths(REPO, directory, args.raw_dir, args.backup_dir, history, raw_root)
    if args.command == "init":
        init(directory)
    elif args.command == "grid":
        grid(directory)
    elif args.command == "record":
        record(directory, args.message)
    elif args.command == "isolation":
        return 1 if isolation(directory) else 0
    elif args.command == "certification-init":
        namespace = certification_init(
            history,
            raw_root,
            args.certification,
            args.claude_bin,
            args.binary_sha256,
            args.account_evidence,
        )
        print(f"created {namespace}, history at {git(history, 'rev-parse', 'HEAD')[:12]}")
    elif args.command in ("isolation-test", "broker-test"):
        namespace, raw = certification_paths(args.certification)
        run_test = isolation_test if args.command == "isolation-test" else broker_test
        code = run_test(namespace, raw, args.claude_bin, args.env_extra)
        record(history, f"certification {args.certification}: {args.command}, exit {code}")
        return code
    elif args.command == "certification-seal":
        namespace, raw = certification_paths(args.certification)
        reports = dict(r.split("=", 1) for r in args.suite_report)
        digest = hcert.seal(namespace, raw, REPO, {n: Path(p) for n, p in reports.items()})
        print(f"sealed {args.certification}: manifest {digest}")
    elif args.command == "certification-verify":
        namespace, raw = certification_paths(args.certification)
        manifest, problems = hcert.verify(namespace, raw, REPO, args.manifest)
        found = {"status": hcert.status(namespace, raw), "sealed": manifest is not None}
        print(json.dumps(found | {"problems": problems}, indent=1))
        return 1 if manifest is None or problems else 0
    elif args.command == "figure-density":
        figure_density(directory)
    elif args.command == "pre-authoring-freeze":
        return pre_authoring_freeze(directory, args.certification, args.certification_manifest)
    elif args.command == "describe":
        print(json.dumps(describe(args.benchmark), indent=1, ensure_ascii=False))
    elif args.command == "overlap-statements":
        overlap_statements(directory)
    elif args.command == "overlap-evidence":
        overlap_evidence(directory)
    elif args.command == "start":
        hproc.install_termination()  # SIGTERM ends every child's group on the way out
        hla.start(paths, args.contract)
    elif args.command == "resume":
        hproc.install_termination()
        hla.resume(paths, args.contract)
    elif args.command == "controller":
        return controller(paths, args.contract, args.policy, "start" if args.start else "resume")
    elif args.command == "freeze":
        print(json.dumps(hla.freeze(paths, args.contract)["files"], indent=1))
    elif args.command == "validate":
        report = hla.validate(paths)
        print(json.dumps(report, indent=1))
        return 1 if report["problems"] else 0
    elif args.command == "record-public-commit":
        print(json.dumps(hla.record_public_commit(paths, args.commit), indent=1))
    elif args.command == "record-amendment-commit":
        print(json.dumps(hla.record_amendment_commit(paths, args.commit), indent=1))
    elif args.command == "amend":
        print(json.dumps(hla.amend(paths, args.contract), indent=1))
    elif args.command == "backup":
        if (directory / hs.STATE_DIR / hs.DB_FILE).exists():
            with hs.Store.open(directory, args.raw_dir) as store:
                done = hs.backup_state(store, REPO, args.folder, "manual backup")
            print(f"bundle {done['bundle']}: the state verified, backed up and restored")
        else:
            bundle = ho.backup(directory, REPO, args.folder)
            print(f"bundle {bundle}: verified and restored into a temporary clone")
    elif args.command == "state":
        with hs.Store.open(directory, args.raw_dir, readonly=True) as store:
            print(json.dumps(store.verify(), indent=1))
    elif args.command == "restore":
        print(json.dumps(hs.restore(args.bundle, args.into, args.raw_dir), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
