"""The real launcher of TEST's construction (D-055), on a copy of the frozen machinery: a public
repository holding exactly the files the pre-authoring contract freezes, pushed to a local
remote; the sealed Gate 1 certification the contract names, made, run and sealed here by the
real code with the stand-in model (tests/certification_fakes.py); and a private TEST directory
of its own, made here: the grid built again from DEV, a fresh salt, and run 6's kind of Gate 1
records as a decoy that is never read. Only the model boundary (a
stand-in executor) and the probes of the binary and of the MLX runtime are replaced. No model
runs, no TEST content exists, and nothing here reads the real private TEST directory.

The checks, the refusals and the commands that must not reach a model stop before any stage
reads the corpus: they run in a clean checkout, CI included, with the corpus stood in for
(no_corpus). The tests that build cells read it, and carry the corpus mark."""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib
import json
import re
import secrets
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_broker as hb
from ingest import holdout_certification as hcert
from ingest import holdout_contract as hc
from ingest import holdout_executor as hx
from ingest import holdout_isolation as hi
from ingest import holdout_launcher as hla
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from ingest import holdout_roles as hr
from ingest import holdout_runtime as hrt
from ingest import holdout_spool as hsp
from ingest import holdout_state as hs
from tests import certification_fakes as cf
from tests.holdout_fakes import (
    FAKE_EFFECTIVE,
    FAKE_MLX,
    PIN,
    Crash,
    FakeExecutor,
    Scenario,
    session,
)

MODEL = FAKE_EFFECTIVE["init"]["model"]
# the pinned binary of the stand-in's Gate 1, signed in to the control account its sessions
# report; it is never run
RUNTIME: dict[str, Any] = {
    "path": str(Path(tempfile.gettempdir()).resolve() / "pinned-claude"),
    "version": PIN["version"],
    "sha256": "b" * 64,
    "model": MODEL,
    "organization_sha256": hi.CONTROL_ORG_SHA256,
}
# the certification the contract names: made by the real code for that binary and model
CERTIFICATION = "gate1-fixture"
# What a TEST folder of run 6's kind holds of its own Gate 1: another binary, another model.
# Contracts name a certification, so it is never read; a fallback to it would show.
DECOY_RUNTIME: dict[str, Any] = {**RUNTIME, "sha256": "d" * 64, "model": "claude-opus-5-4"}
DECOY_GATE_1: dict[str, Any] = {
    "gate_1": "PASS",
    "run": "run-4",
    "model": DECOY_RUNTIME["model"],
    "binary": {"version": DECOY_RUNTIME["version"], "sha256": DECOY_RUNTIME["sha256"]},
    "audit": {"module_sha256": hashlib.sha256(hla.AUDIT_MODULE.read_bytes()).hexdigest()},
    "environment_allowlist": {"inherited": [*hi.ENV_INHERITED, "HOME"]},
}
WORKDIR = str(Path(tempfile.gettempdir()).resolve() / "tmp.fake")


def probe(binary: Path) -> dict[str, str]:
    """The pinned binary's identity, without running it."""
    return {"path": str(binary), "version": RUNTIME["version"], "sha256": RUNTIME["sha256"]}


def mlx(**changes: Any) -> dict[str, Any]:
    """The MLX runtime the machinery is frozen with here (a stand-in), or one changed."""
    found: dict[str, Any] = json.loads(json.dumps(FAKE_MLX))
    for dotted, value in changes.items():
        place = found
        *path, last = dotted.split("__")
        for key in path:
            place = place[key]
        place[last] = value
    return found


PROBES = hla.Probes(binary=probe, mlx=mlx)


@functools.cache
def grid() -> ho.Grid:
    """The grid the grid command writes: built from DEV and the protocol, both versioned."""
    return ho.build_grid(bm.load_benchmark(bm.DEV_DIR), ho.load_areas())


def order() -> list[str]:
    return ho.cell_order(grid())


def run_6_provenance(test: Path) -> None:
    """The decoy: run 6's kind of Gate 1 records in TEST's own provenance, and their sessions."""
    provenance = test / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    ho.write_json(provenance / "runtime.json", DECOY_RUNTIME)
    ho.write_json(provenance / "gate1.json", DECOY_GATE_1)
    ho.write_json(provenance / "broker_test.json", {"run": "run-1"})
    for kind, run in (("isolation", "run-4"), ("broker", "run-1")):
        evidence = test / "audits" / kind / run
        evidence.mkdir(parents=True)
        texts = session(["P"], ["R"])
        for name in ("stream", "transcript"):
            (evidence / f"{name}.jsonl").write_text(texts[name], encoding="utf-8")


def gate_1(paths: hla.Paths) -> Path:
    """The provenance of the certification the contract names, in this test's copy of it."""
    return paths.certifications / CERTIFICATION / hcert.PROVENANCE


@pytest.fixture(scope="module")
def machinery(tmp_path_factory):
    """The public repository as the pre-authoring commit holds it, pushed: the pinned contract,
    its construction record built again from the code for the stand-in's runtimes, and every
    file it freezes. Before it, the certified code's commit, pushed, and the certification
    made, run and sealed from it, which the contract names. Not hc.build: a shallow clone has
    no history for the code record."""
    # names as distinctive as the real ones: the project's terms forbid the repository's
    # name and its folder's in every session
    root = tmp_path_factory.mktemp("frozen-machinery-fixture")
    repo = cf.public_repo(root, name="fundamento-fixture-public")
    evidence = root / "evidence"
    evidence.mkdir()
    ho.write_json(evidence / ho.GRID_FILE, grid().model_dump(mode="json"))
    plan, roles = ho.plan_record(grid()), hr.contracts_record()
    binary = {k: RUNTIME[k] for k in ("path", "version", "sha256")}
    with pytest.MonkeyPatch.context() as patch:
        cf.stand_ins(patch, repo, binary)
        patch.setattr(cf.cli(), "HEADLESS_MODEL", MODEL)
        namespace, _, _ = cf.certified(
            patch, root, repo, CERTIFICATION, extra_env=("USER",), identity=binary
        )
    claude = hrt.claude_runtime(namespace / hcert.PROVENANCE)
    reference = hcert.reference(namespace)
    record = hc.construction(evidence / ho.GRID_FILE, plan, roles, claude, FAKE_MLX, reference)
    pinned = json.loads(hc.CONTRACT_PATH.read_text(encoding="utf-8"))
    # a base contract, as the pre-authoring commit holds it: an amendment is added by a test
    contract = {k: v for k, v in pinned.items() if k != "amends"} | {"construction": record}
    for f in hc.frozen_files(contract):
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, repo / f)
    sha = hc.write(contract, repo / hc.CONTRACT_PATH)
    (repo / ".gitignore").write_text("data/benchmark/test/\nprivate-*/\n", encoding="utf-8")
    ho.git(repo, "add", "-A")
    ho.git(repo, "commit", "-q", "-m", "the pre-authoring machinery, frozen")
    ho.git(repo, "push", "-q", "origin", "main")
    ho.git(repo, "fetch", "-q", "origin")
    grid_file = evidence / ho.GRID_FILE
    certifications = (root / "certifications", root / "certification-raw")
    return {
        "repo": repo,
        "sha": sha,
        "plan": plan,
        "roles": roles,
        "grid": grid_file,
        "certifications": certifications,
    }


@pytest.fixture
def repo(machinery):
    return machinery["repo"]


@pytest.fixture
def sha(machinery):
    return machinery["sha"]


@pytest.fixture
def paths(machinery, tmp_path):
    """A private TEST directory of its own: run 6's kind of Gate 1 records as a decoy, the
    provenance of that contract, the grid, and a fresh salt; and a copy of the certification
    the contract names, for this test alone."""
    repo = machinery["repo"]
    history, raw_root = machinery["certifications"]
    certifications, certification_raw = tmp_path / "certifications", tmp_path / "certification-raw"
    shutil.copytree(history, certifications)
    shutil.copytree(raw_root, certification_raw)
    test = repo / f"private-{tmp_path.name}"
    run_6_provenance(test)
    provenance = test / "provenance"
    pin = {"path": hc.CONTRACT_PATH.as_posix(), "sha256": machinery["sha"]}
    ho.write_json(provenance / "contract_pin.json", pin)
    ho.write_json(provenance / "plan.json", machinery["plan"])
    ho.write_json(provenance / "role_contracts.json", machinery["roles"])
    ho.write_json(provenance / "frozen_templates.json", {"files": hr.template_hashes()})
    (provenance / "opaque_salt.hex").write_text(secrets.token_hex(32) + "\n")
    shutil.copyfile(machinery["grid"], test / ho.GRID_FILE)
    for folder in ("overlap", "packets", "audits"):
        (test / folder).mkdir(exist_ok=True)
        (test / folder / ".keep").write_text("")
    ho.git(test, "init", "-q", "-b", "main")
    ho.guard_private_history(test)
    ho.git(test, "add", "-A")
    ho.git(test, "commit", "-q", "-m", "fixture: private history begins")
    raw, backups = tmp_path / "raw", tmp_path / "backups"
    return hla.Paths(repo, test, raw, backups, certifications, certification_raw)


@pytest.fixture
def no_corpus(monkeypatch):
    """A clean checkout has no corpus (D-002). These tests stop before any stage reads it: its
    identity check finds nothing to differ and it loads empty, and each test asserts where it
    stopped. The check itself is tests/integration/test_holdout_contract.py's (corpus), and the
    launcher's stop on it is the "corpus_on_disk" change below."""
    monkeypatch.setattr(hc, "corpus_problems", lambda contract: [])
    monkeypatch.setattr(bm, "load_corpus", lambda *args, **kwargs: {})


class Fakes:
    """The stand-in executor at the model boundary, and what the launcher built around it."""

    def __init__(self, monkeypatch: Any, scenario: Scenario | None = None) -> None:
        self.scenario = scenario or Scenario()
        self.executors: list[FakeExecutor] = []
        self.pipelines: list[hpl.Pipeline] = []
        real = hla._pipeline

        def attach(paths, frozen, world, store, executor):
            pipeline = real(paths, frozen, world, store, executor)
            if isinstance(executor, FakeExecutor):
                executor.pipeline = pipeline
            self.pipelines.append(pipeline)
            return pipeline

        monkeypatch.setattr(hla, "_pipeline", attach)

    def __call__(self, frozen: hla.Frozen, paths: hla.Paths) -> FakeExecutor:
        runtime = {"claude_runtime": frozen.claude_runtime, "mlx_runtime": frozen.mlx_runtime}
        executor = FakeExecutor(self.scenario, paths.raw / "fake-spool", WORKDIR, runtime)
        self.executors.append(executor)
        return executor

    @property
    def calls(self) -> list[tuple[str, str, int]]:
        return [c for e in self.executors for c in e.calls]


def started(paths: hla.Paths, sha: str, fakes: Fakes) -> None:
    hla.record_public_commit(paths, "HEAD")
    hla.start(paths, sha, fakes, PROBES)


def stopped(paths: hla.Paths, sha: str, fakes: Fakes) -> str:
    """A construction STARTed and stopped by a technical failure at its first stage, which
    reads no corpus: a durable state to resume or to refuse."""
    cell = order()[0]
    fakes.scenario.technical = {cell: "concept_author"}
    with pytest.raises(hpl.TechnicalStop):
        started(paths, sha, fakes)
    assert fakes.calls == [("concept_author", cell, 1)]
    fakes.scenario.technical = {}
    return cell


# ---------------------------------------------------------------------------
# START and RESUME
# ---------------------------------------------------------------------------


@pytest.mark.corpus
def test_start_builds_the_real_world_with_fakes_only_at_the_model_boundary(paths, sha, monkeypatch):
    cell = order()[2]
    fakes = Fakes(monkeypatch, Scenario(technical={cell: "judge_b"}))
    with pytest.raises(hpl.TechnicalStop):
        started(paths, sha, fakes)
    pipeline = fakes.pipelines[0]
    world = pipeline.world
    frozen, problems = hla.contract_problems(paths)
    assert problems == [] and frozen is not None
    assert world.order == order() and world.omission == ho.omission_cells(frozen.grid)
    assert world.salt == bytes.fromhex((paths.provenance / "opaque_salt.hex").read_text().strip())
    assert world.pin == {
        "version": RUNTIME["version"],
        "organization_sha256": RUNTIME["organization_sha256"],
    }
    assert world.identities == hpl.frozen_identities(
        sha,
        paths.repo / hc.PLAN_PATH,
        frozen.grid,
        RUNTIME,
        hla.static_terms(),
        paths.repo,
        frozen.claude_runtime,
        frozen.mlx_runtime,
    )
    assert world.runtime == {
        "claude_runtime": frozen.contract["construction"]["claude_runtime"],
        "mlx_runtime": FAKE_MLX,
    }
    with hs.Store.open(paths.test, paths.raw, readonly=True) as store:
        store.check_identities(world.identities, world.runtime)  # persisted at START
        assert store.runtime() == world.runtime
        assert store.verify()["waiting"][0]["cell_id"] == cell
    assert {t.label for t in hla.static_terms()} <= {t.label for t in world.terms}
    assert isinstance(pipeline.executor, FakeExecutor) and pipeline.backup is not None
    real = hla.real_executor(frozen, paths)  # what START builds without the stand-in
    assert isinstance(real, hx.RealExecutor) and real.spool == paths.spool
    assert real.extra_env == ["USER"] and real.model == RUNTIME["model"]
    assert real.model == frozen.claude_runtime["requested_model"]
    assert real.local_runtime == FAKE_MLX


def test_start_refuses_an_existing_state_and_resume_an_absent_one(
    paths, sha, monkeypatch, no_corpus
):
    fakes = Fakes(monkeypatch)
    hla.record_public_commit(paths, "HEAD")
    with pytest.raises(hla.LaunchRefused, match="never makes one up"):
        hla.resume(paths, sha, fakes, PROBES)
    with pytest.raises(hla.LaunchRefused, match="typed again"):
        hla.start(paths, sha[:12], fakes, PROBES)
    stopped(paths, sha, fakes)
    calls = len(fakes.calls)
    with pytest.raises(hla.LaunchRefused, match="never starts it again"):
        hla.start(paths, sha, fakes, PROBES)
    assert len(fakes.calls) == calls


def test_start_needs_the_public_commit_recorded_first(paths, sha, monkeypatch, no_corpus):
    fakes = Fakes(monkeypatch)
    with pytest.raises(hla.LaunchRefused, match="public pre-authoring commit"):
        hla.start(paths, sha, fakes, PROBES)
    assert fakes.calls == [] and not (paths.test / hs.STATE_DIR).exists()


def test_with_nothing_changed_every_check_passes(paths, sha, no_corpus):
    """The control of the mismatches below: each stops on its own change, nothing else."""
    hla.record_public_commit(paths, "HEAD")
    report = hla.validate(paths, PROBES)
    assert report["problems"] == [] and report["public_commit_problems"] == []
    assert hla.preflight(paths, "start", sha, PROBES).contract_sha256 == sha


@pytest.mark.corpus
def test_a_technical_failure_waits_for_an_explicit_resume(paths, sha, monkeypatch):
    cell = order()[0]
    fakes = Fakes(monkeypatch, Scenario(technical={cell: "judge_b"}))
    with pytest.raises(hpl.TechnicalStop):
        started(paths, sha, fakes)
    calls = list(fakes.calls)
    assert calls[-1] == ("judge_b", cell, 1)  # START returned: nothing goes on by itself
    fakes.scenario.technical = {}
    hla.resume(paths, sha, fakes, PROBES)
    resumed = fakes.calls[len(calls) :]
    assert resumed[0] == ("judge_b", cell, 1)  # the same stage, the same attempt
    assert fakes.pipelines[-1].world.identities == fakes.pipelines[0].world.identities
    with hs.Store.open(paths.test, paths.raw, readonly=True) as store:
        summary = store.verify()
        assert summary["cells"] == {"ready": 60} and summary["technical_failures"] == {
            "model_api": 1
        }


# one frozen identity changed, everything else as it was, and what the refusal names
CHANGES = {
    "runtime": "the Claude Code binary changed: ['sha256']",
    "binary_version": "the Claude Code binary changed: ['version']",
    "contract": "files differ from the contract: ['ingest/holdout_ledger.py']",
    # not "corpus": a test id is a keyword, and conftest.py skips the corpus mark by keyword
    "corpus_on_disk": "the corpus differs from the contract",
    # a sealed certification's record changed: its seal no longer verifies
    "gate_1_audit": f"{CERTIFICATION} does not verify",
    "corrupt_state": "the state does not verify",
    "claude_model": f"{CERTIFICATION} does not verify",
    "mlx_python": "python.version",
    "mlx_mlx": "packages.mlx'",
    "mlx_lm": "packages.mlx-lm",
    "mlx_tokenizers": "packages.tokenizers",
    "host_macos": "host.macos_product_version",
    "host_build": "host.macos_build_version",
    "host_model": "host.hardware_model",
}


def changed(paths: hla.Paths, change: str, monkeypatch: Any) -> hla.Probes:
    """One frozen identity changed, everything else as it was: the probes to run with."""
    if change == "runtime":
        return hla.Probes(binary=lambda b: {**probe(b), "sha256": "0" * 64}, mlx=mlx)
    if change == "binary_version":
        newer = "2.1.302 (Claude Code)"
        return hla.Probes(binary=lambda b: {**probe(b), "version": newer}, mlx=mlx)
    if change == "corpus_on_disk":  # not the corpus the contract records
        found = ["DORA: not the contract's source"]
        monkeypatch.setattr(hc, "corpus_problems", lambda contract: found)
    if change == "gate_1_audit":  # the audit's module is not the one Gate 1 passed with
        gate1 = json.loads((gate_1(paths) / "gate1.json").read_text())
        ho.write_json(gate_1(paths) / "gate1.json", gate1 | {"audit": {"module_sha256": "0"}})
    if change == "claude_model":  # same binary, code and contract files, another model
        pin = json.loads((gate_1(paths) / "runtime.json").read_text())
        ho.write_json(gate_1(paths) / "runtime.json", {**pin, "model": "claude-opus-5-4"})
    mlx_changes = {
        "mlx_python": {"python__version": "3.13.0"},
        "mlx_mlx": {"packages__mlx": "0.33.0"},
        "mlx_lm": {"packages__mlx-lm": "0.33.0"},
        "mlx_tokenizers": {"packages__tokenizers": "0.24.0"},
        "host_macos": {"host__macos_product_version": "27.1.0"},
        "host_build": {"host__macos_build_version": "26B100"},
        "host_model": {"host__hardware_model": "MacBookPro18,3"},  # same arm64
    }.get(change, {})
    return hla.Probes(binary=probe, mlx=lambda: mlx(**mlx_changes))


@pytest.mark.parametrize("change", [c for c in CHANGES if c != "corrupt_state"])
def test_a_mismatch_stops_start_before_any_model(paths, sha, monkeypatch, no_corpus, change):
    fakes = Fakes(monkeypatch)
    hla.record_public_commit(paths, "HEAD")
    frozen_file = paths.repo / "ingest" / "holdout_ledger.py"
    original = frozen_file.read_bytes()
    try:
        probes = changed(paths, change, monkeypatch)
        if change == "contract":
            frozen_file.write_bytes(original + b"\n")
        refused = (hs.StateError, hla.LaunchRefused)
        with pytest.raises(refused, match=re.escape(CHANGES[change])):
            hla.start(paths, sha, fakes, probes)
    finally:
        frozen_file.write_bytes(original)
    assert fakes.calls == [] and fakes.executors == []
    assert not (paths.test / hs.STATE_DIR).exists()


def test_the_effective_model_of_every_session_is_checked_not_only_the_binary(
    paths, sha, monkeypatch, no_corpus
):
    """Same binary, code and contract files; the runtime serves another model in a session:
    that output is never admitted, and the construction waits for an explicit RESUME."""
    cell = order()[0]
    frozen, _ = hla.contract_problems(paths)
    assert frozen is not None
    usage = frozen.claude_runtime["effective"]["model_usage"]
    model = next(iter(usage))
    other = {"model_usage": {model: {**usage[model], "canonicalModel": "claude-opus-5-4"}}}
    fakes = Fakes(monkeypatch, Scenario(model={(cell, "concept_author"): other}))
    with pytest.raises(hpl.TechnicalStop, match="runtime_identity"):
        started(paths, sha, fakes)
    assert fakes.calls == [("concept_author", cell, 1)]
    with hs.Store.open(paths.test, paths.raw, readonly=True) as store:
        assert store.stage(f"{cell}-a1", "concept") is None
        assert store.verify()["technical_failures"] == {"runtime_identity": 1}


@pytest.mark.parametrize("change", CHANGES)
def test_a_mismatch_stops_resume_before_any_model(paths, sha, monkeypatch, no_corpus, change):
    fakes = Fakes(monkeypatch)
    stopped(paths, sha, fakes)
    calls = len(fakes.calls)
    with hs.Store.open(paths.test, paths.raw, readonly=True) as store:
        events = len(store.events())
    frozen_file = paths.repo / "ingest" / "holdout_ledger.py"
    original = frozen_file.read_bytes()
    try:
        probes = changed(paths, change, monkeypatch)
        if change == "contract":
            frozen_file.write_bytes(original + b"\n")
        elif change == "corrupt_state":
            with hs.Store.open(paths.test, paths.raw) as store:
                store.db.execute("UPDATE cells SET state = 'ready' WHERE position = 1")
        executors = len(fakes.executors)
        with pytest.raises(hs.StateError, match=re.escape(CHANGES[change])):
            hla.resume(paths, sha, fakes, probes)
    finally:
        frozen_file.write_bytes(original)
    assert len(fakes.executors) == executors  # refused before an executor was even built
    assert len(fakes.calls) == calls
    with hs.Store.open(paths.test, paths.raw, readonly=True) as store:
        assert len(store.events()) == events


# ---------------------------------------------------------------------------
# the certification the contract names, and only it
# ---------------------------------------------------------------------------


def test_the_launcher_reads_the_named_certification_never_tests_own_provenance(paths, no_corpus):
    frozen, problems = hla.contract_problems(paths)
    assert problems == [] and frozen is not None
    assert frozen.runtime == json.loads((gate_1(paths) / "runtime.json").read_text())
    assert frozen.runtime["sha256"] == RUNTIME["sha256"] != DECOY_RUNTIME["sha256"]
    assert frozen.extra_env == ["USER"]  # the decoy's Gate 1 inherited HOME
    construction = frozen.contract["construction"]
    assert hla.certification_provenance(paths, construction) == gate_1(paths)
    unnamed = {k: v for k, v in construction.items() if k != "certification"}
    with pytest.raises(hla.LaunchRefused, match="names no Gate 1 certification"):
        hla.certification_provenance(paths, unnamed)
    other = construction | {"certification": construction["certification"] | {"gate1_run": "x"}}
    with pytest.raises(hla.LaunchRefused, match="not the certification the contract records"):
        hla.certification_provenance(paths, other)


def test_without_its_certification_a_contract_never_falls_back(paths, sha, monkeypatch, no_corpus):
    (paths.certifications / CERTIFICATION).rename(paths.certifications / "moved-away")
    frozen, problems = hla.contract_problems(paths)
    assert frozen is None and any(f"no certification {CERTIFICATION}" in p for p in problems)
    fakes = Fakes(monkeypatch)
    for refused in (
        lambda: hla.record_public_commit(paths, "HEAD"),
        lambda: hla.start(paths, sha, fakes, PROBES),
    ):
        with pytest.raises(hs.StateError, match=f"no certification {CERTIFICATION}"):
            refused()
    assert fakes.calls == [] and not (paths.test / hs.STATE_DIR).exists()


# ---------------------------------------------------------------------------
# no model outside START and RESUME
# ---------------------------------------------------------------------------


class ModelInvoked(BaseException):
    pass


def no_models(monkeypatch: Any) -> None:
    """Every way the code has to reach a model, made to blow up."""

    def blow(*args: Any, **kwargs: Any) -> Any:
        raise ModelInvoked

    monkeypatch.setattr(hx.RealExecutor, "__init__", blow)
    monkeypatch.setattr(hsp, "run_headless", blow)
    monkeypatch.setattr(hb.StreamJsonConversation, "__init__", blow)
    monkeypatch.setattr(hla, "real_executor", blow)
    monkeypatch.setattr(hla, "start", blow)
    monkeypatch.setattr(hla, "resume", blow)


def cli() -> ModuleType:
    return importlib.import_module("scripts.holdout")  # scripts/ is no package for mypy


def commands_without_models(paths: hla.Paths, monkeypatch: Any) -> list[str]:
    """validate, state, record-public-commit, backup and restore through the CLI, with every
    model boundary made to blow up and the stand-in probes: the arguments they ran with."""
    no_models(monkeypatch)
    monkeypatch.setattr(hla.hi, "binary_identity", probe)
    monkeypatch.setattr(hla.hrt, "mlx_runtime", mlx)
    monkeypatch.setattr(cli(), "REPO", paths.repo)
    monkeypatch.setattr(hcert, "CERTIFICATIONS_DIR", paths.certifications)
    monkeypatch.setattr(hcert, "CERTIFICATION_RAW_DIR", paths.certification_raw)
    common = ["--dir", str(paths.test), "--raw-dir", str(paths.raw)]
    common += ["--backup-dir", str(paths.backups)]
    assert cli().main([*common, "validate"]) == 0
    assert cli().main([*common, "state"]) == 0
    assert cli().main([*common, "record-public-commit", "HEAD"]) == 0
    assert cli().main([*common, "backup", "--folder", str(paths.backups)]) == 0
    bundle = sorted(paths.backups.glob("*.bundle"))[-1]
    restored_raw = paths.raw.with_name("raw-restored")
    into = paths.test.with_name(paths.test.name + "-restored")
    restore = ["--raw-dir", str(restored_raw), "restore", str(bundle), "--into", str(into)]
    assert cli().main(restore) == 0
    return common


def test_no_command_but_start_and_resume_can_reach_a_model(paths, sha, monkeypatch, no_corpus):
    """The commands that never construct, on a construction STARTed with the stand-in and
    stopped: each runs with every model boundary made to blow up, and freeze refuses an
    unfinished construction before it writes anything."""
    fakes = Fakes(monkeypatch)
    stopped(paths, sha, fakes)
    common = commands_without_models(paths, monkeypatch)
    with hs.Store.open(paths.test, paths.raw, readonly=True) as store:
        events = len(store.events())
    with pytest.raises(hs.StateError, match="has no admitted output: this replay runs none"):
        cli().main([*common, "freeze", "--contract", sha])
    with hs.Store.open(paths.test, paths.raw, readonly=True) as store:
        assert len(store.events()) == events
    assert not (paths.repo / ho.COMMITMENT_PATH).exists()
    assert fakes.calls == [("concept_author", order()[0], 1)]


@pytest.mark.corpus
def test_the_cli_freezes_a_finished_construction_without_a_model(paths, sha, monkeypatch):
    fakes = Fakes(monkeypatch)
    started(paths, sha, fakes)
    common = commands_without_models(paths, monkeypatch)
    try:
        assert cli().main([*common, "freeze", "--contract", sha]) == 0
    finally:
        (paths.repo / ho.COMMITMENT_PATH).unlink(missing_ok=True)


def test_in_the_cli_only_start_and_resume_launch_a_construction():
    """In the CLI's code only the commands start, resume, isolation-test and broker-test name a
    model boundary (the last two run synthetic packets only, never TEST)."""
    source = cli().__file__
    assert source is not None
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    boundaries = {"run_headless", "StreamJsonConversation", "RealExecutor", "start", "resume"}
    reaching = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            if names & boundaries and node.name != "main":
                reaching.add(node.name)
    assert reaching == {"isolation_test", "broker_test"}
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    launched = {
        n.attr
        for n in ast.walk(main)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "hla"
    }
    assert launched & {"start", "resume"} == {"start", "resume"}
    assert not launched & {"real_executor", "build_world", "_pipeline"}


# ---------------------------------------------------------------------------
# the freeze, the audit and the commit
# ---------------------------------------------------------------------------


@pytest.mark.corpus
def test_the_freeze_runs_no_model_and_writes_test_1_0_0(paths, sha, monkeypatch):
    fakes = Fakes(monkeypatch)
    started(paths, sha, fakes)
    no_models(monkeypatch)
    try:
        frozen_tree(paths, sha)
    finally:
        (paths.repo / ho.COMMITMENT_PATH).unlink(missing_ok=True)


def frozen_tree(paths: hla.Paths, sha: str) -> None:
    record = hla.freeze(paths, sha, PROBES)
    manifest = json.loads((paths.test / "manifest.json").read_text())
    assert (manifest["role"], manifest["benchmark_version"]) == ("test", "1.0.0")
    freeze = json.loads((paths.test / ho.FREEZE_FILE).read_text())
    assert freeze["pre_authoring"]["contract_sha256"] == sha
    commitment = json.loads((paths.repo / ho.COMMITMENT_PATH).read_text())
    assert not ho.TEST_ID.search(json.dumps(commitment))
    assert len(record["needs"]) == 60 and set(record["files"]) >= {"needs.jsonl", "freeze.json"}
    again = hla.freeze(paths, sha, PROBES)  # run again: the same files, one frozen event
    assert again["files"] == record["files"]
    with hs.Store.open(paths.test, paths.raw, readonly=True) as store:
        assert [e.kind for e in store.events()].count("frozen") == 1


@pytest.mark.corpus
def test_a_failed_audit_has_no_consumer_and_admission_needs_the_log(paths, sha, monkeypatch):
    cell = order()[0]
    scenario = Scenario(technical={cell: "judge_b"}, technical_kind="contaminate")
    fakes = Fakes(monkeypatch, scenario)
    with pytest.raises(hpl.TechnicalStop):
        started(paths, sha, fakes)
    fakes.scenario.technical = {}
    hla.resume(paths, sha, fakes, PROBES)
    with hs.Store.open(paths.test, paths.raw) as store:
        runs = store.runs(f"{cell}-a1", "judge_b")
        assert [r["status"] for r in runs] == ["failed", "admitted"]
        assert json.loads(runs[0]["audit"])["ok"] is False
        assert store.consumption_problems() == []
        failed = runs[0]["run_id"]
        assert failed not in fakes.pipelines[-1].consumed
        # a stage written around the event log, from the failed run: the state refuses it
        store.db.execute(
            "UPDATE stages SET run_id = ? WHERE run_id = ?", (failed, runs[1]["run_id"])
        )
        problems = store.consumption_problems()
        assert any("not admitted" in p for p in problems)
        with pytest.raises(hs.IntegrityError):
            store.verify()


def test_the_public_commit_is_provenance_never_frozen_content(
    paths, sha, monkeypatch, repo, no_corpus
):
    first = hla.record_public_commit(paths, "HEAD")
    identities = hla.validate(paths, PROBES)
    assert identities["problems"] == [] and identities["public_commit"] == first["commit"]
    # a later commit that touches nothing frozen changes nothing the contract checks
    (repo / "NOTES.md").write_text("a note\n")
    ho.git(repo, "add", "NOTES.md")
    ho.git(repo, "commit", "-q", "-m", "a later commit")
    try:
        assert hc.verify(sha, paths.contract, repo, check_corpus=False)["schema"] == hc.SCHEMA
        report = hla.validate(paths, PROBES)
        assert report["problems"] == [] and report["public_commit_problems"] == []
        contract = json.loads(paths.contract.read_text())
        assert hc.committed_problems(contract, sha, repo, "HEAD") == []
        assert "built_on_commit" in contract["code"] and "head" not in contract["code"]
        # a commit whose frozen content differs is never a pre-authoring timestamp
        frozen_file = repo / "ingest" / "holdout_ledger.py"
        original = frozen_file.read_bytes()
        frozen_file.write_bytes(original + b"\n")
        ho.git(repo, "commit", "-q", "-am", "a change of method")
        frozen_file.write_bytes(original)
        assert hc.committed_problems(contract, sha, repo, "HEAD") == [
            "ingest/holdout_ledger.py: in HEAD with other content"
        ]
        with pytest.raises(hla.LaunchRefused):
            hla.record_public_commit(paths, "HEAD")
    finally:
        ho.git(repo, "reset", "-q", "--hard", first["commit"])


# ---------------------------------------------------------------------------
# a documented amendment (CONSTRUCTION_AMENDMENT_1)
# ---------------------------------------------------------------------------


def amended(paths: hla.Paths, repo: Path, sha: str, cell: str) -> str:
    """The base contract with an amendment of the point the log is at, committed, pushed and
    pinned: its SHA-256."""
    with hs.Store.open(paths.test, paths.raw, readonly=True) as store:
        summary = store.verify()
    public = json.loads((paths.provenance / hla.PUBLIC_COMMIT_FILE).read_text())
    note = repo / "amendment-fixture.json"
    note.write_text('{"id": "CONSTRUCTION_AMENDMENT_1"}\n', encoding="utf-8")
    amends = {
        "amendment": "CONSTRUCTION_AMENDMENT_1",
        "amendment_file": note.name,
        "amendment_sha256": hashlib.sha256(note.read_bytes()).hexdigest(),
        "base_contract_sha256": sha,
        "base_public_commit": public["commit"],
        "after_event": {"seq": summary["events"], "sha256": summary["head_sha256"]},
        "identity_changes": ["pre_authoring_contract"],  # the same code here: the contract only
        "first_transition": {
            "refused_write": "attempt_closed",
            "cell_id": cell,
            "proposal_id": f"{cell}-a1",
            "outcome": "concept_invalid",
        },
    }
    contract = json.loads(paths.contract.read_text()) | {"amends": amends}
    paths.contract.unlink()
    new = hc.write(contract, paths.contract)
    ho.git(repo, "add", "-A")
    ho.git(repo, "commit", "-q", "-m", "the amended contract")
    ho.git(repo, "push", "-q", "--force", "origin", "HEAD:main")  # the fixture's own remote
    ho.git(repo, "fetch", "-q", "origin")
    pin = {"path": hc.CONTRACT_PATH.as_posix(), "sha256": new}
    ho.write_json(paths.provenance / "contract_pin.json", pin)
    ho.git(paths.test, "add", "-A")
    ho.git(paths.test, "commit", "-q", "-m", "pin the amended contract")
    return new


@pytest.mark.corpus
def test_an_amendment_continues_the_same_construction_from_the_point_it_names(
    paths, sha, repo, monkeypatch
):
    cell = order()[2]
    guard = hpl._packet_guard

    def base(writer: Any, words: str, facts: list[str]) -> str:
        # the base code: the guard's PacketError had no transition and ended the process
        try:
            return str(writer(words, facts))
        except hp.PacketError as error:
            raise Crash("no transition") from error

    monkeypatch.setattr(hpl, "_packet_guard", base)
    fakes = Fakes(monkeypatch, Scenario(copied_concept={(cell, 1)}))
    with pytest.raises(Crash):
        started(paths, sha, fakes)
    monkeypatch.setattr(hpl, "_packet_guard", guard)  # the amended code
    with hs.Store.open(paths.test, paths.raw, readonly=True) as store:
        before = store.events()
        ready_before = {r["cell_id"]: r["record_sha256"] for r in store.ready()}
    assert len(ready_before) == 2  # the two cells before it are ready
    first = ho.git(repo, "rev-parse", "HEAD")
    try:
        new = amended(paths, repo, sha, cell)
        with pytest.raises(hla.LaunchRefused, match="not recorded"):
            hla.amend(paths, new, PROBES)  # the amendment's public commit first
        hla.record_amendment_commit(paths, "HEAD")
        assert hla.validate(paths, PROBES)["public_commit_problems"] == []
        record = hla.amend(paths, new, PROBES)
        assert record["changed_identities"] == ["pre_authoring_contract"]
        assert record["first_transition"]["proposal_id"] == f"{cell}-a1"
        assert hla.amend(paths, new, PROBES)["seq"] == len(before) + 1  # recorded once
        calls = len(fakes.calls)
        hla.resume(paths, new, fakes, PROBES)
        resumed = fakes.calls[calls:]
        with hs.Store.open(paths.test, paths.raw, readonly=True) as store:
            events = store.events()
            ready = {r["cell_id"]: r["record_sha256"] for r in store.ready()}
            attempts = [
                (a["number"], a["outcome"]) for a in store.attempts() if a["cell_id"] == cell
            ]
            summary = store.verify()
        # the log: unchanged up to the point the amendment names, then the amendment
        assert [e.sha256 for e in events[: len(before)]] == [e.sha256 for e in before]
        (event,) = [e for e in events if e.kind == "construction_amended"]
        assert event.seq == len(before) + 1 and event.payload["after_event"]["seq"] == len(before)
        # the needs ready before it: untouched, never built again
        assert {c: ready[c] for c in ready_before} == ready_before
        assert not [c for _, c, _ in resumed if c in ready_before]
        # the cell it was stopped at: its concept rejected, a new blind attempt
        assert attempts == [(1, "concept_invalid"), (2, "accepted")]
        assert summary["cells"] == {"ready": 60} and len(summary["amendments"]) == 1
        no_models(monkeypatch)
        frozen = hla.freeze(paths, new, PROBES)
        assert len(frozen["needs"]) == 60
        pre = json.loads((paths.test / ho.FREEZE_FILE).read_text())["pre_authoring"]
        assert (pre["contract_sha256"], pre["amended_contract_sha256"]) == (sha, new)
        assert [a["amendment"] for a in pre["amendments"]] == ["CONSTRUCTION_AMENDMENT_1"]
    finally:
        (paths.repo / ho.COMMITMENT_PATH).unlink(missing_ok=True)
        ho.git(repo, "reset", "-q", "--hard", first)


def test_an_amendment_is_refused_away_from_the_point_it_names(
    paths, sha, repo, monkeypatch, no_corpus
):
    fakes = Fakes(monkeypatch)
    stopped(paths, sha, fakes)
    first = ho.git(repo, "rev-parse", "HEAD")
    try:
        new = amended(paths, repo, sha, order()[0])
        hla.record_amendment_commit(paths, "HEAD")
        with hs.Store.open(paths.test, paths.raw) as store:
            store.record("resumed", None, {"status": "pending_technical", "waiting": []})
        with pytest.raises(hla.LaunchRefused, match="not at the point"):
            hla.amend(paths, new, PROBES)
        with hs.Store.open(paths.test, paths.raw, readonly=True) as store:
            assert store.verify()["amendments"] == []
    finally:
        ho.git(repo, "reset", "-q", "--hard", first)
