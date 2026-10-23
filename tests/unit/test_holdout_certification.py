"""Gate 1 certifications: a namespace of their own, made from a clean pushed commit; run numbers
reserved in the raw folder they are written to; evidence never overwritten; one final run of
each test; a seal that any later change breaks; and a contract that relies only on the sealed
certification it names. Every model boundary is replaced, and the binary's, host's and
substrate's identities are stand-ins: nothing here runs Claude Code."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_broker as hb
from ingest import holdout_certification as hcert
from ingest import holdout_contract as hc
from ingest import holdout_controller as hct
from ingest import holdout_isolation as hi
from ingest import holdout_runtime as hrt
from ingest import holdout_state as hs
from tests import certification_fakes as cf
from tests.holdout_fakes import FAKE_EFFECTIVE, FAKE_MLX

BINARY, SHA, CERTIFICATION = cf.BINARY, cf.SHA, cf.CERTIFICATION


@pytest.fixture
def repo(tmp_path):
    return cf.public_repo(tmp_path)


@pytest.fixture
def cli(monkeypatch, repo):
    cf.stand_ins(monkeypatch, repo)
    return cf.cli()


@pytest.fixture
def roots(tmp_path):
    """The history of certifications, the raw folder beside it, and the account evidence."""
    history, raw_root = tmp_path / "private" / "certifications", tmp_path / "private" / "raw"
    return history, raw_root, cf.account_evidence(tmp_path)


def make(cli, roots, certification=CERTIFICATION):
    history, raw_root, evidence = roots
    namespace = cli.certification_init(history, raw_root, certification, BINARY, SHA, evidence)
    return namespace, raw_root / certification


def files(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def git_log(history: Path) -> list[str]:
    done = subprocess.run(
        ["git", "-C", str(history), "log", "--format=%s"], capture_output=True, text=True
    )
    return done.stdout.splitlines()


def manifest_of(namespace: Path) -> dict:
    found: dict = json.loads((namespace / hcert.MANIFEST).read_text())
    return found


# ---------------------------------------------------------------------------
# the namespace, its runtime pin, its code and its substrate
# ---------------------------------------------------------------------------


def test_init_makes_a_private_history_and_a_pin_read_from_its_sources(cli, roots, repo):
    history, raw_root, evidence = roots
    namespace, raw = make(cli, roots)
    pin = json.loads((namespace / "provenance" / "runtime.json").read_text())
    assert {k: pin[k] for k in cf.IDENTITY} == cf.IDENTITY
    assert pin["model"] == cli.HEADLESS_MODEL
    assert pin["organization_sha256"] == hi.CONTROL_ORG_SHA256
    source = pin["identity_source"]
    assert source["sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert source["logical"] == str(evidence.resolve()) and source["attachment"] == "credential_org"
    assert pin["host"] == cf.HOST and pin["certification"] == CERTIFICATION
    record = json.loads((namespace / "provenance" / "certification.json").read_text())
    assert record["substrate"] == cf.SUBSTRATE and record["schema"] == hcert.SCHEMA
    assert (history / ".git" / "hooks" / "pre-push").read_text() == ho.NO_PUSH_HOOK
    assert git_log(history) == [
        f"certification {CERTIFICATION}: namespace and runtime pin",
        "certifications: private history begins",
    ]
    assert not ho.git(history, "status", "--porcelain")
    for kind in hcert.KINDS:
        assert (raw / kind).stat().st_mode & 0o777 == 0o700
        assert not list((raw / kind).iterdir())


def test_init_refuses_a_binary_other_than_the_one_typed_and_makes_nothing(cli, roots):
    history, raw_root, evidence = roots
    with pytest.raises(ho.HoldoutError, match="not the SHA-256 typed"):
        cli.certification_init(history, raw_root, CERTIFICATION, BINARY, "2" * 64, evidence)
    assert not history.exists() and not raw_root.exists()


@pytest.mark.parametrize(
    "change, refusal",
    [
        ({"vscode": {"update_mode": "manual"}}, "update.mode is 'manual'"),
        ({"extension": {"pinned": False}}, "is not pinned"),
        ({"macos_automatic_installs": {"CriticalUpdateInstall": True}}, "CriticalUpdateInstall"),
        ({"macos_automatic_installs": {"ConfigDataInstall": None}}, "ConfigDataInstall"),
    ],
    ids=["vscode-updates", "unpinned", "security-responses", "unknown"],
)
def test_init_refuses_a_substrate_free_to_change(cli, roots, monkeypatch, change, refusal):
    history, raw_root, evidence = roots
    substrate = json.loads(json.dumps(cf.SUBSTRATE))
    for part, values in change.items():
        substrate[part] |= values
    monkeypatch.setattr(hcert, "execution_substrate", lambda binary: substrate)
    with pytest.raises(ho.HoldoutError, match=refusal):
        cli.certification_init(history, raw_root, CERTIFICATION, BINARY, SHA, evidence)
    assert not history.exists() and not raw_root.exists()


def test_the_substrate_is_read_where_vscode_and_macos_keep_it(tmp_path, monkeypatch):
    extensions = tmp_path / "extensions"
    folder = extensions / "anthropic.claude-code-2.1.301-darwin-arm64"
    binary = folder / "resources" / "native-binary" / "claude"
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    entries = [
        {"identifier": {"id": "other.extension"}, "version": "1.0.0", "relativeLocation": "o"},
        {
            "identifier": {"id": "anthropic.claude-code"},
            "version": "2.1.301",
            "relativeLocation": folder.name,
            "metadata": {"pinned": True},
        },
    ]
    (extensions / "extensions.json").write_text(json.dumps(entries))
    product, settings = tmp_path / "product.json", tmp_path / "settings.json"
    product.write_text(json.dumps({"version": "1.143.1", "commit": "e" * 40}))
    settings.write_text(json.dumps({"update.mode": "none"}))
    monkeypatch.setattr(hcert, "VSCODE_PRODUCT", product)
    monkeypatch.setattr(hcert, "VSCODE_SETTINGS", settings)
    monkeypatch.setattr(hcert, "_preference", lambda domain, key: False)
    found = hcert.execution_substrate(binary)
    assert found == cf.SUBSTRATE and hcert.substrate_problems(found) == []
    monkeypatch.setattr(hcert, "_preference", lambda domain, key: key != "ConfigDataInstall")
    assert (
        "AutomaticallyInstallMacOSUpdates"
        in hcert.substrate_problems(hcert.execution_substrate(binary))[0]
    )
    settings.write_text('{"update.mode": "none", // a comment VS Code allows\n}')
    with pytest.raises(hcert.CertificationError, match="cannot be read"):
        hcert.execution_substrate(binary)


@pytest.mark.parametrize(
    "orgs", [[], [hi.CONTROL_ORG, "11111111-1111-4111-8111-111111111111"]], ids=["none", "two"]
)
def test_the_account_evidence_must_name_exactly_one_organization(cli, tmp_path, orgs):
    lines = [{"attachment": {"type": "credential_org", "organizationUuid": o}} for o in orgs]
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(ho.HoldoutError, match=f"holds {len(orgs)} organizations"):
        cli.account_organization(evidence)


def test_a_run_without_a_pin_stops_before_any_number_is_taken(cli, tmp_path):
    namespace, raw = tmp_path / "ns", tmp_path / "raw"
    (namespace / "provenance").mkdir(parents=True)
    (raw / "isolation").mkdir(parents=True)
    assert cli.isolation_test(namespace, raw, BINARY, []) == 2
    assert not list((raw / "isolation").iterdir())


# ---------------------------------------------------------------------------
# the run number, reserved in the raw folder it is written to (A-F of the raw id)
# ---------------------------------------------------------------------------


def test_a_the_number_comes_from_the_raw_folder_never_from_the_audits(cli, roots, monkeypatch):
    namespace, raw = make(cli, roots)
    for n in (1, 2, 3):
        (raw / "isolation" / f"run-{n}").mkdir()  # runs that died: no complete.json
    (namespace / "audits" / "isolation" / "run-9").mkdir()  # an audit folder decides nothing
    cf.fake_headless(monkeypatch)
    assert cli.isolation_test(namespace, raw, BINARY, []) == 0
    assert json.loads((namespace / "provenance" / "gate1.json").read_text())["run"] == "run-4"
    assert (raw / "isolation" / "run-4" / "complete.json").exists()
    assert (namespace / "audits" / "isolation" / "run-4" / "report.json").exists()


def test_b_an_empty_raw_folder_gives_run_1(cli, tmp_path):
    tmp_path = tmp_path / "raw-root"  # a raw folder of its own
    tmp_path.mkdir()
    assert cli.next_run(tmp_path) == "run-1"
    run = cli.reserve_run(tmp_path)
    assert run == tmp_path / "run-1"
    assert json.loads((run / "reserved.json").read_text())["run"] == "run-1"


def test_c_the_next_number_is_one_above_the_highest_any_name_holds(cli, tmp_path):
    tmp_path = tmp_path / "raw-root"  # a raw folder of its own
    tmp_path.mkdir()
    for name in ("run-1", "run-2", "run-5", "run-x", "run-0", "run-05"):
        (tmp_path / name).mkdir()
    (tmp_path / "run-7").write_text("a file holds its number too")
    (tmp_path / "notes.txt").write_text("")
    assert cli.next_run(tmp_path) == "run-8"


def test_d_an_existing_run_is_never_overwritten(cli, roots, monkeypatch):
    namespace, raw = make(cli, roots)
    old = raw / "isolation" / "run-1"
    old.mkdir()
    (old / "transcript.jsonl").write_text("the evidence of an earlier run")
    before = files(old)
    cf.fake_headless(monkeypatch)
    assert cli.isolation_test(namespace, raw, BINARY, []) == 0
    assert files(old) == before
    assert (raw / "isolation" / "run-2" / "complete.json").exists()


def test_e_the_exclusive_mkdir_is_the_reservation(cli, tmp_path, monkeypatch):
    tmp_path = tmp_path / "raw-root"  # a raw folder of its own
    tmp_path.mkdir()
    taken = tmp_path / "run-1"
    taken.mkdir()
    (taken / "transcript.jsonl").write_text("taken")
    monkeypatch.setattr(cli, "next_run", lambda root: "run-1")  # a listing already stale
    with pytest.raises(ho.HoldoutError, match="another run took it"):
        cli.reserve_run(tmp_path)
    assert files(taken) == {"transcript.jsonl": hashlib.sha256(b"taken").hexdigest()}


def test_f_concurrent_reservations_of_one_number_cannot_both_win(cli, tmp_path, monkeypatch):
    tmp_path = tmp_path / "raw-root"  # a raw folder of its own
    tmp_path.mkdir()
    workers = 8
    barrier = threading.Barrier(workers, timeout=30)
    listed = cli.next_run

    def racing(root):
        name = listed(root)
        barrier.wait()  # every worker holds the same number before any of them creates it
        return name

    monkeypatch.setattr(cli, "next_run", racing)
    won: list[Path] = []
    lost: list[ho.HoldoutError] = []

    def attempt():
        try:
            won.append(cli.reserve_run(tmp_path))
        except ho.HoldoutError as error:
            lost.append(error)

    threads = [threading.Thread(target=attempt) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert won == [tmp_path / "run-1"] and len(lost) == workers - 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run-1"]
    assert sorted(p.name for p in (tmp_path / "run-1").iterdir()) == ["reserved.json"]


# ---------------------------------------------------------------------------
# G-J of the raw id: new files only; interrupted runs; the two kinds apart
# ---------------------------------------------------------------------------


def test_g_an_existing_raw_file_is_never_truncated(cli, tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    out.mkdir()
    (raw / "transcript.jsonl").write_text("earlier evidence")
    with pytest.raises(FileExistsError):
        cli.keep_evidence(out, raw, {"transcript.jsonl": "new"}, [])
    assert (raw / "transcript.jsonl").read_text() == "earlier evidence"
    (out / "copy.txt").write_text("earlier copy")
    with pytest.raises(FileExistsError):
        hcert.write_new(out / "copy.txt", "new")
    assert (out / "copy.txt").read_text() == "earlier copy"


def test_h_i_an_interrupted_run_keeps_its_number_and_the_next_takes_another(
    cli, roots, monkeypatch
):
    namespace, raw = make(cli, roots)

    def killed(*args):
        raise KeyboardInterrupt  # the process dies while the model runs

    monkeypatch.setattr(hi, "run_headless", killed)
    with pytest.raises(KeyboardInterrupt):
        cli.isolation_test(namespace, raw, BINARY, [])
    died = raw / "isolation" / "run-1"
    assert sorted(p.name for p in died.iterdir()) == ["reserved.json"]  # no complete.json
    assert hcert.runs(raw, "isolation") == {"run-1": "incomplete"}
    assert hcert.status(namespace, raw) == "OPEN"  # a run that died is no final run
    before = files(died)
    cf.fake_headless(monkeypatch)
    assert cli.isolation_test(namespace, raw, BINARY, []) == 0
    assert files(died) == before
    assert json.loads((namespace / "provenance" / "gate1.json").read_text())["run"] == "run-2"
    complete = json.loads((raw / "isolation" / "run-2" / "complete.json").read_text())
    assert complete["verdict"] == "PASS" and set(complete["raw_evidence_sha256"]) == {
        "transcript.jsonl",
        "stream.jsonl",
        "stderr.txt",
        "launcher-context.json",
    }


def test_j_isolation_and_broker_number_their_runs_apart(cli, roots, monkeypatch):
    namespace, raw = make(cli, roots)
    for n in (1, 2, 3):
        (raw / "isolation" / f"run-{n}").mkdir()
    cf.fake_headless(monkeypatch)
    cf.fake_broker(monkeypatch)
    assert cli.isolation_test(namespace, raw, BINARY, []) == 0
    assert cli.broker_test(namespace, raw, BINARY, []) == 0
    assert sorted(p.name for p in (raw / "broker").iterdir()) == ["run-1"]
    assert sorted(p.name for p in (raw / "isolation").iterdir()) == [
        "run-1",
        "run-2",
        "run-3",
        "run-4",
    ]
    broker = json.loads((namespace / "provenance" / "broker_test.json").read_text())
    assert broker["run"] == "run-1" and hcert.status(namespace, raw) == "PASS"


# ---------------------------------------------------------------------------
# the lifecycle: one final run of each test; a FAIL is final
# ---------------------------------------------------------------------------


def test_a_second_gate1_in_one_certification_stops_before_any_run(cli, roots, monkeypatch):
    namespace, raw = make(cli, roots)
    cf.fake_headless(monkeypatch)
    assert cli.isolation_test(namespace, raw, BINARY, []) == 0
    assert cli.isolation_test(namespace, raw, BINARY, []) == 2
    assert sorted(p.name for p in (raw / "isolation").iterdir()) == ["run-1"]


def test_the_broker_test_runs_only_after_gate1_passed(cli, roots, monkeypatch):
    namespace, raw = make(cli, roots)
    cf.fake_broker(monkeypatch)
    assert cli.broker_test(namespace, raw, BINARY, []) == 2
    assert not list((raw / "broker").iterdir())


def test_a_failed_gate1_fails_the_certification_for_good(cli, roots, monkeypatch):
    namespace, raw = make(cli, roots)
    cf.fake_headless(monkeypatch, ok=False)
    assert cli.isolation_test(namespace, raw, BINARY, []) == 1
    assert hcert.status(namespace, raw) == "FAILED"
    assert not (namespace / "provenance" / "gate1.json").exists()
    cf.fake_headless(monkeypatch)
    cf.fake_broker(monkeypatch)
    assert cli.isolation_test(namespace, raw, BINARY, []) == 2
    assert cli.broker_test(namespace, raw, BINARY, []) == 2
    assert hcert.runs(raw, "isolation") == {"run-1": "FAIL"} and not hcert.runs(raw, "broker")


def test_a_failed_broker_test_fails_the_certification_for_good(cli, roots, monkeypatch):
    namespace, raw = make(cli, roots)
    cf.fake_headless(monkeypatch)
    cf.fake_broker(monkeypatch)
    assert cli.isolation_test(namespace, raw, BINARY, []) == 0
    with monkeypatch.context() as patch:
        patch.setattr(hb, "foreign_user_messages", lambda *args: ["not the broker's"])
        assert cli.broker_test(namespace, raw, BINARY, []) == 1
    assert hcert.status(namespace, raw) == "FAILED"
    assert cli.broker_test(namespace, raw, BINARY, []) == 2
    assert hcert.runs(raw, "broker") == {"run-1": "FAIL"}


# ---------------------------------------------------------------------------
# K-L of the raw id: nothing outside the certification; an id is used once
# ---------------------------------------------------------------------------


def test_k_a_certification_writes_only_inside_its_namespace(cli, roots, tmp_path, monkeypatch):
    """Stand-ins for run 6's TEST folder, its raw artifacts, its backups and logs, and Gate 1's
    historical transcripts, and the public repository: none of them changes, and every new file
    is the certification's."""
    historical = tmp_path / "historical"
    for name in ("transcripts", "holdout-raw", "holdout-backups", "holdout-logs", "test"):
        run = historical / name / "isolation" / "run-2"
        run.mkdir(parents=True)
        (run / "transcript.jsonl").write_text(f"{name} evidence")
    monkeypatch.setattr(cli, "RAW_DIR", historical / "transcripts")
    monkeypatch.setattr(hs, "RAW_DIR", historical / "holdout-raw")
    monkeypatch.setattr(ho, "BACKUP_DIR", historical / "holdout-backups")
    monkeypatch.setattr(hct, "LOG_DIR", historical / "holdout-logs")
    before = files(tmp_path)
    namespace, raw = make(cli, roots)
    cf.fake_headless(monkeypatch)
    cf.fake_broker(monkeypatch)
    assert cli.isolation_test(namespace, raw, BINARY, []) == 0
    assert cli.broker_test(namespace, raw, BINARY, []) == 0
    after = files(tmp_path)
    assert {k: v for k, v in after.items() if k in before} == before
    history, raw_root, _ = roots
    owned = (history.relative_to(tmp_path).as_posix(), raw_root.relative_to(tmp_path).as_posix())
    assert all(k.startswith(owned) for k in set(after) - set(before))


@pytest.mark.parametrize(
    "where",
    ["repository", "test", "historical transcripts", "around holdout-raw", "raw in history"],
)
def test_k_a_certification_never_lives_in_or_around_historical_evidence(
    cli, roots, tmp_path, monkeypatch, where
):
    history, raw_root, evidence = roots
    monkeypatch.setattr(cli, "RAW_DIR", tmp_path / "historical" / "transcripts")
    monkeypatch.setattr(hs, "RAW_DIR", tmp_path / "historical" / "holdout-raw")
    history, raw_root = {
        "repository": (cli.REPO / "certifications", raw_root),
        "test": (cli.REPO / ho.HOLDOUT_DIR / "certifications", raw_root),
        "historical transcripts": (history, cli.RAW_DIR / "certifications"),
        "around holdout-raw": (tmp_path / "historical", raw_root),
        "raw in history": (history, history / "raw"),
    }[where]
    existed = [p for p in (history, raw_root) if p.exists()]
    with pytest.raises(ho.HoldoutError, match="overlaps"):
        cli.certification_init(history, raw_root, CERTIFICATION, BINARY, SHA, evidence)
    assert [p for p in (history, raw_root) if p.exists()] == existed


def test_l_an_existing_certification_is_a_stop_and_never_reused(cli, roots):
    history, raw_root, evidence = roots
    make(cli, roots)
    before, log = files(history.parent), git_log(history)
    with pytest.raises(ho.HoldoutError, match="used once"):
        make(cli, roots)
    assert files(history.parent) == before and git_log(history) == log


def test_l_a_raw_folder_left_by_another_is_a_stop_and_kept(cli, roots):
    history, raw_root, evidence = roots
    left = raw_root / CERTIFICATION / "isolation" / "run-1"
    left.mkdir(parents=True)
    (left / "transcript.jsonl").write_text("someone else's evidence")
    with pytest.raises(ho.HoldoutError, match="used once"):
        make(cli, roots)
    assert (left / "transcript.jsonl").read_text() == "someone else's evidence"


# ---------------------------------------------------------------------------
# A-F: the code a certification is made from, and its seal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dirt", ["modified", "staged", "untracked"])
def test_code_a_a_dirty_working_tree_refuses_certification_init(cli, roots, repo, dirt):
    history, raw_root, evidence = roots
    module = repo / hcert.CERTIFIED_CODE[1]
    if dirt == "untracked":
        (repo / "ingest" / "new_module.py").write_text("x = 1\n")
    else:
        module.write_text(module.read_text() + "\n# changed\n")
        if dirt == "staged":
            ho.git(repo, "add", "--", hcert.CERTIFIED_CODE[1])
    with pytest.raises(ho.HoldoutError, match="working tree is not clean"):
        cli.certification_init(history, raw_root, CERTIFICATION, BINARY, SHA, evidence)
    assert not history.exists() and not raw_root.exists()


def test_code_b_a_clean_pushed_tree_records_its_exact_commit(cli, roots, repo):
    namespace, _ = make(cli, roots)
    code = json.loads((namespace / "provenance" / "certification.json").read_text())["code"]
    assert code["commit"] == ho.git(repo, "rev-parse", "HEAD")
    assert code["remote_branches"] == ["origin/main"]
    assert code["repository_roots"] == [ho.git(repo, "rev-list", "--max-parents=0", "HEAD")]
    assert code["certified_code_sha256"] == {
        f: hashlib.sha256((repo / f).read_bytes()).hexdigest() for f in hcert.CERTIFIED_CODE
    }
    assert set(code["implementation_sha256"]) == set(hcert.IMPLEMENTATION)


def test_code_b_a_commit_no_remote_holds_refuses_certification_init(cli, roots, repo):
    history, raw_root, evidence = roots
    (repo / "notes.txt").write_text("not pushed\n")
    ho.git(repo, "add", "-A")
    ho.git(repo, "commit", "-q", "-m", "local only")
    with pytest.raises(ho.HoldoutError, match="on no remote branch"):
        cli.certification_init(history, raw_root, CERTIFICATION, BINARY, SHA, evidence)


def test_code_c_no_manifest_exists_before_a_pass(cli, roots, repo, tmp_path, monkeypatch):
    namespace, raw = make(cli, roots)
    report = {"full": cf.junit(tmp_path / "full.xml")}
    with pytest.raises(hcert.CertificationError, match="is OPEN"):
        hcert.seal(namespace, raw, repo, report)
    cf.fake_headless(monkeypatch, ok=False)
    assert cli.isolation_test(namespace, raw, BINARY, []) == 1
    with pytest.raises(hcert.CertificationError, match="is FAILED"):
        hcert.seal(namespace, raw, repo, report)
    assert not (namespace / hcert.MANIFEST).exists()


def test_code_c_a_pass_with_gate1_only_has_no_manifest(cli, roots, repo, tmp_path, monkeypatch):
    namespace, raw = make(cli, roots)
    cf.fake_headless(monkeypatch)
    assert cli.isolation_test(namespace, raw, BINARY, []) == 0
    with pytest.raises(hcert.CertificationError, match="is OPEN"):
        hcert.seal(namespace, raw, repo, {"full": cf.junit(tmp_path / "full.xml")})
    assert not (namespace / hcert.MANIFEST).exists()


def test_code_d_the_manifest_seals_the_exact_public_commit(tmp_path, repo, monkeypatch):
    cf.stand_ins(monkeypatch, repo)
    namespace, raw, digest = cf.certified(monkeypatch, tmp_path, repo)
    manifest = manifest_of(namespace)
    assert manifest["code"]["commit"] == ho.git(repo, "rev-parse", "HEAD")
    assert manifest["status"] == "PASS" and manifest["certification"] == CERTIFICATION
    assert digest == hashlib.sha256((namespace / hcert.MANIFEST).read_bytes()).hexdigest()
    assert manifest["files_sha256"] == hcert.tree(namespace, (hcert.MANIFEST,))
    assert manifest["raw_sha256"] == hcert.tree(raw)
    assert manifest["runs"] == {"isolation": {"run-1": "PASS"}, "broker": {"run-1": "PASS"}}
    assert manifest["runtime"]["binary_sha256"] == SHA
    assert manifest["runtime"]["organization_sha256"] == hi.CONTROL_ORG_SHA256
    assert manifest["substrate"] == cf.SUBSTRATE
    assert (
        manifest["gate1"]["audit"]["module_sha256"]
        == manifest["code"]["certified_code_sha256"]["ingest/holdout_isolation.py"]
    )
    assert manifest["suites"]["full"]["tests"] == 3
    assert "provenance/suites/full.xml" in manifest["files_sha256"]
    assert hcert.verify(namespace, raw, repo, digest) == (manifest, [])
    assert git_log(namespace.parent)[0] == f"certification {CERTIFICATION}: sealed"
    with pytest.raises(hcert.CertificationError, match="sealed already"):
        hcert.seal(namespace, raw, repo, {"full": cf.junit(tmp_path / "again.xml")})


def test_code_d_the_seal_refuses_code_that_moved_since_init(tmp_path, repo, monkeypatch):
    cf.stand_ins(monkeypatch, repo)
    namespace, raw, _ = cf.certified(monkeypatch, tmp_path, repo, seal=False)
    (repo / "notes.txt").write_text("a later change\n")
    ho.git(repo, "add", "-A")
    ho.git(repo, "commit", "-q", "-m", "after certification-init")
    with pytest.raises(hcert.CertificationError, match="not the clean commit"):
        hcert.seal(namespace, raw, repo, {"full": cf.junit(tmp_path / "full.xml")})
    assert not (namespace / hcert.MANIFEST).exists()


@pytest.mark.parametrize(
    "change, refusal",
    [
        ("failures", "did not pass"),
        ("errors", "did not pass"),
        ("binary", "not the pinned one"),
        ("substrate", "substrate changed"),
        ("evidence", "account evidence changed"),
    ],
)
def test_code_d_the_seal_refuses_a_failed_suite_or_a_changed_runtime(
    tmp_path, repo, monkeypatch, change, refusal
):
    cf.stand_ins(monkeypatch, repo)
    namespace, raw, _ = cf.certified(monkeypatch, tmp_path, repo, seal=False)
    counts = {change: 1} if change in ("failures", "errors") else {}
    report = cf.junit(tmp_path / "full.xml", **counts)
    if change == "binary":
        monkeypatch.setattr(hi, "binary_identity", lambda b: {**cf.IDENTITY, "sha256": "9" * 64})
    if change == "substrate":
        newer = json.loads(json.dumps(cf.SUBSTRATE))
        newer["vscode"]["version"] = "1.144.0"
        monkeypatch.setattr(hcert, "execution_substrate", lambda binary: newer)
    if change == "evidence":
        (tmp_path / "evidence" / "transcript.jsonl").write_text("another login\n")
    with pytest.raises(hcert.CertificationError, match=refusal):
        hcert.seal(namespace, raw, repo, {"full": report})
    assert not (namespace / hcert.MANIFEST).exists()


@pytest.mark.parametrize("mutation", ["raw file", "audit report", "added file", "runtime pin"])
def test_code_e_verification_detects_evidence_mutated_after_the_seal(
    tmp_path, repo, monkeypatch, mutation
):
    cf.stand_ins(monkeypatch, repo)
    namespace, raw, digest = cf.certified(monkeypatch, tmp_path, repo)
    target = {
        "raw file": raw / "isolation" / "run-1" / "transcript.jsonl",
        "audit report": namespace / "audits" / "broker" / "run-1" / "report.json",
        "added file": namespace / "provenance" / "notes.txt",
        "runtime pin": namespace / "provenance" / "runtime.json",
    }[mutation]
    with target.open("a") as file:
        file.write("\n")
    manifest, problems = hcert.verify(namespace, raw, repo, digest)
    assert manifest is not None and any("changed" in p for p in problems)


def test_code_f_verification_detects_a_code_commit_mismatch(tmp_path, repo, monkeypatch):
    cf.stand_ins(monkeypatch, repo)
    namespace, raw, _ = cf.certified(monkeypatch, tmp_path, repo)
    module = repo / hcert.CERTIFIED_CODE[1]
    module.write_text(module.read_text() + "\n# another transport\n")
    ho.git(repo, "add", "-A")
    ho.git(repo, "commit", "-q", "-m", "a later commit")
    manifest = manifest_of(namespace)
    manifest["code"]["commit"] = ho.git(repo, "rev-parse", "HEAD")  # claims the later commit
    (namespace / hcert.MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    _, problems = hcert.verify(namespace, raw, repo)
    assert f"the manifest's code is not its commit's: {hcert.CERTIFIED_CODE[1]}" in problems
    other = cf.public_repo(tmp_path / "elsewhere")  # the same files, another history
    _, problems = hcert.verify(namespace, raw, other)
    assert any(p.startswith("the manifest's code is not its commit's") for p in problems)


# ---------------------------------------------------------------------------
# G-M: the contract relies on the sealed certification it names, and only on it
# ---------------------------------------------------------------------------


def selected(tmp_path, repo, certification, digest):
    roots = (tmp_path / "certifications", tmp_path / "certification-raw")
    return hcert.selected(certification, digest, repo, roots)


def test_contract_g_refuses_an_unsealed_certification(tmp_path, repo, monkeypatch):
    cf.stand_ins(monkeypatch, repo)
    namespace, raw, _ = cf.certified(monkeypatch, tmp_path, repo, seal=False)
    assert hcert.status(namespace, raw) == "PASS"
    with pytest.raises(hcert.CertificationError, match="it is not sealed"):
        selected(tmp_path, repo, CERTIFICATION, "0" * 64)


def test_contract_h_refuses_a_failed_certification(tmp_path, repo, monkeypatch):
    cf.stand_ins(monkeypatch, repo)
    module = cf.cli()
    history, raw_root = tmp_path / "certifications", tmp_path / "certification-raw"
    namespace = module.certification_init(
        history, raw_root, CERTIFICATION, BINARY, SHA, cf.account_evidence(tmp_path)
    )
    cf.fake_headless(monkeypatch, ok=False)
    assert module.isolation_test(namespace, raw_root / CERTIFICATION, BINARY, []) == 1
    with pytest.raises(hcert.CertificationError, match="is FAILED"):
        selected(tmp_path, repo, CERTIFICATION, "0" * 64)


def test_contract_i_refuses_a_manifest_that_is_not_the_one_named(tmp_path, repo, monkeypatch):
    cf.stand_ins(monkeypatch, repo)
    namespace, raw, digest = cf.certified(monkeypatch, tmp_path, repo)
    assert selected(tmp_path, repo, CERTIFICATION, digest) == namespace
    with pytest.raises(hcert.CertificationError, match="not the one named"):
        selected(tmp_path, repo, CERTIFICATION, "0" * 64)
    with (raw / "broker" / "run-1" / "stream.jsonl").open("a") as file:
        file.write("\n")
    with pytest.raises(hcert.CertificationError, match="does not verify"):
        selected(tmp_path, repo, CERTIFICATION, digest)


def test_contract_j_refuses_certified_code_that_changed(tmp_path, repo, monkeypatch):
    cf.stand_ins(monkeypatch, repo)
    namespace, raw, digest = cf.certified(monkeypatch, tmp_path, repo)
    module = repo / hcert.CERTIFIED_CODE[0]
    module.write_text(module.read_text() + "\n# a changed audit\n")
    with pytest.raises(hcert.CertificationError, match="vouches for changed"):
        selected(tmp_path, repo, CERTIFICATION, digest)
    ho.git(repo, "add", "-A")
    ho.git(repo, "commit", "-q", "-m", "the changed audit, committed")
    with pytest.raises(hcert.CertificationError, match="vouches for changed"):
        selected(tmp_path, repo, CERTIFICATION, digest)


def test_contract_j_refuses_a_live_binary_other_than_the_certified(tmp_path, repo, monkeypatch):
    cf.stand_ins(monkeypatch, repo)
    namespace, raw, digest = cf.certified(monkeypatch, tmp_path, repo)
    module = cf.cli()
    monkeypatch.setattr(hcert, "CERTIFICATIONS_DIR", tmp_path / "certifications")
    monkeypatch.setattr(hcert, "CERTIFICATION_RAW_DIR", tmp_path / "certification-raw")
    assert module.certified(CERTIFICATION, digest) == namespace
    monkeypatch.setattr(hi, "binary_identity", lambda b: {**cf.IDENTITY, "version": "2.1.309"})
    with pytest.raises(hcert.CertificationError, match="not the certified one"):
        module.certified(CERTIFICATION, digest)


def test_contract_l_two_certifications_cannot_be_confused(tmp_path, repo, monkeypatch):
    cf.stand_ins(monkeypatch, repo)
    first, _, first_digest = cf.certified(monkeypatch, tmp_path, repo, "gate1-first")
    second, _, second_digest = cf.certified(monkeypatch, tmp_path, repo, "gate1-second")
    assert first_digest != second_digest
    assert selected(tmp_path, repo, "gate1-first", first_digest) == first
    assert selected(tmp_path, repo, "gate1-second", second_digest) == second
    with pytest.raises(hcert.CertificationError, match="not the one named"):
        selected(tmp_path, repo, "gate1-first", second_digest)
    shutil.copyfile(second / hcert.MANIFEST, first / hcert.MANIFEST)  # one passed for another
    with pytest.raises(hcert.CertificationError, match="another certification's"):
        selected(tmp_path, repo, "gate1-first", second_digest)
    assert hcert.reference(second)["id"] == "gate1-second"


# what pre-authoring-freeze is run with here: a TEST folder of its own, with run 6's kind of
# provenance (its own runtime pin, Gate 1 and broker records) as a decoy; the contract's code
# record stood in (it needs the history back to E's DEV result) and its file captured
DECOY = {
    "runtime.json": {"path": "/decoy", "version": "2.1.177", "sha256": "d" * 64, "model": "x"},
    "gate1.json": {"gate_1": "PASS", "run": "run-4", "model": "x"},
    "broker_test.json": {"broker_test": "PASS", "run": "run-1"},
}


def freeze_contract(module, test: Path, certification: str, digest: str, monkeypatch) -> dict:
    written: list[dict] = []
    pinned = json.loads(hc.CONTRACT_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(hc, "code_identity", lambda repo: pinned["code"])

    def write(contract: dict) -> str:
        written.append(contract)
        return "f" * 64

    monkeypatch.setattr(hc, "write", write)
    monkeypatch.setattr(hc, "verify", lambda digest: None)
    monkeypatch.setattr(hrt, "mlx_runtime", lambda: FAKE_MLX)
    code = module.pre_authoring_freeze(test, certification, digest)
    return {"code": code, "contract": written[0] if written else None}


@pytest.fixture
def test_folder(tmp_path):
    """A TEST folder with its grid and template freeze, and run 6's kind of Gate 1 records."""
    test = tmp_path / "test"
    (test / "provenance").mkdir(parents=True)
    grid = ho.build_grid(bm.load_benchmark(bm.DEV_DIR), ho.load_areas())
    ho.write_json(test / ho.GRID_FILE, grid.model_dump(mode="json"))
    ho.write_json(test / "provenance" / "frozen_templates.json", {"files": {}})
    for name, data in DECOY.items():
        ho.write_json(test / "provenance" / name, data)
    return test


def certified_for_contract(tmp_path, repo, monkeypatch):
    """A sealed certification whose sessions report the model it requested, as a real one does."""
    cf.stand_ins(monkeypatch, repo)
    module = cf.cli()
    monkeypatch.setattr(module, "HEADLESS_MODEL", FAKE_EFFECTIVE["init"]["model"])
    namespace, raw, digest = cf.certified(monkeypatch, tmp_path, repo)
    monkeypatch.setattr(hcert, "CERTIFICATIONS_DIR", tmp_path / "certifications")
    monkeypatch.setattr(hcert, "CERTIFICATION_RAW_DIR", tmp_path / "certification-raw")
    return module, namespace, digest


def test_contract_k_reads_the_named_certification_never_tests_own_provenance(
    tmp_path, repo, monkeypatch, test_folder
):
    module, namespace, digest = certified_for_contract(tmp_path, repo, monkeypatch)
    before = files(test_folder)
    refused = freeze_contract(module, test_folder, "gate1-not-made", digest, monkeypatch)
    assert refused == {"code": 1, "contract": None} and files(test_folder) == before
    made = freeze_contract(module, test_folder, CERTIFICATION, digest, monkeypatch)
    assert made["code"] == 0
    construction = made["contract"]["construction"]
    assert construction["certification"] == hcert.reference(namespace)
    assert construction["certification"]["manifest_sha256"] == digest
    assert construction["claude_runtime"] == hrt.claude_runtime(namespace / "provenance")
    assert (
        construction["claude_runtime"]["binary"]["sha256"] == SHA != DECOY["runtime.json"]["sha256"]
    )
    with pytest.raises(SystemExit):
        module.main(["--dir", str(test_folder), "pre-authoring-freeze"])  # no certification


def test_contract_m_changing_run_6_provenance_changes_nothing(
    tmp_path, repo, monkeypatch, test_folder
):
    module, namespace, digest = certified_for_contract(tmp_path, repo, monkeypatch)
    first = freeze_contract(module, test_folder, CERTIFICATION, digest, monkeypatch)["contract"]
    for name in DECOY:
        (test_folder / "provenance" / name).write_text('{"changed": "run 6 provenance"}\n')
    (test_folder / "audits" / "isolation" / "run-4").mkdir(parents=True)
    second = freeze_contract(module, test_folder, CERTIFICATION, digest, monkeypatch)["contract"]
    for key in ("certification", "claude_runtime"):
        assert first["construction"][key] == second["construction"][key]


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def test_the_cli_runs_a_test_only_inside_a_certification_and_records_it(cli, roots, monkeypatch):
    history, raw_root, evidence = roots
    monkeypatch.setattr(hcert, "CERTIFICATIONS_DIR", history)
    monkeypatch.setattr(hcert, "CERTIFICATION_RAW_DIR", raw_root)
    command = ["--claude-bin", str(BINARY)]
    with pytest.raises(ho.HoldoutError, match="certification-init makes one"):
        cli.main(["isolation-test", "--certification", CERTIFICATION, *command])
    init = ["--binary-sha256", SHA, "--account-evidence", str(evidence)]
    assert cli.main(["certification-init", CERTIFICATION, *command, *init]) == 0
    cf.fake_headless(monkeypatch)
    assert cli.main(["isolation-test", "--certification", CERTIFICATION, *command]) == 0
    assert git_log(history)[0] == f"certification {CERTIFICATION}: isolation-test, exit 0"
    tracked = ho.git(history, "ls-files").split()
    assert f"{CERTIFICATION}/audits/isolation/run-1/report.json" in tracked
    assert f"{CERTIFICATION}/provenance/gate1.json" in tracked
    assert not any("raw" in Path(p).parts for p in tracked)
    assert cli.main(["certification-verify", CERTIFICATION]) == 1  # not sealed: OPEN


def test_the_cli_seals_and_verifies(cli, roots, tmp_path, monkeypatch):
    history, raw_root, evidence = roots
    monkeypatch.setattr(hcert, "CERTIFICATIONS_DIR", history)
    monkeypatch.setattr(hcert, "CERTIFICATION_RAW_DIR", raw_root)
    command = ["--claude-bin", str(BINARY)]
    init = ["--binary-sha256", SHA, "--account-evidence", str(evidence)]
    assert cli.main(["certification-init", CERTIFICATION, *command, *init]) == 0
    cf.fake_headless(monkeypatch)
    cf.fake_broker(monkeypatch)
    for test in ("isolation-test", "broker-test"):
        assert cli.main([test, "--certification", CERTIFICATION, *command]) == 0
    report = cf.junit(tmp_path / "full.xml")
    seal = ["certification-seal", CERTIFICATION, "--suite-report", f"full={report}"]
    assert cli.main(seal) == 0
    digest = hashlib.sha256((history / CERTIFICATION / hcert.MANIFEST).read_bytes()).hexdigest()
    assert cli.main(["certification-verify", CERTIFICATION, "--manifest", digest]) == 0
    assert cli.main(["certification-verify", CERTIFICATION, "--manifest", "0" * 64]) == 1
