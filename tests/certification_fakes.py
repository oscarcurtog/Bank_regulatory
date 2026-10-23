"""Stand-ins for a Gate 1 certification (ingest/holdout_certification.py), run by the real code
of scripts/holdout.py: a public repository holding the code a certification records, pushed to
a local remote; the binary's, host's and substrate's identities; the two model boundaries,
replaced by sessions that pass the audit; and the suites' reports. Nothing here runs Claude
Code, reads the real private folders or changes the machine."""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ingest import holdout as ho
from ingest import holdout_broker as hb
from ingest import holdout_certification as hcert
from ingest import holdout_isolation as hi
from ingest import holdout_runtime as hrt
from tests.holdout_fakes import session
from tests.unit.test_holdout_broker import FakeTools, Scripted, req

BINARY = Path("/not-a-real-binary")
SHA = "1" * 64
IDENTITY = {"path": str(BINARY), "version": "2.1.301 (Claude Code)", "sha256": SHA}
HOST = {
    "macos_product_version": "27.0.1",
    "macos_build_version": "27A1",
    "darwin_release": "27.0.0",
    "hardware_model": "Mac16,1",
    "architecture": "arm64",
}
# a substrate that keeps everything from changing: what certification-init requires
SUBSTRATE: dict[str, Any] = {
    "vscode": {"version": "1.143.1", "commit": "e" * 40, "update_mode": "none"},
    "extension": {
        "id": "anthropic.claude-code",
        "version": "2.1.301",
        "pinned": True,
        "folder": "anthropic.claude-code-2.1.301-darwin-arm64",
    },
    "macos_automatic_installs": {k: False for k in hcert.FROZEN_INSTALLS},
}
CERTIFICATION = "gate1-20261023-cc2.1.301-test"


def cli() -> ModuleType:
    return importlib.import_module("scripts.holdout")  # scripts/ is no package for mypy


def git(repo: Path, *args: str) -> str:
    return ho.git(repo, *args)


def public_repo(root: Path, files: tuple[str, ...] = (), name: str = "public") -> Path:
    """A public repository holding the code a certification records (and any other files
    named), committed and pushed to a local remote: a clean checkout of a pushed commit."""
    repo = root / name
    for f in {*hcert.CERTIFIED_CODE, *hcert.IMPLEMENTATION, *files}:
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, repo / f)
    git(root, "init", "-q", "--bare", "remote.git")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "fixture")
    git(repo, "config", "user.email", "fixture@example.org")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "the certified code")
    git(repo, "remote", "add", "origin", str(root / "remote.git"))
    git(repo, "push", "-q", "origin", "main")
    return repo


def stand_ins(patch: pytest.MonkeyPatch, repo: Path, identity: dict[str, str] = IDENTITY) -> None:
    """The binary, the host and the substrate, as they are read, and the CLI's repository."""
    module = cli()
    patch.setattr(hi, "binary_identity", lambda binary: dict(identity))
    patch.setattr(hrt, "host_identity", lambda: dict(HOST))
    patch.setattr(hcert, "execution_substrate", lambda binary: json.loads(json.dumps(SUBSTRATE)))
    patch.setattr(module, "isolation_terms", lambda: hi.forbidden_terms(None, None, []))
    patch.setattr(hi, "positive_controls", lambda *args: [])
    patch.setattr(module, "REPO", repo)


def account_evidence(root: Path) -> Path:
    """A raw transcript of the login: the stand-in session's credential_org."""
    evidence = root / "evidence" / "transcript.jsonl"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(session(["x"], ["y"])["transcript"], encoding="utf-8")
    return evidence


def fake_headless(patch: pytest.MonkeyPatch, ok: bool = True) -> dict[str, Any]:
    """A headless run in place of the binary: one that passes Gate 1's audit, or one whose
    answer lacks a field the packet asks for, which fails it."""
    fields = {"sentence": "A person checks the library hours.", "context_report": "none"}
    answer = json.dumps(fields | ({"working_directory": "/tmp/tmp.x"} if ok else {}))
    text = session([hi.DUMMY_PACKET], [answer])
    context = text["_gate1_context"]
    run = hi.HeadlessRun(
        context["launcher_workdir"],
        0,
        text["stream"],
        "",
        "s",
        None,
        text["transcript"],
        answer,
        context,
    )
    patch.setattr(hi, "run_headless", lambda *args: run)
    found: dict[str, Any] = context
    return found


def fake_broker(patch: pytest.MonkeyPatch) -> None:
    """A judge A conversation that passes the broker's test, in place of the binary."""

    class Conversation(Scripted):
        def __init__(self, *args: Any) -> None:
            super().__init__(
                [
                    req("titles", act="DORA"),
                    req("article", act="DORA", article="16"),
                    {"final": hb.SYNTHETIC_ANSWER},
                ]
            )

        def close(self) -> dict[str, Any]:
            return session(self.received, self.replies) | {"returncode": 0, "command": []}

    patch.setattr(hb, "StreamJsonConversation", Conversation)
    patch.setattr(hb, "corpus_tools", FakeTools)


def junit(path: Path, tests: int = 3, failures: int = 0, errors: int = 0) -> Path:
    """A pytest JUnit report with those counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = f'<testsuite name="pytest" tests="{tests}" failures="{failures}" errors="{errors}" '
    path.write_text(f'<testsuites>{suite}skipped="1" time="1.0"/></testsuites>\n')
    return path


def certified(
    patch: pytest.MonkeyPatch,
    root: Path,
    repo: Path,
    certification: str = CERTIFICATION,
    extra_env: tuple[str, ...] = (),
    seal: bool = True,
    identity: dict[str, str] = IDENTITY,
) -> tuple[Path, Path, str | None]:
    """A certification made, run and sealed by the real code with the stand-ins: init, a Gate 1
    that passes, a broker test that passes, then the seal with one suite report. Returns its
    namespace, raw folder and manifest's SHA-256 (None when not sealed)."""
    module = cli()
    history, raw_root = root / "certifications", root / "certification-raw"
    evidence = account_evidence(root)
    binary = Path(identity["path"])
    namespace = module.certification_init(
        history, raw_root, certification, binary, identity["sha256"], evidence
    )
    raw = raw_root / certification
    fake_headless(patch)
    fake_broker(patch)
    assert module.isolation_test(namespace, raw, binary, list(extra_env)) == 0
    assert module.broker_test(namespace, raw, binary, list(extra_env)) == 0
    module.record(history, f"certification {certification}: tests")
    if not seal:
        return namespace, raw, None
    report = junit(root / "reports" / f"{certification}.xml")
    return namespace, raw, hcert.seal(namespace, raw, repo, {"full": report})
