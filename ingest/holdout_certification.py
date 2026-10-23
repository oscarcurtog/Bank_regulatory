"""Gate 1 certifications (S3): the proof that one Claude Code runtime, launched by this code,
gives a headless author and a broker's judge no context. Made once, sealed once, and named
explicitly by the contract that relies on it.

scripts/holdout.py certification-init makes a certification from a clean checkout of a commit
that a remote branch holds. It lives under one id in two private folders:
- CERTIFICATIONS_DIR/<id>/, in the private git history of certifications: the runtime pin
  (provenance/runtime.json), what it was made from (provenance/certification.json: the public
  code and the execution substrate), Gate 1's and the broker test's records, the redacted
  audits, the suites' reports and, last, the manifest;
- CERTIFICATION_RAW_DIR/<id>/, outside every git history: the raw transcripts, which hold the
  account's email and organization id. A run's number is reserved there with an exclusive
  mkdir; reserved.json is written then, complete.json last.

Its status is read from that evidence and never kept apart. It is FAILED as soon as a run ends
FAIL; PASS once Gate 1 and then the broker test have each ended PASS once; OPEN otherwise. A run
that died (reserved, never complete) is no final run: its number stays taken and the next run
takes another. A FAILED certification never runs again: a new attempt is a new id.

seal writes a PASS's manifest, once, after it checks that the public code is still the clean
commit it was made from and that the runtime, host, substrate and account evidence are
unchanged. The manifest holds the suites' reports, the private history's commit and the SHA-256
of every file of both folders, and verify recomputes all of it. A contract names a certification
by its id and its manifest's SHA-256 (reference). selected refuses that certification unless it
verifies, its commit is in the checkout's history, and the code it vouches for (CERTIFIED_CODE)
still holds the bytes it sealed. Nothing here falls back to another certification or to TEST's
own provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ingest import holdout as ho
from ingest import holdout_isolation as hi
from ingest import holdout_runtime as hrt

PRIVATE_ROOT = Path.home() / ".fundamento-private"
CERTIFICATIONS_DIR = PRIVATE_ROOT / "certifications"
CERTIFICATION_RAW_DIR = PRIVATE_ROOT / "certification-raw"
SCHEMA = "gate1-certification.v1"
ID = re.compile(r"[a-z0-9][a-z0-9.-]*")
KINDS = ("isolation", "broker")
RUN_NAME = re.compile(r"run-([1-9][0-9]*)")
RESERVED = "reserved.json"  # written when a run's number is taken, before anything runs
COMPLETE = "complete.json"  # written last: a reserved run without it died before its end
PROVENANCE = "provenance"
IDENTITY = "certification.json"  # the public code and the substrate, at certification-init
SUITES = "suites"  # the suites' JUnit reports, copied at the seal
MANIFEST = "manifest.json"
# The code a certification vouches for, byte for byte: the audit, the headless and stream-json
# transports and the process groups between the launcher and the binary. A construction relies
# on a certification only while these files hold the bytes it sealed.
CERTIFIED_CODE = (
    "ingest/holdout_isolation.py",
    "ingest/holdout_broker.py",
    "ingest/holdout_spool.py",
    "ingest/holdout_process.py",
)
# The code that makes and runs a certification: sealed with it, free to change after it
IMPLEMENTATION = (
    "scripts/holdout.py",
    "ingest/holdout_certification.py",
    "ingest/holdout_runtime.py",
    "ingest/holdout_launcher.py",
    "ingest/holdout.py",
    "ingest/benchmark.py",
    "ingest/corpus_tools.py",
)
# What runs the binary and what could replace it while it is pinned
VSCODE_PRODUCT = Path("/Applications/Visual Studio Code.app/Contents/Resources/app/product.json")
VSCODE_SETTINGS = (
    Path.home() / "Library" / "Application Support" / "Code" / "User" / "settings.json"
)
SOFTWARE_UPDATE = "/Library/Preferences/com.apple.SoftwareUpdate"
# the automatic installs the person turns off for the frozen window: macOS updates, and the
# security responses and system files (one switch in System Settings, two keys)
FROZEN_INSTALLS = ("AutomaticallyInstallMacOSUpdates", "CriticalUpdateInstall", "ConfigDataInstall")


class CertificationError(ho.HoldoutError):
    """A certification that cannot be made, run, sealed or relied on."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, text: str) -> None:
    """A new file, never an existing one: mode "x" refuses a name that exists, so evidence is
    never truncated or replaced."""
    with path.open("x", encoding="utf-8") as file:
        file.write(text)


def write_new_json(path: Path, data: Any) -> None:
    write_new(path, json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise CertificationError(f"{path} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise CertificationError(f"git {' '.join(args)} in {repo}: {done.stderr.strip()}")
    return done.stdout.strip()


def folders(history: Path, raw_root: Path, certification: str) -> tuple[Path, Path]:
    """A certification's namespace and raw folder, by its id."""
    if not ID.fullmatch(certification):
        raise CertificationError(f"{certification!r} is not a certification id")
    return history / certification, raw_root / certification


def tree(root: Path, leave_out: tuple[str, ...] = ()) -> dict[str, str]:
    """Every file under a folder, by its path from there, with its SHA-256."""
    return {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.relative_to(root).as_posix() not in leave_out
    }


# ---------------------------------------------------------------------------
# the status, read from the evidence
# ---------------------------------------------------------------------------


def runs(raw: Path, kind: str) -> dict[str, str]:
    """Every run of a kind that the raw folder holds, and how it ended: its verdict, or
    incomplete (reserved, never complete: the process died)."""
    if not (raw / kind).is_dir():
        raise CertificationError(f"{raw / kind} does not exist")
    found = {}
    for run in sorted((raw / kind).iterdir()):
        if RUN_NAME.fullmatch(run.name):
            done = run / COMPLETE
            found[run.name] = str(read_json(done)["verdict"]) if done.is_file() else "incomplete"
    return found


def finals(raw: Path, kind: str) -> list[str]:
    """The verdicts of the runs of a kind that ended."""
    return [v for v in runs(raw, kind).values() if v != "incomplete"]


def status(namespace: Path, raw: Path) -> str:
    """FAILED once any run ended other than PASS; PASS once Gate 1 and the broker test have
    each ended PASS exactly once, with their records; OPEN otherwise."""
    ended = {kind: finals(raw, kind) for kind in KINDS}
    if any(v != "PASS" for verdicts in ended.values() for v in verdicts):
        return "FAILED"
    provenance = namespace / PROVENANCE
    records = all((provenance / f).is_file() for f in ("gate1.json", "broker_test.json"))
    if ended["isolation"] == ["PASS"] and ended["broker"] == ["PASS"] and records:
        return "PASS"
    return "OPEN"


# ---------------------------------------------------------------------------
# what a certification is made from
# ---------------------------------------------------------------------------


def code_identity(repo: Path) -> dict[str, Any]:
    """The public code a certification is made from: the commit checked out, which a remote
    branch holds, with a clean working tree (no change to a tracked file, staged or not, and no
    file that git does not ignore), and the SHA-256 of the code it vouches for and of the code
    that makes it. The repository is named by its root commits."""
    dirty = git(repo, "--no-optional-locks", "status", "--porcelain")
    if dirty:
        raise CertificationError(f"the working tree is not clean, commit or remove:\n{dirty}")
    commit = git(repo, "rev-parse", "HEAD")
    listed = git(repo, "branch", "-r", "--contains", commit).splitlines()
    remotes = sorted({line.strip().split(" -> ")[0] for line in listed if line.strip()})
    if not remotes:
        raise CertificationError(f"{commit[:12]} is on no remote branch: push it first")
    return {
        "commit": commit,
        "remote_branches": remotes,
        "repository_roots": git(repo, "rev-list", "--max-parents=0", commit).split(),
        "certified_code_sha256": {f: sha256_file(repo / f) for f in CERTIFIED_CODE},
        "implementation_sha256": {f: sha256_file(repo / f) for f in IMPLEMENTATION},
        "implementation": SCHEMA,
    }


def _preference(domain: str, key: str) -> bool | None:
    """A macOS preference as the system reads it: None when it is not set or not a boolean."""
    done = subprocess.run(
        ["/usr/bin/defaults", "read", domain, key],
        capture_output=True,
        text=True,
        check=False,
        timeout=hrt.PROBE_SECONDS,
    )
    if done.returncode != 0:
        return None
    return {"0": False, "1": True}.get(done.stdout.strip())


def execution_substrate(binary: Path) -> dict[str, Any]:
    """What runs the binary and what could replace it while it is pinned: VS Code (version,
    commit, update.mode), the extension that ships the binary (id, version, pin) and macOS's
    automatic installs. Read only, never changed; anything unreadable is a stop."""
    try:
        product = json.loads(VSCODE_PRODUCT.read_text(encoding="utf-8"))
        settings = json.loads(VSCODE_SETTINGS.read_text(encoding="utf-8"))
        folder = binary.resolve().parents[2]  # <extension>/resources/native-binary/claude
        registry = json.loads((folder.parent / "extensions.json").read_text(encoding="utf-8"))
        entries = [e for e in registry if e.get("relativeLocation") == folder.name]
        if len(entries) != 1:
            raise CertificationError(f"{folder.name} is not exactly one installed extension")
        entry = entries[0]
        return {
            "vscode": {
                "version": product["version"],
                "commit": product["commit"],
                "update_mode": settings.get("update.mode"),
            },
            "extension": {
                "id": entry["identifier"]["id"],
                "version": entry["version"],
                "pinned": entry.get("metadata", {}).get("pinned") is True,
                "folder": folder.name,
            },
            "macos_automatic_installs": {
                k: _preference(SOFTWARE_UPDATE, k) for k in FROZEN_INSTALLS
            },
        }
    except (OSError, ValueError, IndexError, KeyError, TypeError) as error:
        raise CertificationError(f"the execution substrate cannot be read: {error}") from error


def substrate_problems(substrate: Mapping[str, Any]) -> list[str]:
    """What would let the substrate change under a pinned runtime: VS Code free to update, the
    extension unpinned, or macOS free to install updates, security responses or system files."""
    problems = []
    mode = substrate["vscode"]["update_mode"]
    if mode != "none":
        problems.append(f"VS Code's update.mode is {mode!r}, not 'none'")
    if not substrate["extension"]["pinned"]:
        problems.append(f"{substrate['extension']['id']} is not pinned")
    on = [k for k, v in substrate["macos_automatic_installs"].items() if v is not False]
    if on:
        problems.append(f"macOS installs automatically or cannot tell ({', '.join(on)})")
    return problems


# ---------------------------------------------------------------------------
# the seal and its verification
# ---------------------------------------------------------------------------


def _junit(path: Path) -> dict[str, int]:
    """The counts of a pytest JUnit report, over its test suites."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise CertificationError(f"{path} is not a JUnit report: {error}") from error
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if not suites:
        raise CertificationError(f"{path} holds no test suite")
    return {
        k: sum(int(s.get(k, "0")) for s in suites)
        for k in ("tests", "failures", "errors", "skipped")
    }


def seal(namespace: Path, raw: Path, repo: Path, suites: Mapping[str, Path]) -> str:
    """Seal a PASS, once: its manifest, committed to the private history. Refused unless the
    certification is PASS and not sealed; the public code is still the clean commit
    certification-init recorded; the binary, host, substrate and account evidence are
    unchanged; and every suite report has tests and no failure or error. Returns the
    manifest's SHA-256."""
    certification, history = namespace.name, namespace.parent
    if raw.name != certification:
        raise CertificationError(f"{raw} is not {certification}'s raw folder")
    if (namespace / MANIFEST).exists():
        raise CertificationError(f"{certification} is sealed already")
    found = status(namespace, raw)
    if found != "PASS":
        raise CertificationError(f"{certification} is {found}: only a PASS is sealed")
    provenance = namespace / PROVENANCE
    identity = read_json(provenance / IDENTITY)
    pin = read_json(provenance / "runtime.json")
    gate1 = read_json(provenance / "gate1.json")
    broker = read_json(provenance / "broker_test.json")
    code, source = identity["code"], pin["identity_source"]
    problems = []
    if git(repo, "rev-parse", "HEAD") != code["commit"] or git(
        repo, "--no-optional-locks", "status", "--porcelain"
    ):
        problems.append("the public code is not the clean commit certification-init recorded")
    binary = Path(pin["path"])
    live = hi.binary_identity(binary)
    if (live["version"], live["sha256"]) != (pin["version"], pin["sha256"]):
        problems.append("the binary is not the pinned one")
    if hrt.host_identity() != pin["host"]:
        problems.append("the host is not the pinned one")
    substrate = execution_substrate(binary)
    if substrate != identity["substrate"]:
        problems.append("the execution substrate changed since certification-init")
    problems += substrate_problems(substrate)
    if sha256_file(Path(source["path"])) != source["sha256"]:
        problems.append("the account evidence changed")
    if gate1["audit"]["module_sha256"] != code["certified_code_sha256"][CERTIFIED_CODE[0]]:
        problems.append("Gate 1's audit module is not the certified one")
    counts = {}
    for name, path in suites.items():
        counts[name] = _junit(path)
        if not ID.fullmatch(name):
            problems.append(f"{name!r} is not a suite name")
        elif counts[name]["tests"] == 0 or counts[name]["failures"] or counts[name]["errors"]:
            problems.append(f"the suite {name} did not pass: {counts[name]}")
    if not suites:
        problems.append("no suite report: the suites run before the seal")
    if problems:
        raise CertificationError("; ".join(problems))
    (provenance / SUITES).mkdir()
    for name, path in suites.items():
        with (provenance / SUITES / f"{name}.xml").open("xb") as report:
            report.write(path.read_bytes())
    git(history, "add", "-A", "--", certification)
    git(history, "commit", "-q", "-m", f"certification {certification}: suite reports")
    if git(history, "--no-optional-locks", "status", "--porcelain"):
        raise CertificationError(f"{history} holds changes not recorded")
    inherited = gate1["environment_allowlist"]["inherited"]
    manifest = {
        "schema": SCHEMA,
        "certification": certification,
        "status": "PASS",
        "code": code,
        "runtime": {
            "claude_code_version": pin["version"],
            "binary_sha256": pin["sha256"],
            "requested_model": pin["model"],
            "organization_sha256": pin["organization_sha256"],
            "host": pin["host"],
        },
        "account_evidence": {k: source[k] for k in ("logical", "sha256", "attachment")},
        "substrate": substrate,
        "policy": hrt.claude_policy([v for v in inherited if v not in hi.ENV_INHERITED]),
        "gate1": {
            "run": gate1["run"],
            "audit": gate1["audit"],
            "attachment_controls": gate1["attachment_controls"],
            "provenance_controls": gate1["provenance_controls"],
        },
        "broker_test": {"run": broker["run"]},
        "runs": {kind: runs(raw, kind) for kind in KINDS},
        "suites": {n: counts[n] | {"sha256": sha256_file(p)} for n, p in suites.items()},
        "private_history": {
            "commit": git(history, "rev-parse", "HEAD"),
            "namespace_tree_sha256": ho.tree_sha256(namespace),
        },
        "files_sha256": tree(namespace),
        "raw_sha256": tree(raw),
        "sealed_at": datetime.now(UTC).isoformat(),
    }
    write_new_json(namespace / MANIFEST, manifest)
    git(history, "add", "--", f"{certification}/{MANIFEST}")
    git(history, "commit", "-q", "-m", f"certification {certification}: sealed")
    return sha256_file(namespace / MANIFEST)


def verify(
    namespace: Path, raw: Path, repo: Path, manifest_sha256: str | None = None
) -> tuple[dict[str, Any] | None, list[str]]:
    """A sealed certification, recomputed: the manifest (the one named, if one is), its id, its
    status and the evidence's, every file of both folders with the SHA-256 it sealed and none
    added, the code it records against its own commit, and the private history holding the
    manifest as it is. (None, problems) when it is not sealed."""
    certification, path = namespace.name, namespace / MANIFEST
    if not path.is_file():
        return None, [f"{certification} has no manifest: it is not sealed"]
    problems = []
    if manifest_sha256 is not None and sha256_file(path) != manifest_sha256:
        problems.append("the manifest is not the one named")
    manifest: dict[str, Any] = read_json(path)
    if manifest.get("certification") != certification or raw.name != certification:
        problems.append("the manifest or the raw folder is another certification's")
    if manifest.get("status") != "PASS" or status(namespace, raw) != "PASS":
        problems.append("the certification is not PASS")
    for name, root, leave_out, key in (
        ("evidence", namespace, (MANIFEST,), "files_sha256"),
        ("raw evidence", raw, (), "raw_sha256"),
    ):
        sealed, found = manifest.get(key) or {}, tree(root, leave_out)
        changed = sorted(k for k in set(sealed) | set(found) if sealed.get(k) != found.get(k))
        if changed:
            problems.append(f"the sealed {name} changed: {changed[:5]}")
    code = manifest.get("code") or {}
    if not code.get("commit"):
        problems.append("the manifest records no public commit")
    files = {**code.get("certified_code_sha256", {}), **code.get("implementation_sha256", {})}
    for f, h in files.items():
        blob = subprocess.run(
            ["git", "-C", str(repo), "show", f"{code.get('commit')}:{f}"], capture_output=True
        )
        if blob.returncode != 0 or hashlib.sha256(blob.stdout).hexdigest() != h:
            problems.append(f"the manifest's code is not its commit's: {f}")
    history = namespace.parent
    committed = subprocess.run(
        ["git", "-C", str(history), "show", f"HEAD:{certification}/{MANIFEST}"],
        capture_output=True,
    )
    if committed.returncode != 0 or committed.stdout != path.read_bytes():
        problems.append("the private history does not hold this manifest")
    elif git(history, "--no-optional-locks", "status", "--porcelain", "--", certification):
        problems.append("the private history holds changes to it not recorded")
    return manifest, problems


# ---------------------------------------------------------------------------
# the certification a contract relies on
# ---------------------------------------------------------------------------


def selected(
    certification: str,
    manifest_sha256: str,
    repo: Path,
    roots: tuple[Path, Path] | None = None,
) -> Path:
    """The certification a contract names, by its id and its manifest's SHA-256, or a stop: it
    must be PASS, sealed by that manifest and verify; its commit must be in this checkout's
    history; and the code it vouches for must still hold the bytes it sealed. Returns its
    namespace. Never another certification, and never TEST's own provenance."""
    history, raw_root = roots or (CERTIFICATIONS_DIR, CERTIFICATION_RAW_DIR)
    namespace, raw = folders(history, raw_root, certification)
    if not namespace.is_dir() or not raw.is_dir():
        raise CertificationError(f"no certification {certification}")
    found = status(namespace, raw)
    if found != "PASS":
        raise CertificationError(f"{certification} is {found}: only a PASS is relied on")
    manifest, problems = verify(namespace, raw, repo, manifest_sha256)
    if manifest is None or problems:
        raise CertificationError(f"{certification} does not verify: {problems[:5]}")
    commit = manifest["code"]["commit"]
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise CertificationError(f"{certification}'s commit {commit[:12]} is not in this history")
    changed = [
        f
        for f, h in manifest["code"]["certified_code_sha256"].items()
        if not (repo / f).is_file() or sha256_file(repo / f) != h
    ]
    if changed:
        raise CertificationError(f"the code {certification} vouches for changed: {changed}")
    return namespace


def reference(namespace: Path) -> dict[str, Any]:
    """What a contract records of the certification it relies on: no path, no account,
    nothing personal; the manifest's SHA-256 seals the rest."""
    manifest = read_json(namespace / MANIFEST)
    return {
        "id": manifest["certification"],
        "schema": manifest["schema"],
        "manifest_sha256": sha256_file(namespace / MANIFEST),
        "public_commit": manifest["code"]["commit"],
        "private_commit": manifest["private_history"]["commit"],
        "certified_code_sha256": manifest["code"]["certified_code_sha256"],
        "gate1_run": manifest["gate1"]["run"],
        "broker_test_run": manifest["broker_test"]["run"],
    }
