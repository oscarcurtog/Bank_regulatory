"""Explicit approval of a source-only operational supplement to an existing TEST v2.

The base contract and the state's contract identity remain unchanged.  This module
accepts only the exact replacement hashes and new modules named by a separately
reviewed manifest whose SHA-256 the operator supplies explicitly.  A manifest's
presence is never authorization.  The scope declaration is not a proof that the
patch preserves the experiment: review and synthetic equivalence tests are still
required before approving its hash.

This module imports no model or TEST implementation and reads no private state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

SCHEMA = "holdout-v2-operational-amendment.v1"
SCHEMA_V2 = "holdout-v2-operational-amendment.v2"
PREDECESSOR_FILE = "config/holdout/operations-amendment.v1.json"
PREDECESSOR_SHA256 = "a68dc83b8cedc9a6dca59f9faed87acc7067225551f93ca6e3b8eb918e55d1a3"
PURPOSE_V2 = "Provider capacity refusal classification"
SCOPE_V2 = (
    "Prospective classification only; recorded reasons, MAX_RUNS and scientific design unchanged"
)
BASE_FILE = "config/holdout/test-v2-contract.json"
REPLACEMENTS = frozenset(
    {
        "ingest/holdout_v2_engine.py",
        "ingest/holdout_v2_runner.py",
        "ingest/holdout_v2_contract.py",
        "scripts/holdout_v2.py",
    }
)
NEW_FILES = frozenset(
    {
        "ingest/holdout_v2_operations.py",
        "ingest/holdout_v2_operational_amendment.py",
    }
)
PURPOSE = "Operational pacing, cooperative pause, aggregate progress and verified checkpoints"
SCOPE = "No change to scientific inputs, packets, judging, priority, retries or evaluation"
_FIELDS = frozenset(
    {
        "schema",
        "base_contract_file",
        "base_contract_sha256",
        "purpose",
        "scope",
        "replacements",
        "new_files",
    }
)
_SHA = re.compile(r"^[0-9a-f]{64}$")


class AmendmentError(ValueError):
    """The requested operational supplement is unapproved or does not match its files."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash(value: Any) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise AmendmentError("duplicate JSON key in the operational manifest or base contract")
        result[key] = value
    return result


def _json(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AmendmentError("invalid operational manifest or base contract JSON") from error
    if not isinstance(value, dict):
        raise AmendmentError("the operational manifest and base contract must be JSON objects")
    return value


def _path(repo: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or "\\" in relative
        or PurePosixPath(relative).is_absolute()
        or any(p in ("", ".", "..") for p in relative.split("/"))
    ):
        raise AmendmentError("an operational manifest path is not a canonical relative path")
    base = repo.resolve()
    path = base / relative
    if not path.resolve().is_relative_to(base):
        raise AmendmentError("an operational manifest path leaves the repository")
    return path


def _file_sha(repo: Path, relative: str) -> str:
    path = _path(repo, relative)
    try:
        return _sha(path.read_bytes())
    except OSError as error:
        raise AmendmentError(
            f"required operational or frozen file is unavailable: {relative}"
        ) from error


def _base(repo: Path, path: Path) -> tuple[dict[str, Any], str]:
    expected = _path(repo, BASE_FILE)
    supplied = path.resolve()
    if not path.is_absolute() and supplied != expected.resolve():
        supplied = (repo / path).resolve()
    if supplied != expected.resolve():
        raise AmendmentError("the operational supplement must name the original TEST v2 contract")
    try:
        raw = expected.read_bytes()
    except OSError as error:
        raise AmendmentError("the original TEST v2 contract is unavailable") from error
    value = _json(raw)
    if value.get("schema") != "holdout-v2-contract.v1":
        raise AmendmentError("the base is not a TEST v2 contract")
    return value, _sha(raw)


def _frozen(contract: Mapping[str, Any]) -> dict[str, str]:
    """The same public path inventory as holdout_v2_contract.frozen_files, without imports."""
    try:
        files: dict[str, str] = {}

        def add(path: Any, digest: Any) -> None:
            if not isinstance(path, str) or not _hash(digest):
                raise AmendmentError("invalid frozen identity in the base contract")
            if path in files and files[path] != digest:
                raise AmendmentError("the base contract names conflicting frozen identities")
            files[path] = digest

        for source in (contract["code"]["files_sha256"], contract["code"]["inputs_sha256"]):
            for path, digest in source.items():
                add(path, digest)
        protocol = contract["code"]["protocol"]
        add(protocol["file"], protocol["sha256"])
        evaluation = contract["evaluation"]
        add(evaluation["plan_file"], evaluation["plan_sha256"])
        retrieval = contract["retrieval"]
        for path, digest in retrieval["code_sha256"].items():
            add(path, digest)
        roles = contract["roles"]
        for name, digest in roles["templates_sha256"].items():
            add(roles["template_files"][name], digest)
        for system in retrieval["pool_only"].values():
            add(system["config_file"], system["config_sha256"])
        for system in retrieval["systems"].values():
            for part in (system, *(v for v in system.values() if isinstance(v, dict))):
                if "config_file" in part:
                    add(part["config_file"], part["config_sha256"])
    except (KeyError, AttributeError, TypeError) as error:
        raise AmendmentError("the base contract has an incomplete frozen-file inventory") from error
    return files


def _shape(value: Mapping[str, Any]) -> None:
    second = value.get("schema") == SCHEMA_V2
    if set(value) != (_FIELDS | {"predecessor"} if second else _FIELDS):
        raise AmendmentError("unexpected operational manifest fields")
    if (
        value["schema"] not in (SCHEMA, SCHEMA_V2)
        or value["base_contract_file"] != BASE_FILE
        or value["purpose"] != (PURPOSE_V2 if second else PURPOSE)
        or value["scope"] != (SCOPE_V2 if second else SCOPE)
        or not _hash(value["base_contract_sha256"])
    ):
        raise AmendmentError("invalid operational manifest identity or scope")
    if second and value["predecessor"] != {
        "file": PREDECESSOR_FILE,
        "sha256": PREDECESSOR_SHA256,
    }:
        raise AmendmentError("the second supplement must name the exact first approval")
    replacements, new = value["replacements"], value["new_files"]
    if not isinstance(replacements, dict) or set(replacements) != REPLACEMENTS:
        raise AmendmentError(
            "the operational supplement must name exactly the allowed source files"
        )
    if not isinstance(new, dict) or set(new) != NEW_FILES or not all(map(_hash, new.values())):
        raise AmendmentError("the operational supplement must name exactly the two new modules")
    for pair in replacements.values():
        if (
            not isinstance(pair, dict)
            or set(pair) != {"before_sha256", "after_sha256"}
            or not _hash(pair["before_sha256"])
            or not _hash(pair["after_sha256"])
        ):
            raise AmendmentError("invalid operational replacement hashes")


def _chain(repo: Path, value: Mapping[str, Any]) -> str | None:
    """V2 composes BASE -> V1 -> V2 without reauthorizing any other change."""
    if value["schema"] == SCHEMA:
        return None
    path = _path(repo, PREDECESSOR_FILE)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise AmendmentError("the first approved supplement is unavailable") from error
    if _sha(raw) != PREDECESSOR_SHA256:
        raise AmendmentError("the first approved supplement differs from its pinned identity")
    previous = _json(raw)
    _shape(previous)
    if previous["schema"] != SCHEMA or (
        previous["base_contract_sha256"] != value["base_contract_sha256"]
    ):
        raise AmendmentError("the first supplement names another schema or base")
    engine = "ingest/holdout_v2_engine.py"
    loader = "ingest/holdout_v2_operational_amendment.py"
    for path, pair in value["replacements"].items():
        old = previous["replacements"][path]
        if pair["before_sha256"] != old["before_sha256"] or (path != engine and pair != old):
            raise AmendmentError("the second supplement changes an unapproved source")
    for path, digest in value["new_files"].items():
        if path != loader and digest != previous["new_files"][path]:
            raise AmendmentError("the second supplement changes an unapproved module")
    if (
        value["replacements"][engine] == previous["replacements"][engine]
        or value["new_files"][loader] == previous["new_files"][loader]
    ):
        raise AmendmentError("the second supplement must identify both exact replacements")
    return PREDECESSOR_SHA256


def canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode()


def build_manifest(repo: Path, base_contract_path: Path) -> dict[str, Any]:
    """Prepare public source hashes only.  This neither approves nor activates them."""
    contract, digest = _base(repo, base_contract_path)
    try:
        original = contract["code"]["files_sha256"]
        replacements = {}
        for path in sorted(REPLACEMENTS):
            old = original[path]
            if not _hash(old):
                raise AmendmentError("an allowed source lacks its original frozen hash")
            replacements[path] = {"before_sha256": old, "after_sha256": _file_sha(repo, path)}
    except (KeyError, TypeError) as error:
        raise AmendmentError("an allowed source is not frozen in the original contract") from error
    value = {
        "schema": SCHEMA,
        "base_contract_file": BASE_FILE,
        "base_contract_sha256": digest,
        "purpose": PURPOSE,
        "scope": SCOPE,
        "replacements": replacements,
        "new_files": {path: _file_sha(repo, path) for path in sorted(NEW_FILES)},
    }
    _shape(value)
    return value


def write_exclusive(path: Path, manifest: Mapping[str, Any]) -> str:
    """Write one prepared manifest without replacing an existing one; return its review hash."""
    _shape(manifest)
    data = canonical(manifest)
    # No directory creation: the operator deliberately chooses where the review artifact lives.
    with path.open("xb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    return _sha(data)


@dataclass(frozen=True)
class Amendment:
    base_contract_sha256: str
    sha256: str
    replacements: Mapping[str, tuple[str, str]]
    new_files: Mapping[str, str]
    predecessor_sha256: str | None = None

    def permitted_change(self, path: str, old_sha: str, actual_sha: str) -> bool:
        return self.replacements.get(path) == (old_sha, actual_sha)

    def validate(self, repo: Path, contract: Mapping[str, Any]) -> None:
        """Recheck the base, all frozen files and both new modules at every phase preflight."""
        current, digest = _base(repo, repo / BASE_FILE)
        if digest != self.base_contract_sha256 or current != contract:
            raise AmendmentError("the operational supplement is bound to another base contract")
        if self.predecessor_sha256 is not None and (
            _file_sha(repo, PREDECESSOR_FILE) != self.predecessor_sha256
        ):
            raise AmendmentError("the first approved supplement changed after authorization")
        frozen = _frozen(contract)
        if NEW_FILES & set(frozen):
            raise AmendmentError("a new operational module already belongs to the base contract")
        for path, expected in frozen.items():
            actual = _file_sha(repo, path)
            if actual != expected and not self.permitted_change(path, expected, actual):
                raise AmendmentError(f"a file differs outside the approved supplement: {path}")
        original = contract["code"]["files_sha256"]
        for path, (before, after) in self.replacements.items():
            if original.get(path) != before or _file_sha(repo, path) != after:
                raise AmendmentError(
                    f"an approved replacement differs from its exact identity: {path}"
                )
        for path, expected in self.new_files.items():
            if _file_sha(repo, path) != expected:
                raise AmendmentError(
                    f"a new operational module differs from its approved hash: {path}"
                )


def load_authorized(
    path: Path, expected_sha256: str, base_contract_sha256: str, repo: Path
) -> Amendment:
    """Load only with an explicit approval hash, then verify every binding before use."""
    if not _hash(expected_sha256) or not _hash(base_contract_sha256):
        raise AmendmentError("explicit full approval and base-contract SHA-256 values are required")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise AmendmentError("the operational supplement is unavailable") from error
    digest = _sha(data)
    if digest != expected_sha256:
        raise AmendmentError(
            "the operational supplement does not have its explicitly approved hash"
        )
    value = _json(data)
    _shape(value)
    if value["base_contract_sha256"] != base_contract_sha256:
        raise AmendmentError("the operational supplement names another base contract")
    amendment = Amendment(
        base_contract_sha256,
        digest,
        MappingProxyType(
            {p: (v["before_sha256"], v["after_sha256"]) for p, v in value["replacements"].items()}
        ),
        MappingProxyType(dict(value["new_files"])),
        _chain(repo, value),
    )
    contract, _ = _base(repo, repo / BASE_FILE)
    amendment.validate(repo, contract)
    return amendment


_CURRENT: ContextVar[Amendment | None] = ContextVar(
    "holdout_v2_operational_amendment", default=None
)


def get_current() -> Amendment | None:
    return _CURRENT.get()


@contextmanager
def authorized(amendment: Amendment) -> Iterator[None]:
    """Bind an explicitly loaded supplement to this invocation, never to the environment."""
    if not isinstance(amendment, Amendment):
        raise AmendmentError("an explicitly loaded operational supplement is required")
    token = _CURRENT.set(amendment)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--repo", type=Path, default=Path("."))
    prepare.add_argument("--contract", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest(args.repo, args.contract)
        digest = write_exclusive(args.output, manifest)
    except (AmendmentError, OSError) as error:
        parser.exit(1, f"STOP: {error}\n")
    print(f"Prepared operational supplement SHA-256: {digest}")
    print("Not activated. Review the patch and explicitly authorize this hash before use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
