"""Runtime image contract check (D-010, D-011).

Runs inside the runtime image, as the image's default user. The script is fed
through stdin, so the image is tested exactly as built, with no mounts and no
extra files:

    docker run --rm -i fundamento:ci python - < ci/check_runtime_image.py

Exits 1 if any invariant fails. It uses the standard library only, because the
runtime image has no test tooling and this check must not need any.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import sys
import tempfile

EXPECTED_PYTHON = "3.12.14"
EXPECTED_UID = 10001
DEV_TOOLS = ("pytest", "ruff", "mypy")
READ_ONLY_TREES = ("/app/ingest", "/app/scripts")
WRITABLE_DIRS = ("/app/data", "/app/.cache/tiktoken")


def find_tool(tool: str) -> str | None:
    """Where `tool` is installed, or None.

    Checks the installed distribution and an executable on PATH, which are the
    two ways the tool could be run: `python -m tool` and the console script.
    """
    try:
        return f"distribution {importlib.metadata.version(tool)}"
    except importlib.metadata.PackageNotFoundError:
        pass
    path = shutil.which(tool)
    return f"executable {path}" if path else None


def can_create_file(directory: str) -> bool:
    try:
        fd, path = tempfile.mkstemp(dir=directory)
    except OSError:
        return False
    os.close(fd)
    os.unlink(path)
    return True


def writable_paths(tree: str) -> list[str]:
    """Directories under `tree` where a file can be created, and files that
    can be opened for writing."""
    found = []
    for root, _dirs, files in os.walk(tree):
        if can_create_file(root):
            found.append(root)
        for name in files:
            path = os.path.join(root, name)
            try:
                with open(path, "a"):
                    found.append(path)
            except OSError:
                pass
    return found


def main() -> int:
    results: list[tuple[bool, str, str]] = []

    def check(ok: bool, name: str, detail: str) -> None:
        results.append((ok, name, detail))

    version = platform.python_version()
    check(version == EXPECTED_PYTHON, f"Python == {EXPECTED_PYTHON}", version)

    uid = os.getuid()
    check(uid == EXPECTED_UID, f"UID == {EXPECTED_UID} (not root)", f"uid {uid}")

    for tool in DEV_TOOLS:
        where = find_tool(tool)
        check(where is None, f"{tool} not installed", where or "absent")

    tests_present = os.path.exists("/app/tests")
    check(not tests_present, "/app/tests absent", "present" if tests_present else "absent")

    for tree in READ_ONLY_TREES:
        # A missing tree would make "nothing writable" pass vacuously.
        if not os.path.isdir(tree):
            check(False, f"{tree} not writable", "directory missing")
            continue
        writable = writable_paths(tree)
        detail = f"{len(writable)} writable: {', '.join(writable)}" if writable else "read-only"
        check(not writable, f"{tree} not writable", detail)

    for directory in WRITABLE_DIRS:
        ok = can_create_file(directory)
        check(ok, f"{directory} writable", "writable" if ok else "not writable")

    width = max(len(name) for _, name, _ in results)
    for ok, name, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = sum(not ok for ok, _, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} runtime invariants hold")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
