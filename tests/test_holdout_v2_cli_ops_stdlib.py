"""Real CLI dispatch functions with inert dependencies: no repo imports, models or TEST data.

The public script's main, validate, construct and approval wrapper are compiled directly
from their AST.  Only preflight/actions are replaced; argument parsing and authorization
control flow are the implementation under review.
"""

from __future__ import annotations
import __future__

import argparse
import ast
import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

BASE_SHA = "a" * 64
APPROVAL_SHA = "b" * 64


class FakeError(Exception):
    pass


class FakePolicy:
    def __init__(self, interval: float, cap: int | None, window: float):
        self.min_interval_seconds = interval
        self.max_starts = cap
        self.window_seconds = window


class CliOperationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[tuple] = []
        self.active: Any = None
        self.approved = SimpleNamespace(sha256=APPROVAL_SHA)
        script = Path(__file__).resolve().parents[1] / "scripts" / "holdout_v2.py"
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        names = {"main", "_with_supplement", "validate", "construct"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in nodes}, names)

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("dry CLI verification reached a model, state or world constructor")

        def load(path: Path, expected: str, base: str, repo: Path) -> Any:
            self.calls.append(("load", path, expected, base, repo))
            if expected != APPROVAL_SHA or base != BASE_SHA:
                raise FakeError("unapproved supplement")
            return self.approved

        @contextlib.contextmanager
        def authorized(value: Any):
            self.assertIs(value, self.approved)
            old = self.active
            self.active = value
            self.calls.append(("enter",))
            try:
                yield
            finally:
                self.active = old
                self.calls.append(("exit",))

        def preflight(pinned: str, repo: Path) -> tuple:
            self.calls.append(("preflight", pinned, self.active))
            return {"schema": "synthetic-contract"}, None, []

        def public(*args: Any) -> list:
            self.calls.append(("public", self.active))
            return []

        def evaluate(contract: str) -> int:
            self.calls.append(("evaluate", contract, self.active))
            return 0

        def audit(labels: Path, contract: str) -> int:
            self.calls.append(("audit", labels, contract, self.active))
            return 0

        def construct(action: str, contract: str, workers: int, policy: Any) -> int:
            self.calls.append(("construct", action, contract, workers, policy, self.active))
            return 0

        fake_contract = SimpleNamespace(
            problems=preflight,
            public_commit_problems=public,
            NAMESPACE=Path("synthetic-private-namespace"),
            ContractV2Error=FakeError,
        )
        self.ns: dict[str, Any] = {
            "__doc__": "synthetic execution of actual CLI functions",
            "argparse": argparse,
            "Path": Path,
            "REPO": Path("synthetic-repo"),
            "oa": SimpleNamespace(
                load_authorized=load, authorized=authorized, AmendmentError=FakeError
            ),
            "ops": SimpleNamespace(Policy=FakePolicy, OperationsError=FakeError),
            "ho": SimpleNamespace(HoldoutError=FakeError),
            "hcert": SimpleNamespace(CertificationError=FakeError),
            "hc2": fake_contract,
            "vs": SimpleNamespace(StateError=FakeError),
            "_pinned": lambda value: value,
            "_print": lambda value: self.calls.append(("print", value)),
            "_test_world": forbidden,
            "_store": forbidden,
            "_executor_and_audit": forbidden,
            "evaluate": evaluate,
            "audit_sa7": audit,
            "_construct": construct,
        }
        body: list[ast.stmt] = [n for n in nodes]
        module = ast.Module(body=body, type_ignores=[])
        exec(
            compile(module, str(script), "exec", flags=__future__.annotations.compiler_flag),
            self.ns,
        )

    def invoke(self, *args: str) -> int:
        with contextlib.redirect_stdout(io.StringIO()):
            return cast(int, self.ns["main"](list(args)))

    @staticmethod
    def approved_args() -> tuple[str, ...]:
        return (
            "--operations-manifest",
            "reviewed.json",
            "--operations-sha256",
            APPROVAL_SHA,
        )

    def actions(self, kind: str) -> list[tuple]:
        return [entry for entry in self.calls if entry[0] == kind]

    def test_validate_authorizes_preflight_and_does_not_construct_a_world(self) -> None:
        result = self.invoke("validate", "--contract", BASE_SHA, *self.approved_args())
        self.assertEqual(result, 0)
        self.assertEqual(len(self.actions("load")), 1)
        self.assertEqual(self.actions("preflight"), [("preflight", BASE_SHA, self.approved)])
        self.assertEqual(self.actions("public"), [("public", self.approved)])
        self.assertFalse(self.actions("construct"))
        self.assertFalse(self.actions("evaluate"))
        self.assertIsNone(self.active)

    def test_evaluate_receives_authorization_and_keeps_base_pin(self) -> None:
        result = self.invoke("evaluate", "--contract", BASE_SHA, *self.approved_args())
        self.assertEqual(result, 0)
        self.assertEqual(self.actions("evaluate"), [("evaluate", BASE_SHA, self.approved)])
        self.assertFalse(self.actions("construct"))
        self.assertIsNone(self.active)

    def test_audit_sa7_receives_labels_and_authorization(self) -> None:
        result = self.invoke(
            "audit-sa7", "synthetic-labels.json", "--contract", BASE_SHA, *self.approved_args()
        )
        self.assertEqual(result, 0)
        self.assertEqual(
            self.actions("audit"),
            [("audit", Path("synthetic-labels.json"), BASE_SHA, self.approved)],
        )
        self.assertIsNone(self.active)

    def test_original_validate_is_not_silently_authorized(self) -> None:
        self.assertEqual(self.invoke("validate", "--contract", BASE_SHA), 0)
        self.assertFalse(self.actions("load"))
        self.assertEqual(self.actions("preflight"), [("preflight", BASE_SHA, None)])

    def test_half_an_approval_is_rejected_for_all_three_routes(self) -> None:
        for command in ("validate", "evaluate", "audit-sa7"):
            for flag, value in (
                ("--operations-manifest", "reviewed.json"),
                ("--operations-sha256", APPROVAL_SHA),
            ):
                self.calls.clear()
                argv = [command]
                if command == "audit-sa7":
                    argv.append("synthetic-labels.json")
                argv += ["--contract", BASE_SHA, flag, value]
                with self.subTest(command=command, flag=flag):
                    self.assertEqual(self.invoke(*argv), 1)
                    self.assertFalse(self.actions("load"))
                    self.assertFalse(self.actions("preflight"))
                    self.assertFalse(self.actions("evaluate"))
                    self.assertFalse(self.actions("audit"))
                    self.assertIsNone(self.active)

    def test_wrong_approval_never_reaches_the_action(self) -> None:
        self.assertEqual(
            self.invoke(
                "evaluate",
                "--contract",
                BASE_SHA,
                "--operations-manifest",
                "reviewed.json",
                "--operations-sha256",
                "c" * 64,
            ),
            1,
        )
        self.assertFalse(self.actions("evaluate"))
        self.assertFalse(self.actions("enter"))

    def test_authorization_is_removed_when_action_fails(self) -> None:
        def failed(contract: str) -> int:
            self.assertIs(self.active, self.approved)
            raise FakeError("synthetic failure")

        self.ns["evaluate"] = failed
        self.assertEqual(self.invoke("evaluate", "--contract", BASE_SHA, *self.approved_args()), 1)
        self.assertEqual(len(self.actions("exit")), 1)
        self.assertIsNone(self.active)

    def test_resume_parses_and_passes_operational_parameters(self) -> None:
        result = self.invoke(
            "resume",
            "--contract",
            BASE_SHA,
            "--workers",
            "1",
            *self.approved_args(),
            "--min-start-interval-seconds",
            "300",
            "--max-executions-per-window",
            "120",
            "--execution-window-seconds",
            "86400",
        )
        self.assertEqual(result, 0)
        self.assertEqual(len(self.actions("construct")), 1)
        call = self.actions("construct")[0]
        self.assertEqual(call[:4], ("construct", "resume", BASE_SHA, 1))
        policy = call[4]
        self.assertEqual(policy.min_interval_seconds, 300)
        self.assertEqual(policy.max_starts, 120)
        self.assertEqual(policy.window_seconds, 86400)
        self.assertIs(call[5], self.approved)
        self.assertIsNone(self.active)

    def test_resume_defaults_to_300_seconds_only_after_explicit_approval(self) -> None:
        self.assertEqual(
            self.invoke("resume", "--contract", BASE_SHA, "--workers", "1", *self.approved_args()),
            0,
        )
        policy = self.actions("construct")[0][4]
        self.assertEqual(policy.min_interval_seconds, 300)
        self.assertIsNone(policy.max_starts)

    def test_resume_does_not_accept_pacing_without_an_approved_supplement(self) -> None:
        self.assertEqual(
            self.invoke(
                "resume",
                "--contract",
                BASE_SHA,
                "--workers",
                "1",
                "--min-start-interval-seconds",
                "300",
            ),
            1,
        )
        self.assertFalse(self.actions("construct"))
        self.assertFalse(self.actions("load"))

    def test_resume_rejects_manifest_without_approval_sha(self) -> None:
        self.assertEqual(
            self.invoke(
                "resume",
                "--contract",
                BASE_SHA,
                "--workers",
                "1",
                "--operations-manifest",
                "reviewed.json",
            ),
            1,
        )
        self.assertFalse(self.actions("construct"))
        self.assertFalse(self.actions("load"))

    def test_original_resume_has_no_pacing_or_authorization(self) -> None:
        self.assertEqual(self.invoke("resume", "--contract", BASE_SHA, "--workers", "2"), 0)
        self.assertEqual(
            self.actions("construct"), [("construct", "resume", BASE_SHA, 2, None, None)]
        )
        self.assertFalse(self.actions("load"))

    def test_start_cannot_activate_operational_supplement(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            self.invoke("start", "--contract", BASE_SHA, "--workers", "1", *self.approved_args())
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.actions("construct"))

    def test_nonpositive_workers_are_rejected_before_opening_state(self) -> None:
        self.assertEqual(self.invoke("resume", "--contract", BASE_SHA, "--workers", "0"), 1)
        self.assertFalse(self.actions("construct"))
        self.assertFalse(self.actions("load"))


if __name__ == "__main__":
    unittest.main()
