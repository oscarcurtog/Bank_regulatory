"""The CI evidence that tests/conftest.py prints as workflow commands in GitHub
Actions, which GitHub turns into check-run annotations anyone can read."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from tests.conftest import ci_annotations


def _skipped(reason: str) -> SimpleNamespace:
    return SimpleNamespace(longrepr=("tests/x.py", 3, f"Skipped: {reason}"))


def test_the_summary_counts_and_groups_the_skip_reasons():
    stats: dict[str, list[Any]] = {
        "passed": [object()] * 3,
        "skipped": [_skipped("data/raw/ is missing"), _skipped("data/raw/ is missing")],
    }
    assert ci_annotations(stats, "ir-measures 0.4.3; Linux x86_64") == [
        "::notice title=pytest::3 passed, 2 skipped, 0 failed, 0 errors, 0 warnings",
        "::notice title=skipped (2)::data/raw/ is missing",
        "::notice title=evaluator backend::ir-measures 0.4.3; Linux x86_64",
    ]


def test_a_multiline_message_stays_one_command():
    stats: dict[str, list[Any]] = {"warnings": [SimpleNamespace(message="first line\nsecond: 50%")]}
    assert ci_annotations(stats, "b")[-1] == (
        "::warning title=pytest warning::first line%0Asecond: 50%25"
    )
