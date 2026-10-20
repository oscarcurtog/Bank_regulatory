"""The Gate 2 closure conditions on the construction (Codex): a diagnostic is admitted only if
complete, and an output only if produced under the frozen runtimes, Claude's effective model
and the MLX runtime of the diagnostic. On the real grid and corpus with synthetic replies; no
model runs."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from ingest import holdout as ho
from ingest import holdout_judging as hj
from ingest import holdout_pipeline as hpl
from ingest import holdout_state as hs
from tests.holdout_fakes import (
    FAKE_EFFECTIVE,
    FAKE_MLX,
    FAKE_RUNTIME,
    Crash,
    Scenario,
    load_base,
    make,
    reopen,
)

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def base():
    return load_base()


@pytest.fixture(scope="module")
def first(base):
    return ho.cell_order(base["grid"])[0]


def outcome(pipeline: hpl.Pipeline) -> dict[str, Any]:
    store = pipeline.store
    return {
        "attempts": [(a["proposal_id"], a["need_id"], a["outcome"]) for a in store.attempts()],
        "ready": sorted((r["cell_id"], r["record_sha256"]) for r in store.ready()),
    }


@pytest.fixture(scope="module")
def reference(base, tmp_path_factory):
    pipeline, _ = make(base, tmp_path_factory.mktemp("reference") / "test")
    pipeline.build()
    found = outcome(pipeline)
    pipeline.store.close()
    return found


def failed_runs(pipeline: hpl.Pipeline, cell: str, stage: str) -> list[dict[str, Any]]:
    return [r for r in pipeline.store.runs(f"{cell}-a1", stage) if r["status"] == "failed"]


def check_of(pipeline: hpl.Pipeline, run_id: str) -> str | None:
    event = next(e for e in pipeline.store.events() if e.key == f"{run_id}/ended")
    found: str | None = event.payload["check"]
    return found


# ---------------------------------------------------------------------------
# 1. the diagnostic's completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "how",
    [
        "empty",
        "subset",
        "duplicate",
        "unknown",
        "invalid_grade",
        "missing_answerability",
        "invalid_answerability",
        "missing_ambiguous",
    ],
)
def test_an_incomplete_diagnostic_is_a_technical_retry_of_the_same_need(
    base, first, tmp_path, reference, how
):
    directory = tmp_path / "test"
    pipeline, executor = make(base, directory, Scenario(qwen_bad={first: how}))
    with pytest.raises(hpl.TechnicalStop):
        pipeline.build()
    assert pipeline.store.cells()[0]["state"] == "pending_technical"
    assert pipeline.store.stage(f"{first}-a1", "diagnostic") is None  # never admitted
    (run,) = failed_runs(pipeline, first, "diagnostic")
    assert check_of(pipeline, run["run_id"]) == "judge_answer_complete"
    roles = [role for role, cell, _ in executor.calls if cell == first]
    assert roles[-1] == "qwen_diagnostic" and "judge_d" not in roles  # no D from a partial
    attempts = [a for a in pipeline.store.attempts() if a["cell_id"] == first]
    assert [(a["number"], a["outcome"]) for a in attempts] == [(1, "accepted")]
    pipeline.store.close()
    resumed, _ = reopen(base, directory, Scenario())  # an explicit resume, a healthy answer
    resumed.resume()
    runs = resumed.store.runs(f"{first}-a1", "diagnostic")
    assert [r["status"] for r in runs] == ["failed", "admitted"]
    assert outcome(resumed) == reference  # the same need, not a replacement


def test_the_freeze_proves_every_admitted_diagnostic_complete(base, tmp_path, monkeypatch):
    cell = ho.cell_order(base["grid"])[0]
    pipeline, _ = make(base, tmp_path / "test", Scenario(qwen_bad={cell: "subset"}))
    monkeypatch.setattr(hj, "diagnostic_problems", lambda *a: [])  # a broken gate, for this test
    pipeline.build()
    monkeypatch.undo()
    problems = pipeline.freeze_problems()
    assert any("an admitted diagnostic is incomplete" in p and cell in p for p in problems)
    others = [p for p in problems if "diagnostic" not in p]
    assert others == []


# ---------------------------------------------------------------------------
# 2. Claude's effective model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"model_usage": {"m": {**FAKE_EFFECTIVE["model_usage"]["m"], "canonicalModel": "other"}}},
        {"assistant_model": "m-latest"},  # an alias is not the model
        {"init": {**FAKE_EFFECTIVE["init"], "model": "other"}},
        {"model_attachment": {**FAKE_EFFECTIVE["model_attachment"], "modelId": "other"}},
    ],
)
def test_a_session_on_another_claude_model_is_never_admitted(base, first, tmp_path, change):
    scenario = Scenario(model={(first, "judge_b"): change})
    pipeline, executor = make(base, tmp_path / "test", scenario)
    with pytest.raises(hpl.TechnicalStop, match="runtime_identity"):
        pipeline.build()
    assert pipeline.store.stage(f"{first}-a1", "judge_b") is None
    (run,) = failed_runs(pipeline, first, "judge_b")
    assert check_of(pipeline, run["run_id"]) == "model_identity"
    assert executor.calls[-1] == ("judge_b", first, 1)  # nothing consumed it


def test_judge_a_through_the_broker_is_checked_too(base, first, tmp_path):
    change = {"assistant_model": "other"}
    pipeline, _ = make(base, tmp_path / "test", Scenario(model={(first, "judge_a"): change}))
    with pytest.raises(hpl.TechnicalStop, match="runtime_identity"):
        pipeline.build()
    assert pipeline.store.stage(f"{first}-a1", "judge_a") is None


# ---------------------------------------------------------------------------
# 3. the MLX runtime of the diagnostic
# ---------------------------------------------------------------------------


def test_a_diagnostic_of_another_mlx_runtime_is_never_admitted(base, first, tmp_path):
    other = {"packages": {**FAKE_MLX["packages"], "mlx-lm": "0.33.0"}}
    pipeline, _ = make(base, tmp_path / "test", Scenario(local_runtime={first: other}))
    with pytest.raises(hpl.TechnicalStop, match="runtime_identity"):
        pipeline.build()
    assert pipeline.store.stage(f"{first}-a1", "diagnostic") is None
    (run,) = failed_runs(pipeline, first, "diagnostic")
    assert check_of(pipeline, run["run_id"]) == "runtime_identity"


@pytest.mark.parametrize(
    "other",
    [
        {"python": {**FAKE_MLX["python"], "version": "3.13.0"}},
        {"host": {**FAKE_MLX["host"], "macos_build_version": "26B100"}},
        {"host": {**FAKE_MLX["host"], "hardware_model": "MacBookPro18,3"}},
    ],
)
def test_a_recovered_spool_keeps_the_runtime_that_produced_it(
    base, first, tmp_path, reference, other
):
    """The diagnostic answered under another runtime or on another host, and the orchestrator
    died before recording it. Resume recovers it from the spool with that identity, and
    refuses it."""
    directory = tmp_path / "test"
    scenario = Scenario(die_after=(first, "qwen_diagnostic"), local_runtime={first: other})
    pipeline, _ = make(base, directory, scenario)
    with pytest.raises(Crash):
        pipeline.build()
    pipeline.store.close()
    resumed, executor = reopen(base, directory, Scenario())
    with pytest.raises(hpl.TechnicalStop, match="runtime_identity"):
        resumed.resume()
    assert len(executor.recovered) == 1
    assert "qwen_diagnostic" not in [role for role, _, _ in executor.calls]
    responded = [
        e for e in resumed.store.events() if e.kind == "run_responded" and e.stage == "diagnostic"
    ]
    assert [e.payload["recovered"] for e in responded if e.cell_id == first] == [True]
    assert resumed.store.stage(f"{first}-a1", "diagnostic") is None
    resumed.store.close()
    again, _ = reopen(base, directory, Scenario())
    again.resume()  # a new execution, under the frozen runtime
    runs = again.store.runs(f"{first}-a1", "diagnostic")
    assert [r["status"] for r in runs] == ["failed", "admitted"]
    assert outcome(again) == reference


def test_an_output_of_the_frozen_runtimes_is_admitted(base, first, tmp_path):
    pipeline, _ = make(base, tmp_path / "test")
    pipeline.build_cell(first)
    assert pipeline.store.stage(f"{first}-a1", "diagnostic") is not None
    assert pipeline.store.runtime() == FAKE_RUNTIME


# ---------------------------------------------------------------------------
# 4. the durable state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "path", "value"),
    [
        ("mlx_runtime", ("packages", "tokenizers"), "0.24.0"),
        ("mlx_runtime", ("python", "version"), "3.13.0"),
        ("mlx_runtime", ("host", "macos_product_version"), "27.1.0"),
        ("mlx_runtime", ("host", "macos_build_version"), "26B100"),
        ("mlx_runtime", ("host", "hardware_model"), "MacBookPro18,3"),
        ("claude_runtime", ("requested_model",), "other"),
        ("claude_runtime", ("effective", "assistant_model"), "other"),
    ],
)
def test_the_durable_state_refuses_a_resume_under_another_runtime(
    base, first, tmp_path, name, path, value
):
    directory = tmp_path / "test"
    pipeline, _ = make(base, directory, Scenario(technical={first: "judge_b"}))
    with pytest.raises(hpl.TechnicalStop):
        pipeline.build()
    pipeline.store.close()
    runtime = copy.deepcopy(FAKE_RUNTIME)
    place = runtime[name]
    for key in path[:-1]:
        place = place[key]
    place[path[-1]] = value
    resumed, executor = reopen(base, directory, Scenario(), runtime=runtime)
    with pytest.raises(hs.IncompatibleState, match=name) as refusal:
        resumed.resume()
    assert ".".join(path) in str(refusal.value)
    assert executor.calls == []
