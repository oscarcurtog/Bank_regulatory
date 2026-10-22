"""The Gate 2 closure conditions on the construction (Codex): a diagnostic is admitted only if
complete, and is otherwise unavailable, with a blind judge D in its place (after the first
START was aborted); and an output only if produced under the frozen runtimes, Claude's
effective model and the MLX runtime of the diagnostic. On the real grid and corpus with
synthetic replies; no model runs."""

from __future__ import annotations

import copy
import inspect
import json
import re
from typing import Any

import pytest

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_judging as hj
from ingest import holdout_packets as hp
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
# 1. the diagnostic's answer: admitted complete, or unavailable, never retried
# (diagnostic_admissible, after the first START was aborted on 2026-10-20)
# ---------------------------------------------------------------------------

# how the fake breaks the answer -> the classes of defect the stage records
DEFECTS = {
    "empty": ["missing_candidate"],
    "subset": ["missing_candidate"],
    "duplicate": ["duplicate_candidate"],
    "unknown": ["unknown_candidate"],
    "invalid_grade": ["invalid_grade"],
    "missing_answerability": ["invalid_answerability"],
    "invalid_answerability": ["invalid_answerability"],
    "missing_ambiguous": ["invalid_ambiguous"],
    "outer_array": ["not_an_object"],
    "not_json": ["not_json"],
}


def labels(pipeline: hpl.Pipeline, cell: str) -> dict[str, Any]:
    """What the need became: its final labels and its attempts, not how it was checked."""
    ready = next(r for r in pipeline.ready.values() if r.cell_id == cell)
    attempts = [a for a in pipeline.store.attempts() if a["cell_id"] == cell]
    return {
        "statement": ready.need.statement,
        "answerability": ready.need.answerability,
        "qrels": [q.model_dump(mode="json") for q in ready.qrels],
        "adjudications": [a.model_dump(mode="json") for a in ready.adjudications],
        "attempts": [(a["number"], a["outcome"]) for a in attempts],
    }


def items_of(packet: str) -> list[str]:
    """What judge D is asked to grade, by the targets its items' ids name in its packet."""
    section = packet.split("ITEMS TO GRADE", 2)[-1].split("ALL THE CANDIDATES", 1)[0]
    node_of = dict(
        re.findall(r"^CANDIDATE (C-[0-9a-f]{12})\n\[((?:DORA|MiCA|PSD2):[^\]]+)\]", packet, re.M)
    )
    return [
        bm.ANSWERABILITY_TARGET if text.startswith("the answerability") else node_of[i]
        for i, text in re.findall(r"^ITEM (C-[0-9a-f]{12}): (.*)$", section, re.M)
    ]


@pytest.fixture(scope="module")
def healthy(base, first, tmp_path_factory):
    """The first cell with a complete diagnostic: its labels, and its candidates."""
    pipeline, executor = make(base, tmp_path_factory.mktemp("healthy") / "test")
    pipeline.build_cell(first)
    adjudication = pipeline.store.stage(f"{first}-a1", "adjudication")
    assert adjudication is not None
    candidates = [
        a["target"]
        for a in adjudication["value"]["adjudications"]
        if a["target"] != bm.ANSWERABILITY_TARGET
    ]
    found = {
        "labels": labels(pipeline, first),
        "candidates": candidates,
        "roles": [role for role, cell, _ in executor.calls if cell == first],
    }
    pipeline.store.close()
    return found


def test_a_complete_diagnostic_keeps_the_frozen_behaviour(healthy):
    assert "qwen_diagnostic" in healthy["roles"] and "judge_d" not in healthy["roles"]
    assert healthy["labels"]["attempts"] == [(1, "accepted")]


@pytest.mark.parametrize("how", list(DEFECTS))
def test_an_inadmissible_diagnostic_is_unavailable_and_judge_d_grades_everything(
    base, first, tmp_path, healthy, how
):
    pipeline, executor = make(base, tmp_path / "test", Scenario(qwen_bad={first: how}))
    pipeline.build_cell(first)  # no technical stop: executing it again would repeat it
    pid = f"{first}-a1"
    stage = pipeline.store.stage(pid, "diagnostic")
    assert stage is not None
    value = stage["value"]
    assert value == hj.unavailable_diagnostic(DEFECTS[how]) and hj.diagnostic_unavailable(value)
    (run,) = pipeline.store.runs(pid, "diagnostic")  # one execution, admitted as a run
    assert run["status"] == "admitted" and json.loads(run["audit"])["ok"] is True
    kept = json.loads(pipeline.store.text(json.loads(run["response"])["raw"], "raw"))
    assert kept["record"]["output"]  # the answer stays with its run, never read again
    roles = [role for role, cell, _ in executor.calls if cell == first]
    assert roles.count("qwen_diagnostic") == 1 and roles.count("judge_d") == 1
    d_packet = executor.packets[("judge_d", first, 1)]
    assert items_of(d_packet) == [*healthy["candidates"], bm.ANSWERABILITY_TARGET]
    log = pipeline.diagnostic_log[-1]
    assert (log["status"], log["defects"], log["discrepancies"]) == (
        "unavailable",
        DEFECTS[how],
        [],
    )
    assert set(log["judge_d"].values()) == {"consensus_held"}
    # no label, no vote and no replacement from it: the same need, with the same labels
    assert labels(pipeline, first) == healthy["labels"]


def test_judge_d_in_place_of_the_diagnostic_learns_nothing_of_it(base, first, tmp_path):
    """The same packet whatever the answer lacked, built by the frozen builder that takes no
    label of anyone: no Qwen output, no failed field, no vote of A, B or C."""
    packets = []
    for how in ("outer_array", "subset", "not_json", "invalid_answerability"):
        pipeline, executor = make(base, tmp_path / how / "test", Scenario(qwen_bad={first: how}))
        pipeline.build_cell(first)
        packets.append(executor.packets[("judge_d", first, 1)])
        pipeline.store.close()
    assert all(p == packets[0] for p in packets)
    assert "Qwen" not in packets[0] and "not_an_object" not in packets[0]
    parameters = list(inspect.signature(hp.judge_d_packet).parameters)
    assert parameters == [
        "corpus",
        "need",
        "candidates",
        "scope",
        "targets",
        "ids",
        "opening",
        "ref",
    ]


def test_a_judge_d_that_does_not_hold_rejects_by_the_rule_of_a_discrepancy(base, first, tmp_path):
    scenario = Scenario(qwen_bad={first: "outer_array"}, d_answer={first: {"ambiguous": True}})
    pipeline, _ = make(base, tmp_path / "test", scenario)
    pipeline.build_cell(first)
    attempts = [a for a in pipeline.store.attempts() if a["cell_id"] == first]
    # the first attempt's D does not hold: uncertain, a new blind attempt; the second has a
    # complete diagnostic with nothing to send to D
    assert [(a["number"], a["outcome"]) for a in attempts] == [
        (1, "adjudication_uncertain"),
        (2, "accepted"),
    ]


def test_a_crash_of_the_diagnostic_is_still_technical(base, first, tmp_path, reference):
    directory = tmp_path / "test"
    scenario = Scenario(technical={first: "qwen_diagnostic"}, technical_times=1)
    pipeline, executor = make(base, directory, scenario)
    with pytest.raises(hpl.TechnicalStop):
        pipeline.build()
    pid = f"{first}-a1"
    assert pipeline.store.stage(pid, "diagnostic") is None
    (run,) = failed_runs(pipeline, first, "diagnostic")
    assert check_of(pipeline, run["run_id"]) != "diagnostic_admissible"
    assert "judge_d" not in [role for role, cell, _ in executor.calls if cell == first]
    pipeline.store.close()
    resumed, _ = reopen(base, directory, Scenario())  # the same need, executed again
    resumed.resume()
    assert [r["status"] for r in resumed.store.runs(pid, "diagnostic")] == ["failed", "admitted"]
    assert outcome(resumed) == reference


def test_the_report_and_the_freeze_record_the_diagnostics_availability(base, tmp_path):
    cell = ho.cell_order(base["grid"])[0]
    pipeline, _ = make(base, tmp_path / "test", Scenario(qwen_bad={cell: "outer_array"}))
    pipeline.build()
    diagnostic = pipeline.report()["diagnostic"]
    assert (diagnostic["unavailable"], diagnostic["available"]) == (1, diagnostic["needs"] - 1)
    assert diagnostic["unavailable_defects"] == {"not_an_object": 1}
    held = diagnostic["judge_d_in_place_of_the_diagnostic"]
    assert set(held) == {"consensus_held"} and held["consensus_held"] >= 2
    assert pipeline.freeze_problems() == []


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


def test_a_search_through_the_broker_is_checked_too(base, first, tmp_path):
    change = {"assistant_model": "other"}
    pipeline, _ = make(base, tmp_path / "test", Scenario(model={(first, "searcher_1"): change}))
    with pytest.raises(hpl.TechnicalStop, match="runtime_identity"):
        pipeline.build()
    assert pipeline.store.stage(f"{first}-a1", "search_1") is None


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
