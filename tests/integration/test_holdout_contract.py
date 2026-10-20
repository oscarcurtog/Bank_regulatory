"""The pre-authoring evaluation contract (Codex Gate 2, P1.5): built from the files that define
each value, and immutable once pinned. No system runs."""

from __future__ import annotations

import ast
import inspect
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_contract as hc
from ingest import holdout_isolation as hi
from ingest import holdout_pipeline as hpl
from ingest import holdout_roles as hr
from ingest import holdout_state as hs
from tests.holdout_fakes import FAKE_RUNTIME


def _history_reaches_e() -> bool:
    found = subprocess.run(
        ["git", "cat-file", "-e", f"{hc.E_RESULT_COMMIT}^{{commit}}"], capture_output=True
    )
    return found.returncode == 0


@pytest.fixture(scope="module")
def contract(tmp_path_factory):
    """Every section built from its source, but the code record, which needs the history back
    to E's DEV result that a shallow clone (CI's) lacks: the pinned contract's, which
    test_the_code_record_is_the_history_since_e_result builds again where that history is."""
    grid = ho.build_grid(bm.load_benchmark(bm.DEV_DIR), ho.load_areas())
    folder = tmp_path_factory.mktemp("contract")
    grid_path = folder / "grid.json"
    grid_path.write_text(json.dumps(grid.model_dump(mode="json")), encoding="utf-8")
    runtime = FAKE_RUNTIME
    record = hc.construction(
        grid_path,
        ho.plan_record(grid),
        hr.contracts_record(),
        runtime["claude_runtime"],
        runtime["mlx_runtime"],
    )
    pinned = json.loads(hc.CONTRACT_PATH.read_text(encoding="utf-8"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hc, "code_identity", lambda repo: pinned["code"])
        return hc.build(construction_record=record)


@pytest.mark.skipif(
    not _history_reaches_e(),
    reason=f"the history does not reach E's DEV result {hc.E_RESULT_COMMIT}: a shallow clone",
)
def test_the_code_record_is_the_history_since_e_result():
    found = hc.code_identity(Path("."))
    pinned = json.loads(hc.CONTRACT_PATH.read_text(encoding="utf-8"))["code"]
    assert set(found["changes_since_e_result"]) <= set(hc.CHANGES_SINCE_E)
    del found["built_on_commit"], pinned["built_on_commit"]  # provenance only
    assert found == pinned


def test_the_values_come_from_their_sources(contract):
    e5 = json.loads(open(hc.CONFIG_FILES["E5"], encoding="utf-8").read())
    assert contract["systems"]["E5"]["config"] == e5
    assert contract["systems"]["BM25"]["config_hash"].startswith("e086bdb4")
    e = contract["systems"]["E"]
    assert (e["assembler"]["K"], e["assembler"]["keep"], e["assembler"]["depth"]) == (5, 6, 10)
    assert e["requirement_retrieval"]["depth"] == 50 and e["decomposer"]["excerpts"] == 3
    assert e["decomposer"]["max_new_tokens"] == 256
    assert contract["analysis"]["bootstrap"]["seed"] == 20261013
    assert contract["analysis"]["bootstrap"]["resamples"] == 10_000
    assert set(contract["code"]["changes_since_e_result"]) <= set(hc.CHANGES_SINCE_E)
    assert contract["construction"]["broker"] == {
        "escalation_at": 16,
        "ceiling": 32,
        "max_payload_characters": 8000,
        "violations_forgiven": 2,
    }
    construction = contract["construction"]
    assert construction["technical_executions_per_stage"] == 3
    assert construction["operational_checkpoints"] == [20, 40, 60]
    assert construction["state_schema"] == hs.SCHEMA
    assert "ingest/holdout_state.py" in construction["code_sha256"]
    roles = hr.contracts_record()
    assert {r["authentication"] for n, r in roles.items() if n != "qwen_diagnostic"} == {
        hr.KEYCHAIN
    }
    assert roles["qwen_diagnostic"]["authentication"] == "none"


def test_the_frozen_identities_name_everything_a_resume_checks():
    grid = ho.build_grid(bm.load_benchmark(bm.DEV_DIR), ho.load_areas())
    binary = {"version": "2.1.301 (Claude Code)", "sha256": "f" * 64}
    terms = hi.forbidden_terms(None, None, [])
    claude, mlx = FAKE_RUNTIME["claude_runtime"], FAKE_RUNTIME["mlx_runtime"]
    here = Path(".")

    def identities(**changes):
        given = {"binary": binary, "terms": terms, "claude": claude, "mlx": mlx} | changes
        return hpl.frozen_identities(
            "c" * 64,
            hc.PLAN_PATH,
            grid,
            given["binary"],
            given["terms"],
            here,
            given["claude"],
            given["mlx"],
        )

    found = identities()
    assert set(found) == {
        "pre_authoring_contract",
        "analysis_plan",
        "systems",
        "role_contracts",
        "templates",
        "order",
        "omission_audit",
        "corpus",
        "binary",
        "audit",
        "construction_code",
        "claude_runtime",
        "mlx_runtime",
    }
    assert found == identities()
    other = identities(binary={**binary, "sha256": "0"})
    assert [k for k in found if found[k] != other[k]] == ["binary"]
    fewer = identities(terms=terms[1:])
    assert [k for k in found if found[k] != fewer[k]] == ["audit"]
    model = identities(claude={**claude, "requested_model": "other"})
    assert [k for k in found if found[k] != model[k]] == ["claude_runtime"]
    packages = {**mlx["packages"], "mlx": "9.9.9"}
    runtime = identities(mlx={**mlx, "packages": packages})
    assert [k for k in found if found[k] != runtime[k]] == ["mlx_runtime"]


def test_the_contract_carries_every_new_identity_and_no_account(contract):
    construction = contract["construction"]
    claude, mlx = construction["claude_runtime"], construction["mlx_runtime"]
    assert set(claude) == {"requested_model", "binary", "effective", "policy"}
    effective = {"init", "assistant_model", "model_usage", "model_attachment"}
    assert set(claude["effective"]) == effective
    assert set(mlx) == {"python", "machine", "host", "packages", "model", "generation"}
    assert set(mlx["host"]) == {
        "macos_product_version",
        "macos_build_version",
        "darwin_release",
        "hardware_model",
        "architecture",
    }
    assert "ingest/holdout_runtime.py" in construction["code_sha256"]
    checks = construction["checks"]["checks"]
    assert checks["model_identity"]["kind"] == "technical_integrity"
    assert list(checks["runtime_identity"]["stages"]) == ["diagnostic"]
    assert "diagnostic" not in checks["judge_answer_complete"]["stages"]
    assert "diagnostic" not in checks["reply_is_json"]["stages"]
    admissible = checks["diagnostic_admissible"]
    assert (admissible["kind"], admissible["retriable"]) == ("diagnostic_only", False)
    assert "diagnostic_defects" in admissible["rule"] and "not agreement" in admissible["rule"]
    # the construction this contract replaces, aborted with no ready need, and why
    (aborted,) = contract["aborted_runs"]
    assert aborted["status"] == "ABORTED_PROTOCOL_DEFECT_QWEN_DIAGNOSTIC"
    assert aborted["contract_sha256"].startswith("fe1891755127") and aborted["ready_needs"] == 0
    assert aborted["public_commit"] == "3226afad8fdb8f424b6486f540312e31cb16278a"
    with pytest.raises(hc.ContractError, match="account"):
        hc.construction(
            Path("config/holdout/s4-analysis-plan.md"),
            {},
            {},
            {**claude, "organization_sha256": "x"},
            mlx,
        )


def test_a_pinned_contract_cannot_change(contract, tmp_path):
    path = tmp_path / "contract.json"
    digest = hc.write(contract, path)
    assert hc.verify(digest, path, check_corpus=False)["schema"] == hc.SCHEMA
    with pytest.raises(hc.ContractError, match="written once"):
        hc.write(contract, path)
    edited = json.loads(path.read_text())
    edited["systems"]["BM25"]["config"]["k1"] = 1.5
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(hc.ContractError, match="changed since it was pinned"):
        hc.verify(digest, path, check_corpus=False)


def test_a_code_file_that_no_longer_matches_stops_s4(contract, tmp_path):
    forged = json.loads(json.dumps(contract))
    forged["code"]["files_sha256"]["ingest/decomposition.py"] = "0" * 64
    path = tmp_path / "contract.json"
    digest = hc.write(forged, path)
    with pytest.raises(hc.ContractError, match="files differ"):
        hc.verify(digest, path, check_corpus=False)


def test_an_open_plan_cannot_be_frozen(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("# plan\n\nThe combination of metrics is OPEN.\n", encoding="utf-8")
    with pytest.raises(hc.ContractError, match="OPEN"):
        hc.analysis(plan)


def test_the_real_plan_has_nothing_open():
    text = hc.PLAN_PATH.read_text(encoding="utf-8")
    assert "OPEN" not in text and "DRAFT" not in text


@pytest.mark.corpus
def test_the_corpus_on_disk_matches_the_contract(contract):
    assert hc.corpus_problems(contract) == []
    forged = json.loads(json.dumps(contract))
    forged["corpus"]["layer"]["content_sha256"] = "0" * 64
    forged["corpus"]["acts"]["DORA"]["source_sha256"] = "0" * 64
    problems = hc.corpus_problems(forged)
    assert (
        "layer content_sha256 differs" in problems and "DORA: not the contract's source" in problems
    )


def _ingest_imports(path: str) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module == "ingest":
            found |= {f"ingest/{a.name}.py" for a in node.names}
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ingest."):
            found.add("ingest/" + str(node.module).split(".", 1)[1] + ".py")
    return found


def test_every_module_the_construction_runs_is_frozen_by_content():
    pinned = set(hc.CODE_FILES) | set(hc.CONSTRUCTION_FILES)
    seen: set[str] = set()
    todo = list(hc.CONSTRUCTION_FILES)
    while todo:
        f = todo.pop()
        if f not in seen and Path(f).exists():
            seen.add(f)
            todo += sorted(_ingest_imports(f))
    assert seen <= pinned, sorted(seen - pinned)
    for f in (
        "ingest/holdout_launcher.py",
        "ingest/holdout_checks.py",
        "ingest/holdout_spool.py",
    ):
        assert f in hc.CONSTRUCTION_FILES
    assert "data/benchmark/PROTOCOL.md" in hc.CONSTRUCTION_INPUTS
    assert "config/retrieval/qwen3-8b-mlx.json" in hc.CONSTRUCTION_INPUTS


def test_the_contract_freezes_content_and_records_commits_only_as_provenance(contract, tmp_path):
    files = hc.frozen_files(contract)
    assert set(contract["construction"]["code_sha256"]) <= set(files)
    assert set(contract["construction"]["template_files"].values()) <= set(files)
    assert contract["analysis"]["plan_file"] in files
    assert {"built_on_commit", "e_result_commit"} <= set(contract["code"])
    assert "head" not in contract["code"]
    # the functions that decide validity name no commit, no git, no HEAD
    for function in (hc.verify, hc.frozen_files):
        body = inspect.getsource(function)
        assert body.startswith("def ") and len(body.splitlines()) > 5  # really the function
        assert not any(w in body for w in ("HEAD", "rev-parse", "_git(", "subprocess", "git "))


def test_verification_is_content_based_and_never_reads_git(contract, tmp_path, monkeypatch):
    """P2 (Codex): the contract verifies in a directory that is not a git repository at all,
    with every way of reaching git made to fail; and it stops as soon as one frozen byte
    changes. The commit checked out is never a condition of validity."""
    copy = tmp_path / "content-only"
    for f in hc.frozen_files(contract):
        (copy / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, copy / f)
    path = copy / "contract.json"
    digest = hc.write(contract, path)
    assert not (copy / ".git").exists()

    def no_git(*args, **kwargs):
        raise AssertionError("verify() reached git")

    monkeypatch.setattr(hc, "_git", no_git)
    monkeypatch.setattr(hc.subprocess, "run", no_git)
    assert hc.verify(digest, path, copy, check_corpus=False)["schema"] == hc.SCHEMA
    frozen = copy / "ingest" / "holdout_ledger.py"
    frozen.write_bytes(frozen.read_bytes() + b"\n")
    with pytest.raises(hc.ContractError, match="holdout_ledger"):
        hc.verify(digest, path, copy, check_corpus=False)
