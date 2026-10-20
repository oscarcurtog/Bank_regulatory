"""The runtime identities of S3's model executions (Codex Gate 2 closure): Claude's effective
model as the runtime reports it, and the MLX runtime of the diagnostic. No model runs."""

from __future__ import annotations

import ast
import copy
import getpass
import json
import os
import platform
import re
import socket
import sys
import time

import pytest

from ingest import holdout as ho
from ingest import holdout_runtime as hrt
from tests.holdout_fakes import FAKE_EFFECTIVE, FAKE_MLX, session

EXPECTED = hrt.claude_effective(session(["P"], ["R"]))


def test_the_effective_model_is_read_from_every_place_the_runtime_reports_it():
    assert EXPECTED == FAKE_EFFECTIVE
    assert hrt.claude_problems(session(["P", "Q"], ["R", "S"]), FAKE_EFFECTIVE) == []


@pytest.mark.parametrize(
    ("part", "change"),
    [
        ("init", {**FAKE_EFFECTIVE["init"], "model": "other"}),
        ("init", {**FAKE_EFFECTIVE["init"], "fast_mode_state": "on"}),
        ("assistant_model", "other"),
        ("model_usage", {"m": {**FAKE_EFFECTIVE["model_usage"]["m"], "canonicalModel": "o"}}),
        ("model_usage", {"m": {**FAKE_EFFECTIVE["model_usage"]["m"], "provider": "bedrock"}}),
        ("model_attachment", {**FAKE_EFFECTIVE["model_attachment"], "modelId": "other"}),
    ],
)
def test_any_other_effective_model_is_a_mismatch(part, change):
    texts = session(["P"], ["R"], identity={**FAKE_EFFECTIVE, part: change})
    assert any(part in p for p in hrt.claude_problems(texts, FAKE_EFFECTIVE))


def test_an_alias_is_never_taken_for_the_canonical_model():
    alias = copy.deepcopy(FAKE_EFFECTIVE)
    alias["assistant_model"] = "m-latest"
    assert hrt.claude_problems(session(["P"], ["R"], identity=alias), FAKE_EFFECTIVE)


def test_a_session_that_does_not_report_its_model_is_a_mismatch():
    texts = session(["P"], ["R"])
    texts = {**texts, "stream": texts["stream"].splitlines()[0]}  # no result, no usage
    assert hrt.claude_effective(texts)["model_usage"] is None
    assert hrt.claude_problems(texts, FAKE_EFFECTIVE)


def test_the_generator_records_exactly_these_versions_under_a_runtime():
    assert hrt.generator_versions(FAKE_MLX) == {
        "mlx": "1.0.0",
        "mlx_lm": "1.0.0",
        "python": "3.12.0",
        "machine": "arm64",
        "model_id": "fake/qwen",
        "revision": "0" * 40,
        "max_new_tokens": 2048,
        "decoding": "greedy",
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("python", "version"), "3.13.0"),
        (("packages", "mlx"), "0.33.0"),
        (("packages", "mlx-lm"), "0.33.0"),
        (("packages", "tokenizers"), "0.24.0"),
        (("packages", "transformers"), "6.0.0"),
        (("model", "files", "model.safetensors"), "e" * 64),
        (("generation", "max_new_tokens"), 1024),
        (("host", "macos_product_version"), "27.1.0"),
        (("host", "macos_build_version"), "26B100"),
        (("host", "hardware_model"), "MacBookPro18,3"),  # another Mac, the same arm64
    ],
)
def test_an_output_of_another_runtime_is_never_admitted(path, value):
    other = copy.deepcopy(FAKE_MLX)
    place = other
    for key in path[:-1]:
        place = place[key]
    place[path[-1]] = value
    record = {"runtime": other, "versions": hrt.generator_versions(other)}
    problems = hrt.local_problems(record, FAKE_MLX)
    assert any(".".join(path) in p for p in problems), problems
    assert (
        hrt.local_problems(
            {"runtime": FAKE_MLX, "versions": hrt.generator_versions(FAKE_MLX)}, FAKE_MLX
        )
        == []
    )
    assert hrt.local_problems({"runtime": None, "versions": None}, FAKE_MLX)
    wrong = {"runtime": FAKE_MLX, "versions": {**hrt.generator_versions(FAKE_MLX), "mlx": "9"}}
    assert hrt.local_problems(wrong, FAKE_MLX) == [
        "the generator's own versions are not those of the frozen runtime"
    ]


def test_the_model_files_are_named_by_their_blobs_without_reading_them(tmp_path):
    snapshot = tmp_path / "models--org--m" / "snapshots" / "abc"
    blobs = tmp_path / "models--org--m" / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    for name, blob in (("config.json", "1" * 40), ("model.safetensors", "2" * 64)):
        (blobs / blob).write_text("x")
        os.symlink(f"../../blobs/{blob}", snapshot / name)
    (snapshot / "README.md").write_text("not a link")
    assert hrt._model_files("org/m", "abc", tmp_path) == {
        "config.json": "1" * 40,
        "model.safetensors": "2" * 64,
    }
    (snapshot / "tokenizer.json").write_text("a copy, not the hub's link")
    with pytest.raises(hrt.RuntimeIdentityError, match="not a cache link"):
        hrt._model_files("org/m", "abc", tmp_path)


def test_a_missing_mlx_environment_cannot_be_read(tmp_path):
    with pytest.raises(hrt.RuntimeIdentityError, match="MLX environment is missing"):
        hrt.mlx_runtime(python=tmp_path / "python")
    tree = ast.parse(hrt._PROBE)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imported == {"importlib.metadata", "json", "platform", "sys"}  # nothing heavy


@pytest.mark.skipif(not hrt.MLX_PYTHON.exists(), reason="needs .cache/mlx-venv (local only)")
def test_the_real_mlx_runtime_is_read_from_metadata_without_the_model():
    started = time.perf_counter()
    found = hrt.mlx_runtime()
    assert time.perf_counter() - started < 10  # 4.5 GB of weights are never read
    assert set(found["packages"]) == set(hrt.MLX_PACKAGES)
    assert found["host"] == hrt.host_identity()
    assert found["model"]["revision"] == json.loads(hrt.DIAGNOSTIC_CONFIG.read_text())["revision"]
    assert "model.safetensors" in found["model"]["files"]
    assert found == hrt.mlx_runtime()


@pytest.mark.requires_private_holdout_environment
def test_claudes_identity_comes_from_gate_1_and_the_broker_test_and_carries_no_account():
    found = hrt.claude_runtime(ho.HOLDOUT_DIR / "provenance")
    pin = json.loads((ho.HOLDOUT_DIR / "provenance" / "runtime.json").read_text())
    assert found["requested_model"] == pin["model"]
    assert found["effective"]["init"]["model"] == pin["model"]
    usage = found["effective"]["model_usage"][pin["model"]]
    assert usage["canonicalModel"] == pin["model"]
    text = json.dumps(found).lower()
    assert "email" not in text and "organization" not in text
    assert pin["organization_sha256"] not in text and pin["email_sha256"] not in text


@pytest.mark.skipif(sys.platform != "darwin", reason="the host identity is macOS's")
def test_the_host_identity_names_the_system_and_the_model_and_nothing_personal():
    found = hrt.host_identity()
    assert set(found) == set(hrt.HOST_FIELDS)
    assert found["architecture"] == platform.machine()
    assert found["macos_product_version"] == platform.mac_ver()[0]
    assert found["darwin_release"] == platform.release()
    assert re.fullmatch(r"[A-Za-z]+\d+,\d+", found["hardware_model"])
    personal = {socket.gethostname(), getpass.getuser(), os.environ.get("USER", "")}
    assert not any(p and p in v for p in personal for v in found.values())
    commands = " ".join(" ".join(c) for c in hrt.HOST_FIELDS.values())
    assert not any(w in commands for w in ("serial", "UUID", "uuid", "hostname", "ioreg"))


def test_a_host_that_cannot_be_read_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(hrt, "HOST_FIELDS", {"hardware_model": (str(tmp_path / "missing"),)})
    with pytest.raises(hrt.RuntimeIdentityError, match="hardware_model"):
        hrt.host_identity()


def test_a_runtime_identity_of_another_shape_names_what_differs():
    other = copy.deepcopy(FAKE_MLX)
    other["packages"]["Jinja2"] = "4.0.0"
    del other["generation"]["max_prompt_tokens"]
    assert hrt.runtime_differences(other, FAKE_MLX) == [
        "generation.max_prompt_tokens",
        "packages.Jinja2",
    ]
