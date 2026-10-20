"""The runtime identities of S3's model executions (D-055, Codex Gate 2 closure P1s).

Two model runtimes take part in TEST's construction, and each is frozen in the pre-authoring
contract, kept in the durable state, and checked before START, before every RESUME and before
any of their outputs is admitted.

CLAUDE (authors, writers, gatekeepers, judges A to D, the second searcher, the metadata
adjudicator). Its identity has four parts:
- the requested model, the --model of every command (provenance/runtime.json);
- the binary, its version and SHA-256 (the one Gate 1 passed with);
- the static policy: the command's flags, the environment allowlist and the built-in plugins
  (ingest/holdout_isolation.py, byte for byte Gate 1's module);
- the EFFECTIVE model, as the authenticated runtime itself reports it in every session: the
  stream's init event (model and the settings that select or shape it), the model of every
  assistant message, every result's model usage with its canonical model and provider, and
  the transcript's model attachment. The expected values are read from Gate 1's own session
  (run 4) and must equal those of the broker's synthetic test: nothing here is typed in.
  No alias is taken for another: every value is compared exactly.
The account's email and organization never enter this identity.

QWEN3-8B through MLX (the diagnostic judge). Its identity is the runtime that produces its
output, read from the installed packages' metadata by the MLX environment's own Python,
without importing or loading anything heavy:
- the HOST: the macOS product and build versions (the build stands for the Metal and system
  runtime the OS provides), the Darwin release, the Mac's model identifier and the
  architecture. Nothing personal: no serial number, hardware UUID, hostname or user. A partly
  built TEST therefore never moves to another macOS build or another Mac model: macOS is not
  updated during S3;
- the Python version and build, and the machine;
- the packages on the path from prompt to text (MLX_PACKAGES, each with the reason);
- the model's files, by the content-addressed blob each file of the pinned snapshot points to;
- the generation parameters (the token limit, greedy decoding, the template's arguments).
An output keeps the identity it was produced under, written into its spool before the run, and
is admitted only if that identity is the frozen one.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ingest import holdout_isolation as hi

MLX_PYTHON = Path(".cache/mlx-venv/bin/python")
DIAGNOSTIC_CONFIG = Path("config/retrieval/qwen3-8b-mlx.json")
HF_CACHE = Path(".cache/huggingface/hub")
# Room for judge B's JSON over a long candidate list; a cut answer is a malformed session,
# a technical failure, never a judgment.
DIAGNOSTIC_MAX_NEW_TOKENS = 2048
# The packages whose code runs between the diagnostic's prompt and its text, found by tracing
# scripts/mlx_generate.py's imports and its path (D-055); the others the process imports
# (networking, progress bars, numpy, sentencepiece for SentencePiece models) do not touch it.
MLX_PACKAGES = {
    "mlx": "the array computation of every layer",
    "mlx-metal": "the GPU kernels mlx runs on Apple silicon",
    "mlx-lm": "the Qwen3 model code, the generation loop, greedy sampling and the prompt cache",
    "transformers": "the tokenizer class (Qwen2Tokenizer) and the chat template's rendering",
    "tokenizers": "the fast BPE tokenizer behind it, which encodes the prompt and decodes text",
    "Jinja2": "the engine that renders the chat template",
    "MarkupSafe": "the string type Jinja2 renders with",
    "huggingface_hub": "the resolution of the pinned snapshot's files",
}
# What the authenticated runtime reports in its init event that selects or shapes the model
INIT_KEYS = (
    "model",
    "fast_mode_state",
    "per_turn_effort_active",
    "permissionMode",
    "output_style",
    "apiKeySource",
)
USAGE_KEYS = ("canonicalModel", "provider", "contextWindow", "maxOutputTokens")
BROKER_FLAGS = ("--input-format", "stream-json")


class RuntimeIdentityError(Exception):
    """A runtime identity cannot be read, or is not the frozen one."""


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


def _records(text: str) -> list[dict[str, Any]]:
    out = []
    for line in text.splitlines():
        if line.startswith("{"):
            try:
                found = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(found, dict):
                out.append(found)
    return out


def _one(values: Sequence[Any]) -> Any:
    """The single value every occurrence agrees on; None if there is none, or several."""
    distinct = {json.dumps(v, sort_keys=True) for v in values}
    return values[0] if len(distinct) == 1 else None


def claude_effective(texts: Mapping[str, str]) -> dict[str, Any]:
    """The model a session ran on, as the runtime reported it, from its stream and transcript.
    A part that is absent or that disagrees with itself across the session is None."""
    stream = _records(texts.get("stream", ""))
    transcript = _records(texts.get("transcript", ""))
    inits = [e for e in stream if e.get("type") == "system" and e.get("subtype") == "init"]
    init = inits[0] if inits else None
    models = [
        (r.get("message") or {}).get("model")
        for r in [*stream, *transcript]
        if r.get("type") == "assistant" and isinstance(r.get("message"), dict)
    ]
    usage = []
    for e in stream:
        if e.get("type") == "result" and isinstance(e.get("modelUsage"), dict):
            usage.append(
                {
                    model: {k: entry.get(k) for k in USAGE_KEYS}
                    for model, entry in e["modelUsage"].items()
                    if isinstance(entry, dict)
                }
            )
    attachments = [
        r["attachment"].get("identity")
        for r in transcript
        if isinstance(r.get("attachment"), dict) and r["attachment"].get("type") == "model"
    ]
    return {
        "init": None if init is None else {k: init.get(k) for k in INIT_KEYS},
        "assistant_model": _one(models) if models else None,
        "model_usage": _one(usage) if usage else None,
        "model_attachment": _one(attachments) if attachments else None,
    }


def claude_problems(texts: Mapping[str, str], expected: Mapping[str, Any]) -> list[str]:
    """Where a session's effective model is not, exactly, the frozen one."""
    found = claude_effective(texts)
    return [
        f"the session's {part} is {found.get(part)!r}, not the frozen {expected[part]!r}"
        for part in sorted(expected)
        if found.get(part) != expected[part]
    ]


def claude_policy(extra_env: Sequence[str]) -> dict[str, Any]:
    """The static policy every Claude execution of S3 runs under, from Gate 1's module."""
    return {
        "command": hi.headless_command(Path("<pinned binary>"), "<requested model>"),
        "broker_flags": list(BROKER_FLAGS),
        "environment": {
            "inherited": [*hi.ENV_INHERITED, *extra_env],
            "set": dict(sorted(hi.ENV_SET.items())),
            "path": hi.SYSTEM_PATH,
            "refused": sorted(hi.BANNED_ENV),
        },
        "plugins": [dict(p) for p in hi.BUILTIN_PLUGINS],
    }


def _evidence(audits: Path, run: str) -> dict[str, str]:
    folder = audits / run
    texts = {}
    for name in ("stream", "transcript"):
        path = folder / f"{name}.jsonl"
        if not path.exists():
            raise RuntimeIdentityError(f"{path} does not exist: the runtime's own record")
        texts[name] = path.read_text(encoding="utf-8")
    return texts


def claude_runtime(provenance: Path) -> dict[str, Any]:
    """Claude's identity for S3, read from the private provenance: the pin, Gate 1's record and
    the sessions of Gate 1 (run 4) and of the broker's synthetic test, which must agree."""
    runtime = json.loads((provenance / "runtime.json").read_text(encoding="utf-8"))
    gate1 = json.loads((provenance / "gate1.json").read_text(encoding="utf-8"))
    broker = json.loads((provenance / "broker_test.json").read_text(encoding="utf-8"))
    audits = provenance.parent / "audits"
    effective = claude_effective(_evidence(audits / "isolation", gate1["run"]))
    if any(v is None for v in effective.values()):
        raise RuntimeIdentityError(f"Gate 1's session does not report its model: {effective}")
    if claude_effective(_evidence(audits / "broker", broker["run"])) != effective:
        raise RuntimeIdentityError("Gate 1 and the broker's test report different models")
    if gate1["model"] != runtime["model"] or effective["init"]["model"] != runtime["model"]:
        raise RuntimeIdentityError("the requested model is not the one Gate 1 ran")
    inherited = list(gate1["environment_allowlist"]["inherited"])
    return {
        "requested_model": runtime["model"],
        "binary": {"version": runtime["version"], "sha256": runtime["sha256"]},
        "effective": effective,
        "policy": claude_policy([v for v in inherited if v not in hi.ENV_INHERITED]),
    }


# ---------------------------------------------------------------------------
# MLX and Qwen3-8B
# ---------------------------------------------------------------------------

# Run by the MLX environment's own Python, isolated (-I): metadata only, nothing imported
_PROBE = """
import importlib.metadata as md, json, platform, sys
names = json.loads(sys.argv[1])
print(json.dumps({
    "python": {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "build": sys.version,
    },
    "machine": platform.machine(),
    "packages": {n: md.version(n) for n in names},
}))
"""


# The host's fields and where each is read: system tools by their absolute paths, so that no
# PATH decides; nothing that identifies the machine or its owner
HOST_FIELDS = {
    "macos_product_version": ("/usr/bin/sw_vers", "-productVersion"),
    "macos_build_version": ("/usr/bin/sw_vers", "-buildVersion"),
    "darwin_release": ("/usr/bin/uname", "-r"),
    "hardware_model": ("/usr/sbin/sysctl", "-n", "hw.model"),
    "architecture": ("/usr/bin/uname", "-m"),
}


def host_identity() -> dict[str, str]:
    """The host the construction runs on, as the system reports it, with nothing personal."""
    found = {}
    for name, command in HOST_FIELDS.items():
        try:
            done = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as error:
            raise RuntimeIdentityError(f"the host's {name} cannot be read: {error}") from error
        value = done.stdout.strip()
        if done.returncode != 0 or not value:
            raise RuntimeIdentityError(f"the host's {name} cannot be read")
        found[name] = value
    return found


def _model_files(model_id: str, revision: str, cache: Path) -> dict[str, str]:
    """Each file of the pinned snapshot, by the blob it points to: the cache names a blob by its
    content (SHA-256 for large files, the git blob hash for small ones), so this fixes the
    content without reading the weights."""
    snapshot = cache / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if not snapshot.is_dir():
        raise RuntimeIdentityError(f"{snapshot} does not exist: the pinned snapshot is missing")
    files = {}
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and path.name not in ("README.md", ".gitattributes"):
            # the snapshot's own link, one step: the blob the hub named by its content (where
            # that blob is stored after it is another matter)
            target = Path(os.readlink(path)) if path.is_symlink() else None
            if target is None or target.parent.name != "blobs":
                raise RuntimeIdentityError(f"{path} is not a cache link to a blob")
            files[path.relative_to(snapshot).as_posix()] = target.name
    return files


def mlx_runtime(
    python: Path = MLX_PYTHON,
    config_path: Path = DIAGNOSTIC_CONFIG,
    cache: Path = HF_CACHE,
    max_new_tokens: int = DIAGNOSTIC_MAX_NEW_TOKENS,
    host: Callable[[], Mapping[str, str]] = host_identity,
) -> dict[str, Any]:
    """The effective runtime of the Qwen diagnostic, with its host, read without loading the
    model."""
    if not python.exists():
        raise RuntimeIdentityError(f"{python} does not exist: the MLX environment is missing")
    done = subprocess.run(
        [str(python), "-I", "-c", _PROBE, json.dumps(sorted(MLX_PACKAGES))],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeIdentityError(f"the MLX environment cannot be read: {done.stderr[-300:]}")
    found: dict[str, Any] = json.loads(done.stdout)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return found | {
        "host": dict(host()),
        "model": {
            "config_file": config_path.as_posix(),
            "model_id": config["model_id"],
            "revision": config["revision"],
            "files": _model_files(config["model_id"], config["revision"], cache),
        },
        "generation": {
            "max_new_tokens": max_new_tokens,
            "decoding": config["decoding"],
            "chat_template_kwargs": config.get("chat_template_kwargs", {}),
            "max_prompt_tokens": config["max_prompt_tokens"],
        },
    }


def generator_versions(identity: Mapping[str, Any]) -> dict[str, Any]:
    """The versions scripts/mlx_generate.py writes into each output it makes under this
    identity, as it writes them."""
    versions: dict[str, Any] = {
        "mlx": identity["packages"]["mlx"],
        "mlx_lm": identity["packages"]["mlx-lm"],
        "python": identity["python"]["version"],
        "machine": identity["machine"],
        "model_id": identity["model"]["model_id"],
        "revision": identity["model"]["revision"],
        "max_new_tokens": identity["generation"]["max_new_tokens"],
        "decoding": "greedy",
    }
    if identity["generation"]["chat_template_kwargs"]:
        versions["chat_template_kwargs"] = identity["generation"]["chat_template_kwargs"]
    return versions


def runtime_differences(found: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    """The parts of a runtime identity that differ, by their dotted names."""
    out = []
    for key in sorted(set(found) | set(expected)):
        a, b = found.get(key), expected.get(key)
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            out += [f"{key}.{d}" for d in runtime_differences(a, b)]
        elif a != b:
            out.append(key)
    return out


def local_problems(record: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    """Where a diagnostic output was not produced under the frozen runtime: the identity its
    spool recorded before the run, and the versions the generator wrote into the output."""
    runtime = record.get("runtime")
    if not isinstance(runtime, Mapping):
        return ["the output carries no runtime identity"]
    problems = [f"produced under another {d}" for d in runtime_differences(runtime, expected)]
    if record.get("versions") != generator_versions(expected):
        problems.append("the generator's own versions are not those of the frozen runtime")
    return problems
