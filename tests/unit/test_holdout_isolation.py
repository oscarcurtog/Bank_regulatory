"""The isolation of TEST's authors (S3, D-054): the forbidden strings, the audit of a run
and its positive controls, and the command and environment of a headless author. Nothing
here launches a model."""

from __future__ import annotations

import json

import pytest

from ingest import holdout_isolation as hi

LINE = "The guide — not the architect — explains every decision to the owner."
INDEX_LINE = "- [Retrieval round closed](x.md) — E frozen as dev candidate; no DEV tests"


@pytest.fixture
def terms(tmp_path) -> list[hi.Term]:
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        f"# Fundamento — working rules\n\n{LINE}\n",
        encoding="utf-8",
    )
    memory = tmp_path / "MEMORY.md"
    memory.write_text(INDEX_LINE + "\n", encoding="utf-8")
    return hi.forbidden_terms(claude_md, memory, ["feat(t4): the external validation's results"])


def _run(*records) -> str:
    init = {
        "type": "system",
        "subtype": "init",
        "tools": [],
        "cwd": "/tmp/tmp.x",
        "model": "m",
        "plugins": list(hi.BUILTIN_PLUGINS),
    }
    return "\n".join(json.dumps(r) for r in (init, *records)) + "\n"


def test_a_clean_run_passes(terms):
    reply = {"type": "assistant", "message": {"content": [{"type": "text", "text": "none"}]}}
    report = hi.audit_run({"transcript": _run(reply)}, terms, allowed_cwd_prefix="/tmp/")
    assert report.ok and report.findings == {} and report.attachment_problems == []
    assert not report.account_identity_metadata_present


def test_a_line_of_claude_md_is_found_through_json_escapes(terms):
    leak = {
        "type": "attachment",
        "attachment": {"content": f"a pasted file\n{LINE}\nand more"},
    }
    text = _run(leak)
    assert "\\u2014" in text  # json.dumps escaped the dash
    report = hi.audit_run({"transcript": text}, terms)
    assert "claude_md" in report.findings and not report.ok


@pytest.mark.parametrize(
    ("leak", "category"),
    [
        (INDEX_LINE, "memory_index"),
        ("feat(t4): the external validation's results", "git_status"),
        ("Recent commits: abc", "git_status"),
        ("the E5 baseline", "systems"),
        ("Recall@10 0.824", "dev_metrics"),
        ("need N017", "dev_metrics"),
        ("JuriFindIT", "external_validation"),
        ("an OOM on MPS", "failure_modes"),
        ("see D-050", "retrieval_decisions"),
        ("the BM25 run", "retrieval_decisions"),
    ],
)
def test_each_kind_of_leak_is_found(terms, leak, category):
    record = {"type": "attachment", "attachment": {"type": "x", "content": leak}}
    report = hi.audit_run({"stream": _run(record)}, terms)
    assert category in report.findings


def test_project_context_tools_and_directory_are_checked(terms):
    attached = {
        "type": "attachment",
        "attachment": {"type": "instructions", "files": [{"type": "Project"}, {"type": "AutoMem"}]},
    }
    report = hi.audit_run({"transcript": _run(attached)}, terms)
    assert report.attachment_problems == ["attachment 'instructions' is not allowed"]
    with_tools = '{"type": "system", "subtype": "init", "tools": ["Read"], "cwd": "/repo"}\n'
    report = hi.audit_run({"stream": with_tools}, [], allowed_cwd_prefix="/tmp/")
    assert any("tools offered" in p for p in report.problems)
    assert any("working directory" in p for p in report.problems)
    assert (
        "no init event: the run cannot be checked"
        in hi.audit_run({"stream": "plain text\n"}, []).problems
    )


@pytest.mark.parametrize(("name", "init", "attachments", "should_pass"), hi.ATTACHMENT_CONTROLS)
def test_the_attachment_policy(terms, name, init, attachments, should_pass):
    first = {
        "type": "system",
        "subtype": "init",
        "tools": [],
        "cwd": "/tmp/tmp.x",
        "model": "m",
        "plugins": list(hi.BUILTIN_PLUGINS),
    } | init
    records = [first, *({"type": "attachment", "attachment": a} for a in attachments)]
    text = "\n".join(json.dumps(r) for r in records)
    report = hi.audit_run(
        {"transcript": text},
        hi.forbidden_terms(None, None, []) + terms,
        expected_org_sha256=hi.CONTROL_ORG_SHA256,
    )
    assert report.ok == should_pass, (name, report.problems)


def test_the_generic_phrase_passes_and_a_real_git_status_does_not():
    terms = hi.forbidden_terms(None, None, [])
    assert not any(t.found_in(hi.GENERIC_PROMPT, hi.normalize(hi.GENERIC_PROMPT)) for t in terms)
    assert any(t.category == "git_status" and t.found_in(hi._GIT, hi._GIT) for t in terms)


def test_no_agents_md_where_the_plugin_could_read_one(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    work = tmp_path / "work" / "tmp.x"
    work.mkdir(parents=True)
    assert hi.agents_md_problems(work) == []
    (tmp_path / "work" / "AGENTS.md").write_text("x", encoding="utf-8")
    assert hi.agents_md_problems(work) == [f"{(tmp_path / 'work').resolve() / 'AGENTS.md'} exists"]


def test_the_audit_has_an_identity(terms):
    identity = hi.audit_identity(terms)
    assert len(identity["module_sha256"]) == 64 and identity["terms"] == str(len(terms))
    assert hi.schema_description()["prompt_snapshot"][1]["tools"] == hi.EMPTY


def test_identity_values_are_redacted_in_plain_and_escaped_form():
    records: list[dict] = [
        {"attachment": {"type": "session_context", "context": {"userEmail": hi.CONTROL_EMAIL}}},
        {"attachment": {"type": "credential_org", "organizationUuid": hi.CONTROL_ORG}},
    ]
    values = hi.identity_values(records)
    assert values == sorted({"someone@example.org", hi.CONTROL_ORG}, key=len, reverse=True)
    text = json.dumps(records) + " someone@example.org"
    redacted = hi.redact(text, values)
    assert "someone@example.org" not in redacted and hi.CONTROL_ORG not in redacted
    assert redacted.count(hi.REDACTED) == 3


def test_the_positive_controls_pass_on_a_transcript_with_project_context(tmp_path, terms):
    dev = tmp_path / "agent.jsonl"
    dev.write_text(
        json.dumps(
            {
                "type": "attachment",
                "attachment": {
                    "type": "instructions",
                    "files": [
                        {
                            "type": "Project",
                            "content": f"# Fundamento\n{LINE}\n{hi._GIT}",
                        }
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert hi.positive_controls(terms, dev) == []
    assert hi.positive_controls(terms, tmp_path / "missing.jsonl") != []


def test_the_headless_command_has_no_tools_and_safe_mode(tmp_path):
    command = hi.headless_command(tmp_path / "claude", "claude-opus-5-5")
    assert command[0] == str(tmp_path / "claude")
    assert command[1:3] == ["-p", "--safe-mode"]
    assert command[command.index("--tools") + 1] == ""
    assert "stream-json" in command and "claude-opus-5-5" in command


def test_the_environment_is_an_allowlist(monkeypatch):
    monkeypatch.setenv("PWD", "/Users/x/Bank_regulatory")
    monkeypatch.setenv("VIRTUAL_ENV", "/Users/x/Bank_regulatory/.venv")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "abc")
    monkeypatch.setenv("HOME", "/Users/x")
    monkeypatch.setenv("USER", "x")
    env = hi.clean_environment()
    assert set(env) == {"HOME", "PATH", *hi.ENV_SET}
    assert env["HOME"] == "/Users/x" and env["PATH"] == hi.SYSTEM_PATH
    assert hi.clean_environment(["USER"])["USER"] == "x"
    for banned in ("PWD", "VIRTUAL_ENV", "CLAUDE_CODE_SESSION_ID", "ANTHROPIC_API_KEY"):
        with pytest.raises(ValueError, match="never passed"):
            hi.clean_environment([banned])


def _fake_binary(tmp_path, version: str):
    binary = tmp_path / "claude"
    binary.write_text(f"#!/bin/sh\necho '{version} (Claude Code)'\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def test_a_binary_is_its_version_and_its_hash(tmp_path):
    binary = _fake_binary(tmp_path, "2.1.301")
    pinned = hi.binary_identity(binary)
    assert pinned["version"] == "2.1.301 (Claude Code)" and len(pinned["sha256"]) == 64
    assert hi.check_pinned(binary, pinned) == pinned
    binary.write_text(binary.read_text() + "# rebuilt\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256"):
        hi.check_pinned(binary, pinned)


def test_the_audit_checks_the_pinned_version(terms):
    plugins = json.dumps(list(hi.BUILTIN_PLUGINS))
    init = (
        '{"type": "system", "subtype": "init", "tools": [], "claude_code_version": "2.1.177", '
        f'"plugins": {plugins}}}\n'
    )
    report = hi.audit_run({"stream": init}, terms, expected_version="2.1.301 (Claude Code)")
    assert any("pinned 2.1.301" in p for p in report.problems)
    ok = init.replace("2.1.177", "2.1.301")
    assert hi.audit_run({"stream": ok}, terms, expected_version="2.1.301 (Claude Code)").ok


def test_the_transcript_folder_is_named_like_claude_code_names_it():
    assert hi.encoded_project_dir("/private/var/folders/y2/T/tmp.ab_c") == (
        "-private-var-folders-y2-T-tmp-ab-c"
    )
