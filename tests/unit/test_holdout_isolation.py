"""The isolation of TEST's authors (S3, D-054): the forbidden strings, the audit of a run
and its positive controls, and the command and environment of a headless author. Nothing
here launches a model."""

from __future__ import annotations

import ast
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from ingest import holdout_isolation as hi

TEST_CWD = str(Path("/tmp/tmp.x").resolve(strict=False))
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


def _run(*records, cwd=TEST_CWD) -> str:
    init = {
        "type": "system",
        "subtype": "init",
        "tools": [],
        "cwd": cwd,
        "model": "m",
        "plugins": list(hi.BUILTIN_PLUGINS),
    }
    return "\n".join(json.dumps(r) for r in (init, *records)) + "\n"


def test_a_clean_run_passes(terms):
    reply = {"type": "assistant", "message": {"content": [{"type": "text", "text": "none"}]}}
    report = hi.audit_run({"transcript": _run(reply)}, terms, expected_cwd=TEST_CWD)
    assert report.ok and report.findings == {} and report.attachment_problems == []
    assert not report.account_identity_metadata_present


def _clock(instant: str) -> dict[str, str]:
    utc = datetime.fromisoformat(instant).astimezone(UTC)
    zone = ZoneInfo("Europe/Madrid")
    return {
        "utc": utc.isoformat(timespec="microseconds"),
        "local_date": utc.astimezone(zone).date().isoformat(),
        "timezone": "Europe/Madrid",
    }


def _runtime_context(start: str, end: str, cwd: str = TEST_CWD) -> dict:
    return {
        "launcher_workdir": cwd,
        "clock_start": _clock(start),
        "clock_end": _clock(end),
    }


def _date(value: str, changed: bool | None = None) -> dict:
    body: dict[str, Any] = {"type": "date", "date": value}
    if changed is not None:
        body["changed"] = changed
    return {"type": "attachment", "attachment": body}


def test_a_date_rollover_is_accepted_only_when_the_launcher_observed_it(terms):
    text = _run(
        _date("2026-10-22"),
        _date("2026-10-23", changed=True),
    )
    report = hi.audit_run(
        {"transcript": text},
        terms,
        expected_cwd=TEST_CWD,
        runtime_context=_runtime_context(
            "2026-10-22T21:59:00+00:00",
            "2026-10-22T22:01:00+00:00",
        ),
    )
    assert report.ok, report.problems
    assert '"changed": true' in text

    leak = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "The E5 run is relevant."}]},
    }
    report = hi.audit_run(
        {"transcript": _run(_date("2026-10-22"), _date("2026-10-23", changed=True), leak)},
        terms,
        expected_cwd=TEST_CWD,
        runtime_context=_runtime_context(
            "2026-10-22T21:59:00+00:00",
            "2026-10-22T22:01:00+00:00",
        ),
    )
    assert "systems" in report.findings


@pytest.mark.parametrize(
    ("attachments", "start", "end", "problem"),
    [
        (
            [_date("2026-10-21")],
            "2026-10-22T21:59:00+00:00",
            "2026-10-22T22:01:00+00:00",
            "outside",
        ),
        (
            [_date("2026-02-30")],
            "2026-02-28T12:00:00+00:00",
            "2026-02-28T12:01:00+00:00",
            "calendar",
        ),
        (
            [_date("2026-10-23", changed=False)],
            "2026-10-22T21:59:00+00:00",
            "2026-10-22T22:01:00+00:00",
            "changed=false",
        ),
        (
            [_date("2026-10-23", changed=True)],
            "2026-10-22T21:59:00+00:00",
            "2026-10-22T22:01:00+00:00",
            "rollover",
        ),
    ],
)
def test_invalid_date_provenance_fails_closed(terms, attachments, start, end, problem):
    report = hi.audit_run(
        {"transcript": _run(*attachments)},
        terms,
        expected_cwd=TEST_CWD,
        runtime_context=_runtime_context(start, end),
    )
    assert not report.ok
    assert any(problem in p for p in report.attachment_problems)


def test_only_the_verified_cwd_fields_are_normalized_for_leak_scanning():
    cwd = str(Path("/tmp/tmp.nDCG").resolve(strict=False))
    snapshot = {**hi._SNAPSHOT, "workingDirectory": cwd}
    prompt = {
        "type": "attachment",
        "attachment": {
            "type": "prompt_snapshot",
            "systemPrompt": [f"# Environment\n - Primary working directory: {cwd}"],
            "reminderFold": False,
            "echoWireToolInputs": False,
            "contextRendering": "x",
        },
    }
    environment = {
        "type": "attachment",
        "attachment": {"type": "environment", "snapshot": snapshot},
    }
    text = _run(
        environment, prompt, _date(_clock("2026-10-22T10:00:00+00:00")["local_date"]), cwd=cwd
    )
    context = _runtime_context("2026-10-22T10:00:00+00:00", "2026-10-22T10:01:00+00:00", cwd)
    term = hi._literal("metric", "nDCG")
    report = hi.audit_run(
        {"transcript": text},
        [term],
        expected_cwd=cwd,
        runtime_context=context,
    )
    assert report.ok, report.problems

    leak = {"type": "assistant", "message": {"content": [{"type": "text", "text": cwd}]}}
    report = hi.audit_run(
        {
            "transcript": _run(
                environment, prompt, _date(context["clock_start"]["local_date"]), leak, cwd=cwd
            )
        },
        [term],
        expected_cwd=cwd,
        runtime_context=context,
    )
    assert "metric" in report.findings


def test_a_temp_path_prefix_is_not_a_cwd_identity():
    actual = f"{TEST_CWD}-sibling"
    record = {
        "type": "attachment",
        "attachment": {
            "type": "environment",
            "snapshot": {**hi._SNAPSHOT, "workingDirectory": actual},
        },
    }
    problems, _ = hi.attachment_problems([record], expected_cwd=TEST_CWD)
    assert problems == ["environment: workingDirectory differs from the launcher path"]


def test_unknown_date_attachment_fields_are_still_rejected(terms):
    date_record = _date("2026-10-22")
    date_record["attachment"]["retrieval_results"] = []
    report = hi.audit_run(
        {"transcript": _run(date_record)},
        terms,
        expected_cwd=TEST_CWD,
        runtime_context=_runtime_context(
            "2026-10-22T10:00:00+00:00",
            "2026-10-22T10:01:00+00:00",
        ),
    )
    assert any("expected exactly" in p for p in report.attachment_problems)


def _real_cwd_records(cwd: str) -> list[dict[str, Any]]:
    """Metadata positions sampled from the pinned runtime's synthetic Gate 1 transcript.
    The model's message content stays separate from its runtime-owned cwd envelope."""
    environment = f"# Environment\n - Primary working directory: {cwd}\n - Platform: darwin"
    return [
        {
            "type": "system",
            "subtype": "init",
            "tools": [],
            "cwd": cwd,
            "original_cwd": cwd,
            "plugins": list(hi.BUILTIN_PLUGINS),
        },
        {
            "type": "attachment",
            "cwd": cwd,
            "attachment": {
                "type": "environment",
                "snapshot": {**hi._SNAPSHOT, "workingDirectory": cwd},
            },
            "rendered": [{"content": f"<system-reminder>\n{environment}\n</system-reminder>"}],
        },
        {**_date("2026-10-22"), "cwd": cwd},
        {
            "type": "attachment",
            "cwd": cwd,
            "attachment": {**hi._PROMPT, "systemPrompt": [environment]},
        },
        {
            "type": "api-request-blob",
            "message": {"role": "system", "content": [{"type": "text", "text": environment}]},
        },
        {"type": "user", "cwd": cwd, "message": {"content": "a synthetic packet"}},
        {"type": "assistant", "cwd": cwd, "message": {"content": "none"}},
    ]


def test_real_cwd_envelopes_and_renderings_pass_and_other_occurrences_remain_audited():
    cwd = str(Path("/tmp/tmp.abnDCGxy").resolve(strict=False))
    records = _real_cwd_records(cwd)
    raw = "\n".join(map(json.dumps, records))
    context = _runtime_context("2026-10-22T10:00:00+00:00", "2026-10-22T10:01:00+00:00", cwd)
    term = hi._literal("metric", "nDCG")
    report = hi.audit_run({"transcript": raw}, [term], expected_cwd=cwd, runtime_context=context)
    assert report.ok and report.cwd_validated, report.problems
    assert cwd in raw  # normalization preserves the original artifact
    for where in ("assistant", "unknown", "system_extra"):
        extra = {
            "assistant": {"type": "assistant", "message": {"content": cwd}},
            "unknown": {"type": "unknown", "path": cwd},
            "system_extra": {
                "type": "api-request-blob",
                "message": {"role": "system", "content": f"Unverified path: {cwd}"},
            },
        }[where]
        report = hi.audit_run(
            {"transcript": raw + "\n" + json.dumps(extra)},
            [term],
            expected_cwd=cwd,
            runtime_context=context,
        )
        assert "metric" in report.findings, where


@pytest.mark.parametrize("where", ["cwd", "original_cwd", "rendered", "environment"])
def test_each_runtime_cwd_must_match_even_without_a_forbidden_term(where):
    records = _real_cwd_records(TEST_CWD)
    other = TEST_CWD + "-sibling"
    if where in ("cwd", "original_cwd"):
        records[0][where] = other
    elif where == "rendered":
        records[1]["rendered"][0]["content"] = f" - Primary working directory: {other}"
    else:
        records[1]["attachment"]["snapshot"]["workingDirectory"] = other
    report = hi.audit_run(
        {"transcript": "\n".join(map(json.dumps, records))}, [], expected_cwd=TEST_CWD
    )
    assert not report.ok and not report.cwd_validated


@pytest.mark.parametrize("rollover", [False, True])
def test_the_date_rendering_must_match_the_pinned_template(rollover):
    dates = (
        [_date("2026-10-22"), _date("2026-10-23", changed=True)]
        if rollover
        else [_date("2026-10-22")]
    )
    for record in dates:
        attachment = record["attachment"]
        text = (
            f"The date has changed. Today's date is now {attachment['date']}. "
            "No need to announce the new date — the user's own clock shows it."
            if attachment.get("changed") is True
            else f"Today's date is {attachment['date']}."
        )
        record["rendered"] = [{"content": f"<system-reminder>\n{text}\n</system-reminder>"}]
    context = _runtime_context("2026-10-22T21:59:00+00:00", "2026-10-22T22:01:00+00:00")
    report = hi.audit_run({"transcript": _run(*dates)}, [], runtime_context=context)
    assert report.ok and report.date_validated
    dates[-1]["rendered"][0]["content"] += " unexpected metadata"
    report = hi.audit_run({"transcript": _run(*dates)}, [], runtime_context=context)
    assert not report.ok and not report.date_validated
    assert any("rendered template" in p for p in report.problems)


@pytest.mark.parametrize("source", ["transcript", "stream"])
def test_duplicate_json_keys_fail_closed_without_discarding_a_leak(source):
    raw = _run() + '{"type":"assistant","message":{"content":"nDCG","content":"none"}}'
    report = hi.audit_run({source: raw}, [hi._literal("metric", "nDCG")])
    assert not report.ok
    assert any("duplicate JSON key" in p for p in report.problems)
    assert "metric" in report.findings


@pytest.mark.parametrize("raw", ["[]", "null", '{"type":"assistant",', "42"])
def test_non_object_or_malformed_records_fail_closed(raw):
    report = hi.audit_run({"transcript": _run() + raw}, [])
    assert not report.ok
    assert any("transcript:" in p for p in report.problems)


def test_a_cwd_context_mismatch_never_enables_normalization():
    context = _runtime_context("2026-10-22T10:00:00+00:00", "2026-10-22T10:01:00+00:00")
    report = hi.audit_run(
        {"transcript": _run(_date("2026-10-22"))},
        [],
        expected_cwd=TEST_CWD + "-other",
        runtime_context=context,
    )
    assert not report.ok and not report.cwd_validated and not report.date_validated


@pytest.mark.parametrize("value", [None, [], 42])
def test_malformed_attachment_bodies_and_plugin_lists_fail_without_crashing(value):
    report = hi.audit_run(
        {"transcript": _run({"type": "attachment", "attachment": value})},
        [],
    )
    assert not report.ok and "attachment body is not an object" in report.problems
    init = {"type": "system", "subtype": "init", "plugins": [value]}
    assert not hi.audit_run({"stream": json.dumps(init)}, []).ok


def test_a_second_init_cannot_hide_its_tools_behind_the_first():
    extra = {"type": "system", "subtype": "init", "tools": ["Bash"], "cwd": TEST_CWD}
    report = hi.audit_run({"stream": _run(extra)}, [])
    assert not report.ok and any("conflicting init" in p for p in report.problems)


def test_an_unknown_record_type_cannot_crash_cwd_validation():
    record = {"type": [], "cwd": TEST_CWD}
    report = hi.audit_run({"transcript": _run(record)}, [], expected_cwd=TEST_CWD)
    assert not report.ok and not report.cwd_validated


@pytest.mark.parametrize("version", ["2.1", "2.1.3", "2.1.30"])
def test_a_version_prefix_is_not_the_pinned_runtime_identity(version):
    init = json.loads(_run().splitlines()[0])
    init["claude_code_version"] = version
    report = hi.audit_run(
        {"stream": json.dumps(init)}, [], expected_version="2.1.301 (Claude Code)"
    )
    assert not report.ok and any("Claude Code" in p for p in report.problems)


def test_plugin_identity_ignores_object_key_order_but_preserves_every_value():
    init = json.loads(_run().splitlines()[0])
    init["plugins"] = [dict(reversed(list(p.items()))) for p in hi.BUILTIN_PLUGINS]
    assert hi.audit_run({"stream": json.dumps(init)}, []).ok


def test_a_recovered_rollover_without_an_observed_end_time_is_not_admitted():
    context = _runtime_context("2026-10-22T21:59:00+00:00", "2026-10-22T22:01:00+00:00")
    context["clock_end"] = None
    report = hi.audit_run(
        {"transcript": _run(_date("2026-10-22"), _date("2026-10-23", changed=True))},
        [],
        runtime_context=context,
    )
    assert not report.ok and not report.date_validated
    assert any("end-clock proof" in p for p in report.problems)


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
    report = hi.audit_run({"stream": with_tools}, [], expected_cwd=TEST_CWD)
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
        "cwd": TEST_CWD,
        "model": "m",
        "plugins": list(hi.BUILTIN_PLUGINS),
    } | init
    records = [first, *({"type": "attachment", "attachment": a} for a in attachments)]
    text = "\n".join(json.dumps(r) for r in records)
    runtime_context = hi.attachment_control_context(attachments)
    report = hi.audit_run(
        {"transcript": text},
        hi.forbidden_terms(None, None, []) + terms,
        expected_cwd=TEST_CWD if runtime_context else None,
        expected_org_sha256=hi.CONTROL_ORG_SHA256,
        runtime_context=runtime_context,
    )
    assert report.ok == should_pass, (name, report.problems)


# the deviation each failing provenance control is built around, as the audit names it
PROVENANCE_REASONS = {
    "the verified cwd copied into a model reply": "dev_metrics",
    "a leak beside a verified cwd": "systems",
    "a leak beside a verified rollover": "systems",
    "a sibling directory": "cwd is not the exact launcher-created path",
    "a directory under the launcher's": "cwd is not the exact launcher-created path",
    "the launcher directory's parent": "workingDirectory differs from the launcher path",
    "provenance for another directory": "workingDirectory differs from the launcher path",
    "a launcher directory that is not canonical": "provenance is not canonical",
    "a relative launcher directory": "provenance is invalid",
    "a cwd on a record kind never seen": "progress: cwd is not the exact",
    "a rollover flag as text": "str, expected bool",
    "a rollover flag as a number": "int, expected bool",
    "a null rollover flag": "NoneType, expected bool",
    "an impossible calendar date": "not a calendar date",
    "a date in another form": "canonical YYYY-MM-DD",
    "a date going back": "rollover is not supported",
    "a date change without the rollover flag": "without the pinned rollover marker",
    "two rollovers": "rollover is not supported",
    "a rollover inside one day": "outside the launcher's observed date window",
    "a rollover without the launcher's end clock": "end-clock proof",
    "a rollover rendered with another text": "unknown rendered template",
    "a duplicate key in a record": "duplicate JSON key",
    "a duplicate key nested in a message": "duplicate JSON key",
    "a cut record": "transcript:6:",
    "no launcher provenance": "no launcher clock provenance",
    "a launcher clock that changes timezone": "changed timezone",
    "a launcher clock that moves backwards": "changed timezone or moved backwards",
    "a launcher window longer than a day": "unexplained date change",
    "a local date its instant contradicts": "start-clock provenance is invalid",
}


def _provenance_report(name: str, terms) -> tuple[str, hi.IsolationReport]:
    transcript, context = next((t, c) for n, t, c, _ in hi.PROVENANCE_CONTROLS if n == name)
    report = hi.audit_run(
        {"transcript": transcript},
        terms,
        expected_cwd=context["launcher_workdir"] if context else None,
        runtime_context=context,
    )
    return transcript, report


@pytest.mark.parametrize(
    ("name", "should_pass"),
    [(name, should_pass) for name, _, _, should_pass in hi.PROVENANCE_CONTROLS],
)
def test_the_provenance_controls(terms, name, should_pass):
    transcript, report = _provenance_report(name, terms)
    assert report.ok == should_pass, (name, report.problems)
    if should_pass:
        # run 3's collision is in the artifact; only the verified fields were normalized
        assert "ndcg" in transcript.lower() and report.cwd_validated and report.date_validated
    else:
        assert PROVENANCE_REASONS[name] in f"{report.problems} {sorted(report.findings)}"


def test_every_failing_provenance_control_names_its_deviation():
    failing = {name for name, _, _, should_pass in hi.PROVENANCE_CONTROLS if not should_pass}
    assert failing == set(PROVENANCE_REASONS)


@pytest.mark.parametrize(
    "name", ["a leak beside a verified cwd", "a leak beside a verified rollover"]
)
def test_verified_metadata_hides_nothing_but_itself(terms, name):
    _, report = _provenance_report(name, terms)
    assert report.cwd_validated and report.date_validated
    assert sorted(report.findings) == ["systems"]  # the leak, and not the verified cwd


def test_a_nested_duplicate_key_keeps_its_first_value_audited(terms):
    _, report = _provenance_report("a duplicate key nested in a message", terms)
    assert "systems" in report.findings


def test_the_audit_never_changes_what_it_reads():
    cwd = str(Path("/tmp/tmp.abnDCGxy").resolve(strict=False))
    texts = {"transcript": "\n".join(map(json.dumps, _real_cwd_records(cwd)))}
    context = _runtime_context("2026-10-22T10:00:00+00:00", "2026-10-22T10:01:00+00:00", cwd)
    before = copy.deepcopy((texts, context))
    report = hi.audit_run(
        texts, [hi._literal("metric", "nDCG")], expected_cwd=cwd, runtime_context=context
    )
    assert report.ok and (texts, context) == before
    # the copy the leak scan reads is another text; the recorded artifact stays as it was
    assert hi.audit_content_copy(texts["transcript"], cwd, True) != texts["transcript"]


def test_only_the_two_audits_make_the_normalized_copy():
    """The copy with verified metadata normalized exists only for the leak scan: no replay,
    answer, state, judgment, provenance or admission reads it."""
    callers = set()
    for path in [*Path("ingest").glob("*.py"), *Path("scripts").glob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef):
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and "audit_content_copy" in ast.unparse(node.func):
                    callers.add((path.as_posix(), function.name))
    assert callers == {
        ("ingest/holdout_isolation.py", "audit_run"),
        ("ingest/holdout_roles.py", "audit_role_run"),
    }


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
