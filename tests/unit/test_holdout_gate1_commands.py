"""Gate 1's synthetic CLI checks, with every model boundary replaced: no real calls."""

from __future__ import annotations

import json

from ingest import holdout_certification as hcert
from ingest import holdout_isolation as hi
from tests.certification_fakes import BINARY, cli, fake_broker, fake_headless

PIN = {
    "path": str(BINARY),
    "version": "2.1.301 (Claude Code)",
    "sha256": "0" * 64,
    "organization_sha256": hi.CONTROL_ORG_SHA256,
}


def certification_layout(tmp_path):
    """A certification's namespace and raw folder, as certification-init lays them out."""
    namespace = tmp_path / "certifications" / "synthetic-check"
    raw = tmp_path / "certification-raw" / "synthetic-check"
    for kind in hcert.KINDS:
        (namespace / "audits" / kind).mkdir(parents=True)
        (raw / kind).mkdir(parents=True)
    (namespace / "provenance").mkdir()
    return namespace, raw


def setup_cli(monkeypatch, tmp_path):
    module = cli()
    monkeypatch.setattr(module, "pinned_runtime", lambda *args: PIN)
    monkeypatch.setattr(module, "isolation_terms", lambda: hi.forbidden_terms(None, None, []))
    monkeypatch.setattr(hi, "positive_controls", lambda *args: [])
    return (module, *certification_layout(tmp_path))


def test_isolation_command_passes_launcher_context_and_keeps_its_evidence(tmp_path, monkeypatch):
    module, namespace, raw = setup_cli(monkeypatch, tmp_path)
    context = fake_headless(monkeypatch)
    assert module.isolation_test(namespace, raw, BINARY, []) == 0
    report = json.loads((namespace / "audits" / "isolation" / "run-1" / "report.json").read_text())
    assert report["report"]["date_validated"] and report["report"]["cwd_validated"]
    assert "launcher-context.json" in report["raw_evidence_sha256"]
    assert report["runtime_context"] == context
    gate = json.loads((namespace / "provenance" / "gate1.json").read_text())
    assert gate["run"] == "run-1"
    assert gate["audit"] == hi.audit_identity(module.isolation_terms())
    assert gate["attachment_controls"] == len(hi.ATTACHMENT_CONTROLS)
    assert gate["provenance_controls"] == len(hi.PROVENANCE_CONTROLS)


def test_broker_command_passes_the_same_context_and_keeps_its_evidence(tmp_path, monkeypatch):
    module, namespace, raw = setup_cli(monkeypatch, tmp_path)
    fake_headless(monkeypatch)
    fake_broker(monkeypatch)
    assert module.isolation_test(namespace, raw, BINARY, []) == 0  # the broker test comes after
    assert module.broker_test(namespace, raw, BINARY, []) == 0
    report = json.loads((namespace / "audits" / "broker" / "run-1" / "report.json").read_text())
    assert report["verdict"] == "PASS"
    assert report["runtime_context"]["clock_end"] is not None
    assert "launcher-context.json" in report["raw_evidence_sha256"]
    assert (namespace / "provenance" / "broker_test.json").exists()
