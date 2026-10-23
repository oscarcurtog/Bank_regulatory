"""The role audit on REAL packets (Codex Gate 2, P1.1): every packet is built by the code that
will build it in S3, from the real grid, DEV, the fixtures and the corpus, and placed in a
session shaped like the runtime of Gate 1. The judges' needs are the pipeline fixtures, never
TEST. A correct run passes; injected context, a changed packet, an unexpected message or an
unexpected attachment fails."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_broker as hb
from ingest import holdout_isolation as hi
from ingest import holdout_packets as hp
from ingest import holdout_roles as hr
from ingest.corpus_tools import CorpusTools
from tests.holdout_fakes import session as runtime_session

pytestmark = pytest.mark.corpus

PIN = {"version": "2.1.301 (Claude Code)", "organization_sha256": hi.CONTROL_ORG_SHA256}
GENERIC = "/code-review ultra launches a multi-agent cloud review of the current branch."


def _attachments(workdir: str, local_date: str) -> list[dict]:
    snapshot = {**hi._SNAPSHOT, "workingDirectory": workdir}
    return [
        {"type": "environment", "snapshot": snapshot},
        {
            "type": "model",
            "identity": {"modelId": "m", "marketingName": "M", "knowledgeCutoff": "x"},
            "text": "x",
        },
        {"type": "total_tokens_reminder", "text": "<total_tokens>1</total_tokens>"},
        {"type": "session_context", "context": {"userEmail": hi.CONTROL_EMAIL}},
        {"type": "date", "date": local_date},
        {"type": "credential_org", "organizationUuid": hi.CONTROL_ORG},
        {
            "type": "prompt_snapshot",
            "systemPrompt": [GENERIC],
            "reminderFold": False,
            "echoWireToolInputs": True,
            "contextRendering": "x",
        },
    ]


def session(sent, replies, reminder="", extra=()) -> dict:
    workdir = str(Path("/tmp/tmp.x").resolve(strict=False))
    clock = hi.runtime_clock_observation()
    records: list[dict] = [
        {"type": "attachment", "attachment": a}
        for a in [*_attachments(workdir, clock["local_date"]), *extra]
    ]
    records.append(
        {
            "type": "api-request-blob",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<system-reminder>{hi.CONTROL_EMAIL}{reminder}</system-reminder>",
                    }
                ],
            },
        }
    )
    records.append(
        {
            "type": "api-request-blob",
            "message": {
                "role": "system",
                "content": [
                    {"type": "text", "text": f"# Environment\nPrimary working directory: {workdir}"}
                ],
            },
        }
    )
    for i, message in enumerate(sent):
        records.append(
            {
                "type": "user",
                "message": {"role": "user", "content": [{"type": "text", "text": message}]},
            }
        )
        records.append(
            {
                "type": "api-request-blob",
                "message": {"role": "user", "content": [{"type": "text", "text": message}]},
            }
        )
        if i < len(replies):
            records.append(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": replies[i]}]},
                }
            )
    init = {
        "type": "system",
        "subtype": "init",
        "tools": [],
        "mcp_servers": [],
        "cwd": workdir,
        "model": "m",
        "plugins": list(hi.BUILTIN_PLUGINS),
        "claude_code_version": "2.1.301",
    }
    return {
        "transcript": "\n".join(json.dumps(r) for r in records),
        "stream": json.dumps(init),
        "stderr": "",
        "_gate1_context": {
            "launcher_workdir": workdir,
            "clock_start": clock,
            "clock_end": clock,
        },
    }


def run_of(role, packet, replies=("{}",), sent=None, log=(), extra=(), reminder="") -> hr.RoleRun:
    sent = list(sent or [packet])
    return hr.RoleRun(
        role=role,
        run_id="R-000000000000",
        packet=packet,
        packet_sha256=hashlib.sha256(packet.encode()).hexdigest(),
        sent=sent,
        replies=list(replies),
        session=session(sent, list(replies), reminder, extra),
        broker_log=list(log),
    )


@pytest.fixture(scope="module")
def world():
    dev = bm.load_benchmark(bm.DEV_DIR)
    fixtures = bm.load_benchmark(bm.FIXTURES_DIR)
    docs = bm.load_corpus()
    grid = ho.build_grid(dev, ho.load_areas())
    return {
        "dev": dev,
        "fixtures": fixtures,
        "grid": grid,
        "corpus": hp.Corpus(docs),
        "tools": CorpusTools(docs),
        "forbidden": [n.statement for n in dev.needs + fixtures.needs],
        "terms": hi.forbidden_terms(None, None, ["feat(t4): the external validation's results"]),
        "frozen": hr.template_hashes(),
    }


def audit(world, run):
    return hr.audit_role_run(
        run,
        world["terms"],
        world["frozen"],
        world["forbidden"],
        pin=PIN,
        tmp_root="/tmp/",
        tools=world["tools"],
    )


def test_role_audit_normalizes_the_real_cwd_positions_and_keeps_unverified_content(world):
    run = real_runs(world)["concept_author"]
    run.session = runtime_session(run.sent, run.replies, workdir="/tmp/tmp.abnDCGxy")
    report = audit(world, run)
    assert report.ok, report.problems
    # This is outside any authorized input or reply and must still be caught.
    run.session["transcript"] += '\n{"type":"unknown","content":"nDCG"}'
    report = audit(world, run)
    assert not report.ok and "dev_metrics" in report.findings


def _need(world) -> bm.Need:
    need: bm.Need = world["fixtures"].needs[0]
    return need


def _candidates(world) -> list[str]:
    need = _need(world)
    return [f"{r.act}:{r.path}" for r in world["fixtures"].qrels_of(need.need_id)]


def real_runs(world) -> dict[str, hr.RoleRun]:
    """One correct run per role, with the packet the S3 code builds."""
    cell, need, corpus = world["grid"].cells[0], _need(world), world["corpus"]
    ref = "R-0123456789ab"
    candidates = _candidates(world)
    concept = hp.concept_packet(cell, world["forbidden"], ref)
    writer = hp.writer_packet(cell, ref, "A firm asks a question.", ["a fact"], world["forbidden"])
    statement = (
        "A small payment firm wonders whether a new rule applies to its card service. Does it?"
    )
    query = bm.render_query_prompt(
        statement, [("en", "formal"), ("en", "natural"), ("es", "natural")]
    )
    existing = [hp.Existing(n.need_id, "DEV", n.statement) for n in world["dev"].needs]
    gate = hp.gate_packet(ref, "A firm asks a question.", existing)
    ids = {t: ho.item_id(b"test-salt", "N101:stage", t) for t in [*candidates, "answerability"]}
    judge_b = hp.judge_b_packet(corpus, need, candidates, {}, ids, ref=ref)
    a = {c: {"grade": 1, "ambiguous": False, "rationale": "A"} for c in candidates}
    b = {c: {"grade": 2, "ambiguous": False, "rationale": "B"} for c in candidates}
    a["answerability"] = {"answerability": "complete", "rationale": "A"}
    b["answerability"] = {"answerability": "complete", "rationale": "B"}
    judge_c, _ = hp.judge_c_packet(
        corpus, need, candidates, {}, a, b, candidates[:1], False, ids, ref=ref
    )
    judge_d = hp.judge_d_packet(corpus, need, candidates, {}, candidates[:1], ids, ref=ref)
    metadata = hp.metadata_packet(
        corpus, need, world["fixtures"].qrels_of(need.need_id), "complete", None, ref=ref
    )
    pool = hp.judge_b_pool_packet(corpus, need, candidates, ids, ref=ref)
    search = hp.judge_a_broker_packet(need, cell, ref=ref)

    class Judge:
        replies = [
            json.dumps({"request": {"op": "titles", "args": {"act": "DORA"}}}),
            json.dumps({"request": {"op": "article", "args": {"act": "DORA", "article": "19"}}}),
            json.dumps({"final": {"need_id": ref, "candidates": []}}),
        ]
        received: list[str] = []

        def send(self, text):
            self.received.append(text)
            return self.replies[len(self.received) - 1]

    judge = Judge()
    broker = hb.run_judge_a(judge, search, world["tools"])
    runs = {
        "concept_author": run_of("concept_author", concept),
        "statement_writer": run_of("statement_writer", writer),
        "query_writer": run_of("query_writer", query),
        "gatekeeper": run_of("gatekeeper", gate),
        "judge_b": run_of("judge_b", judge_b),
        "judge_c": run_of("judge_c", judge_c),
        "judge_d": run_of("judge_d", judge_d),
        "metadata_adjudicator": run_of("metadata_adjudicator", metadata),
        # judge B's pool variant: its first and its second blind judgment
        "pool_judge_1": run_of("pool_judge_1", pool),
        "pool_judge_2": run_of("pool_judge_2", pool),
        # every search: one packet and broker, each in a run of its own
        **{
            role: run_of(role, search, judge.replies, broker.sent, broker.log)
            for role in hr.SEARCHERS
        },
    }
    messages = hp.diagnostic_messages(corpus, need, candidates, {}, ids, ref=ref)
    local = hr.RoleRun(
        "qwen_diagnostic",
        "R-0",
        messages[0]["content"],
        hashlib.sha256(messages[0]["content"].encode()).hexdigest(),
        [messages[0]["content"]],
        [],
        local={
            "messages": messages,
            "prompt_sha256": hashlib.sha256(
                json.dumps(messages, ensure_ascii=False).encode()
            ).hexdigest(),
        },
    )
    runs["qwen_diagnostic"] = local
    return runs


def test_every_real_authorized_run_passes(world):
    runs = real_runs(world)
    assert set(runs) == set(hr.ROLE_CONTRACTS) - {"gatekeeper_evidence"}
    for role, run in runs.items():
        result = audit(world, run)
        assert result.ok, (role, result.problems, result.findings)


def test_the_judges_packets_hold_forbidden_words_that_the_mask_lets_through(world):
    runs = real_runs(world)
    raw = hi.audit_run(runs["judge_b"].session, world["terms"])
    assert raw.findings, "the protocol inside the packet trips the global scan"
    assert audit(world, runs["judge_b"]).ok


@pytest.mark.parametrize(
    "injection",
    [
        " The E5 baseline reached Recall@10 0.824 on DEV.",
        " Current branch: main\n\nRecent commits:\nabc",
        " JuriFindIT showed an OOM on MPS.",
        " See D-050: E frozen as dev candidate.",
    ],
)
def test_injected_context_fails_for_every_role(world, injection):
    for role, run in real_runs(world).items():
        if role == "qwen_diagnostic":
            continue
        tainted = run_of(
            role, run.packet, run.replies, run.sent, run.broker_log, reminder=injection
        )
        result = audit(world, tainted)
        assert not result.ok and result.findings, (role, injection)


def test_a_fixture_statement_in_a_query_packet_is_a_leak(world):
    leaking = bm.render_query_prompt(_need(world).statement, [("en", "formal")])
    result = audit(world, run_of("query_writer", leaking))
    assert any("leaks" in p for p in result.problems)


def test_a_dev_statement_outside_the_packet_fails_an_author_but_dev_inside_passes_the_gatekeeper(
    world,
):
    runs = real_runs(world)
    dev_statement = world["dev"].needs[0].statement
    tainted = run_of("concept_author", runs["concept_author"].packet, reminder=" " + dev_statement)
    assert "dev_statement" in audit(world, tainted).findings
    assert dev_statement in runs["gatekeeper"].packet and audit(world, runs["gatekeeper"]).ok


def test_a_modified_packet_fails(world):
    runs = real_runs(world)
    for role in ("concept_author", "judge_b", "searcher_1", "pool_judge_1", "gatekeeper"):
        run = runs[role]
        changed = run.packet.replace("information need", "information request", 1)
        assert changed != run.packet
        forged = run_of(role, changed, run.replies, [changed, *run.sent[1:]], run.broker_log)
        assert any("not an instance" in p for p in audit(world, forged).problems), role
        stale = run_of(role, changed, run.replies, [changed, *run.sent[1:]], run.broker_log)
        stale.packet_sha256 = run.packet_sha256
        assert any("not the one recorded" in p for p in audit(world, stale).problems), role


def test_an_unexpected_orchestrator_message_fails(world):
    runs = real_runs(world)
    hint = "Hint: the answer is in Article 19."
    for role in ("statement_writer", "judge_b", "searcher_2", "pool_judge_2"):
        run = runs[role]
        forged = run_of(role, run.packet, [*run.replies, "{}"], [*run.sent, hint], run.broker_log)
        problems = audit(world, forged).problems
        assert any("unexpected orchestrator message" in p for p in problems), role


def test_a_tampered_broker_result_fails(world):
    run = real_runs(world)["searcher_3"]
    tampered = [run.sent[0], run.sent[1].replace("DORA", "DORA (see DEV N014)", 1), *run.sent[2:]]
    forged = run_of("searcher_3", run.packet, run.replies, tampered, run.broker_log)
    assert any("not what the corpus returns" in p for p in audit(world, forged).problems)


def test_an_unexpected_runtime_attachment_fails(world):
    for role, run in real_runs(world).items():
        if role == "qwen_diagnostic":
            continue
        forged = run_of(
            role,
            run.packet,
            run.replies,
            run.sent,
            run.broker_log,
            extra=[{"type": "skill_listing", "skills": []}],
        )
        assert any("not allowed" in p for p in audit(world, forged).problems), role


def test_the_local_diagnostic_must_answer_its_own_prompt(world):
    run = real_runs(world)["qwen_diagnostic"]
    run.local = {**run.local, "prompt_sha256": "0" * 64}
    assert any("not generated from this prompt" in p for p in audit(world, run).problems)


def test_a_template_changed_after_its_freeze_fails(world):
    run = real_runs(world)["judge_b"]
    frozen = {**world["frozen"], "judge_b.v2": "0" * 64}
    result = hr.audit_role_run(
        run, world["terms"], frozen, world["forbidden"], pin=PIN, tmp_root="/tmp/"
    )
    assert any("frozen hash" in p for p in result.problems)
