"""TEST v2 (D-067), the tests that need no corpus: the pure core (anchors by legal unit,
deduplication, the pool, evaluability), the engine (retries, outages, isolation, model identity),
the state, the leakage filter, the admissions, the evaluation's statistics, the stops and Gate 1.
The construction itself, on the real corpus, is tested in
tests/integration/test_holdout_v2_construction.py; the two files hold the 29 tests the
specification requires, numbered as it numbers them."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from ingest import benchmark as bm
from ingest import holdout_packets as hp
from ingest import holdout_v2 as v2
from ingest import holdout_v2_dedup as dd
from ingest import holdout_v2_engine as en
from ingest import holdout_v2_evaluation as hev
from ingest import holdout_v2_packets as v2p
from ingest import holdout_v2_state as vs
from ingest import run_report as rr
from tests import holdout_fakes as hf
from tests import holdout_v2_fakes as f

V2_MODULES = sorted(Path("ingest").glob("holdout_v2*.py")) + [Path("scripts/holdout_v2.py")]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_03_anchor_sampling_by_legal_unit_not_chunk_frequency() -> None:
    """Every demand takes an article, uniformly and without replacement while the act has
    articles left; an article with many provisions is no likelier than one with a single one."""
    slot = v2.Slot("P01", ("DORA",), "direct", "complete", "easy", False)
    big = v2.LegalUnit("DORA", "1", "t", "area", tuple(f"1.{i}" for i in range(1, 101)))
    small = v2.LegalUnit("DORA", "2", "t", "area", ("2.1",))
    others = {a: [v2.LegalUnit(a, "9", "t", "area", ("9.1",))] for a in ("MiCA", "PSD2")}
    first: Counter[str] = Counter()
    for n in range(400):
        bank = v2.build_bank(
            [slot], {"DORA": [big, small], **others}, f"salt-{n}".encode(), universe=2
        )
        articles = [c.anchors[0].article for c in bank.candidates]
        assert sorted(articles) == ["1", "2"]  # two demands, two distinct articles
        first[articles[0]] += 1
        for c in bank.candidates:
            unit = big if c.anchors[0].article == "1" else small
            assert c.anchors[0].node.split(":")[1] in unit.nodes
    assert abs(first["1"] - first["2"]) < 60  # about 200 each, whatever their node counts
    # past the capacity, reuse is deterministic and counted
    bank = v2.build_bank([slot], {"DORA": [big, small], **others}, b"s", universe=5)
    assert bank.reuse["DORA"] == 3
    assert [c.anchors[0].reuse for c in bank.candidates] == [0, 0, 1, 1, 2]


def test_04_no_none_profiles_enter_test_v2() -> None:
    """The slots are DEV's answerable profiles only; the bank check refuses a `none` (the bank
    itself, on the corpus: tests/integration/test_holdout_v2_construction.py)."""
    dev = bm.load_benchmark(bm.DEV_DIR)
    slots = v2.slots(dev)
    assert slots and all(s.answerability in ("complete", "partial") for s in slots)
    nones = [n for n in dev.needs if n.answerability == "none"]
    assert nones  # DEV has some, and they are left out
    assert len(slots) == len(dev.needs) - len(nones)
    anchor = v2.Anchor("DORA", "1", "DORA:1.1", "area", 0)
    good = v2.Candidate("P01-01", slots[0], "0" * 64, 1, "N401", (anchor,) * len(slots[0].acts))
    bad = dataclasses.replace(good, slot=dataclasses.replace(slots[0], answerability="none"))
    bank = v2.Bank(1, (bad,), {}, {})
    assert "a `none` profile entered the bank" in v2.bank_problems(bank, slots, 1)


def _items(n: int) -> tuple[list[dd.Item], list[dd.Reference], dict[str, Any]]:
    items = [
        dd.Item(
            f"P{i % 4:02d}-{i:02d}",
            i,
            f"statement about topic {i % 4} number {i}",
            (("DORA", str(i % 3)),),
        )
        for i in range(1, n + 1)
    ]
    refs = [dd.Reference(f"N{i:03d}", "DEV", f"a dev need {i}") for i in range(1, 9)]
    vectors = f.fake_vectors([r.statement for r in refs] + [i.statement for i in items])
    keyed = dict(zip([r.key for r in refs] + [i.candidate_id for i in items], vectors, strict=True))
    return items, refs, keyed


def _oracle(cid: str, n: int, shown: dd.Shown) -> dict[str, Any]:
    """Two candidates of the same topic repeat each other."""
    mine = int(cid.split("-")[0][1:])
    for s, kind, text in shown.existing:
        if kind == "TEST" and text.startswith(f"statement about topic {mine} "):
            return {"verdict": "paraphrase", "closest": s}
    return {"verdict": "independent", "closest": None}


def test_08_global_dedup_invariant_to_input_order() -> None:
    items, refs, vectors = _items(12)
    hoods = dd.neighborhoods(items, refs, vectors)
    by_key = {r.key: r for r in refs}
    first = dd.resolve(items, hoods, by_key, b"s", _oracle)
    for seed in range(5):
        shuffled = items[:]
        random.Random(seed).shuffle(shuffled)
        rr_ = refs[:]
        random.Random(seed + 1).shuffle(rr_)
        again_hoods = dd.neighborhoods(shuffled, rr_, vectors)
        assert {k: (sorted(h.references), h.bank) for k, h in again_hoods.items()} == {
            k: (sorted(h.references), h.bank) for k, h in hoods.items()
        }
        assert dd.resolve(shuffled, again_hoods, by_key, b"s", _oracle) == first
    kept = [
        cid for cid, d in sorted(first.items(), key=lambda x: int(x[0].split("-")[1])) if d.kept
    ]
    assert kept == ["P01-01", "P02-02", "P03-03", "P00-04"]  # priority wins


def test_09_prefilter_never_omits_mandatory_comparisons() -> None:
    target = dd.Item(
        "P01-01", 1, "a firm asks how fast it must report a major incident", (("DORA", "19"),)
    )
    earlier = dd.Item("P02-01", 0, "something else entirely about crypto", (("DORA", "19"),))
    noise = [dd.Reference(f"N{i:03d}", "DEV", f"close noise statement {i}") for i in range(40)]
    exact = dd.Reference(
        "historical:x", "DEVELOPMENT", "A firm asks how fast it must report a major incident"
    )
    lexical = dd.Reference(
        "historical:y", "DEVELOPMENT", "a firm asks how fast it must report an incident"
    )
    refs = [*noise, exact, lexical]
    vectors: dict[str, Any] = {}
    import numpy as np

    base_vec = np.ones(8)
    for r in noise:
        vectors[r.key] = base_vec + 0.001 * np.arange(8)
    vectors[exact.key] = -base_vec
    vectors[lexical.key] = -base_vec
    vectors[target.candidate_id] = base_vec
    vectors[earlier.candidate_id] = -base_vec
    hoods = dd.neighborhoods([earlier, target], refs, vectors)
    hood = hoods[target.candidate_id]
    assert exact.key in hood.references and lexical.key in hood.references
    assert hood.references[:2] == (exact.key, lexical.key) or set(hood.references[:2]) == {
        exact.key,
        lexical.key,
    }
    assert earlier.candidate_id in hood.bank  # same anchor article, whatever its similarity
    many = dd.Neighborhood(tuple(r.key for r in refs), ())
    shown = dd.packets("P01-01", many, {}, {r.key: r for r in refs}, b"s")
    assert sum(len(s.existing) for s in shown) == len(refs)  # never cut, several packets
    assert all(len(s.existing) <= dd.MAX_SHOWN for s in shown)


def test_11_pool_construction_exact() -> None:
    chunks = [f"DORA:{i}" for i in range(1, 61)]
    qids = ["N1-en-1", "N1-en-2", "N1-es-1"]

    def order(system: str, query: str) -> list[str]:
        return sorted(chunks, key=lambda c: f._h(system, query, c))

    rankings = {s: {q: order(s, q) for q in qids} for s in f.SYSTEMS}
    rankings["E"].pop("N1-es-1")  # a missing ranking contributes nothing
    pooled = v2.pool(rankings, qids, ["PSD2:1"])
    expected = {"PSD2:1"}
    for s, depth in v2.POOL:
        for q in qids:
            expected |= set(rankings.get(s, {}).get(q, [])[:depth])
    assert [p.chunk_id for p in pooled] == sorted(expected)
    by = {p.chunk_id: p.sources for p in pooled}
    assert ("anchor", "", 0) in by["PSD2:1"]
    bge = rankings["BGE-M3"]["N1-en-1"]
    eleventh = bge[10]
    assert all(src[0] != "BGE-M3" or src[2] <= 10 for p in pooled for src in p.sources)
    assert not any(s == ("BGE-M3", "N1-en-1", 11) for s in by.get(eleventh, ()))
    assert [n for n, _ in v2.POOL] == ["E", "E5", "BM25", "BGE-M3", "Qwen3-Embedding-0.6B"]
    assert [d for _, d in v2.POOL] == [20, 20, 20, 10, 20]


def test_18_mechanical_evaluability_rule() -> None:
    assert v2.evaluable({"a": {"grade": 2, "ambiguous": False}})
    assert v2.evaluable(
        {"a": {"grade": 0, "ambiguous": False}, "b": {"grade": 3, "ambiguous": False}}
    )
    assert not v2.evaluable({"a": {"grade": 1, "ambiguous": False}})
    assert not v2.evaluable({"a": {"grade": 3, "ambiguous": True}})
    assert not v2.evaluable({"a": {"grade": None, "ambiguous": False}})
    assert not v2.evaluable({})


def test_24_no_metric_code_outside_the_evaluation() -> None:
    """Only the evaluation computes a metric, and it refuses qrels that are not frozen (the
    construction's own refusal: tests/integration/test_holdout_v2_construction.py)."""
    with pytest.raises(hev.EvaluationRefused):
        hev.frozen_inputs({"qrels": {"schema": "not-frozen"}}, None)  # type: ignore[arg-type]
    scoring = re.compile(
        r"^\s*(?:from ingest import (?:ir_metrics|run_report)|import ingest\.ir_metrics)", re.M
    )
    for path in V2_MODULES:
        if path.name == "holdout_v2_evaluation.py":
            continue
        text = path.read_text()
        assert not scoring.search(text), path
        assert "standard_scores(" not in text and "paired_bootstrap(" not in text, path


def test_26_no_old_minimum_hard_stop_in_the_code() -> None:
    for path in V2_MODULES:
        text = path.read_text()
        assert "MIN_EVALUABLE" not in text and "min_evaluable" not in text, path
    assert "minimum" not in en.INTEGRITY


def test_27_no_old_omission_audit_hard_stop_in_the_code() -> None:
    for path in V2_MODULES:
        text = path.read_text()
        assert "ConstructionStop" not in text and "omission_audit(" not in text, path
    assert not any("omission" in r for r in en.INTEGRITY)


def test_25_hard_stops_limited_to_integrity_classes() -> None:
    with pytest.raises(ValueError):
        en.V2Stop("minimum", "x")
    with pytest.raises(ValueError):
        en.V2Stop("omission_audit", "x")
    pattern = re.compile(r"(?:V2Stop|self\.stop)\(\s*\"([a-z_]+)\"")
    found = set()
    for path in V2_MODULES:
        found |= set(pattern.findall(path.read_text()))
    assert found and found <= set(en.INTEGRITY), found - set(en.INTEGRITY)


def test_29_gate1_certification_hashes_unchanged() -> None:
    contract = json.loads(Path("config/holdout/pre-authoring-contract.json").read_text())
    certified = contract["construction"]["certification"]["certified_code_sha256"]
    assert set(certified) == {
        "ingest/holdout_isolation.py",
        "ingest/holdout_broker.py",
        "ingest/holdout_spool.py",
        "ingest/holdout_process.py",
    }
    for path, digest in certified.items():
        assert _sha(Path(path)) == digest, path
    v2_contract = Path("config/holdout/test-v2-contract.json")
    if v2_contract.exists():
        found = json.loads(v2_contract.read_text())["certification"]["certified_code_sha256"]
        assert found == certified


def _engine(tmp_path: Path, executor: Any) -> en.Engine:
    store = vs.Store.create(tmp_path / "e", tmp_path / "e-raw", {"kind": "test"})
    engine = en.Engine(store, executor, f.audit_context(), workers=3, backoff_seconds=0.0)
    engine.sleep = lambda seconds: None
    return engine


def _gate_job(n: int) -> en.Job:
    packet = hp.gate_packet(f"R-{n:04x}", "A synthetic firm on topic T001 asks.", [])

    def admit(reply: str) -> Any:
        value = en.reply_json(reply)
        if value.get("verdict") not in ("independent", "paraphrase"):
            raise en.Inadmissible("gate_contract", ["verdict"])
        return value

    return en.Job(f"j/{n}", "gatekeeper", packet, admit)


def test_engine_retries_then_exhausts_after_max_runs(tmp_path: Path) -> None:
    """An answer that breaks its contract three times: three executions, the job exhausted and
    recorded once; a resume never runs a fourth."""
    fake = f.V2Fake(tmp_path / "spool")
    engine = _engine(tmp_path, fake)

    def refuse(reply: str) -> Any:
        raise en.Inadmissible("gate_contract", ["never"])

    bad = dataclasses.replace(_gate_job(1), admit=refuse)
    out = engine.run([bad, _gate_job(2)])
    assert out["j/1"].exhausted and engine.ledger.failures("j/1") == en.MAX_RUNS
    assert "j/1" in engine.ledger.exhausted and not out["j/2"].exhausted
    assert engine.run([bad])["j/1"].exhausted  # never a fourth execution
    assert len(engine.ledger.jobs["j/1"]) == en.MAX_RUNS


def test_engine_capacity_refusals_never_count(tmp_path: Path) -> None:
    fake = f.V2Fake(tmp_path / "spool", capacity={"j/": en.MAX_RUNS * 2})
    engine = _engine(tmp_path, fake)
    out = engine.run([_gate_job(n) for n in range(1, 7)])
    assert all(not o.exhausted for o in out.values())
    kinds = Counter(e.kind for e in engine.store.events())
    assert kinds["capacity_wait"] >= 1 and kinds["run_failed"] == en.MAX_RUNS * 2
    assert all(engine.ledger.failures(f"j/{n}") == 0 for n in range(1, 7))


def test_engine_isolation_finding_is_an_integrity_stop(tmp_path: Path) -> None:
    class Leaky(f.V2Fake):
        def headless(self, role: str, run_id: str, packet: str) -> Any:
            s = super().headless(role, run_id, packet)
            s.session["transcript"] += "\n" + json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "E5 reached Recall@10 0.8"}],
                    },
                }
            )
            return s

    engine = _engine(tmp_path, Leaky(tmp_path / "spool"))
    with pytest.raises(en.V2Stop) as stop:
        engine.run([_gate_job(1)])
    assert stop.value.reason == "isolation"


def test_engine_model_identity_mismatch_stops(tmp_path: Path) -> None:
    store = vs.Store.create(tmp_path / "m", tmp_path / "m-raw", {"kind": "test"})
    other = {**hf.FAKE_EFFECTIVE, "assistant_model": "another"}
    audit = dataclasses.replace(f.audit_context(), claude_effective=other)
    engine = en.Engine(store, f.V2Fake(tmp_path / "spool"), audit, 1, 0.0, lambda s: None)
    with pytest.raises(en.V2Stop) as stop:
        engine.run([_gate_job(1)])
    assert stop.value.reason == "model_identity"


def test_engine_inadmissible_answer_is_retried_with_the_same_packet(tmp_path: Path) -> None:
    fake = f.V2Fake(tmp_path / "spool")
    engine = _engine(tmp_path, fake)
    calls: Counter[str] = Counter()

    def admit(reply: str) -> Any:
        calls["n"] += 1
        if calls["n"] < 3:
            raise en.Inadmissible("gate_contract", ["not yet"])
        return en.reply_json(reply)

    job = dataclasses.replace(_gate_job(9), admit=admit)
    out = engine.run([job])
    assert not out["j/9"].exhausted and calls["n"] == 3
    packets = {p for r, p in fake.packets}
    assert len(packets) == 1  # one task, three contexts


def test_state_is_append_only_and_chained(tmp_path: Path) -> None:
    store = vs.Store.create(tmp_path / "s", tmp_path / "s-raw", {"kind": "test"})
    store.record("a", "k", {"x": 1})
    assert store.record("a", "k", {"x": 1}).seq == 1  # idempotent
    with pytest.raises(vs.Diverged):
        store.record("a", "k", {"x": 2})
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.db.execute("UPDATE events SET payload = '{}'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.db.execute("DELETE FROM events")
    store.verify()
    with pytest.raises(vs.StateBusy):
        vs.Store.open(tmp_path / "s", tmp_path / "s-raw")
    store.close()


def test_leakage_tells_terminology_from_near_copying() -> None:
    anchor = (
        "Financial entities shall report major ICT-related incidents to the relevant competent "
        "authority without undue delay after becoming aware of them."
    )
    term = "Must my firm, a critical ICT third-party service provider, report this incident?"
    copied = (
        "Do financial entities report major ICT-related incidents to the relevant competent "
        "authority?"
    )
    assert not v2p.leakage(term, [term, term, term], [anchor]).rejected
    assert v2p.leakage(copied, [term, term, term], [anchor]).rejected


def test_screen_and_answerability_admissions() -> None:
    ok = {"passages": [{"item_id": "P-0123456789ab", "flag": False, "reason": "x"}]}
    assert not v2p.screen_problems(ok, ["P-0123456789ab"])
    assert v2p.screen_problems(ok, ["P-0123456789ab", "P-ba9876543210"])
    assert v2p.screen_problems(
        {"passages": [{"item_id": "P-0123456789ab", "flag": "no"}]}, ["P-0123456789ab"]
    )
    assert not v2p.answerability_problems({"answerability": "complete", "reason": "r"})
    assert v2p.answerability_problems({"answerability": "partial", "reason": "r"})  # no gap


def test_bootstrap_is_s4s() -> None:
    values = {f"N{i:03d}": random.Random(i).random() - 0.4 for i in range(40)}
    queries = [
        bm.Query(
            query_id=f"{n}-en-1",
            need_id=n,
            lang="en",
            style="formal",
            origin="llm_from_need",
            template=bm.QUERY_TEMPLATE_ID,
            text="q?",
        )
        for n in values
    ]
    expected = rr.paired_bootstrap({f"{n}-en-1": v for n, v in values.items()}, queries, 2000)
    found = hev.bootstrap_mean(values, 2000)
    assert (found["mean"], found["ci95_low"], found["ci95_high"]) == pytest.approx(
        (expected["mean"], expected["ci95_low"], expected["ci95_high"])
    )


def test_claims_follow_s4s_table() -> None:
    assert hev.claim("holds", "holds") == "E improves retrieval over E5."
    assert "coverage" in hev.claim("holds", "inconclusive")
    assert "ranking quality" in hev.claim("inconclusive", "holds")
    assert hev.claim("inconclusive", "inconclusive").startswith("We do not have evidence")
    assert hev.claim("reverses", "holds").startswith("E is worse than E5 on Recall@10")
    assert hev.verdict(0.01, 0.2) == "holds" and hev.verdict(-0.2, -0.01) == "reverses"
    assert hev.verdict(-0.01, 0.2) == "inconclusive"


def test_fixed_scale_gain_and_ndcg() -> None:
    chunks = {"a": 3, "b": 2, "c": 0}
    assert hev.g_at_10(["a"] * 0 + ["a", "b"], chunks) == pytest.approx(
        (3 + 2 / 1.5849625007211562) / sum(3 / __import__("math").log2(i + 1) for i in range(1, 11))
    )
    assert hev.g_at_10(["x"] * 10, chunks) == 0.0
    assert hev.ndcg_at_10(["a", "b"], chunks) == pytest.approx(1.0)
    assert hev.ndcg_at_10(["b", "a"], chunks) < 1.0
    assert hev.ndcg_at_10(["a", "b"], chunks, [3]) < 1.0  # a residual relevant chunk


# ---------------------------------------------------------------------------
# the independent implementation review's findings, pure parts (P1-2, P1-3)
# ---------------------------------------------------------------------------


def _logged(store: vs.Store, job: str, n: int, reason: str | None) -> None:
    run = f"{job}#{n}"
    store.record(
        "run_started",
        f"{run}/started",
        {
            "job_id": job,
            "run_id": run,
            "n": n,
            "role": "gatekeeper",
            "packet_sha256": "0" * 64,
            "inputs": {},
        },
    )
    if reason is not None:
        store.record(
            "run_failed",
            f"{run}/failed",
            {"run_id": run, "reason": reason, "detail": "x", "problems": []},
        )


def test_review_p1_2_capacity_refusals_shared_by_jobs_exhaust_none(tmp_path: Path) -> None:
    """The review's case, with the provider refusing for its capacity: A and B refused in
    turn, three times each; neither is exhausted, both run again after a wait."""
    fake = f.V2Fake(tmp_path / "spool", capacity={"j/1#": 3, "j/2#": 3})
    engine = _engine(tmp_path, fake)
    engine.workers = 1
    out = engine.run([_gate_job(1), _gate_job(2)])
    assert not out["j/1"].exhausted and not out["j/2"].exhausted
    assert Counter(e.kind for e in engine.store.events())["job_exhausted"] == 0


def test_review_n1_a_job_failing_the_same_way_every_time_is_exhausted(tmp_path: Path) -> None:
    """A failure that is no capacity refusal (an API error 400, a crash) counts every time: the
    job is exhausted after MAX_RUNS executions, even alone at a stage's tail, and no pause."""
    fake = f.V2Fake(tmp_path / "spool", fail={"j/1#": 99})
    engine = _engine(tmp_path, fake)
    out = engine.run([_gate_job(1)])
    assert out["j/1"].exhausted and len(engine.ledger.jobs["j/1"]) == en.MAX_RUNS
    assert Counter(e.kind for e in engine.store.events())["paused"] == 0


@pytest.mark.parametrize(
    "reasons", [("model_api",) * 3, ("model_api", "execution_deadline", "execution_deadline")]
)
def test_review_n4_a_resume_decides_as_the_live_run(
    tmp_path: Path, reasons: tuple[str, ...]
) -> None:
    """A log with three counted failures and no exhaustion recorded (a crash between them):
    the resume exhausts the job at once, as the live run would have, and never runs it again."""
    store = vs.Store.create(tmp_path / "e", tmp_path / "e-raw", {"kind": "test"})
    for n, reason in enumerate(reasons, start=1):
        _logged(store, "j/1", n, reason)
    engine = en.Engine(
        store, f.V2Fake(tmp_path / "spool"), f.audit_context(), 1, 0.0, lambda s: None
    )
    out = engine.run([_gate_job(1)])
    assert out["j/1"].exhausted and "j/1" in engine.ledger.exhausted
    assert engine.ledger.jobs["j/1"] == ["j/1#1", "j/1#2", "j/1#3"]


def test_review_capacity_refusals_on_replay_do_not_count(tmp_path: Path) -> None:
    store = vs.Store.create(tmp_path / "e", tmp_path / "e-raw", {"kind": "test"})
    for n in (1, 2, 3, 4):
        _logged(store, "j/1", n, en.CAPACITY)
    engine = en.Engine(
        store, f.V2Fake(tmp_path / "spool"), f.audit_context(), 1, 0.0, lambda s: None
    )
    out = engine.run([_gate_job(1)])
    assert not out["j/1"].exhausted and engine.ledger.jobs["j/1"][-1] == "j/1#5"


def test_review_a_long_capacity_outage_pauses(tmp_path: Path) -> None:
    fake = f.V2Fake(tmp_path / "spool", capacity={"j/1#": 10_000})
    engine = _engine(tmp_path, fake)
    with pytest.raises(en.Paused):
        engine.run([_gate_job(1)])
    kinds = Counter(e.kind for e in engine.store.events())
    assert kinds["paused"] == 1 and kinds["job_exhausted"] == 0


def test_review_capacity_is_read_from_the_cli_result_only() -> None:
    error = json.dumps(
        {"type": "result", "is_error": True, "result": "Claude AI usage limit reached|1761200000"}
    )
    assert en.capacity_refusal(error, "")
    overloaded = json.dumps(
        {"type": "result", "is_error": True, "result": 'API Error: 529 {"type":"overloaded_error"}'}
    )
    assert en.capacity_refusal(overloaded, "")
    assert en.capacity_refusal("", "API Error: 429 rate_limit_error")
    other = json.dumps(
        {"type": "result", "is_error": True, "result": "API Error: 400 invalid request"}
    )
    assert not en.capacity_refusal(other, "")
    # a reply or a packet that mentions a usage limit is never a refusal
    reply = json.dumps(
        {"type": "assistant", "message": {"content": "the usage limit of the payment account"}}
    )
    ok = json.dumps({"type": "result", "is_error": False, "result": "fine"})
    assert not en.capacity_refusal(reply + "\n" + ok, "")


def test_review_p1_3_the_contract_freezes_the_metric_code() -> None:
    from ingest import holdout_v2_contract as hc2

    files = set(hc2.code_files())
    for path in (
        "ingest/ir_metrics.py",
        "ingest/evaluation.py",
        "ingest/benchmark.py",
        "ingest/run_report.py",
        "ingest/retrieval.py",
        "ingest/holdout_isolation.py",
        "scripts/holdout_v2.py",
    ):
        assert path in files, path
    assert set(hc2.CODE_FILES) <= files


# ---------------------------------------------------------------------------
# the third review's findings (N7 to N9), pure parts
# ---------------------------------------------------------------------------


def test_review_n7_a_pause_records_in_flight_capacity_refusals_as_such(tmp_path: Path) -> None:
    """Every call refused for capacity until the pause: no run is left open, every failure is a
    capacity refusal, and a resume, with the provider back, exhausts nothing."""
    fake = f.V2Fake(tmp_path / "spool", capacity={"j/": 10_000})
    engine = _engine(tmp_path, fake)
    engine.workers = 6
    jobs = [_gate_job(n) for n in range(1, 7)]
    with pytest.raises(en.Paused):
        engine.run(jobs)
    runs = engine.ledger.runs.values()
    assert not [r for r in runs if r["status"] == "started"]
    assert {r.get("reason") for r in runs if r["status"] == "failed"} == {en.CAPACITY}
    assert all(engine.ledger.failures(j.job_id) == 0 for j in jobs)
    again = en.Engine(
        engine.store, f.V2Fake(tmp_path / "spool2"), f.audit_context(), 6, 0.0, lambda s: None
    )
    out = again.run(jobs)
    assert all(not o.exhausted for o in out.values())


def test_review_n7_a_recovered_capacity_refusal_is_not_counted(tmp_path: Path) -> None:
    """A call that was running when the orchestrator died, whose spool holds a capacity
    refusal: recorded as that refusal, never as an interruption."""

    class Refused(f.V2Fake):
        def recover_headless(self, role: str, run_id: str, packet: str) -> Any:
            return None

        def recover_failure(self, role: str, run_id: str) -> str | None:
            return en.CAPACITY if run_id == "j/1#1" else None

    store = vs.Store.create(tmp_path / "e", tmp_path / "e-raw", {"kind": "test"})
    _logged(store, "j/1", 1, None)  # started, then the process died
    engine = en.Engine(
        store, Refused(tmp_path / "spool"), f.audit_context(), 1, 0.0, lambda s: None
    )
    out = engine.run([_gate_job(1)])
    assert not out["j/1"].exhausted
    assert engine.ledger.runs["j/1#1"]["reason"] == en.CAPACITY
    assert engine.ledger.failures("j/1") == 0


def test_review_n8_only_the_last_stderr_line_can_name_a_capacity_refusal() -> None:
    earlier = "API Error (overloaded_error): retrying in 5 s\nError: 400 invalid request"
    assert not en.capacity_refusal("", earlier)
    assert en.capacity_refusal("", "a notice\nAPI Error: 429 rate_limit_error")
