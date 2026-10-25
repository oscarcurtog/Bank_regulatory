"""TEST v2 (D-067), the construction on the real corpus, DEV, fixtures and anchors, with
synthetic replies (tests/holdout_v2_fakes.py) whose sessions pass Gate 1's real audit: the tests
of the specification that need the whole construction, numbered as it numbers them (the others
are in tests/unit/test_holdout_v2.py). No model and no TEST content."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
import shutil
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from ingest import benchmark as bm
from ingest import holdout_roles as hr
from ingest import holdout_v2 as v2
from ingest import holdout_v2_audit as hau
from ingest import holdout_v2_engine as en
from ingest import holdout_v2_evaluation as hev
from ingest import holdout_v2_judging as jv
from ingest import holdout_v2_packets as v2p
from ingest import holdout_v2_retrieval as rv
from ingest import holdout_v2_runner as hrun
from ingest import holdout_v2_state as vs
from tests import holdout_fakes as hf
from tests import holdout_v2_fakes as f

pytestmark = pytest.mark.corpus


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(directory: Path, **kwargs: Any) -> tuple[hrun.Runner, f.V2Fake, list[str]]:
    """A whole construction with the fakes, evaluation included, in directory."""
    evaluate = kwargs.pop("evaluate", True)
    w, fake, picked = f.world(directory, **kwargs)
    if evaluate:
        layer = w.layer
        w.evaluate = lambda inputs: hev.evaluate(hev.frozen_inputs(inputs, layer))
    runner = f.runner(w, fake)
    runner.build()
    return runner, fake, picked


def frozen_hashes(directory: Path) -> dict[str, str]:
    out = {}
    for name in (
        "bank/bank.json",
        "topics/topics.json",
        "pools/pools.json",
        "qrels/qrels.json",
        "completeness/completeness.json",
        "evaluation/evaluation.json",
    ):
        path = directory / name
        if path.exists():
            out[name] = _sha(path)
    return out


@pytest.fixture(scope="module")
def complete(tmp_path_factory: pytest.TempPathFactory) -> tuple[hrun.Runner, f.V2Fake, Path]:
    directory = tmp_path_factory.mktemp("v2") / "ns"
    runner, fake, _ = build(directory)
    return runner, fake, directory


def test_01_candidate_priority_independent_of_completion_order(tmp_path: Path) -> None:
    """The bank is a function of its seed and salt; the topics frozen from it are the same
    whether the calls run one at a time or six at a time, finishing in another order."""
    assert f.small_bank(8).sha256() == f.small_bank(8).sha256()

    class Jittery(f.V2Fake):
        def headless(self, role: str, run_id: str, packet: str) -> Any:
            time.sleep(random.Random(run_id).random() / 200)
            return super().headless(role, run_id, packet)

    hashes = []
    for workers in (1, 6):
        d = tmp_path / f"w{workers}"
        fake = Jittery(d.with_name(d.name + "-spool"))
        runner, _, _ = build(d, executor=fake, workers=workers, universe=8, evaluate=False)
        hashes.append(frozen_hashes(d))
        ranks = [t["rank"] for t in runner.topics()]
        assert ranks == sorted(ranks)
    assert hashes[0] == hashes[1]


def test_02_legal_anchor_selection_deterministic() -> None:
    a = f.small_bank(12)
    b = f.small_bank(12)
    assert [c.anchors for c in a.candidates] == [c.anchors for c in b.candidates]
    other = f.small_bank(12, salt=b"another-salt-for-the-same-bank-xx")
    assert [c.anchors for c in a.candidates] != [c.anchors for c in other.candidates]


def test_04_the_bank_holds_no_none_profile() -> None:
    bank = f.small_bank(27)
    assert not any(c.slot.answerability == "none" for c in bank.candidates)
    assert len({c.slot.slot_id for c in bank.candidates}) == 27


def _packets(fake: f.V2Fake, role: str) -> list[str]:
    return [p for r, p in fake.packets if r == role]


def test_05_topic_author_isolation(complete: Any) -> None:
    """The concept author reads its specification and anchors: never DEV, the fixtures, the
    exclusions, another candidate, a system, a ranking, a qrel or a metric."""
    runner, fake, _ = complete
    refs = f.references()
    concepts = _packets(fake, "v2_concept_author")
    assert len(concepts) == len(runner.w.bank.candidates)
    for packet in concepts:
        for r in refs:
            assert r.statement not in packet
        for word in ("E5", "BM25", "BGE", "Qwen", "Recall", "nDCG", "qrel", "ranking"):
            assert word not in packet
    for c, packet in zip(runner.w.bank.candidates, concepts, strict=True):
        others = [o for o in runner.w.bank.candidates if o.candidate_id != c.candidate_id]
        assert c.candidate_id not in packet and c.need_id not in packet
        for o in others:
            if o.anchors[0].node != c.anchors[0].node:
                theirs = runner.w.corpus.texts[o.anchors[0].node_key].strip()
                if len(theirs) > 40:
                    assert f"ANCHOR: {theirs}" not in packet
    contract = hr.ROLE_CONTRACTS["v2_concept_author"]
    assert {"DEV", "the fixtures", "the historical exclusions", "other TEST v2 candidates"} <= set(
        contract.forbidden
    )


def test_06_final_need_and_query_authors_cannot_see_anchor(complete: Any) -> None:
    runner, fake, _ = complete
    corpus = runner.w.corpus
    for role in ("statement_writer", "query_writer"):
        for packet in _packets(fake, role):
            assert "ANCHOR" not in packet
            assert not re.search(r"\b(?:DORA|MiCA|PSD2):\S", packet)
            for c in runner.w.bank.candidates:
                for a in c.anchors:
                    text = corpus.texts[a.node_key].strip()
                    title = corpus.titles[(a.act, a.article)]
                    assert v2p.longest_run(packet, [text]) < v2p.ANCHOR_COPY_WORDS
                    if len(title) > 25:
                        assert title not in packet
    for packet in _packets(fake, "query_writer"):
        assert "topic T" in packet  # the statement, and nothing of the concept's facts
        assert "works on case" not in packet


def test_07_historical_exclusions_hidden_from_authors(complete: Any) -> None:
    runner, fake, _ = complete
    historical = [r for r in f.references() if r.kind == "DEVELOPMENT"]
    for role in ("v2_concept_author", "statement_writer", "query_writer"):
        for packet in _packets(fake, role):
            for r in historical:
                assert r.statement not in packet and r.key not in packet
    gates = _packets(fake, "gatekeeper")
    assert gates
    for packet in gates:
        for r in historical:
            assert r.key not in packet  # shown, if nominated, under an opaque id only


def test_08b_runner_dedup_keeps_the_earlier_duplicate(tmp_path: Path) -> None:
    d = tmp_path / "dup"
    fake = f.V2Fake(d.with_name("dup-spool"), dup_all=True)
    runner, _, _ = build(d, executor=fake, universe=30, n_target=2, evaluate=False)
    decided = runner.read_frozen("topics", hrun.TOPICS_FILE)["deduplication"]
    dropped = [cid for cid, x in decided.items() if not x["kept"]]
    assert dropped  # some profile has two candidates, and the later one goes
    rank = {c.candidate_id: c.rank for c in runner.w.bank.candidates}
    topic = {
        cid: re.search(r"topic (T\d{3})", runner.keyed[f"generated/{cid}"]["statement"]).group(1)  # type: ignore[union-attr]
        for cid in decided
    }
    for cid in dropped:
        same = [o for o in decided if topic[o] == topic[cid] and rank[o] < rank[cid]]
        assert any(decided[o]["kept"] for o in same)  # the earlier one of its topic stays
    for cid, x in decided.items():
        if x["kept"]:
            assert not any(
                topic[o] == topic[cid] and decided[o]["kept"] for o in decided if o != cid
            )


def test_10_topic_freeze_prevents_later_mutation(tmp_path: Path) -> None:
    d = tmp_path / "freeze"
    runner, fake, _ = build(d, evaluate=False, n_target=1)
    path = d / hrun.TOPICS_FILE
    data = json.loads(path.read_text())
    data["topics"][0]["statement"] += " Changed."
    path.write_text(json.dumps(data))
    runner.store.close()
    again = f.runner(dataclasses.replace(runner.w), fake, create=False)
    with pytest.raises(en.V2Stop) as stop:
        again.topics()
    assert stop.value.reason == "topic_mutation"
    with pytest.raises(en.V2Stop) as diverged:
        again.record(
            "candidate_generated", f"generated/{runner.w.bank.candidates[0].candidate_id}", {"x": 1}
        )
    assert diverged.value.reason == "state"


def test_12_pool_provenance_hidden_from_judges(complete: Any) -> None:
    runner, fake, _ = complete
    qids = {q["query_id"] for t in runner.topics() for q in t["queries"]}
    for role in ("v2_pool_judge_1", "v2_pool_judge_2", "judge_c", "v2_screen", "v2_answerability"):
        for packet in _packets(fake, role):
            for word in ("E5", "BM25", "BGE", "Qwen", "rank ", "anchor", "contributor"):
                assert word not in packet.replace("ranking", "")
            assert not any(q in packet for q in qids)


def _qrels(runner: hrun.Runner) -> dict[str, Any]:
    found: dict[str, Any] = runner.qrels()["qrels"]
    return found


def test_13_b1_exact_coverage(complete: Any) -> None:
    runner, fake, _ = complete
    pools = runner.need_pools()
    for need, record in _qrels(runner).items():
        assert set(record) == set(pools[need].nodes)  # every node, once, nothing else
        assert all(r["b1"] is not None for r in record.values())
    ids = {"C-000000000001": "DORA:1", "C-000000000002": "DORA:2"}
    answer = {
        "grades": [{"item_id": "C-000000000001", "grade": 1, "ambiguous": False, "reason": "x"}]
    }
    assert v2p.pool_problems(answer, ids)  # a node left out breaks the contract
    groups = [["DORA:1.1", "DORA:1.2"], ["DORA:2.1"], ["DORA:3.1", "DORA:3.2", "DORA:3.3"]]
    cut = v2.batches(groups, lambda b: " ".join(b), lambda t: len(t.split()), 3, 10_000)
    assert cut == [["DORA:1.1", "DORA:1.2", "DORA:2.1"], ["DORA:3.1", "DORA:3.2", "DORA:3.3"]]
    assert v2.batches([["a"] * 5], str, len, 3, 10_000) == [["a"] * 5]  # oversized: whole


def test_14_explicit_grade_0_persistence(complete: Any) -> None:
    runner, _, _ = complete
    zeros = [
        r
        for record in _qrels(runner).values()
        for r in record.values()
        if r["grade"] == 0 and r["outcome"] == "b1_negative"
    ]
    assert zeros and all(r["b2"] is None and not r["ambiguous"] for r in zeros)
    grades = jv.evaluation_grades(next(iter(_qrels(runner).values())))
    assert 0 in grades.values()


def test_15_b2_includes_every_decisive_candidate(tmp_path: Path) -> None:
    def zero(node: str, role: str) -> tuple[int, bool]:
        return (0, False) if role == "v2_pool_judge_1" else f.default_grade(node, role)

    d = tmp_path / "b2"
    fake = f.V2Fake(d.with_name("b2-spool"), grade=zero)
    runner, _, _ = build(d, executor=fake, evaluate=False, n_target=3)
    pools = runner.need_pools()
    checked = 0
    judged = {
        k.split("/", 1)[1]: json.loads(runner.store.text(p["qrels"], "kept"))
        for k, p in runner.keyed.items()
        if k.startswith("judged/") and "qrels" in p
    }
    assert judged
    for need, record in judged.items():
        decisive = set(pools[need].decisive_nodes())
        assert decisive
        for node in decisive:
            assert record[node]["b2"] is not None  # B1 said 0, B2 judged it anyway
            checked += 1
        for node, r in record.items():
            if node not in decisive:
                assert r["b2"] is None  # B1 0 and not decisive: final
    assert checked


def test_16_c_invoked_exactly_on_disagreement(complete: Any) -> None:
    runner, fake, _ = complete
    for record in _qrels(runner).values():
        for r in record.values():
            if r["b2"] is None:
                assert r["c"] is None
                continue
            differ = (r["b1"]["grade"], r["b1"]["ambiguous"]) != (
                r["b2"]["grade"],
                r["b2"]["ambiguous"],
            )
            assert (r["c"] is not None) == differ
            assert r["outcome"] == ("judge_c" if differ else "agreed")
    c_nodes = sum(len(re.findall(r"^ITEM C-", p, re.M)) for p in _packets(fake, "judge_c"))
    disputes = sum(
        r["c"] is not None for record in _qrels(runner).values() for r in record.values()
    )
    assert c_nodes >= disputes  # every dispute went to C (and the completeness follow-up)


def test_17_final_ambiguity_does_not_delete_need(complete: Any) -> None:
    runner, _, _ = complete
    qrels = _qrels(runner)
    ambiguous = [n for n, record in qrels.items() if any(r["ambiguous"] for r in record.values())]
    assert ambiguous and set(ambiguous) <= set(runner.primary())
    for n in ambiguous:
        assert v2.evaluable(qrels[n])


def test_18b_non_evaluable_needs_are_replaced_by_the_next_in_priority(tmp_path: Path) -> None:
    def low(node: str, role: str) -> tuple[int, bool]:
        return (1, False) if f._h("low", node) % 2 else (0, False)

    d = tmp_path / "low"
    fake = f.V2Fake(d.with_name("low-spool"), grade=low)
    runner, _, _ = build(d, executor=fake, evaluate=False, n_target=2)
    q = runner.qrels()
    assert q["n"] == 0 and len(q["not_evaluable"]) == len(runner.topics())  # the bank exhausted
    assert not runner.engine.ledger.stops


def test_19_answerability_does_not_filter_primary_sample(tmp_path: Path) -> None:
    d = tmp_path / "none"
    fake = f.V2Fake(d.with_name("none-spool"), answerability="none", answerability_2="partial")
    runner, fake, _ = build(d, executor=fake, evaluate=False, n_target=3)
    primary = runner.primary()
    assert len(primary) == 3
    paths = Counter(runner.keyed[f"answerability/{n}"]["path"] for n in primary)
    assert paths == {"judge_c": 3}  # a contradiction, a second judgment, then C
    assert fake.calls["v2_answerability_2"] == 3
    d2 = tmp_path / "none2"
    fake2 = f.V2Fake(d2.with_name("none2-spool"), answerability="none", answerability_2="none")
    runner2, _, _ = build(d2, executor=fake2, evaluate=False, n_target=3)
    assert runner2.primary() == primary
    assert {runner2.keyed[f"answerability/{n}"]["path"] for n in primary} == {
        "contradiction_confirmed"
    }


def test_20_qrel_freeze_immutable(complete: Any, tmp_path: Path) -> None:
    runner, fake, directory = complete
    copy = tmp_path / "copy"
    shutil.copytree(directory, copy)
    shutil.copytree(f.raw(directory), f.raw(copy))
    w = dataclasses.replace(runner.w, namespace=copy)
    again = f.runner(w, fake, create=False)
    before = _sha(copy / hrun.QRELS_FILE)
    again.freeze_qrels()  # recorded: never written again
    assert _sha(copy / hrun.QRELS_FILE) == before
    data = json.loads((copy / hrun.QRELS_FILE).read_text())
    data["primary"] = data["primary"][:-1]
    (copy / hrun.QRELS_FILE).write_text(json.dumps(data))
    with pytest.raises(en.V2Stop) as stop:
        again.qrels()
    assert stop.value.reason == "topic_mutation"
    again.store.close()


def test_21_completeness_cannot_mutate_primary_qrels(tmp_path: Path) -> None:
    d = tmp_path / "c"
    fake = f.V2Fake(d.with_name("c-spool"), flag_every=1)  # every passage flagged
    w, fake, _ = f.world(d, executor=fake, n_target=3)
    runner = f.runner(w, fake)
    original = runner.completeness
    seen = {}

    def spy() -> None:
        seen["before"] = _sha(d / hrun.QRELS_FILE)
        original()
        seen["after"] = _sha(d / hrun.QRELS_FILE)

    runner.completeness = spy  # type: ignore[method-assign]
    runner.build()
    assert seen["before"] == seen["after"]
    manifest = runner.read_frozen("completeness", hrun.COMPLETENESS_FILE)
    assert manifest["needs"] and any(r["residual_chunks"] for r in manifest["needs"].values())
    assert not runner.engine.ledger.stops  # residual relevance is a measurement, never a stop


def test_22_human_audit_cannot_mutate_primary_qrels(complete: Any) -> None:
    runner, _, directory = complete
    inputs = runner.evaluation_inputs()
    plain = hev.evaluate(hev.frozen_inputs(inputs, runner.w.layer))
    need = runner.primary()[0]
    pool = inputs["pools"]["pools"][need]
    labels = [
        {"need_id": need, "chunk_id": c, "grade": 3, "decoy": False} for c in pool["decisive"]
    ]
    audited = hev.evaluate(
        dataclasses.replace(hev.frozen_inputs(inputs, runner.w.layer), human={"labels": labels})
    )
    assert audited["primary"] == plain["primary"]
    assert audited["sensitivity"]["SA7_human_audit"]["status"] == "run"
    assert json.loads((directory / hrun.QRELS_FILE).read_text()) == inputs["qrels"]


def test_23_reserve_selection_independent_of_e_and_e5_metrics(tmp_path: Path) -> None:
    """Two constructions whose E rankings differ only inside the top 20 (so the pools are the
    same): the same primary sample, and different E metrics."""
    d1, d2 = tmp_path / "a", tmp_path / "b"
    r1, _, picked = build(d1, n_target=3)
    w, _, picked2 = f.world(d2, n_target=3)
    assert picked == picked2

    def reorder(system: str, queries: Any, work: Path) -> rv.Run:
        rankings = f.fake_rankings(queries, picked)[system]
        if system == "E":
            rankings = {q: list(reversed(r[:20])) + r[20:] for q, r in rankings.items()}
        return rv.Run(system, rankings)

    w.rank = reorder
    layer = w.layer
    w.evaluate = lambda inputs: hev.evaluate(hev.frozen_inputs(inputs, layer))
    fake = f.V2Fake(d2.with_name("b-spool"))
    r2 = f.runner(w, fake)
    r2.build()
    assert r1.primary() == r2.primary()
    e1 = r1.read_frozen("evaluation", hrun.EVALUATION_FILE)["primary"]["nDCG@10"]["mean_a"]
    e2 = r2.read_frozen("evaluation", hrun.EVALUATION_FILE)["primary"]["nDCG@10"]["mean_a"]
    assert e1 != e2


def test_24_no_aggregate_metric_before_qrel_freeze(tmp_path: Path) -> None:
    with pytest.raises(hev.EvaluationRefused):
        hev.frozen_inputs({"qrels": {"schema": "not-frozen"}}, f.base()["layer"].layer)
    d = tmp_path / "early"
    w, fake, _ = f.world(d, n_target=1)
    runner = f.runner(w, fake)
    runner.freeze_bank()
    with pytest.raises(hrun.NotReady):
        runner.evaluation_inputs()


def test_26_no_old_minimum_hard_stop(tmp_path: Path) -> None:
    d = tmp_path / "small"
    fake = f.V2Fake(d.with_name("small-spool"), grade=lambda n, r: (0, False))
    runner, _, _ = build(d, executor=fake, n_target=200, evaluate=False)
    assert runner.qrels()["n"] == 0 and not runner.engine.ledger.stops


def test_27_no_old_omission_audit_hard_stop(complete: Any) -> None:
    runner, _, _ = complete
    manifest = runner.read_frozen("completeness", hrun.COMPLETENESS_FILE)
    assert manifest["needs"]
    assert not runner.engine.ledger.stops


def test_28_restart_produces_same_final_scientific_state(tmp_path: Path) -> None:
    whole = tmp_path / "whole"
    build(whole, n_target=3)
    expected = frozen_hashes(whole)
    for crash_at in ("gen/writer", "judge/b2", "completeness/screen"):
        d = tmp_path / crash_at.replace("/", "-")
        fake = f.V2Fake(d.with_name(d.name + "-spool"), die_after=crash_at)
        w, _, _ = f.world(d, executor=fake, n_target=3)
        layer = w.layer
        w.evaluate = lambda inputs, layer=layer: hev.evaluate(hev.frozen_inputs(inputs, layer))
        runner = f.runner(w, fake)
        with pytest.raises(hf.Crash):
            runner.build()
        runner.store.close()
        again = f.runner(w, fake, create=False)
        again.build()
        assert frozen_hashes(d) == expected, crash_at
        events = again.store.head()[0]
        again.build()  # a complete state: nothing new
        assert again.store.head()[0] == events
        again.store.close()


def test_metadata_packet_shows_only_grade_1_or_more() -> None:
    corpus = f.base()["corpus"]
    finals = {
        "DORA:28.1": {"grade": 0, "ambiguous": False},
        "DORA:28.2": {"grade": 2, "ambiguous": False},
    }
    packet = v2p.metadata_packet(
        corpus, v2p.NeedText("R-1", "A question?"), finals, "complete", None
    )
    assert "[DORA:28.2]" in packet and "[DORA:28.1]" not in packet
    assert [v.path for v in v2p.qrel_views(finals)] == ["28.2"]


def test_answerability_c_packet_is_judge_c_v2() -> None:
    corpus = f.base()["corpus"]
    finals = {"DORA:28.2": {"grade": 3, "ambiguous": False}}
    a = {"answerability": "none", "reason": "nothing"}
    b = {"answerability": "complete", "reason": "all"}
    packet, items = v2p.answerability_c_packet(
        corpus, v2p.NeedText("R-1", "Q?"), finals, a, b, "C-0123456789ab", True
    )
    assert hr.is_instance("judge_c.v2", packet)
    assert items[0].target == bm.ANSWERABILITY_TARGET and "DORA:28.2" in packet


def test_evaluation_matches_the_evaluator(complete: Any) -> None:
    runner, _, _ = complete
    result = runner.read_frozen("evaluation", hrun.EVALUATION_FILE)
    assert result["schema"] == hev.SCHEMA and result["n"] == len(runner.primary())
    assert set(result["sensitivity"]) == {
        "SA1_ambiguity",
        "SA2_augmented_qrels",
        "SA3_fixed_scale_gain",
        "SA4_residual_relevance",
        "SA5_gains",
        "SA6_inference",
        "SA7_human_audit",
        "SA8_leave_one_contributor_out",
    }
    diag = result["pool_diagnostics"]
    for key in ("Judged@10 E", "Judged@10 E5", "Judged@20 E", "Judged@20 E5"):
        assert 0 <= diag[key]["mean"] <= 1
    assert diag["decisive_set"]["with_b2"] == diag["decisive_set"]["nodes"]
    assert runner.keyed["sensitivity"]["sha256"]


def test_human_audit_packet_is_blind_and_read_back_exactly(complete: Any) -> None:
    runner, _, _ = complete
    pools = runner.read_frozen("pools", hrun.POOLS_FILE)["pools"]
    need = runner.primary()[0]
    its = hau.items(need, pools[need], f.SALT)
    assert sum(i.decoy for i in its) == min(
        hau.DECOYS_PER_NEED, len(pools[need]["chunks"]) - len(pools[need]["decisive"])
    )
    texts = {c: runner.w.chunk_text[c] for c in runner.w.chunk_text}
    packet, template = hau.packet([(need, "A question?", its)], texts, f.SALT)
    assert (
        need not in packet and "E5" not in packet and "grade 3" not in packet.split("## Need", 1)[1]
    )
    for i in its:
        assert i.chunk_id not in packet
    labels = {k: 1 for k in template["labels"]}
    read = hau.read_labels(json.dumps({"labels": labels}), its)
    assert len(read) == len(its)
    with pytest.raises(hau.AuditError):
        hau.read_labels(json.dumps({"labels": {**labels, "x": 1}}), its)
    with pytest.raises(hau.AuditError):
        hau.read_labels(json.dumps({"labels": {k: True for k in labels}}), its)
    assert runner.keyed["human_audit/selection"]["needs"]


def test_status_and_summary_hold_counts_only(complete: Any) -> None:
    runner, _, directory = complete
    status = json.loads((directory / hrun.STATUS_FILE).read_text())
    text = json.dumps(status) + json.dumps(runner.summary())
    for t in runner.topics():
        assert t["statement"] not in text
        assert all(q["text"] not in text for q in t["queries"])
    assert statistics.fmean([1.0]) == 1.0


# ---------------------------------------------------------------------------
# the independent implementation review's findings (P1-1 to P1-5, P2-1 to P2-11)
# ---------------------------------------------------------------------------


def test_review_p1_1_anchor_copy_drop_survives_a_crash_between_its_records(tmp_path: Path) -> None:
    """A crash after `candidate_generated` with a rejected leakage and before its drop: the
    candidate is still never a topic."""
    d = tmp_path / "leak"
    w, fake, _ = f.world(d, universe=4, n_target=1)
    runner = f.runner(w, fake)
    runner.freeze_bank()
    c = w.bank.candidates[0]
    runner.record(
        "candidate_generated",
        f"generated/{c.candidate_id}",
        {
            "candidate_id": c.candidate_id,
            "need_id": c.need_id,
            "statement": "A synthetic firm on topic T001 asks what it must do in case 1.",
            "queries": [],
            "leakage": {"longest_run": 9, "share": 0.5, "rejected": True},
        },
    )
    assert c.candidate_id not in {g["candidate_id"] for g in runner.generated()}
    runner.generate()
    assert runner.dropped(c.candidate_id)
    runner.deduplicate()
    runner.freeze_topics()
    assert c.candidate_id not in {t["candidate_id"] for t in runner.topics()}


def test_review_p1_4_evaluation_waits_for_the_whole_construction(tmp_path: Path) -> None:
    d = tmp_path / "partial"
    w, fake, _ = f.world(d, n_target=2)
    runner = f.runner(w, fake)
    runner.freeze_bank()
    runner.generate()
    runner.deduplicate()
    runner.freeze_topics()
    runner.retrieve()
    runner.pools()
    runner.judge()
    runner.freeze_qrels()
    with pytest.raises(hrun.NotReady):  # no answerability, metadata or completeness yet
        runner.evaluation_inputs()
    assert "evaluation" not in runner.keyed


def test_review_p1_5_judge_c_packet_names_the_need_by_its_opaque_reference(complete: Any) -> None:
    runner, fake, _ = complete
    packets = [
        p for r, p in fake.packets if r == "judge_c" and ": the answerability of the need" not in p
    ]
    assert packets
    for packet in packets:
        need = re.search(r"^THE NEED\n(\S+):", packet, re.M)
        assert need and need.group(1).startswith("R-"), need


def test_review_p2_1_a_reused_article_gives_another_provision() -> None:
    bank = f.small_bank(260)  # DORA's demand is over its articles: reuse happens
    used: dict[tuple[str, str], list[str]] = {}
    for c in bank.candidates:
        for a in c.anchors:
            used.setdefault((a.act, a.article), []).append(a.node)
    units = {(u.act, u.article): u for us in f.base()["units"].values() for u in us}
    checked = 0
    for key, nodes in used.items():
        if len(nodes) > 1:
            distinct = min(len(nodes), len(units[key].nodes))
            assert len(set(nodes)) == distinct, key
            checked += 1
    assert checked


def test_review_p2_2_gate_shows_no_reference_key(complete: Any) -> None:
    _, fake, _ = complete
    dev = f.base()["dev"]
    for role, packet in fake.packets:
        if role == "gatekeeper":
            listed = packet.split("THE NEEDS THAT ALREADY EXIST", 1)[1]
            assert not any(f"- {n.need_id} " in listed for n in dev.needs)


def test_review_p2_3_an_exhausted_second_answerability_is_unknown(tmp_path: Path) -> None:
    d = tmp_path / "a2"
    fake = f.V2Fake(
        d.with_name("a2-spool"), answerability="none", broken=frozenset({"v2_answerability_2"})
    )
    runner, _, _ = build(d, executor=fake, evaluate=False, n_target=2)
    for n in runner.primary():
        record = runner.keyed[f"answerability/{n}"]
        assert record["answerability"] == "unknown"
        assert record["path"] == "contradiction_second_exhausted"


def test_review_p2_4_an_interrupted_wave_is_judged_again_as_planned(tmp_path: Path) -> None:
    whole = tmp_path / "whole"
    reference, _, _ = build(whole, n_target=3, evaluate=False)
    planned = [
        reference.keyed[k]["needs"] for k in sorted(reference.keyed) if k.startswith("wave/")
    ]
    d = tmp_path / "cut"
    fake = f.V2Fake(d.with_name("cut-spool"), die_after="judge/c")
    w, _, _ = f.world(d, executor=fake, n_target=3)
    runner = f.runner(w, fake)
    with pytest.raises(hf.Crash):
        runner.build()
    runner.store.close()
    again = f.runner(w, fake, create=False)
    again.build()
    assert [
        again.keyed[k]["needs"] for k in sorted(again.keyed) if k.startswith("wave/")
    ] == planned
    assert again.qrels() == reference.qrels()


def test_review_p2_5_an_admitted_job_whose_packet_changed_stops(
    complete: Any, tmp_path: Path
) -> None:
    runner, fake, directory = complete
    copy = tmp_path / "copy"
    shutil.copytree(directory, copy)
    shutil.copytree(f.raw(directory), f.raw(copy))
    w = dataclasses.replace(runner.w, namespace=copy)
    again = f.runner(w, fake, create=False)
    job_id = next(j for j in again.engine.ledger.admitted if j.startswith("gen/queries/"))
    changed = en.Job(job_id, "query_writer", "another packet", lambda reply: reply)
    with pytest.raises(en.V2Stop) as stop:
        again.engine.run([changed])
    assert stop.value.reason == "state"
    again.store.close()


def test_review_p2_6_a_failing_system_runs_once_more_then_is_not_completed(tmp_path: Path) -> None:
    d = tmp_path / "fail"
    w, fake, picked = f.world(d, n_target=2)
    calls: Counter[str] = Counter()
    rank = w.rank

    def flaky(system: str, queries: Any, work: Path) -> rv.Run:
        calls[system] += 1
        if system == "BM25" and calls[system] == 1:
            raise RuntimeError("out of memory")
        if system == "Qwen3-Embedding-0.6B":
            raise RuntimeError("out of memory")
        return rank(system, queries, work)

    w.rank = flaky
    layer = w.layer
    w.evaluate = lambda inputs: hev.evaluate(hev.frozen_inputs(inputs, layer))
    runner = f.runner(w, fake)
    runner.build()
    assert calls["BM25"] == 2 and calls["Qwen3-Embedding-0.6B"] == 2
    assert runner.keyed["runs"]["not_completed"] == ["Qwen3-Embedding-0.6B"]
    result = runner.read_frozen("evaluation", hrun.EVALUATION_FILE)
    assert result["primary"]["Recall@10"]["needs"] == len(runner.primary())


def test_review_p2_6_without_e_the_primary_comparison_is_not_made(tmp_path: Path) -> None:
    d = tmp_path / "noe"
    w, fake, _ = f.world(d, n_target=2)
    rank = w.rank

    def no_e(system: str, queries: Any, work: Path) -> rv.Run:
        if system == "E":
            raise RuntimeError("out of memory")
        return rank(system, queries, work)

    w.rank = no_e
    layer = w.layer
    w.evaluate = lambda inputs: hev.evaluate(hev.frozen_inputs(inputs, layer))
    runner = f.runner(w, fake)
    runner.build()
    result = runner.read_frozen("evaluation", hrun.EVALUATION_FILE)
    assert result["primary"] == {"status": "not_made", "systems_not_completed": ["E"]}


def test_review_p2_10_a_ranking_with_a_chunk_twice_stops(tmp_path: Path) -> None:
    d = tmp_path / "dup"
    w, fake, _ = f.world(d, n_target=1)
    rank = w.rank

    def twice(system: str, queries: Any, work: Path) -> rv.Run:
        run = rank(system, queries, work)
        if system == "BM25":
            first = next(iter(run.rankings))
            run.rankings[first] = [run.rankings[first][0], *run.rankings[first]]
        return run

    w.rank = twice
    runner = f.runner(w, fake)
    with pytest.raises(en.V2Stop) as stop:
        runner.build()
    assert stop.value.reason == "retrieval_identity"


def test_review_p2_11_metadata_sees_no_ambiguous_label() -> None:
    finals = {
        "DORA:28.1": {"grade": 3, "ambiguous": True},
        "DORA:28.2": {"grade": 2, "ambiguous": False},
    }
    assert [v.path for v in v2p.qrel_views(finals)] == ["28.2"]
    assert vs.SCHEMA  # the state schema the contract binds


def test_review_n2_interruptions_never_use_a_systems_retry(tmp_path: Path) -> None:
    """Two process interruptions while E ranks (a kill, a sleep): neither is a technical
    failure, so E is ranked on the third process, never `not completed`."""
    d = tmp_path / "interrupted"
    w, fake, _ = f.world(d, n_target=1)
    rank = w.rank
    seen: Counter[str] = Counter()

    def killed(system: str, queries: Any, work: Path) -> rv.Run:
        seen[system] += 1
        if system == "E" and seen[system] <= 2:
            raise hf.Crash("the process is killed while E ranks")
        return rank(system, queries, work)

    w.rank = killed
    for _ in range(2):
        runner = f.runner(w, fake)
        with pytest.raises(hf.Crash):
            runner.build()
        runner.store.close()
    runner = f.runner(w, fake, create=False)
    runner.build()
    assert runner.keyed["runs"]["not_completed"] == []
    assert "retrieval/E/ranked" in runner.keyed
    assert sum(k.startswith("retrieval/E/interrupted/") for k in runner.keyed) == 2


def test_review_n5_an_integrity_stop_in_retrieval_stays_a_stop(tmp_path: Path) -> None:
    d = tmp_path / "bad"
    w, fake, _ = f.world(d, n_target=1)
    rank = w.rank

    def twice(system: str, queries: Any, work: Path) -> rv.Run:
        run = rank(system, queries, work)
        if system == "BM25":
            first = next(iter(run.rankings))
            run.rankings[first] = [run.rankings[first][0], *run.rankings[first]]
        return run

    w.rank = twice
    for _ in range(4):
        runner = f.runner(w, fake)
        with pytest.raises(en.V2Stop) as stop:
            runner.build()
        assert stop.value.reason == "retrieval_identity"
        assert "retrieval/BM25/not_completed" not in runner.keyed
        runner.store.close()


def test_review_n9_a_system_that_ends_its_process_every_time_is_not_completed(
    tmp_path: Path,
) -> None:
    """E kills its process on every attempt: two interruptions are free, then each counts, and
    on the fifth process E is not completed and the construction goes on without it."""
    d = tmp_path / "killer"
    w, fake, _ = f.world(d, n_target=1)
    rank = w.rank
    seen: Counter[str] = Counter()

    def killed(system: str, queries: Any, work: Path) -> rv.Run:
        seen[system] += 1
        if system == "E":
            raise hf.Crash("the system ends its process")
        return rank(system, queries, work)

    w.rank = killed
    for _ in range(4):
        runner = f.runner(w, fake)
        with pytest.raises(hf.Crash):
            runner.build()
        runner.store.close()
    runner = f.runner(w, fake, create=False)
    runner.build()
    assert seen["E"] == 4
    assert runner.keyed["runs"]["not_completed"] == ["E"]
