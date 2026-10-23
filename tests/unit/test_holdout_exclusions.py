"""The development need exclusions of a confirmatory TEST (ingest/holdout_exclusions.py, D-066):
one canonical artifact, read once, whose identity is its bytes'; collected from durable states
made by the real Store, under the construction's own lock, verified whole and read twice to the
same head; every admitted statement and nothing else; written once; and the backstop that
refuses a READY need repeating one. Synthetic statements only: no TEST text."""

from __future__ import annotations

import dataclasses
import importlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from ingest import holdout as ho
from ingest import holdout_contract as hc
from ingest import holdout_exclusions as hxe
from ingest import holdout_state as hs
from tests.holdout_fakes import FAKE_EXCLUSIONS, fake_sources, write_exclusions

CONTRACT = "c" * 64


# ---------------------------------------------------------------------------
# the artifact
# ---------------------------------------------------------------------------


def test_one_set_has_one_canonical_text_and_its_identity_is_its_bytes(tmp_path):
    found = write_exclusions(tmp_path)
    read = hxe.read(tmp_path / hxe.FILE)
    assert found == read.identity == {"schema": hxe.SCHEMA, "sha256": read.sha256, "count": 2}
    assert read.entries == tuple(sorted(FAKE_EXCLUSIONS, key=lambda e: e.key))
    assert read.sources == fake_sources()
    # the same set in another order: the same text, the same identity
    other = tmp_path / "other"
    assert write_exclusions(other, tuple(reversed(FAKE_EXCLUSIONS))) == found
    with pytest.raises(FileExistsError):  # written once: a file that exists is never replaced
        write_exclusions(tmp_path)


def change_entry(i: int, **fields: Any):
    return lambda d: d["entries"][i].update(fields)


def change_source(label: str, **fields: Any):
    return lambda d: next(s for s in d["sources"] if s["label"] == label).update(fields)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (change_entry(0, key="G01-a1"), "<run>/<attempt>"),
        (change_entry(0, key="run-9/G01-a1"), "of no source"),
        (change_entry(0, statement=" "), "an empty statement"),
        (change_entry(0, locked="no"), "neither true nor false"),
        (change_entry(1, record_sha256="abc"), "neither null nor a SHA-256"),
        (change_entry(0, record_sha256="b" * 64), "a READY need without its lock"),
        (lambda d: d["entries"].append(dict(d["entries"][0])), "a key appears twice"),
        (change_entry(0, extra=1), "not exactly key"),
        (change_source("run-1", statements=5), "its counts are not its entries'"),
        (change_source("run-2", head_sha256="x"), "head_sha256 is not a SHA-256"),
        (lambda d: d["sources"].append(dict(d["sources"][0])), "a source appears twice"),
        (lambda d: d.update(schema="development-need-exclusions.v1"), "schema is not"),
        (lambda d: d.pop("sources"), "exactly schema, sources and entries"),
    ],
)
def test_an_artifact_with_a_problem_is_refused_never_repaired(tmp_path, change, message):
    write_exclusions(tmp_path)
    path = tmp_path / hxe.FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    with pytest.raises(hxe.ExclusionsError, match=message) as refused:
        hxe.read(path)
    assert not any(e.statement in str(refused.value) for e in FAKE_EXCLUSIONS)


def test_a_text_that_is_not_its_canonical_form_is_refused(tmp_path):
    write_exclusions(tmp_path)
    path = tmp_path / hxe.FILE
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(hxe.ExclusionsError, match="canonical"):
        hxe.read(path)
    with pytest.raises(hxe.ExclusionsError, match="does not exist"):
        hxe.read(tmp_path / "absent.json")


def test_one_read_gives_the_entries_and_the_identity_never_two_versions(tmp_path, monkeypatch):
    """P1 (D-066): the bytes are read exactly once, and everything used comes from them. A file
    replaced A -> B between any two reads gives A whole (A's entries, sources and identity),
    never B's identity with A's entries."""
    a = write_exclusions(tmp_path / "a")
    b = write_exclusions(tmp_path / "b", (FAKE_EXCLUSIONS[0],))
    assert a != b
    target = tmp_path / "provenance" / hxe.FILE
    target.parent.mkdir()
    versions: list[bytes] = [(tmp_path / v / hxe.FILE).read_bytes() for v in ("a", "b")]
    target.write_bytes(versions[0])
    reads: list[str] = []
    real = Path.read_bytes

    def replaced_after_the_first_read(self: Path) -> bytes:
        if self == target:
            reads.append("read_bytes")
            return versions[0] if len(reads) == 1 else versions[1]
        return bytes(real(self))

    def no_other_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the artifact read by another way")

    monkeypatch.setattr(Path, "read_bytes", replaced_after_the_first_read)
    for name in ("read_text", "open"):
        real_other = getattr(Path, name)

        def guarded(self: Path, *args: Any, _real: Any = real_other, **kwargs: Any) -> Any:
            if self == target:
                no_other_read()
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(Path, name, guarded)
    found = hxe.read(target)
    assert reads == ["read_bytes"]
    assert found.identity == a and found.entries == tuple(
        sorted(FAKE_EXCLUSIONS, key=lambda e: e.key)
    )
    assert hxe.identity(target) == b  # a later, separate read sees B whole


def test_write_returns_the_identity_of_the_bytes_written_and_checks_them(tmp_path, monkeypatch):
    entries, sources = FAKE_EXCLUSIONS, fake_sources()
    other = hxe.parse(hxe.render(entries[:1], fake_sources(entries[:1])).encode(), "other")
    monkeypatch.setattr(hxe, "read", lambda path: other)  # the file changed under the writer
    with pytest.raises(hxe.ExclusionsError, match="changed while it was written"):
        hxe.write(tmp_path / hxe.FILE, entries, sources)
    with pytest.raises(hxe.ExclusionsError, match="counts"):  # checked before any byte is written
        hxe.write(tmp_path / "bad.json", entries, fake_sources(entries[:1]))
    assert not (tmp_path / "bad.json").exists()


# ---------------------------------------------------------------------------
# collecting them from durable states made by the real Store
# ---------------------------------------------------------------------------


def admitted(store: hs.Store, cell: str, pid: str, stage: str, value: Any, state: str) -> None:
    store.record(
        "stage_admitted",
        f"{pid}/{stage}/admitted",
        {
            "inputs_sha256": "0" * 64,
            "value": value,
            "value_sha256": hs.sha_value(value),
            "audit": None,
            "state": state,
        },
        cell=cell,
        proposal=pid,
        stage=stage,
    )


def development_state(
    directory: Path, attempts: list[tuple[str, str, str]], contract: str = CONTRACT
) -> Path:
    """A construction state written by the real Store, as an aborted run leaves it. Each attempt
    (proposal id, statement, fate): its statement run "failed" (never admitted), or its admitted
    statement was "rejected" by the gate, or "locked", or "ready"; "ready_without_lock" and
    "lock_differs" are the defects the collection must refuse."""
    cells = sorted({pid.split("-")[0] for pid, _, _ in attempts})
    raw = directory.with_name(directory.name + "-raw")
    with hs.Store.create(directory, raw, {"pre_authoring_contract": contract}, cells, []) as store:
        for pid, statement, fate in attempts:
            cell, number = pid.split("-a")
            store.record(
                "attempt_opened", f"{pid}/opened", {"number": int(number)}, cell=cell, proposal=pid
            )
            if fate == "failed":
                continue
            written = {"proposal_id": "R-x", "primary": f" {statement} ", "alternative": None}
            admitted(store, cell, pid, "statement", written, "candidate")
            if fate == "rejected":
                closed = {"outcome": "overlap_paraphrase", "against": "dev", "note": None}
                store.record(
                    "attempt_closed",
                    f"{pid}/closed",
                    closed | {"state": "rejected_methodological"},
                    cell=cell,
                    proposal=pid,
                    stage="overlap_gate",
                )
                continue
            if fate in ("locked", "ready", "lock_differs"):
                locked = statement if fate != "lock_differs" else statement + " Changed."
                admitted(store, cell, pid, "lock", {"need": {"statement": locked}}, "locked")
            if fate in ("ready", "ready_without_lock"):
                record = {"need": {"statement": statement}, "qrels": []}
                store.record(
                    "need_ready",
                    f"{cell}/ready",
                    {
                        "need_id": f"N-{pid}",
                        "record": record,
                        "record_sha256": hs.sha_value(record),
                    },
                    cell=cell,
                    proposal=pid,
                    stage="ready",
                )
    return raw


def source(label: str, directory: Path, raw: Path, contract: str = CONTRACT) -> hxe.Source:
    return hxe.Source(label, directory, raw, contract)


def files_of(directory: Path) -> dict[str, bytes]:
    return {
        p.name: p.read_bytes() for p in sorted((directory / hs.STATE_DIR).rglob("*")) if p.is_file()
    }


def test_collect_takes_every_admitted_statement_with_the_inventory_and_writes_nothing(tmp_path):
    one, two = tmp_path / "run-1", tmp_path / "run-2"
    raw_one = development_state(
        one,
        [
            ("G01-a1", "First synthetic need.", "rejected"),  # the gate rejected it: in
            ("G01-a2", "Second synthetic need.", "ready"),
            ("G02-a1", "Never admitted.", "failed"),  # technical garbage: out
            ("G03-a1", "Third synthetic need.", "locked"),
        ],
    )
    raw_two = development_state(two, [("G07-a3", "Fourth synthetic need.", "locked")])
    before = {d: files_of(d) for d in (one, two)}
    entries, states = hxe.collect([source("run-2", two, raw_two), source("run-1", one, raw_one)])
    assert [(e.key, e.statement, e.locked, e.record_sha256 is not None) for e in entries] == [
        ("run-1/G01-a1", "First synthetic need.", False, False),
        ("run-1/G01-a2", "Second synthetic need.", True, True),
        ("run-1/G03-a1", "Third synthetic need.", True, False),
        ("run-2/G07-a3", "Fourth synthetic need.", True, False),
    ]
    with hs.Store.open(one, raw_one, readonly=True) as store:
        events = store.events()
        ready = store.ready()[0]
    assert entries[1].record_sha256 == ready["record_sha256"]
    assert states[0] == hxe.SourceState(
        "run-1", CONTRACT, len(events), events[-1].sha256, "building", 3, 2, 1
    )
    assert (states[1].label, states[1].statements, states[1].locked) == ("run-2", 1, 1)
    assert {d: files_of(d) for d in (one, two)} == before  # nothing written, no journal left
    written = hxe.write(tmp_path / hxe.FILE, entries, states)
    assert written["count"] == 4 and hxe.read(tmp_path / hxe.FILE).sources == tuple(states)


def test_a_construction_process_holding_the_state_stops_the_collection(tmp_path):
    raw = development_state(tmp_path / "run-1", [("G01-a1", "A synthetic need.", "locked")])
    writer = hs.Store.open(tmp_path / "run-1", raw)  # the construction's lock, exclusive
    try:
        assert not list((tmp_path / "run-1" / hs.STATE_DIR).glob("*-journal"))  # no journal
        with pytest.raises(hxe.ExclusionsError, match="a construction process holds its state"):
            hxe.collect([source("run-1", tmp_path / "run-1", raw)])
    finally:
        writer.close()
    assert len(hxe.collect([source("run-1", tmp_path / "run-1", raw)])[0]) == 1


def test_no_construction_process_can_take_the_state_while_it_is_collected(tmp_path, monkeypatch):
    raw = development_state(tmp_path / "run-1", [("G01-a1", "A synthetic need.", "locked")])
    real = hs.Store.verify
    refused: list[str] = []

    def verify_while_a_writer_tries(self: hs.Store) -> dict[str, Any]:
        try:
            hs.Store.open(tmp_path / "run-1", raw).close()
        except hs.StateBusy as error:
            refused.append(str(error))
        return real(self)

    monkeypatch.setattr(hs.Store, "verify", verify_while_a_writer_tries)
    assert len(hxe.collect([source("run-1", tmp_path / "run-1", raw)])[0]) == 1
    assert refused and "one construction process at a time" in refused[0]


def test_a_write_while_the_state_is_read_stops_the_collection(tmp_path, monkeypatch):
    """A writer that ignores the lock appends a valid event between the two reads of the log:
    the heads differ and the collection stops, never a mixture of two states."""
    directory = tmp_path / "run-1"
    raw = development_state(directory, [("G01-a1", "A synthetic need.", "locked")])
    real = hs.Store.verify

    def verify_then_a_rogue_append(self: hs.Store) -> dict[str, Any]:
        summary = real(self)
        db = sqlite3.connect(directory / hs.STATE_DIR / hs.DB_FILE)
        seq, prev = db.execute("SELECT max(seq), sha256 FROM events ORDER BY seq DESC").fetchone()
        where = (None, None, None, None, hs.canonical({}))
        digest = hs._chain(prev, seq + 1, "rogue", "resumed", "x", where)
        db.execute(
            "INSERT INTO events VALUES "
            "(?, 'rogue', 'resumed', 'x', NULL, NULL, NULL, NULL, ?, ?, ?)",
            (seq + 1, hs.canonical({}), prev, digest),
        )
        db.commit()
        db.close()
        return summary

    monkeypatch.setattr(hs.Store, "verify", verify_then_a_rogue_append)
    with pytest.raises(hxe.ExclusionsError, match="its state changed while it was read"):
        hxe.collect([source("run-1", directory, raw)])


@pytest.mark.parametrize(
    ("fate", "message"),
    [
        ("ready_without_lock", "a READY need without its lock"),
        ("lock_differs", "its lock is not its admitted statement"),
    ],
)
def test_an_inconsistent_state_stops_the_collection(tmp_path, fate, message):
    raw = development_state(tmp_path / "run-1", [("G01-a1", "A synthetic need.", fate)])
    with pytest.raises(hxe.ExclusionsError, match=message) as refused:
        hxe.collect([source("run-1", tmp_path / "run-1", raw)])
    assert "A synthetic need." not in str(refused.value)


def test_a_table_that_is_not_what_its_event_log_builds_stops_the_collection(tmp_path):
    raw = development_state(tmp_path / "run-1", [("G01-a1", "A synthetic need.", "locked")])
    db = sqlite3.connect(tmp_path / "run-1" / hs.STATE_DIR / hs.DB_FILE)
    db.execute(
        "UPDATE stages SET value = ? WHERE stage = 'statement'",
        (json.dumps({"primary": "Other."}),),
    )
    db.commit()
    db.close()
    with pytest.raises(hxe.ExclusionsError, match="stages: not what the event log builds"):
        hxe.collect([source("run-1", tmp_path / "run-1", raw)])


def test_the_inventory_binds_the_exact_state_its_contract_and_its_head(tmp_path):
    directory = tmp_path / "run-1"
    raw = development_state(directory, [("G01-a1", "A synthetic need.", "locked")])
    with pytest.raises(hxe.ExclusionsError, match="built under another contract"):
        hxe.collect([source("run-1", directory, raw, contract="d" * 64)])
    first = hxe.collect([source("run-1", directory, raw)])
    with hs.Store.open(directory, raw) as store:  # one more event: another head
        store.record("resumed", "resumed-1", {})
    second = hxe.collect([source("run-1", directory, raw)])
    assert first[0] == second[0]  # the same statements
    assert second[1][0].head_sha256 != first[1][0].head_sha256
    assert second[1][0].events == first[1][0].events + 1
    assert hxe.render(*first) != hxe.render(*second)  # and so another identity


def test_a_source_that_is_no_closed_state_stops_the_collection(tmp_path):
    directory = tmp_path / "run-1"
    raw = development_state(directory, [("G01-a1", "A synthetic need.", "locked")])
    (directory / hs.STATE_DIR / f"{hs.DB_FILE}-journal").write_bytes(b"")
    with pytest.raises(hxe.ExclusionsError, match="a journal beside its state"):
        hxe.collect([source("run-1", directory, raw)])
    (directory / hs.STATE_DIR / f"{hs.DB_FILE}-journal").unlink()
    (directory / hs.STATE_DIR / hs.LOCK_FILE).unlink()
    with pytest.raises(hxe.ExclusionsError, match="does not exist"):
        hxe.collect([source("run-1", directory, raw)])
    assert not (directory / hs.STATE_DIR / hs.LOCK_FILE).exists()  # never created by a read
    with pytest.raises(hxe.ExclusionsError, match="no construction state"):
        hxe.collect([source("run-2", tmp_path / "absent", raw)])
    with pytest.raises(hxe.ExclusionsError, match="named twice"):
        hxe.collect([source("run-1", directory, raw), source("run-1", directory, raw)])


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def aborted_states(root: Path) -> list[list[str]]:
    """One synthetic state per aborted run the contract names, built under its contract."""
    given = []
    for n, run in enumerate(hc.ABORTED_RUNS, start=1):
        directory = root / f"run-{n}"
        contract = str(run["contract_sha256"])
        raw = development_state(directory, [(f"G0{n}-a1", f"Need {n}.", "locked")], contract)
        given.append([f"run-{n}", str(directory), str(raw)])
    return given


def test_the_cli_collects_every_aborted_run_once_and_prints_no_statement(
    tmp_path, monkeypatch, capsys
):
    module = importlib.import_module("scripts.holdout")  # scripts/ is no package for mypy
    repo = tmp_path / "public"
    repo.mkdir()
    ho.git(repo, "init", "-q")
    (repo / ".gitignore").write_text("private-*/\n", encoding="utf-8")
    monkeypatch.setattr(module, "REPO", repo)
    test = repo / "private-test"
    (test / "provenance").mkdir(parents=True)
    given = aborted_states(tmp_path / "states")
    command = ["--dir", str(test), "development-exclusions"]
    for run in given:
        command += ["--run", *run]
    assert module.main(command) == 0
    printed = capsys.readouterr().out
    found = hxe.read(test / "provenance" / hxe.FILE)
    assert printed == f"wrote {len(given)} development need exclusions: {found.sha256}\n"
    assert not any(f"Need {n}." in printed for n in range(1, len(given) + 1))
    assert [s.contract_sha256 for s in found.sources] == [
        r["contract_sha256"] for r in hc.ABORTED_RUNS
    ]
    with pytest.raises(FileExistsError):  # written once, never replaced
        module.main(command)
    # every aborted run once: one missing, one twice or one unknown is refused
    for runs, message in (
        (given[:-1], "not named"),
        ([*given, given[0]], "twice"),
        ([*given, ["run-99", given[0][1], given[0][2]]], "none of the aborted runs"),
    ):
        with pytest.raises(ho.HoldoutError, match=message):
            module.development_sources(runs, hc.ABORTED_RUNS)
    shown = repo / "shown"
    (shown / "provenance").mkdir(parents=True)
    with pytest.raises(ho.HoldoutError, match="not ignored"):
        module.main(["--dir", str(shown), *command[2:]])
    assert not (shown / "provenance" / hxe.FILE).exists()


# ---------------------------------------------------------------------------
# the freeze's backstop
# ---------------------------------------------------------------------------


def test_a_ready_need_that_repeats_a_development_need_is_refused_by_id_alone():
    developed, readied = FAKE_EXCLUSIONS
    ready = {
        "T-1": ("A new synthetic need.", "c" * 64),
        # the same words, spaced and cased otherwise
        "T-2": ("  " + developed.statement.upper().replace(" ", "   "), "d" * 64),
        "T-3": ("Another new synthetic need.", str(readied.record_sha256)),  # the same record
    }
    found = hxe.reuse_problems(ready, FAKE_EXCLUSIONS)
    assert found == [
        "T-2: its statement repeats a development need",
        "T-3: its READY record repeats a development need's",
    ]
    assert not any(e.statement in p for p in found for e in FAKE_EXCLUSIONS)
    assert hxe.reuse_problems(ready, ()) == []
    assert dataclasses.replace(developed, locked=True) != developed  # the flag is part of it
