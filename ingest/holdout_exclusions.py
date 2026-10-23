"""The development need exclusions of a confirmatory TEST (S3, D-066): every statement an earlier,
aborted construction admitted, so that none of them reappears in the new TEST.

A private artifact, provenance/development_need_exclusions.json in the private TEST directory,
which git ignores. It holds:
- sources: the canonical inventory of the durable states it was built from, one per aborted run:
  its label, the contract it was built under, its event count and head (the state's identity),
  its status, and its counts of admitted statements, locks and READY needs. They name the exact
  states the frozen universe comes from;
- entries: one per statement a source admitted, before its overlap gate decided: an opaque key
  (run label and attempt), the statement as admitted, whether its attempt reached the lock, and
  its READY record's hash when it became READY. A statement the gate then rejected is in it; a
  statement run that failed technically, never admitted, is not.

It is built mechanically (collect), never by hand and never by reading what a statement means,
written once (write), and never printed. Each source is read under its construction's own lock,
taken shared, so that no construction process holds or takes it meanwhile; it is verified whole
(holdout_state Store.verify: SQLite's integrity, the event log's chain, every table exactly what
the log builds, every artifact with its hash) between two reads of its event log that must end
at the same head; and its entries are taken from the tables that log alone builds. Any
inconsistency stops the collection: never a partial set.

The pre-authoring contract records its identity only (schema, SHA-256, count), fixed before the
first author's call. Every use reads the file once (read): the entries, the sources and the
identity come from the same bytes, so that no identity of one version can vouch for the entries
of another.

Where it is read: the overlap gate compares every proposal with these statements as hidden
references, beside DEV, the fixtures and the ready TEST needs (ingest/holdout_pipeline.py _gate);
no author ever receives them. At the freeze, reuse_problems refuses a READY need whose statement
or whole record repeats one (a backstop: a new construction's state holds nothing of an earlier
one).
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingest import holdout as ho
from ingest import holdout_state as hs

SCHEMA = "development-need-exclusions.v2"
FILE = "development_need_exclusions.json"
# a run's label; an entry's key: <run label>/<attempt>, the attempt's proposal id as its run's
# state names it
LABEL = re.compile(r"[a-z0-9][a-z0-9._-]*")
KEY = re.compile(r"(?P<label>[a-z0-9][a-z0-9._-]*)/[A-Z][0-9]{2}-a[1-9][0-9]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
ENTRY_FIELDS = frozenset({"key", "statement", "locked", "record_sha256"})
SOURCE_FIELDS = frozenset(
    {"label", "contract_sha256", "events", "head_sha256", "status", "statements", "locked", "ready"}
)


class ExclusionsError(ho.HoldoutError):
    """Development need exclusions that cannot be built or used: refused, never repaired."""


@dataclass(frozen=True)
class Exclusion:
    """One statement a development run admitted: its key (<run>/<attempt>), the statement as
    admitted, whether its attempt reached the lock, and its READY record's hash if it became
    READY."""

    key: str
    statement: str
    locked: bool = True
    record_sha256: str | None = None


@dataclass(frozen=True)
class SourceState:
    """One development run's durable state as the collection read it: the contract it was built
    under, its event count and head (the state's identity), its status, and its counts of admitted
    statements, locks and READY needs."""

    label: str
    contract_sha256: str
    events: int
    head_sha256: str
    status: str
    statements: int
    locked: int
    ready: int


@dataclass(frozen=True)
class Exclusions:
    """The artifact as one read of its bytes gave it: its entries, its sources and the SHA-256
    of those same bytes."""

    entries: tuple[Exclusion, ...]
    sources: tuple[SourceState, ...]
    sha256: str

    @property
    def identity(self) -> dict[str, Any]:
        """What a contract records of it: its schema, its SHA-256 and its count of entries,
        never a statement."""
        return {"schema": SCHEMA, "sha256": self.sha256, "count": len(self.entries)}


def render(entries: Sequence[Exclusion], sources: Sequence[SourceState]) -> str:
    """The artifact's canonical text: sources sorted by label, entries by key, so that one set
    has one identity."""
    data = {
        "schema": SCHEMA,
        "sources": [dataclasses.asdict(s) for s in sorted(sources, key=lambda s: s.label)],
        "entries": [
            {
                "key": e.key,
                "statement": e.statement,
                "locked": e.locked,
                "record_sha256": e.record_sha256,
            }
            for e in sorted(entries, key=lambda e: e.key)
        ],
    }
    return json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _source_problems(i: int, s: Any) -> list[str]:
    if not isinstance(s, Mapping) or set(s) != SOURCE_FIELDS:
        return [f"source {i}: not exactly {sorted(SOURCE_FIELDS)}"]
    found = []
    if not isinstance(s["label"], str) or not LABEL.fullmatch(s["label"]):
        found.append(f"source {i}: a label that is not one")
    for name in ("contract_sha256", "head_sha256"):
        if not isinstance(s[name], str) or not SHA256.fullmatch(s[name]):
            found.append(f"source {i}: {name} is not a SHA-256")
    if not isinstance(s["status"], str) or not s["status"]:
        found.append(f"source {i}: no status")
    for name in ("events", "statements", "locked", "ready"):
        if not _count(s[name]):
            found.append(f"source {i}: {name} is not a count")
    return found


def _entry_problems(i: int, e: Any) -> list[str]:
    if not isinstance(e, Mapping) or set(e) != ENTRY_FIELDS:
        return [f"entry {i}: not exactly key, statement, locked and record_sha256"]
    found = []
    if not isinstance(e["key"], str) or not KEY.fullmatch(e["key"]):
        found.append(f"entry {i}: a key that is not <run>/<attempt>")
    if not isinstance(e["statement"], str) or not e["statement"].strip():
        found.append(f"entry {i}: an empty statement")
    if not isinstance(e["locked"], bool):
        found.append(f"entry {i}: locked is neither true nor false")
    sha = e["record_sha256"]
    if sha is not None and not (isinstance(sha, str) and SHA256.fullmatch(sha)):
        found.append(f"entry {i}: record_sha256 is neither null nor a SHA-256")
    if sha is not None and e["locked"] is not True:
        found.append(f"entry {i}: a READY need without its lock")
    return found


def problems(data: Any) -> list[str]:
    """What makes a parsed artifact unusable: its shape, and any disagreement between its
    entries and the inventory of its sources. No statement is ever named."""
    if not isinstance(data, Mapping) or set(data) != {"schema", "sources", "entries"}:
        return ["not an object with exactly schema, sources and entries"]
    if data["schema"] != SCHEMA:
        return [f"schema is not {SCHEMA}"]
    if not isinstance(data["sources"], list) or not isinstance(data["entries"], list):
        return ["sources or entries is not a list"]
    found = [p for i, s in enumerate(data["sources"]) for p in _source_problems(i, s)]
    found += [p for i, e in enumerate(data["entries"]) for p in _entry_problems(i, e)]
    if found:
        return found
    labels = [s["label"] for s in data["sources"]]
    keys = [e["key"] for e in data["entries"]]
    if len(set(labels)) != len(labels):
        found.append("a source appears twice")
    if len(set(keys)) != len(keys):
        found.append("a key appears twice")
    counts = {label: [0, 0, 0] for label in labels}
    for i, e in enumerate(data["entries"]):
        label = KEY.fullmatch(e["key"])["label"]  # type: ignore[index]
        if label not in counts:
            found.append(f"entry {i}: of no source")
            continue
        counts[label][0] += 1
        counts[label][1] += e["locked"]
        counts[label][2] += e["record_sha256"] is not None
    for i, s in enumerate(data["sources"]):
        if counts[s["label"]] != [s["statements"], s["locked"], s["ready"]]:
            found.append(f"source {i}: its counts are not its entries'")
    return found


def parse(data: bytes, where: str) -> Exclusions:
    """The artifact from these bytes: parsed, checked, exactly its canonical form, and identified
    by these same bytes' SHA-256."""
    try:
        text = data.decode("utf-8")
        parsed = json.loads(text)
    except ValueError as error:
        raise ExclusionsError(f"{where} is not JSON: {error.__class__.__name__}") from None
    found = problems(parsed)
    if found:
        raise ExclusionsError(f"{where}: {found[:3]}")
    entries = tuple(
        Exclusion(e["key"], e["statement"], e["locked"], e["record_sha256"])
        for e in parsed["entries"]
    )
    sources = tuple(SourceState(**s) for s in parsed["sources"])
    if render(entries, sources) != text:
        raise ExclusionsError(f"{where} is not in its canonical form")
    return Exclusions(entries, sources, hashlib.sha256(data).hexdigest())


def read(path: Path) -> Exclusions:
    """The artifact at path, read once: whatever is used of it (its entries, its sources and its
    identity) comes from those bytes and from no other read."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise ExclusionsError(f"{path} does not exist") from None
    except OSError as error:
        raise ExclusionsError(f"{path} cannot be read: {error.__class__.__name__}") from None
    return parse(data, str(path))


def identity(path: Path) -> dict[str, Any]:
    """The identity of the artifact at path, from one read (read)."""
    return read(path).identity


def write(
    path: Path, entries: Sequence[Exclusion], sources: Sequence[SourceState]
) -> dict[str, Any]:
    """The artifact, checked before it is written and written once (a file that exists is never
    replaced); its identity is the one of the bytes written, read back to be those bytes."""
    data = render(entries, sources).encode("utf-8")
    made = parse(data, "the exclusions to write")
    with path.open("xb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    if read(path).sha256 != made.sha256:
        raise ExclusionsError(f"{path} changed while it was written")
    return made.identity


# ---------------------------------------------------------------------------
# collecting them from the durable states of the aborted runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One development run to collect, named explicitly: its label, the folder that holds its
    construction state (state/ under it), the folder of its raw artifacts, and the contract its
    state must have been built under."""

    label: str
    directory: Path
    raw_dir: Path
    contract_sha256: str


@contextmanager
def _read_lock(label: str, directory: Path) -> Iterator[None]:
    """The construction's own lock (holdout_state), taken shared: no construction process holds
    the state, and none can take it while it is read. The lock file is opened read-only and is
    never created: nothing is written into a run's state."""
    lock = directory / hs.STATE_DIR / hs.LOCK_FILE
    try:
        fd = os.open(lock, os.O_RDONLY)
    except FileNotFoundError:
        raise ExclusionsError(f"{label}: {lock} does not exist") from None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ExclusionsError(f"{label}: a construction process holds its state") from None
        yield
    finally:
        os.close(fd)


def _log(store: hs.Store) -> tuple[list[hs.Event], str]:
    """The event log in one read snapshot, and its head with the chain recomputed."""
    store.db.execute("BEGIN")
    try:
        events = store.events()
    finally:
        store.db.execute("COMMIT")
    return events, hs.chained(events)


def _meta(tables: sqlite3.Connection, key: str) -> str | None:
    row = tables.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _statement_of(value: Any, *path: str) -> str:
    for name in path:
        value = value.get(name) if isinstance(value, Mapping) else None
    return str(value).strip() if isinstance(value, str) else ""


def _read_source(source: Source) -> tuple[list[hs.Event], str]:
    """A source's event log, under its lock: read, the whole state verified, read again; the two
    reads must end at the same head."""
    label, state = source.label, source.directory / hs.STATE_DIR
    if not (state / hs.DB_FILE).is_file():
        raise ExclusionsError(f"{label}: no construction state in {state}")
    if any((state / f"{hs.DB_FILE}{suffix}").exists() for suffix in ("-journal", "-wal")):
        raise ExclusionsError(f"{label}: a journal beside its state: a write left half done")
    with _read_lock(label, source.directory):
        try:
            with hs.Store.open(source.directory, source.raw_dir, readonly=True) as store:
                before, head = _log(store)
                store.verify()
                events, after = _log(store)
        except (hs.StateError, sqlite3.Error) as error:
            raise ExclusionsError(f"{label}: {error}") from None
    if (len(before), head) != (len(events), after):
        raise ExclusionsError(f"{label}: its state changed while it was read")
    return events, after


def _source_entries(source: Source) -> tuple[list[Exclusion], SourceState]:
    """One source's admitted statements, from the tables its verified event log builds. Every
    lock must be its attempt's admitted statement, and every READY need its locked one, with its
    record's hash; anything else stops."""
    label = source.label
    events, head = _read_source(source)
    tables = hs.rebuild(events)
    try:
        contract = json.loads(_meta(tables, "identities") or "{}").get("pre_authoring_contract")
        if contract != source.contract_sha256:
            raise ExclusionsError(f"{label}: its state was built under another contract")
        rows = tables.execute(
            "SELECT proposal_id, stage, value FROM stages WHERE stage IN ('statement', 'lock')"
        ).fetchall()
        ready = {
            pid: (sha, json.loads(record))
            for pid, sha, record in tables.execute(
                "SELECT proposal_id, record_sha256, record FROM ready"
            )
        }
        status = _meta(tables, "status") or ""
    finally:
        tables.close()
    statements: dict[str, str] = {}
    locks: dict[str, str] = {}
    for pid, stage, value in rows:
        parsed = json.loads(value)
        if stage == "statement":
            statements[pid] = _statement_of(parsed, "primary")
        else:
            locks[pid] = _statement_of(parsed, "need", "statement")
    for pid, text in sorted(statements.items()):
        if not text:
            raise ExclusionsError(f"{label}: {pid}: an admitted statement without its text")
    for pid, text in sorted(locks.items()):
        if pid not in statements:
            raise ExclusionsError(f"{label}: {pid}: a lock without its admitted statement")
        if text != statements[pid]:
            raise ExclusionsError(f"{label}: {pid}: its lock is not its admitted statement")
    for pid, (sha, record) in sorted(ready.items()):
        if pid not in locks:
            raise ExclusionsError(f"{label}: {pid}: a READY need without its lock")
        if hs.sha_value(record) != sha:
            raise ExclusionsError(f"{label}: {pid}: a READY record that is not its hash")
        if _statement_of(record, "need", "statement") != locks[pid]:
            raise ExclusionsError(f"{label}: {pid}: a READY need that is not its locked one")
    entries = []
    for pid, text in sorted(statements.items()):
        key = f"{label}/{pid}"
        if not KEY.fullmatch(key):
            raise ExclusionsError(f"{label}: {pid}: not an attempt's id")
        entries.append(Exclusion(key, text, pid in locks, ready[pid][0] if pid in ready else None))
    state = SourceState(
        label, str(contract), len(events), head, status, len(statements), len(locks), len(ready)
    )
    return entries, state


def collect(sources: Sequence[Source]) -> tuple[list[Exclusion], list[SourceState]]:
    """Every statement the named runs admitted, and the inventory of the states read
    (_source_entries), checked as any read of the artifact is checked. A run named twice, or a
    label that is not one, is refused."""
    labels = [s.label for s in sources]
    if len(set(labels)) != len(labels):
        raise ExclusionsError("a run is named twice")
    if not all(LABEL.fullmatch(label) for label in labels):
        raise ExclusionsError("a run's label is not one")
    entries: list[Exclusion] = []
    states: list[SourceState] = []
    for source in sorted(sources, key=lambda s: s.label):
        found, state = _source_entries(source)
        entries += found
        states.append(state)
    parse(render(entries, states).encode("utf-8"), "the collected exclusions")
    return sorted(entries, key=lambda e: e.key), states


def statement_sha256(statement: str) -> str:
    """A statement's identity for an exact repetition: its words, whatever their spacing or case."""
    return hashlib.sha256(" ".join(statement.split()).casefold().encode("utf-8")).hexdigest()


def reuse_problems(
    ready: Mapping[str, tuple[str, str]], exclusions: Sequence[Exclusion]
) -> list[str]:
    """READY needs that repeat a development need: the same statement (statement_sha256) or the
    same READY record. ready maps a need id to its statement and its record's SHA-256. The
    messages name need ids only, never a statement."""
    statements = {statement_sha256(e.statement) for e in exclusions}
    records = {e.record_sha256 for e in exclusions if e.record_sha256}
    found = []
    for need_id, (statement, record_sha) in sorted(ready.items()):
        if statement_sha256(statement) in statements:
            found.append(f"{need_id}: its statement repeats a development need")
        if record_sha in records:
            found.append(f"{need_id}: its READY record repeats a development need's")
    return found
