"""The durable state of TEST's construction (S3, D-054): the only record of where it is.

One SQLite file in the private TEST directory, state/holdout-state.sqlite3, never in the
public repository. Every change is one EVENT, appended to a log that refuses updates and
deletions and is chained by SHA-256, and applied in the same transaction to the tables that
say where each cell is (cells, attempts, runs, stages, ready): an output is never recorded
without its state, nor a state without its output. The tables can always be rebuilt from the
log alone, and verify() checks that they are.

An event with a key is written at most once. The same key again must carry the same payload:
replaying the construction after a stop writes nothing twice, and a replay that would decide
differently stops as a divergence. A resume appends an event without a key.

A text too large for a row (a reply with its transcript) is a file named by its SHA-256,
written atomically (a temporary file, fsync, rename) before the event that names it. A file
no event names is harmless; a named file that does not hold its hash stops the work. Two
places: 'kept', with the account's identity values redacted, inside the TEST directory and
its history; 'raw', as received, in the private folder outside git, read only to audit again
a response that was recorded just before a crash.

One process at a time: an exclusive lock on a file beside the database, which the system
releases if the process dies.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from ingest import holdout as ho

SCHEMA = "holdout-state.v1"
STATE_DIR = "state"
DB_FILE = "holdout-state.sqlite3"
LOCK_FILE = ho.STATE_LOCK_FILE
KEPT_DIR = "artifacts"
RAW_DIR = Path.home() / ".fundamento-private" / "holdout-raw"
GENESIS = "0" * 64
# The person's rule of 2026-10-19: a stage that fails technically gets at most three
# executions in all, the first and two retries, each in a new isolated context with the same
# packet. A valid answer is never executed again.
MAX_TECHNICAL_RUNS = 3
# Operational checkpoints of the frozen order: a verified backup, an integrity check and a
# report of counts. They never recalibrate, rebalance or change anything.
CHECKPOINTS = (20, 40, 60)

CELL_STATES = (
    "planned",
    "authoring",
    "candidate",
    "locked",
    "judging",
    "pending_technical",
    "pending_technical_unresolved",
    "rejected_methodological",
    "unfilled_after_attempt_limit",
    "ready",
    "frozen",
)
STATUSES = ("building", "pending_technical", "pending_technical_unresolved", "stopped", "frozen")
RUN_STATUSES = ("started", "responded", "admitted", "failed")
Place = Literal["kept", "raw"]
GITIGNORE = "\n".join(ho.STATE_UNTRACKED) + "\n"


class StateError(Exception):
    """The construction's state refuses the operation."""


class StateBusy(StateError):
    """Another process holds the state: one construction process at a time."""


class IntegrityError(StateError):
    """The state does not verify: the database, the event log or an artifact."""


class MediumError(StateError):
    """The state's medium failed (a file of the state that cannot be written or read): never
    written over, never taken for a failure of the construction."""


class ReplayDiverged(IntegrityError):
    """Replaying the construction decides differently from what was recorded."""


class IncompatibleState(StateError):
    """Something frozen changed since the construction began. A partly built TEST is never
    migrated to another method: a person decides."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_value(value: Any) -> str:
    return sha_text(canonical(value))


def _in(values: Sequence[str]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


TRIGGERS = ("events_refuse_update", "events_refuse_delete")
SCHEMA_SQL = f"""
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;
CREATE TABLE events (
    seq INTEGER PRIMARY KEY,
    key TEXT UNIQUE,
    kind TEXT NOT NULL,
    at TEXT NOT NULL,
    cell_id TEXT,
    proposal_id TEXT,
    stage TEXT,
    run_id TEXT,
    payload TEXT NOT NULL,
    prev_sha256 TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE
) STRICT;
CREATE TRIGGER events_refuse_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'the event log is append-only'); END;
CREATE TRIGGER events_refuse_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'the event log is append-only'); END;
CREATE TABLE cells (
    cell_id TEXT PRIMARY KEY,
    position INTEGER NOT NULL UNIQUE,
    omission INTEGER NOT NULL CHECK (omission IN (0, 1)),
    state TEXT NOT NULL CHECK (state IN {_in(CELL_STATES)}),
    stage TEXT,
    proposal_id TEXT,
    updated_at TEXT NOT NULL
) STRICT;
CREATE TABLE attempts (
    proposal_id TEXT PRIMARY KEY,
    cell_id TEXT NOT NULL REFERENCES cells (cell_id),
    number INTEGER NOT NULL CHECK (number BETWEEN 1 AND 3),
    need_id TEXT UNIQUE,
    outcome TEXT,
    against TEXT,
    stage TEXT,
    note TEXT,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    UNIQUE (cell_id, number)
) STRICT;
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL REFERENCES attempts (proposal_id),
    stage TEXT NOT NULL,
    role TEXT NOT NULL,
    ref TEXT NOT NULL,
    technical_run INTEGER NOT NULL CHECK (technical_run BETWEEN 1 AND {MAX_TECHNICAL_RUNS}),
    status TEXT NOT NULL CHECK (status IN {_in(RUN_STATUSES)}),
    packet_sha256 TEXT NOT NULL,
    template TEXT NOT NULL,
    template_sha256 TEXT NOT NULL,
    role_contract_sha256 TEXT NOT NULL,
    inputs TEXT NOT NULL,
    response TEXT,
    audit TEXT,
    failure_reason TEXT,
    failure_detail TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    UNIQUE (proposal_id, stage, technical_run)
) STRICT;
CREATE TABLE stages (
    proposal_id TEXT NOT NULL REFERENCES attempts (proposal_id),
    stage TEXT NOT NULL,
    run_id TEXT UNIQUE REFERENCES runs (run_id),
    inputs_sha256 TEXT NOT NULL,
    value TEXT NOT NULL,
    value_sha256 TEXT NOT NULL,
    admitted_at TEXT NOT NULL,
    PRIMARY KEY (proposal_id, stage)
) STRICT;
CREATE TABLE ready (
    cell_id TEXT PRIMARY KEY REFERENCES cells (cell_id),
    need_id TEXT NOT NULL UNIQUE,
    proposal_id TEXT NOT NULL UNIQUE REFERENCES attempts (proposal_id),
    record TEXT NOT NULL,
    record_sha256 TEXT NOT NULL,
    frozen INTEGER NOT NULL DEFAULT 0 CHECK (frozen IN (0, 1)),
    ready_at TEXT NOT NULL
) STRICT;
CREATE TABLE artifacts (
    sha256 TEXT NOT NULL,
    place TEXT NOT NULL CHECK (place IN ('kept', 'raw')),
    bytes INTEGER NOT NULL,
    PRIMARY KEY (sha256, place)
) STRICT;
"""
# Every table the event log builds, with the order its rows are compared in
PROJECTIONS = {
    "meta": "key",
    "cells": "cell_id",
    "attempts": "proposal_id",
    "runs": "run_id",
    "stages": "proposal_id, stage",
    "ready": "cell_id",
    "artifacts": "sha256, place",
}


@dataclass(frozen=True)
class Event:
    seq: int
    key: str | None
    kind: str
    at: str
    cell_id: str | None
    proposal_id: str | None
    stage: str | None
    run_id: str | None
    payload: dict[str, Any]
    prev_sha256: str
    sha256: str


def _chain(prev: str, seq: int, key: str | None, kind: str, at: str, where: Sequence[Any]) -> str:
    return sha_text(canonical([prev, seq, key, kind, at, *where]))


# ---------------------------------------------------------------------------
# applying an event: the one function that writes the tables, live and when rebuilding
# ---------------------------------------------------------------------------


def _one(cursor: sqlite3.Cursor, what: str) -> None:
    if cursor.rowcount != 1:
        raise IntegrityError(f"a transition the state does not allow: {what}")


def _set_meta(db: sqlite3.Connection, key: str, value: str) -> None:
    db.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _get_meta(db: sqlite3.Connection, key: str) -> str | None:
    row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row[0])


# what the state keeps of each amendment (CONSTRUCTION_AMENDMENT_1 and any later one)
AMENDMENT_FIELDS = (
    "amendment",
    "amendment_sha256",
    "base_contract_sha256",
    "contract_sha256",
    "public_commit",
    "after_event",
    "changed_identities",
    "first_transition",
)


def _set_cell(db: sqlite3.Connection, e: Event, state: str, stage: str | None) -> None:
    _one(
        db.execute(
            "UPDATE cells SET state = ?, stage = ?, proposal_id = ?, updated_at = ? "
            "WHERE cell_id = ?",
            (state, stage, e.proposal_id, e.at, e.cell_id),
        ),
        f"cell {e.cell_id} to {state}",
    )


def _apply(db: sqlite3.Connection, e: Event) -> None:
    p, at = e.payload, e.at
    if e.kind == "created":
        _set_meta(db, "schema", p["schema"])
        _set_meta(db, "identities", canonical(p["identities"]))
        _set_meta(db, "runtime", canonical(p.get("runtime", {})))
        _set_meta(db, "status", "building")
        _set_meta(db, "created_at", at)
        for c in p["cells"]:
            db.execute(
                "INSERT INTO cells VALUES (?, ?, ?, 'planned', NULL, NULL, ?)",
                (c["cell_id"], c["position"], int(c["omission"]), at),
            )
    elif e.kind == "attempt_opened":
        db.execute(
            "INSERT INTO attempts (proposal_id, cell_id, number, opened_at) VALUES (?, ?, ?, ?)",
            (e.proposal_id, e.cell_id, p["number"], at),
        )
        _set_cell(db, e, "authoring", None)
    elif e.kind == "run_started":
        db.execute(
            "INSERT INTO runs (run_id, proposal_id, stage, role, ref, technical_run, status, "
            "packet_sha256, template, template_sha256, role_contract_sha256, inputs, started_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'started', ?, ?, ?, ?, ?, ?)",
            (
                e.run_id,
                e.proposal_id,
                e.stage,
                p["role"],
                p["ref"],
                p["technical_run"],
                p["packet_sha256"],
                p["template"],
                p["template_sha256"],
                p["role_contract_sha256"],
                canonical(p["inputs"]),
                at,
            ),
        )
        _set_cell(db, e, p["state"], e.stage)
        _set_meta(db, "status", "building")
    elif e.kind == "run_responded":
        response = p["response"]
        _one(
            db.execute(
                "UPDATE runs SET status = 'responded', response = ? "
                "WHERE run_id = ? AND status = 'started'",
                (canonical(response), e.run_id),
            ),
            f"run {e.run_id} responded",
        )
        for place in ("kept", "raw"):
            db.execute(
                "INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?)",
                (response[place], place, response[f"{place}_bytes"]),
            )
    elif e.kind == "run_failed":
        _one(
            db.execute(
                "UPDATE runs SET status = 'failed', audit = ?, failure_reason = ?, "
                "failure_detail = ?, ended_at = ? "
                "WHERE run_id = ? AND status IN ('started', 'responded')",
                (
                    None if p["audit"] is None else canonical(p["audit"]),
                    p["reason"],
                    p["detail"],
                    at,
                    e.run_id,
                ),
            ),
            f"run {e.run_id} failed",
        )
        _set_cell(db, e, p["state"], e.stage)
        _set_meta(db, "status", p["state"])
    elif e.kind == "stage_admitted":
        if e.run_id is not None:
            if (p.get("audit") or {}).get("ok") is not True:
                raise IntegrityError(f"run {e.run_id}: an output is admitted only after its audit")
            _one(
                db.execute(
                    "UPDATE runs SET status = 'admitted', audit = ?, ended_at = ? "
                    "WHERE run_id = ? AND status = 'responded'",
                    (canonical(p["audit"]), at, e.run_id),
                ),
                f"run {e.run_id} admitted",
            )
        db.execute(
            "INSERT INTO stages VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                e.proposal_id,
                e.stage,
                e.run_id,
                p["inputs_sha256"],
                canonical(p["value"]),
                p["value_sha256"],
                at,
            ),
        )
        _set_cell(db, e, p["state"], e.stage)
    elif e.kind == "need_id_given":
        _one(
            db.execute(
                "UPDATE attempts SET need_id = ? WHERE proposal_id = ? AND need_id IS NULL",
                (p["need_id"], e.proposal_id),
            ),
            f"{e.proposal_id} given a need id",
        )
    elif e.kind == "attempt_closed":
        _one(
            db.execute(
                "UPDATE attempts SET outcome = ?, against = ?, stage = ?, note = ?, closed_at = ? "
                "WHERE proposal_id = ? AND outcome IS NULL",
                (p["outcome"], p["against"], e.stage, p["note"], at, e.proposal_id),
            ),
            f"{e.proposal_id} closed",
        )
        _set_cell(db, e, p["state"], e.stage)
    elif e.kind == "attempt_failed_after_gate":
        _one(
            db.execute(
                "UPDATE attempts SET outcome = ?, stage = ?, note = ?, closed_at = ? "
                "WHERE proposal_id = ? AND outcome = 'accepted'",
                (p["outcome"], e.stage, p["note"], at, e.proposal_id),
            ),
            f"{e.proposal_id} failed after the gate",
        )
        _set_cell(db, e, p["state"], e.stage)
    elif e.kind == "need_ready":
        db.execute(
            "INSERT INTO ready VALUES (?, ?, ?, ?, ?, 0, ?)",
            (
                e.cell_id,
                p["need_id"],
                e.proposal_id,
                canonical(p["record"]),
                p["record_sha256"],
                at,
            ),
        )
        _set_cell(db, e, "ready", e.stage)
    elif e.kind == "construction_stopped":
        _set_meta(db, "status", "stopped")
        _set_meta(db, "stop", canonical(p))
        if p.get("cell_state"):
            _set_cell(db, e, p["cell_state"], e.stage)
    elif e.kind == "resumed":
        _set_meta(db, "last_resumed_at", at)
    elif e.kind == "construction_amended":
        # a documented amendment, at this exact point of the log: from the identities the state
        # holds to the amended ones, never on a stopped or frozen construction
        status = _get_meta(db, "status")
        if status in ("stopped", "frozen"):
            raise IntegrityError(f"an amendment of a {status} construction")
        if canonical(p["from_identities"]) != _get_meta(db, "identities"):
            raise IntegrityError("an amendment from identities the state does not hold")
        _set_meta(db, "identities", canonical(p["to_identities"]))
        amendments = json.loads(_get_meta(db, "amendments") or "[]")
        amendments.append({k: p[k] for k in AMENDMENT_FIELDS} | {"at": at, "seq": e.seq})
        _set_meta(db, "amendments", canonical(amendments))
    elif e.kind == "checkpoint":
        _set_meta(db, "last_checkpoint", str(p["position"]))
    elif e.kind == "frozen":
        db.execute("UPDATE ready SET frozen = 1")
        db.execute("UPDATE cells SET state = 'frozen', updated_at = ? WHERE state = 'ready'", (at,))
        _set_meta(db, "status", "frozen")
        _set_meta(db, "frozen_at", at)
    else:
        raise IntegrityError(f"an event of unknown kind {e.kind!r}")


def _differences(a: Any, b: Any, prefix: str = "") -> list[str]:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out = []
        for key in sorted(set(a) | set(b)):
            out += _differences(a.get(key), b.get(key), f"{prefix}{key}.")
        return out
    return [] if a == b else [prefix.rstrip(".") or "value"]


def _rows(db: sqlite3.Connection, table: str, order: str) -> list[tuple[Any, ...]]:
    return [tuple(r) for r in db.execute(f"SELECT * FROM {table} ORDER BY {order}")]


def rebuild(events: Sequence[Event]) -> sqlite3.Connection:
    """The tables as the event log alone builds them, in memory."""
    db = sqlite3.connect(":memory:", isolation_level=None)
    db.executescript(SCHEMA_SQL)
    for e in events:
        _apply(db, e)
    return db


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path: Path, data: bytes) -> None:
    """The bytes at path whole or not at all: a temporary file beside it, flushed to disk,
    then renamed over it. A crash leaves at most a temporary file nothing names."""
    tmp = path.with_name(f".tmp-{path.name}-{os.getpid()}")
    with open(tmp, "wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


class Store:
    def __init__(
        self,
        directory: Path,
        raw_dir: Path,
        clock: Callable[[], str] | None = None,
        readonly: bool = False,
    ) -> None:
        self.directory = directory
        self.state = directory / STATE_DIR
        self.raw_dir = raw_dir
        self.clock = clock or _now
        self.readonly = readonly
        self._writes_refused = False
        self._lock: int | None = None
        path = self.state / DB_FILE
        if not readonly:
            fd = os.open(self.state / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                raise StateBusy(
                    f"another process holds {path}: one construction process at a time"
                ) from None
            self._lock = fd
            self.db = sqlite3.connect(path, isolation_level=None)
        else:
            uri = f"{path.resolve().as_uri()}?mode=ro"
            self.db = sqlite3.connect(uri, uri=True, isolation_level=None)
        self.db.execute("PRAGMA foreign_keys = ON")
        if not readonly:
            self.db.execute("PRAGMA journal_mode = DELETE")
            self.db.execute("PRAGMA synchronous = FULL")

    @classmethod
    def create(
        cls,
        directory: Path,
        raw_dir: Path,
        identities: Mapping[str, str],
        order: Sequence[str],
        omission: Sequence[str],
        clock: Callable[[], str] | None = None,
        runtime: Mapping[str, Any] | None = None,
    ) -> Store:
        """runtime: the full records of the model runtimes (Claude, MLX) whose hashes are among
        the identities, kept so that an incompatible resume names what changed."""
        state = directory / STATE_DIR
        if (state / DB_FILE).exists():
            raise StateError(f"{state / DB_FILE} exists: the state is created once")
        state.mkdir(parents=True, exist_ok=True)
        (state / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
        store = cls(directory, raw_dir, clock)
        store.db.executescript(SCHEMA_SQL)
        cells = [
            {"cell_id": c, "position": i, "omission": c in set(omission)}
            for i, c in enumerate(order, start=1)
        ]
        store.record(
            "created",
            "created",
            {
                "schema": SCHEMA,
                "identities": dict(identities),
                "runtime": dict(runtime or {}),
                "cells": cells,
            },
        )
        return store

    @classmethod
    def open(
        cls,
        directory: Path,
        raw_dir: Path,
        clock: Callable[[], str] | None = None,
        readonly: bool = False,
    ) -> Store:
        if not (directory / STATE_DIR / DB_FILE).exists():
            raise StateError(f"no construction state in {directory / STATE_DIR}")
        store = cls(directory, raw_dir, clock, readonly)
        if store.meta("schema") != SCHEMA:
            store.close()
            raise IncompatibleState(f"the state's schema is not {SCHEMA}")
        return store

    def close(self) -> None:
        self.db.close()
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()

    # -- writing ---------------------------------------------------------------

    @property
    def writable(self) -> bool:
        return not self.readonly and not self._writes_refused

    @contextmanager
    def no_writes(self) -> Iterator[None]:
        """While open, every record() that would write is refused before it writes; an event
        replayed exactly as recorded writes nothing and passes. An operation that only reads the
        construction (the freeze's replay) can neither run a model nor change it."""
        self._writes_refused = True
        try:
            yield
        finally:
            self._writes_refused = False

    def record(
        self,
        kind: str,
        key: str | None,
        payload: Mapping[str, Any],
        *,
        cell: str | None = None,
        proposal: str | None = None,
        stage: str | None = None,
        run: str | None = None,
    ) -> bool:
        """Append one event and apply it, in one transaction. A key already recorded with
        the same event is a replay: nothing is written and False is returned. A key recorded
        with another event is a divergence and stops."""
        if self.readonly:
            raise StateError("a read-only state records nothing")
        text = canonical(payload)
        where = (cell, proposal, stage, run)
        db = self.db
        db.execute("BEGIN IMMEDIATE")
        try:
            if key is not None:
                row = db.execute(
                    "SELECT kind, cell_id, proposal_id, stage, run_id, payload FROM events "
                    "WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is not None:
                    if tuple(row) != (kind, *where, text):
                        raise ReplayDiverged(
                            f"{key}: the construction replays differently from what was recorded"
                        )
                    db.execute("ROLLBACK")
                    return False
            if self._writes_refused:
                # a replay that only reads: an event already recorded is fine, a new one is not
                raise StateError(f"{kind}: this operation writes nothing to the state")
            last = db.execute("SELECT seq, sha256 FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            seq, prev = (last[0] + 1, last[1]) if last else (1, GENESIS)
            at = self.clock()
            digest = _chain(prev, seq, key, kind, at, [*where, text])
            db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (seq, key, kind, at, *where, text, prev, digest),
            )
            _apply(db, Event(seq, key, kind, at, *where, json.loads(text), prev, digest))
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        return True

    def artifact_path(self, digest: str, place: Place) -> Path:
        root = self.state / KEPT_DIR if place == "kept" else self.raw_dir
        return root / digest[:2] / digest

    def put(self, text: str, place: Place) -> tuple[str, int]:
        """A text into its place, named by its hash; written once, atomically. A failure of the
        medium is MediumError."""
        data = text.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        path = self.artifact_path(digest, place)
        try:
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise IntegrityError(f"{path} does not hold its hash")
                return digest, len(data)
            mode = 0o700 if place == "raw" else 0o755
            for folder in (path.parent.parent, path.parent):
                if not folder.exists():
                    folder.mkdir(parents=True, mode=mode)
            write_atomic(path, data)
        except OSError as error:
            raise MediumError(
                f"the {place} artifact {digest[:12]} cannot be written: {error}"
            ) from error
        return digest, len(data)

    def text(self, digest: str, place: Place) -> str:
        path = self.artifact_path(digest, place)
        if not path.exists():
            raise IntegrityError(f"the {place} artifact {digest[:12]} is missing")
        try:
            data = path.read_bytes()
        except OSError as error:
            raise MediumError(
                f"the {place} artifact {digest[:12]} cannot be read: {error}"
            ) from error
        if hashlib.sha256(data).hexdigest() != digest:
            raise IntegrityError(f"the {place} artifact {digest[:12]} does not hold its hash")
        return data.decode("utf-8")

    # -- reading ---------------------------------------------------------------

    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def status(self) -> str:
        return self.meta("status") or ""

    def has_event(self, key: str) -> bool:
        return self.db.execute("SELECT 1 FROM events WHERE key = ?", (key,)).fetchone() is not None

    def has_progress(self) -> bool:
        row = self.db.execute("SELECT count(*) FROM events WHERE kind != 'created'").fetchone()
        return bool(row[0])

    def _dicts(self, sql: str, args: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cursor = self.db.execute(sql, args)
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    def cells(self) -> list[dict[str, Any]]:
        return self._dicts("SELECT * FROM cells ORDER BY position")

    def attempts(self) -> list[dict[str, Any]]:
        return self._dicts("SELECT * FROM attempts ORDER BY cell_id, number")

    def runs(
        self, proposal_id: str | None = None, stage: str | None = None
    ) -> list[dict[str, Any]]:
        if proposal_id is None:
            return self._dicts("SELECT * FROM runs ORDER BY started_at, run_id")
        return self._dicts(
            "SELECT * FROM runs WHERE proposal_id = ? AND stage = ? ORDER BY technical_run",
            (proposal_id, stage),
        )

    def stage(self, proposal_id: str, stage: str) -> dict[str, Any] | None:
        found = self._dicts(
            "SELECT * FROM stages WHERE proposal_id = ? AND stage = ?", (proposal_id, stage)
        )
        if not found:
            return None
        row = found[0]
        row["value"] = json.loads(row["value"])
        return row

    def stages(self) -> list[dict[str, Any]]:
        return self._dicts("SELECT proposal_id, stage, run_id, value_sha256 FROM stages")

    def ready(self) -> list[dict[str, Any]]:
        rows = self._dicts("SELECT * FROM ready ORDER BY ready_at, cell_id")
        for row in rows:
            row["record"] = json.loads(row["record"])
        return rows

    def events(self) -> list[Event]:
        return [
            Event(
                r["seq"],
                r["key"],
                r["kind"],
                r["at"],
                r["cell_id"],
                r["proposal_id"],
                r["stage"],
                r["run_id"],
                json.loads(r["payload"]),
                r["prev_sha256"],
                r["sha256"],
            )
            for r in self._dicts("SELECT * FROM events ORDER BY seq")
        ]

    # -- checking --------------------------------------------------------------

    def runtime(self) -> dict[str, Any]:
        found: dict[str, Any] = json.loads(self.meta("runtime") or "{}")
        return found

    def check_identities(
        self, current: Mapping[str, str], runtime: Mapping[str, Any] | None = None
    ) -> None:
        """Every frozen identity the state was created with, again; for the runtimes, the
        component that changed is named."""
        stored: dict[str, str] = json.loads(self.meta("identities") or "{}")
        changed = sorted(k for k in set(stored) | set(current) if stored.get(k) != current.get(k))
        if runtime is not None:
            before = self.runtime()
            for name in sorted(set(before) | set(runtime)):
                a, b = before.get(name), runtime.get(name)
                if a != b:
                    changed.append(f"{name}: {_differences(a, b)[:5]}")
        if changed:
            raise IncompatibleState(
                f"frozen identities changed since the construction began: {changed}. Stop: a "
                "partly built TEST is never migrated to another method; a person decides"
            )

    def check_cells(self, order: Sequence[str], omission: Sequence[str]) -> None:
        cells = self.cells()
        if [c["cell_id"] for c in cells] != list(order) or {
            c["cell_id"] for c in cells if c["omission"]
        } != set(omission):
            raise IncompatibleState("the state's cells are not the frozen order and omission audit")

    def verify(self) -> dict[str, Any]:
        """Everything the state must be before it is read or resumed: the database whole, the
        event log append-only and unbroken, every table exactly what the log builds, every
        artifact present with its hash. Returns the summary; a problem stops."""
        problems: list[str] = []
        db = self.db
        db.execute("BEGIN")
        try:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                problems.append("SQLite integrity_check fails")
            if db.execute("PRAGMA foreign_key_check").fetchall():
                problems.append("a reference to a row that does not exist")
            triggers = {
                r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
            }
            if triggers != set(TRIGGERS):
                problems.append("the event log is no longer append-only")
            prev = GENESIS
            stored = db.execute("SELECT * FROM events ORDER BY seq").fetchall()
            for n, (seq, key, kind, at, *where, prev_sha, digest) in enumerate(stored, start=1):
                text = where[-1]  # the payload exactly as stored, in its canonical form
                if (
                    seq != n
                    or prev_sha != prev
                    or _chain(prev, seq, key, kind, at, where) != digest
                    or canonical(json.loads(text)) != text
                ):
                    problems.append(f"the event log's chain breaks at event {seq}")
                    break
                prev = digest
            events = self.events()
            try:
                rebuilt = rebuild(events)
                for table, order in PROJECTIONS.items():
                    if _rows(db, table, order) != _rows(rebuilt, table, order):
                        problems.append(f"{table}: not what the event log builds")
                rebuilt.close()
            except (IntegrityError, sqlite3.Error) as error:
                problems.append(f"the event log does not build the tables: {error}")
            for digest, place in db.execute("SELECT sha256, place FROM artifacts"):
                try:
                    self.text(digest, place)
                except IntegrityError as error:
                    problems.append(str(error))
            problems += self.consumption_problems()
        finally:
            db.execute("COMMIT")
        if problems:
            raise IntegrityError("the state does not verify: " + "; ".join(problems[:8]))
        return self.summary()

    def consumption_problems(self) -> list[str]:
        """Rejected technical outputs have zero downstream consumers. Every admitted stage that
        a model answered is the output of one admitted run whose audit passed; no stage comes
        from a failed run; every admitted run has its stage; and every input a run was built
        from is the value of an admitted stage of its attempt."""
        problems = []
        runs = {r["run_id"]: r for r in self.runs()}
        stages = {(s["proposal_id"], s["stage"]): s for s in self.stages()}
        for (pid, stage), row in stages.items():
            run = runs.get(row["run_id"]) if row["run_id"] else None
            if row["run_id"] is None:
                continue
            if run is None or run["status"] != "admitted":
                problems.append(f"{pid} {stage}: admitted from a run that is not admitted")
            elif json.loads(run["audit"] or "{}").get("ok") is not True:
                problems.append(f"{pid} {stage}: admitted from a run whose audit did not pass")
        by_run = {s["run_id"] for s in stages.values() if s["run_id"]}
        for run_id, run in runs.items():
            if run["status"] == "admitted" and run_id not in by_run:
                problems.append(f"{run_id}: admitted with no stage")
            if run["status"] == "failed" and run_id in by_run:
                problems.append(f"{run_id}: a failed run has a consumer")
            for name, digest in json.loads(run["inputs"]).items():
                source = stages.get((run["proposal_id"], name))
                if name == "ready_test":
                    continue
                if source is None or source["value_sha256"] != digest:
                    problems.append(f"{run_id}: input {name} is not an admitted output")
        return problems

    def summary(self) -> dict[str, Any]:
        """Counts and identifiers only, never a statement, a query or a judgment: where the
        construction is, and the cell and stage that wait."""
        cells = self.cells()
        attempts = self.attempts()
        runs = self.runs()
        waiting = []
        for c in cells:
            if c["state"] in ("authoring", "candidate", "locked", "judging") or c[
                "state"
            ].startswith("pending_technical"):
                failed = [
                    r
                    for r in runs
                    if r["proposal_id"] == c["proposal_id"]
                    and r["stage"] == c["stage"]
                    and r["status"] == "failed"
                ]
                waiting.append(
                    {
                        "cell_id": c["cell_id"],
                        "position": c["position"],
                        "state": c["state"],
                        "stage": c["stage"],
                        "proposal_id": c["proposal_id"],
                        "failed_technical_runs": len(failed),
                        "last_failure": failed[-1]["failure_reason"] if failed else None,
                    }
                )
        last = self.db.execute(
            "SELECT seq, sha256 FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        stop = self.meta("stop")
        return {
            "schema": self.meta("schema"),
            "status": self.status(),
            "events": last[0] if last else 0,
            "head_sha256": last[1] if last else GENESIS,
            "cells": dict(sorted(Counter(c["state"] for c in cells).items())),
            "attempts": len(attempts),
            "outcomes": dict(sorted(Counter(a["outcome"] or "open" for a in attempts).items())),
            "runs": dict(sorted(Counter(r["status"] for r in runs).items())),
            "technical_failures": dict(
                sorted(Counter(r["failure_reason"] for r in runs if r["failure_reason"]).items())
            ),
            "ready": len(self.ready()),
            "waiting": waiting,
            "stop": None if stop is None else json.loads(stop),
            "last_checkpoint": self.meta("last_checkpoint"),
            "amendments": self.amendments(),
        }

    def amendments(self) -> list[dict[str, Any]]:
        """The amendments the log records, in order, each with the point where it applies."""
        found: list[dict[str, Any]] = json.loads(self.meta("amendments") or "[]")
        return found

    def amend(self, record: Mapping[str, Any]) -> bool:
        """A documented amendment of the frozen identities, as one event at the log's current
        point: everything before it was built under from_identities, everything after under
        to_identities. The launcher checks what it may change (ingest/holdout_launcher.py)."""
        return self.record("construction_amended", f"amendment/{record['amendment']}", dict(record))


# ---------------------------------------------------------------------------
# backup and restore
# ---------------------------------------------------------------------------


def _git(directory: Path, *args: str) -> str:
    return ho.git(directory, *args)


def backup_state(
    store: Store, repo: Path, folder: Path = ho.BACKUP_DIR, label: str = "backup"
) -> dict[str, Any]:
    """The construction's state into the private backups, by the process that holds it:
    the state verified and recorded in the private history; every raw artifact it names
    copied by hash beside the bundles, outside git; a git bundle of the history, verified and
    restored into a temporary clone whose state must verify and equal this one."""
    if store.readonly:
        raise StateError("only the process that holds the state backs it up")
    folder = ho.private_folder(folder, repo)
    summary = store.verify()
    directory = store.directory
    _git(directory, "add", "-A", "--", STATE_DIR)
    if _git(directory, "diff", "--cached", "--name-only"):
        _git(directory, "commit", "-q", "-m", f"state: {label}")
    raw_backup = folder / "raw"
    raw_backup.mkdir(exist_ok=True, mode=0o700)
    raw = sorted(r[0] for r in store.db.execute("SELECT sha256 FROM artifacts WHERE place = 'raw'"))
    for digest in raw:
        target = raw_backup / digest[:2] / digest
        if not target.exists():
            target.parent.mkdir(exist_ok=True, mode=0o700)
            write_atomic(target, store.text(digest, "raw").encode("utf-8"))
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise IntegrityError(f"the backup of raw artifact {digest[:12]} does not hold its hash")

    def check(clone: Path) -> dict[str, Any]:
        with Store.open(clone, raw_backup, readonly=True) as restored:
            found = restored.verify()
        if found != summary:
            raise ho.HoldoutError("the restored state differs from the state backed up")
        return found

    bundle = ho.backup(directory, repo, folder, check=check)
    manifest = folder / f"{bundle.stem}.raw.json"
    write_atomic(manifest, (canonical(raw) + "\n").encode("utf-8"))
    return {
        "bundle": bundle.name,
        "head": _git(directory, "rev-parse", "HEAD"),
        "raw_artifacts": len(raw),
        "state": summary,
    }


def restore(bundle: Path, target: Path, raw_dir: Path) -> dict[str, Any]:
    """A working copy of TEST from a bundle and its raw artifacts, into a directory that
    does not exist; its private history refuses every push again. Returns the restored
    state's summary, verified."""
    if target.exists():
        raise StateError(f"{target} exists: a restore never overwrites")
    folder = bundle.parent
    subprocess.run(
        ["git", "clone", "-q", str(bundle), str(target)], check=True, capture_output=True
    )
    ho.guard_private_history(target)
    raw: list[str] = json.loads((folder / f"{bundle.stem}.raw.json").read_text(encoding="utf-8"))
    for digest in raw:
        source = folder / "raw" / digest[:2] / digest
        destination = raw_dir / digest[:2] / digest
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise IntegrityError(f"the backup of raw artifact {digest[:12]} does not hold its hash")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(source, destination)
    with Store.open(target, raw_dir, readonly=True) as restored:
        return restored.verify()
