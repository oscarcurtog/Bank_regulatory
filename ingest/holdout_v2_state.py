"""The durable state of a TEST v2 construction (D-067): the only record of where it is.

A namespace of its own, never run 7's or any earlier run's: one SQLite file in
<namespace>/state/, beside its kept artifacts, and the raw artifacts in a private folder outside
git. Every change is one EVENT, appended to a log that refuses updates and deletions and is
chained by SHA-256. Nothing else is written: what a construction knows (its jobs, their runs and
admitted values, its freezes, its stop) is folded from the log alone, in memory, every time the
state is opened, so that no table can disagree with it.

An event with a key is written at most once; the same key again must carry the same payload, so
a restart writes nothing twice, and a restart that would decide differently stops as a
divergence. A text too large for an event (a reply with its transcript) is a file named by its
SHA-256, written atomically before the event that names it: 'kept' (the account's identity
values redacted) inside the namespace and its private history, 'raw' (as received) in the raw
folder, read only to audit again a response recorded just before a crash.

One writer at a time: an exclusive lock beside the database, released by the system if the
process dies. Readers take it shared, so nothing is read while a writer works.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from ingest import holdout_state as hs

SCHEMA = "holdout-v2-state.v1"
STATE_DIR = "state"
DB_FILE = "holdout-v2-state.sqlite3"
# the lock keeps the name of every construction state's lock, which the private history
# ignores and its backups never compare (ingest/holdout.py STATE_UNTRACKED)
LOCK_FILE = "holdout-state.lock"
KEPT_DIR = "artifacts"
RAW_DIR = Path.home() / ".fundamento-private" / "holdout-v2-raw"
GENESIS = "0" * 64
GITIGNORE = f"{LOCK_FILE}\n*.sqlite3-journal\n.tmp-*\n"
Place = Literal["kept", "raw"]

SCHEMA_SQL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;
CREATE TABLE events (
    seq INTEGER PRIMARY KEY,
    key TEXT UNIQUE,
    kind TEXT NOT NULL,
    at TEXT NOT NULL,
    payload TEXT NOT NULL,
    prev_sha256 TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE
) STRICT;
CREATE TABLE artifacts (
    sha256 TEXT NOT NULL,
    place TEXT NOT NULL CHECK (place IN ('kept', 'raw')),
    bytes INTEGER NOT NULL,
    PRIMARY KEY (sha256, place)
) STRICT;
CREATE TRIGGER events_refuse_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'the event log is append-only'); END;
CREATE TRIGGER events_refuse_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'the event log is append-only'); END;
CREATE TRIGGER meta_refuse_update BEFORE UPDATE ON meta
BEGIN SELECT RAISE(ABORT, 'the identity of a state never changes'); END;
CREATE TRIGGER meta_refuse_delete BEFORE DELETE ON meta
BEGIN SELECT RAISE(ABORT, 'the identity of a state never changes'); END;
"""


class StateError(Exception):
    """The state cannot be used as asked: nothing was written."""


class StateBusy(StateError):
    """Another process holds the state."""


class IntegrityError(StateError):
    """The state does not hold what its own log and hashes say: an integrity stop."""


class Diverged(IntegrityError):
    """An event key already recorded with another payload: a restart decided differently."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _chain(prev: str, seq: int, key: str | None, kind: str, at: str, payload: str) -> str:
    return sha_text(canonical([prev, seq, key, kind, at, sha_text(payload)]))


@dataclass(frozen=True)
class Event:
    seq: int
    key: str | None
    kind: str
    at: str
    payload: dict[str, Any]
    prev_sha256: str
    sha256: str


class Store:
    def __init__(
        self, directory: Path, raw_dir: Path, db: sqlite3.Connection, lock: int, readonly: bool
    ) -> None:
        self.directory = directory
        self.raw_dir = raw_dir
        self.db = db
        self._lock = lock
        self.readonly = readonly

    # -----------------------------------------------------------------------
    # opening
    # -----------------------------------------------------------------------

    @staticmethod
    def _paths(directory: Path) -> tuple[Path, Path, Path]:
        state = directory / STATE_DIR
        return state, state / DB_FILE, state / LOCK_FILE

    @staticmethod
    def _locked(path: Path, shared: bool) -> int:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise StateBusy(f"{path}: another process holds the state") from None
        return fd

    @classmethod
    def create(cls, directory: Path, raw_dir: Path, meta: Mapping[str, str]) -> Store:
        """A new, empty state with its identity (meta, written once and never again). Refuses a
        namespace that already holds one."""
        state, db_path, lock_path = cls._paths(directory)
        if db_path.exists():
            raise StateError(f"{db_path} exists: a namespace holds one state")
        state.mkdir(parents=True, exist_ok=True)
        (state / KEPT_DIR).mkdir(exist_ok=True)
        (state / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
        raw_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = cls._locked(lock_path, shared=False)
        try:
            db = sqlite3.connect(db_path, isolation_level=None)
            db.execute("PRAGMA synchronous = FULL")
            db.executescript(SCHEMA_SQL)
            with _transaction(db):
                for key, value in sorted({"schema": SCHEMA, **meta}.items()):
                    db.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (key, str(value)))
        except BaseException:
            os.close(fd)
            raise
        return cls(directory, raw_dir, db, fd, readonly=False)

    @classmethod
    def open(cls, directory: Path, raw_dir: Path, readonly: bool = False) -> Store:
        state, db_path, lock_path = cls._paths(directory)
        if not db_path.exists():
            raise StateError(f"{db_path} does not exist")
        fd = cls._locked(lock_path, shared=readonly)
        try:
            uri = f"file:{db_path.resolve()}?mode=ro" if readonly else str(db_path)
            db = sqlite3.connect(uri, uri=readonly, isolation_level=None)
            db.execute("PRAGMA synchronous = FULL")
            store = cls(directory, raw_dir, db, fd, readonly)
            if store.meta("schema") != SCHEMA:
                raise StateError(f"{db_path}: schema {store.meta('schema')!r}, not {SCHEMA}")
        except BaseException:
            os.close(fd)
            raise
        return store

    def close(self) -> None:
        self.db.close()
        os.close(self._lock)

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()

    # -----------------------------------------------------------------------
    # the identity, the log and the artifacts
    # -----------------------------------------------------------------------

    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def meta_all(self) -> dict[str, str]:
        return {k: v for k, v in self.db.execute("SELECT key, value FROM meta ORDER BY key")}

    def events(self) -> list[Event]:
        rows = self.db.execute(
            "SELECT seq, key, kind, at, payload, prev_sha256, sha256 FROM events ORDER BY seq"
        ).fetchall()
        return [
            Event(s, k, kind, at, json.loads(p), prev, sha) for s, k, kind, at, p, prev, sha in rows
        ]

    def head(self) -> tuple[int, str]:
        row = self.db.execute("SELECT seq, sha256 FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        return (0, GENESIS) if row is None else (int(row[0]), str(row[1]))

    def record(self, kind: str, key: str | None, payload: Mapping[str, Any]) -> Event:
        """Append one event. A key already recorded with the same payload returns that event and
        writes nothing; with another payload it is a divergence."""
        if self.readonly:
            raise StateError("a read-only state records nothing")
        text = canonical(dict(payload))
        if key is not None:
            row = self.db.execute(
                "SELECT seq, kind, at, payload, prev_sha256, sha256 FROM events WHERE key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                if row[1] != kind or row[3] != text:
                    raise Diverged(f"event {key!r} is recorded with another payload")
                return Event(int(row[0]), key, kind, row[2], json.loads(row[3]), row[4], row[5])
        with _transaction(self.db):
            seq, prev = self.head()
            seq += 1
            at = _now()
            digest = _chain(prev, seq, key, kind, at, text)
            self.db.execute(
                "INSERT INTO events (seq, key, kind, at, payload, prev_sha256, sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (seq, key, kind, at, text, prev, digest),
            )
        return Event(seq, key, kind, at, json.loads(text), prev, digest)

    def artifact_path(self, digest: str, place: Place) -> Path:
        root = self.directory / STATE_DIR / KEPT_DIR if place == "kept" else self.raw_dir
        return root / digest[:2] / digest

    def put(self, text: str, place: Place) -> tuple[str, int]:
        """A text as a file named by its SHA-256, written atomically, then named in the table.
        The same text twice is one file."""
        if self.readonly:
            raise StateError("a read-only state writes nothing")
        data = text.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        path = self.artifact_path(digest, place)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            hs.write_atomic(path, data)
        with _transaction(self.db):
            self.db.execute(
                "INSERT OR IGNORE INTO artifacts (sha256, place, bytes) VALUES (?, ?, ?)",
                (digest, place, len(data)),
            )
        return digest, len(data)

    def text(self, digest: str, place: Place) -> str:
        path = self.artifact_path(digest, place)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise IntegrityError(f"{path} does not hold its hash")
        return data.decode("utf-8")

    # -----------------------------------------------------------------------
    # verification
    # -----------------------------------------------------------------------

    def verify(self) -> dict[str, Any]:
        """SQLite's integrity, the log's chain from its genesis, and every artifact the state
        names holding its hash. Returns counts and the head; an IntegrityError otherwise."""
        if self.db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise IntegrityError("SQLite's integrity check fails")
        prev = GENESIS
        events = self.db.execute(
            "SELECT seq, key, kind, at, payload, prev_sha256, sha256 FROM events ORDER BY seq"
        ).fetchall()
        for i, (seq, key, kind, at, payload, prev_sha, digest) in enumerate(events, start=1):
            if seq != i or prev_sha != prev or _chain(prev, seq, key, kind, at, payload) != digest:
                raise IntegrityError(f"the event log's chain breaks at event {seq}")
            prev = digest
        artifacts = self.db.execute("SELECT sha256, place FROM artifacts").fetchall()
        for digest, place in artifacts:
            path = self.artifact_path(digest, place)
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise IntegrityError(f"artifact {digest[:12]} ({place}) is missing or changed")
        return {
            "schema": SCHEMA,
            "events": len(events),
            "head_sha256": prev,
            "artifacts": len(artifacts),
            "meta": self.meta_all(),
        }


def backup(
    store: Store, history: Path, repo: Path, folder: Path, label: str = "backup"
) -> dict[str, Any]:
    """The state into the private backups, by the process that holds it: verified, recorded in
    the private history it lives in (history: that history's root), every raw artifact it
    names copied by hash beside the bundles, outside git; then a bundle of the whole history,
    verified and restored into a temporary clone whose state must verify and equal this one
    (ingest/holdout.py backup)."""
    from ingest import holdout as ho

    if store.readonly:
        raise StateError("only the process that holds the state backs it up")
    folder = ho.private_folder(folder, repo)
    summary = store.verify()
    relative = store.directory.resolve().relative_to(history.resolve())
    ho.git(history, "add", "-A", "--", relative.as_posix())
    if ho.git(history, "diff", "--cached", "--name-only"):
        ho.git(history, "commit", "-q", "-m", f"TEST v2 state: {label}")
    raw_backup = folder / "raw"
    raw_backup.mkdir(exist_ok=True, mode=0o700)
    raw = sorted(r[0] for r in store.db.execute("SELECT sha256 FROM artifacts WHERE place = 'raw'"))
    for digest in raw:
        target = raw_backup / digest[:2] / digest
        if not target.exists():
            target.parent.mkdir(exist_ok=True, mode=0o700)
            hs.write_atomic(target, store.text(digest, "raw").encode("utf-8"))
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise IntegrityError(f"the backup of raw artifact {digest[:12]} does not hold its hash")

    def check(clone: Path) -> dict[str, Any]:
        with Store.open(clone / relative, raw_backup, readonly=True) as restored:
            found = restored.verify()
        if found != summary:
            raise IntegrityError("the restored state differs from the state backed up")
        return found

    bundle = ho.backup(history, repo, folder, check=check)
    manifest = folder / f"{bundle.stem}.raw-v2.json"
    hs.write_atomic(
        manifest, (canonical({"state": relative.as_posix(), "raw": raw}) + "\n").encode()
    )
    return {
        "bundle": bundle.name,
        "head": ho.git(history, "rev-parse", "HEAD"),
        "raw_artifacts": len(raw),
        "events": summary["events"],
        "head_sha256": summary["head_sha256"],
    }


@contextmanager
def _transaction(db: sqlite3.Connection) -> Iterator[None]:
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")
