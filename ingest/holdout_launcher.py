"""The only way to run TEST's construction with real models (S3, D-054, D-055).

Two construction actions, and nothing else runs a model:
- START begins a new construction, only where no durable state exists;
- RESUME continues the construction whose durable state exists, by the semantics frozen in
  ingest/holdout_pipeline.py (same cell, need, stage and frozen inputs). It never makes up a
  missing state.
Neither has a default or an automatic start. Each is its own command and needs the full
SHA-256 of the pinned contract typed again. START also needs the public pre-authoring commit
recorded (record_public_commit), so that the commit that dates systems, method and plan comes
before the first need.

Before either can build anything, preflight() checks, in this order:
1. the pre-authoring contract as pinned, and every file it freezes by content (the code of E,
   E5, BM25 and the evaluation, the configs, the S4 plan, the construction's code with this
   launcher, its inputs and templates), with the corpus, snapshot and chunk layer on disk;
2. the systems, recomputed from their sources, equal to the contract's;
3. the grid, the frozen order and the omission audit, against the contract and provenance;
4. the role contracts, the templates and the classification of the checks, against the
   contract and provenance;
5. the audit's implementation: ingest/holdout_isolation.py byte for byte the module Gate 1
   passed with;
6. the Claude Code binary, version and SHA-256, the one pinned and passed at Gate 1;
7. the private paths: the TEST directory ignored by the public repository, with a history of
   its own that refuses every push, the opaque-id salt, and the raw, spool and backup folders
   private and outside the repository;
8. the state: absent for START; present for RESUME.
Only then is the store opened or created, which takes the single-process lock, and the World,
the executors and the orchestrator built. The World's contract_check and runtime_check are
checks 1 to 7 and 6 again, and the pipeline runs them, with the state's own verification,
before its first stage: no path reaches a model without them.

validate() runs checks 1 to 7 and reads the state, building nothing. freeze() replays the
state with every write refused, so it can run no model and change nothing until the
invariants hold; then it exports, seals and freezes. record_public_commit() writes only a
provenance file. None of them builds an executor.

An amended contract (CONSTRUCTION_AMENDMENT_1) carries what it amends: the base contract and
its public commit, the point of the durable state's log where it applies, the identities it may
change and the first transition the amended code must make there. amend() records it as one
event of the state, after checking all of that and a replay of the whole log under the amended
code with every write refused, so that no model runs and nothing admitted is built again. The
needs ready before it stay bound to the base contract; everything after, to the base and the
amendment.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_checks as hk
from ingest import holdout_contract as hc
from ingest import holdout_executor as hx
from ingest import holdout_isolation as hi
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from ingest import holdout_roles as hr
from ingest import holdout_runtime as hrt
from ingest import holdout_state as hs
from ingest import psd2_exclusions as px
from ingest.corpus_tools import CorpusTools

PUBLIC_COMMIT_FILE = "public_commit.json"
# the public commit of an amended contract (CONSTRUCTION_AMENDMENT_1), beside the base's
AMENDMENT_COMMIT_FILE = "amendment_commit.json"
AUDIT_MODULE = Path("ingest/holdout_isolation.py")


class LaunchRefused(Exception):
    """A launch the frozen rules do not allow: nothing ran and nothing was written."""


@dataclass(frozen=True)
class Paths:
    repo: Path = Path(".")
    test: Path = ho.HOLDOUT_DIR
    raw: Path = hs.RAW_DIR
    backups: Path = ho.BACKUP_DIR

    @property
    def provenance(self) -> Path:
        return self.test / "provenance"

    @property
    def spool(self) -> Path:
        return self.raw / "spool"

    @property
    def contract(self) -> Path:
        return self.repo / hc.CONTRACT_PATH


@dataclass(frozen=True)
class Frozen:
    """What the checks found and the construction runs with."""

    contract: dict[str, Any]
    contract_sha256: str
    runtime: dict[str, Any]
    grid: ho.Grid
    salt: bytes
    templates: dict[str, str]
    extra_env: list[str]

    @property
    def claude_runtime(self) -> dict[str, Any]:
        found: dict[str, Any] = self.contract["construction"]["claude_runtime"]
        return found

    @property
    def mlx_runtime(self) -> dict[str, Any]:
        found: dict[str, Any] = self.contract["construction"]["mlx_runtime"]
        return found


@dataclass(frozen=True)
class Probes:
    """How the live runtimes are read: the binary's version and SHA-256 from its file, and the
    MLX runtime from its environment's metadata. Only tests give stand-ins."""

    binary: Callable[[Path], Mapping[str, str]] | None = None
    mlx: Callable[[], Mapping[str, Any]] | None = None

    def read_binary(self, path: Path) -> Mapping[str, str]:
        return (self.binary or hi.binary_identity)(path)

    def read_mlx(self) -> Mapping[str, Any]:
        return (self.mlx or hrt.mlx_runtime)()


ExecutorFactory = Callable[[Frozen, Paths], hpl.Executor]


def project_terms(repo: Path) -> list[hi.Term]:
    """The forbidden terms of this project as it is now: the static ones, the repository's path
    and folders, the names of the private working documents, every distinctive line of
    CLAUDE.md and of the memory index, and the subjects of the recent commits. They change with
    the project, so they are computed again at every start and resume and only add to the
    static terms, which are frozen with the audit's code."""
    repo = repo.resolve()
    places = [str(repo), repo.name, repo.parent.name]
    places += sorted(p.name for p in (repo / "docs").glob("*.md"))
    state = Path.home() / ".claude" / "projects" / hi.encoded_project_dir(str(repo))
    claude_md, memory = repo / "CLAUDE.md", state / "memory" / "MEMORY.md"
    return hi.forbidden_terms(
        claude_md if claude_md.exists() else None,
        memory if memory.exists() else None,
        hi.recent_commit_subjects(repo),
        places,
    )


def static_terms() -> list[hi.Term]:
    """The terms that depend on no file or history of the project."""
    return hi.forbidden_terms(None, None, [])


def _read(path: Path) -> Any:
    if not path.exists():
        raise LaunchRefused(f"{path} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _private(path: Path, repo: Path) -> list[str]:
    problems = []
    if path.resolve().is_relative_to(repo.resolve()):
        problems.append(f"{path} is inside the repository")
    if path.exists() and stat.S_IMODE(path.stat().st_mode) & 0o077:
        problems.append(f"{path} is readable by others")
    return problems


def contract_problems(paths: Paths) -> tuple[Frozen | None, list[str]]:
    """Checks 1 to 5 and 7: everything frozen by content, and the private paths."""
    try:
        pin = _read(paths.provenance / "contract_pin.json")
        contract = hc.verify(pin["sha256"], paths.contract, paths.repo, check_corpus=True)
    except (hc.ContractError, LaunchRefused) as error:
        return None, [f"the pre-authoring contract: {error}"]
    problems = []
    construction = contract["construction"]
    if hc.systems() != contract["systems"]:
        problems.append("the systems recomputed from their sources differ from the contract")
    grid = ho.load_grid(paths.test)
    if _sha(paths.test / ho.GRID_FILE) != construction["grid_sha256"]:
        problems.append("the grid is not the contract's")
    plan = ho.plan_record(grid)
    for key in ("order", "omission_audit"):
        frozen = construction[key]
        if any(plan[key][k] != frozen[k] for k in ("algorithm", "seed", "sha256")):
            problems.append(f"the {key} is not the contract's")
    try:
        if _read(paths.provenance / "plan.json") != plan:
            problems.append("the order or the omission audit differs from provenance")
        roles = hr.contracts_record()
        if _read(paths.provenance / "role_contracts.json") != roles:
            problems.append("the role contracts differ from provenance")
        roles_sha = hashlib.sha256(json.dumps(roles, sort_keys=True).encode()).hexdigest()
        if roles_sha != construction["role_contracts_sha256"]:
            problems.append("the role contracts are not the contract's")
        templates = hr.template_hashes()
        if templates != construction["templates_sha256"]:
            problems.append("the templates are not the contract's")
        frozen_templates = _read(paths.provenance / "frozen_templates.json")["files"]
        if any(frozen_templates.get(t) != h for t, h in templates.items()):
            problems.append("the templates differ from their provenance freeze")
        if json.loads(json.dumps(hk.record())) != construction["checks"]:
            problems.append("the classification of the checks is not the contract's")
        claude = json.loads(json.dumps(hrt.claude_runtime(paths.provenance)))
        differ = hrt.runtime_differences(claude, construction["claude_runtime"])
        if differ:  # the requested model, the binary, the effective model or the policy
            problems.append(f"the Claude runtime is not the contract's: {differ[:5]}")
        gate1 = _read(paths.provenance / "gate1.json")
        if _sha(paths.repo / AUDIT_MODULE) != gate1["audit"]["module_sha256"]:
            problems.append("the audit's implementation is not the one Gate 1 passed with")
        runtime = _read(paths.provenance / "runtime.json")
        if gate1["gate_1"] != "PASS" or any(
            gate1["binary"][k] != runtime[k] for k in ("version", "sha256")
        ):
            problems.append("Gate 1 did not pass with the pinned binary")
        salt = (paths.provenance / "opaque_salt.hex").read_text(encoding="utf-8").strip()
        if len(salt) != 64 or any(c not in "0123456789abcdef" for c in salt):
            problems.append("the opaque-id salt is not 32 bytes in hex")
    except (LaunchRefused, KeyError, OSError, hrt.RuntimeIdentityError) as error:
        return None, [*problems, f"provenance: {error}"]
    problems += _private_problems(paths)
    inherited = list(gate1["environment_allowlist"]["inherited"])
    extra_env = [v for v in inherited if v not in hi.ENV_INHERITED]
    frozen = Frozen(
        contract,
        pin["sha256"],
        runtime,
        grid,
        bytes.fromhex(salt) if not problems else b"",
        templates,
        extra_env,
    )
    return frozen, problems


def _private_problems(paths: Paths) -> list[str]:
    problems = []
    ignored = subprocess.run(
        ["git", "-C", str(paths.repo), "check-ignore", "-q", str(paths.test.resolve() / "x")],
        check=False,
    )
    if ignored.returncode != 0:
        problems.append(f"{paths.test} is not ignored by the public repository")
    hook = paths.test / ".git" / "hooks" / "pre-push"
    if not hook.exists() or hook.read_text(encoding="utf-8") != ho.NO_PUSH_HOOK:
        problems.append(f"{paths.test}: its history does not refuse every push")
    for folder in (paths.raw, paths.spool, paths.backups):
        problems += _private(folder, paths.repo)
    return problems


def runtime_problems(frozen: Frozen, probes: Probes | None) -> list[str]:
    """Check 6: the live runtimes. The binary is the one pinned and passed at Gate 1, by
    version and SHA-256 read from the file; the MLX runtime of the diagnostic, read from its
    environment without loading the model, is exactly the contract's."""
    probes = probes or Probes()
    binary = Path(frozen.runtime["path"])
    problems = []
    try:
        found = probes.read_binary(binary)
        changed = [k for k in ("version", "sha256") if found[k] != frozen.runtime[k]]
        if changed:
            problems.append(f"the Claude Code binary changed: {changed}")
    except (OSError, subprocess.SubprocessError) as error:
        problems.append(f"the pinned binary {binary} cannot be identified: {error}")
    try:
        live = json.loads(json.dumps(probes.read_mlx()))
        differ = hrt.runtime_differences(live, frozen.mlx_runtime)
        if differ:
            problems.append(f"the MLX runtime is not the contract's: {differ[:5]}")
    except (hrt.RuntimeIdentityError, OSError) as error:
        problems.append(f"the MLX runtime cannot be read: {error}")
    return problems


def public_commit_problems(paths: Paths, frozen: Frozen) -> list[str]:
    """START's last condition: the public pre-authoring commit recorded, and still holding
    exactly the frozen content. An amended contract (CONSTRUCTION_AMENDMENT_1) needs both: the
    base contract's commit, read from that commit, and the amendment's commit."""
    path = paths.provenance / PUBLIC_COMMIT_FILE
    if not path.exists():
        return ["the public pre-authoring commit is not recorded: no TEST need before it"]
    record = json.loads(path.read_text(encoding="utf-8"))
    amends = frozen.contract.get("amends")
    if amends is None:
        if record.get("contract_sha256") != frozen.contract_sha256:
            return ["the recorded public commit is of another contract"]
        return hc.committed_problems(
            frozen.contract, frozen.contract_sha256, paths.repo, record["commit"]
        )
    problems = []
    base = (amends["base_contract_sha256"], amends["base_public_commit"])
    if (record.get("contract_sha256"), record.get("commit")) != base:
        problems.append("the recorded public commit is not the amended contract's base")
    else:
        text = ho.git(paths.repo, "show", f"{record['commit']}:{hc.CONTRACT_PATH.as_posix()}")
        problems += hc.committed_problems(
            json.loads(text), record["contract_sha256"], paths.repo, record["commit"]
        )
    amended = paths.provenance / AMENDMENT_COMMIT_FILE
    if not amended.exists():
        return [*problems, "the amendment's public commit is not recorded"]
    found = json.loads(amended.read_text(encoding="utf-8"))
    if found.get("contract_sha256") != frozen.contract_sha256:
        return [*problems, "the recorded amendment commit is of another contract"]
    return problems + hc.committed_problems(
        frozen.contract, frozen.contract_sha256, paths.repo, found["commit"]
    )


def preflight(paths: Paths, action: str, confirm: str | None, probes: Probes | None) -> Frozen:
    """Everything checked before anything is built, or a refusal that ran and wrote nothing."""
    frozen, problems = contract_problems(paths)
    if frozen is None or problems:
        raise hs.IncompatibleState("; ".join(problems[:8]))
    problems = runtime_problems(frozen, probes)
    if problems:
        raise hs.IncompatibleState("; ".join(problems))
    if action in ("start", "resume", "freeze", "amend") and confirm != frozen.contract_sha256:
        raise LaunchRefused(f"{action} needs the full SHA-256 of the pinned contract, typed again")
    exists = (paths.test / hs.STATE_DIR / hs.DB_FILE).exists()
    if action == "start":
        if exists:
            raise LaunchRefused("a construction state exists: START never starts it again")
        problems = public_commit_problems(paths, frozen)
        if problems:
            raise LaunchRefused("; ".join(problems[:8]))
    if action in ("resume", "freeze", "amend") and not exists:
        raise LaunchRefused(f"no construction state: {action} never makes one up")
    return frozen


def build_world(paths: Paths, frozen: Frozen, probes: Probes | None) -> hpl.World:
    """The World of the real construction, from the files the checks have just verified. Its
    contract_check and runtime_check run the same checks again, inside the pipeline."""
    dev = bm.load_benchmark(bm.DEV_DIR)
    docs = bm.load_corpus()
    runtime = frozen.runtime

    def contract_check() -> None:
        again, problems = contract_problems(paths)
        if again is None or problems:
            raise hs.IncompatibleState("; ".join(problems[:8]))
        if again.contract_sha256 != frozen.contract_sha256:
            raise hs.IncompatibleState("another contract is pinned")

    def runtime_check() -> None:
        problems = runtime_problems(frozen, probes)
        if problems:
            raise hs.IncompatibleState("; ".join(problems))

    binary = {"version": runtime["version"], "sha256": runtime["sha256"]}
    return hpl.World(
        grid=frozen.grid,
        order=ho.cell_order(frozen.grid),
        omission=ho.omission_cells(frozen.grid),
        dev=dev,
        fixtures=bm.load_benchmark(bm.FIXTURES_DIR),
        corpus=hp.Corpus(docs),
        tools=CorpusTools(docs),
        texts=bm.node_texts(docs),
        excluded_psd2=px.load_psd2_exclusions(px.PSD2_EXCLUSIONS_PATH).excluded(),
        terms=project_terms(paths.repo),
        frozen_templates=frozen.templates,
        pin={"version": runtime["version"], "organization_sha256": runtime["organization_sha256"]},
        tmp_root=str(Path(tempfile.gettempdir()).resolve()),
        salt=frozen.salt,
        identities=hpl.frozen_identities(
            frozen.contract_sha256,
            paths.repo / hc.PLAN_PATH,
            frozen.grid,
            binary,
            static_terms(),
            paths.repo,
            frozen.claude_runtime,
            frozen.mlx_runtime,
        ),
        contract_check=contract_check,
        runtime_check=runtime_check,
        runtime={"claude_runtime": frozen.claude_runtime, "mlx_runtime": frozen.mlx_runtime},
    )


def run_auditor(
    paths: Paths, frozen: Frozen, probes: Probes | None = None
) -> Callable[[hr.RoleRun], hr.RoleAudit]:
    """The frozen audit of a recorded run, with exactly the arguments the pipeline gives it
    (hpl.Pipeline._audit): what the unattended controller runs again on a failed isolation
    run before it may resume (ingest/holdout_controller.py)."""
    w = build_world(paths, frozen, probes)

    def audit(run: hr.RoleRun) -> hr.RoleAudit:
        return hr.audit_role_run(
            run,
            w.terms,
            w.frozen_templates,
            w.forbidden,
            pin=w.pin,
            tmp_root=w.tmp_root,
            tools=w.tools,
        )

    return audit


def real_executor(frozen: Frozen, paths: Paths) -> hpl.Executor:
    """The executor of the real construction: every command asks for the contract's requested
    model, and the diagnostic runs only under the contract's MLX runtime."""
    runtime = frozen.runtime
    model = frozen.claude_runtime["requested_model"]
    if runtime["model"] != model:
        raise hs.IncompatibleState("the pinned model is not the contract's requested model")
    pin = {"version": runtime["version"], "sha256": runtime["sha256"]}
    return hx.RealExecutor(
        Path(runtime["path"]), model, pin, frozen.extra_env, paths.spool, frozen.mlx_runtime
    )


def _pipeline(
    paths: Paths, frozen: Frozen, world: hpl.World, store: hs.Store, executor: hpl.Executor
) -> hpl.Pipeline:
    def backup(label: str) -> Mapping[str, Any]:
        return hs.backup_state(store, paths.repo, paths.backups, label)

    return hpl.Pipeline(world, executor, store, backup)


def _folders(paths: Paths) -> None:
    for folder in (paths.raw, paths.spool, paths.backups):
        ho.private_folder(folder, paths.repo)


def start(
    paths: Paths,
    confirm: str | None,
    executor: ExecutorFactory = real_executor,
    probes: Probes | None = None,
) -> None:
    """START: a new construction, only where none exists. Its first technical failure stops it;
    nothing continues but an explicit RESUME."""
    frozen = preflight(paths, "start", confirm, probes)
    world = build_world(paths, frozen, probes)
    _folders(paths)
    store = hs.Store.create(
        paths.test,
        paths.raw,
        world.identities,
        world.order,
        world.omission,
        runtime=world.runtime,
    )
    try:
        _pipeline(paths, frozen, world, store, executor(frozen, paths)).build()
    finally:
        store.close()


def resume(
    paths: Paths,
    confirm: str | None,
    executor: ExecutorFactory = real_executor,
    probes: Probes | None = None,
) -> None:
    """RESUME: the construction whose state exists, from its incomplete stage."""
    frozen = preflight(paths, "resume", confirm, probes)
    world = build_world(paths, frozen, probes)
    store = hs.Store.open(paths.test, paths.raw)
    try:
        _compatible(store, world)  # before any executor exists
        _pipeline(paths, frozen, world, store, executor(frozen, paths)).resume()
    finally:
        store.close()


def _compatible(store: hs.Store, world: hpl.World) -> None:
    """The state, held: created with these identities and runtimes, over these cells, whole."""
    store.check_identities(world.identities, world.runtime)
    store.check_cells(world.order, world.omission)
    store.verify()


def validate(paths: Paths, probes: Probes | None = None) -> dict[str, Any]:
    """Checks 1 to 7 and the state, read only: a report, nothing built, nothing run."""
    frozen, problems = contract_problems(paths)
    if frozen is not None:
        problems += runtime_problems(frozen, probes)
    report: dict[str, Any] = {
        "contract_sha256": frozen.contract_sha256 if frozen else None,
        "problems": problems,
        "public_commit": None,
        "state": None,
    }
    if frozen is not None:
        record = paths.provenance / PUBLIC_COMMIT_FILE
        if record.exists():
            report["public_commit"] = json.loads(record.read_text(encoding="utf-8"))["commit"]
            report["public_commit_problems"] = public_commit_problems(paths, frozen)
    if (paths.test / hs.STATE_DIR / hs.DB_FILE).exists():
        try:
            with hs.Store.open(paths.test, paths.raw, readonly=True) as store:
                report["state"] = store.verify()
        except hs.StateError as error:
            problems.append(f"the state: {error}")
    return report


def _record_commit(paths: Paths, commit: str, name: str, amended: bool) -> dict[str, Any]:
    frozen, problems = contract_problems(paths)
    if frozen is None or problems:
        raise hs.IncompatibleState("; ".join(problems[:8]))
    if amended != ("amends" in frozen.contract):
        raise LaunchRefused(
            "the pinned contract amends nothing" if amended else "the pinned contract is amended"
        )
    sha = ho.git(paths.repo, "rev-parse", f"{commit}^{{commit}}")
    problems = hc.committed_problems(frozen.contract, frozen.contract_sha256, paths.repo, sha)
    if problems:
        raise LaunchRefused("; ".join(problems[:8]))
    listed = ho.git(paths.repo, "branch", "-r", "--contains", sha).splitlines()
    remotes = sorted({line.strip().split(" -> ")[0] for line in listed if line.strip()})
    if not remotes:
        raise LaunchRefused(f"{sha[:12]} is on no remote branch: push it first")
    path = paths.provenance / name
    record = {
        "commit": sha,
        "contract_sha256": frozen.contract_sha256,
        "frozen_files": len(hc.frozen_files(frozen.contract)) + 1,
        "remote_branches": remotes,
        "committed_at": ho.git(paths.repo, "show", "-s", "--format=%cI", sha),
    }
    if path.exists():
        found = json.loads(path.read_text(encoding="utf-8"))
        if found["commit"] != sha:
            raise LaunchRefused(f"{path} records another commit: it is written once")
        return dict(found)
    ho.write_json(path, record | {"recorded_at": datetime.now(UTC).isoformat()})
    ho.git(paths.test, "add", "--", str(path.relative_to(paths.test)))
    what = "amendment" if amended else "public pre-authoring"
    ho.git(paths.test, "commit", "-q", "-m", f"record the {what} commit {sha[:12]}")
    return record


def record_public_commit(paths: Paths, commit: str) -> dict[str, Any]:
    """After the public pre-authoring commit is pushed: its SHA in the private provenance,
    once, if the commit holds exactly the frozen content and a remote branch contains it. It
    is a timestamp of the frozen content, never part of it."""
    return _record_commit(paths, commit, PUBLIC_COMMIT_FILE, amended=False)


def record_amendment_commit(paths: Paths, commit: str) -> dict[str, Any]:
    """After the public commit of an amended contract is pushed (CONSTRUCTION_AMENDMENT_1): its
    SHA in the private provenance, once, under the same conditions as the base's."""
    return _record_commit(paths, commit, AMENDMENT_COMMIT_FILE, amended=True)


def _replay_point(
    paths: Paths, frozen: Frozen, world: hpl.World, store: hs.Store, held: Mapping[str, str]
) -> dict[str, Any]:
    """The amended code replays the whole log under the identities the state holds, with every
    write refused, so that it can run no model and change nothing: every admitted stage and
    every decision must be found again exactly, and the first write it would make is where
    the amendment begins."""
    pipeline = _pipeline(
        paths, frozen, dataclasses.replace(world, identities=dict(held)), store, _NoModels()
    )
    try:
        pipeline.replay()
    except hs.StateError as error:
        refused = type(error) is hs.StateError and "writes nothing to the state" in str(error)
        if not refused:
            raise
        last = pipeline.ledger.attempts[-1]
        return {
            "refused_write": str(error).split(":", 1)[0],
            "cell_id": last.cell_id,
            "proposal_id": last.proposal_id,
            "outcome": last.outcome,
        }
    return {"refused_write": None}


def amend(paths: Paths, confirm: str | None, probes: Probes | None = None) -> dict[str, Any]:
    """A documented amendment of the construction (CONSTRUCTION_AMENDMENT_1), recorded as one
    event of the durable state at the exact point the amendment names, after every check and
    a replay under the amended code that runs no model. Before it, everything was built under
    the base contract; after it, under the base and the amendment. Nothing is rebuilt."""
    frozen = preflight(paths, "amend", confirm, probes)
    amends = frozen.contract.get("amends")
    if amends is None:
        raise LaunchRefused("the pinned contract amends nothing")
    problems = public_commit_problems(paths, frozen)
    if problems:
        raise LaunchRefused("; ".join(problems[:8]))
    world = build_world(paths, frozen, probes)
    store = hs.Store.open(paths.test, paths.raw)
    try:
        summary = store.verify()
        done = [a for a in store.amendments() if a["amendment"] == amends["amendment"]]
        if done:
            return dict(done[0])
        held: dict[str, str] = json.loads(store.meta("identities") or "{}")
        if held.get("pre_authoring_contract") != amends["base_contract_sha256"]:
            raise LaunchRefused("the state was not built under the amended contract's base")
        after = amends["after_event"]
        if (summary["events"], summary["head_sha256"]) != (after["seq"], after["sha256"]):
            raise LaunchRefused("the log is not at the point the amendment names")
        changed = sorted(
            k for k in set(held) | set(world.identities) if held.get(k) != world.identities.get(k)
        )
        if changed != sorted(amends["identity_changes"]):
            raise LaunchRefused(
                f"the amendment would change {changed}; it declares {amends['identity_changes']}"
            )
        if store.runtime() != world.runtime:
            raise LaunchRefused("the runtimes are not the state's")
        store.check_cells(world.order, world.omission)
        point = _replay_point(paths, frozen, world, store, held)
        if point != amends["first_transition"]:
            raise LaunchRefused(f"the amended code's first transition is {point}")
        public = json.loads((paths.provenance / AMENDMENT_COMMIT_FILE).read_text(encoding="utf-8"))
        record = {
            "amendment": amends["amendment"],
            "amendment_sha256": amends["amendment_sha256"],
            "base_contract_sha256": amends["base_contract_sha256"],
            "contract_sha256": frozen.contract_sha256,
            "public_commit": public["commit"],
            "after_event": after,
            "changed_identities": changed,
            "first_transition": point,
            "from_identities": held,
            "to_identities": dict(world.identities),
        }
        store.amend(record)
        return {k: record[k] for k in hs.AMENDMENT_FIELDS}
    finally:
        store.close()


class _NoModels:
    """The freeze's executor: it runs nothing. The replay refuses every write before a run
    could start, so none of these is ever reached."""

    def _refuse(self, *args: Any) -> Any:
        raise AssertionError("the freeze runs no model")

    headless = broker = local = _refuse
    recover_headless = recover_broker = recover_local = _refuse


def _export(
    paths: Paths,
    frozen: Frozen,
    docs: Mapping[str, Any],
    amendments: Sequence[Mapping[str, Any]] = (),
) -> Callable[..., dict[str, str]]:
    def export(records: list[dict[str, Any]], report: Mapping[str, Any]) -> dict[str, str]:
        """The benchmark files from the READY records, sealed as TEST 1.0.0, the private freeze
        record, and the private history: written again equal if a freeze is run again."""
        d = paths.test
        bm.write_jsonl(d / bm.NEEDS_FILE, [bm.Need.model_validate(r["need"]) for r in records])
        bm.write_jsonl(
            d / bm.QUERIES_FILE, [bm.Query.model_validate(q) for r in records for q in r["queries"]]
        )
        bm.write_jsonl(
            d / bm.QRELS_FILE, [bm.Qrel.model_validate(q) for r in records for q in r["qrels"]]
        )
        adjudications = [
            bm.Adjudication.model_validate(a) for r in records for a in r["adjudications"]
        ]
        bm.write_jsonl(d / bm.ADJUDICATIONS_FILE, adjudications)
        bm.seal(d, ho.FREEZE_VERSION, "test", docs)
        public = json.loads((paths.provenance / PUBLIC_COMMIT_FILE).read_text(encoding="utf-8"))
        ids = [r["need"]["need_id"] for r in records]
        ids += [q["query_id"] for r in records for q in r["queries"]]
        path = d / ho.FREEZE_FILE
        found = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        frozen_at = (
            found["frozen_at"] if found else datetime.now(UTC).replace(microsecond=0).isoformat()
        )
        record = ho.freeze_record(
            d,
            ho.load_areas(),
            paths.repo / hc.PLAN_PATH,
            {
                "overlap": report["gate"],
                "ledger": report["ledger"],
                "plan_against_realized": report["plan_against_realized"],
                "diagnostic": report["diagnostic"],
                "omission_audit": report["omission_audit"],
                "rejections": report["rejections"],
                "technical": report["technical"],
            },
            ho.isolation_problems(paths.repo, ids),
            bm.TEMPLATES_DIR / "holdout",
            _pre_authoring(frozen, public, amendments),
            {"needs": len(records)},
            frozen_at,
        )
        record = json.loads(json.dumps(record, ensure_ascii=False))  # as written: str keys
        if found is not None and found != record:
            differ = sorted(k for k in set(found) | set(record) if found.get(k) != record.get(k))
            raise ho.HoldoutError(f"{path} exists with other {differ}: it is written once")
        if found is None:
            ho.write_json(path, record)
        ho.git(d, "add", "-A")
        if ho.git(d, "diff", "--cached", "--name-only"):
            ho.git(d, "commit", "-q", "-m", f"freeze TEST {ho.FREEZE_VERSION}")
        names = (*bm.DATA_FILES, bm.ADJUDICATIONS_FILE, bm.MANIFEST_FILE, ho.FREEZE_FILE)
        return {name: bm.sha256_file(d / name) for name in names}

    return export


def _pre_authoring(
    frozen: Frozen, public: Mapping[str, Any], amendments: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The contract TEST was frozen under: the base, and every amendment with the point of the
    log where it applies (CONSTRUCTION_AMENDMENT_1)."""
    amends = frozen.contract.get("amends")
    if amends is None:
        return {"contract_sha256": frozen.contract_sha256, "public_commit": public["commit"]}
    return {
        "contract_sha256": amends["base_contract_sha256"],
        "public_commit": public["commit"],
        "amended_contract_sha256": frozen.contract_sha256,
        "amendments": [dict(a) for a in amendments],
    }


def _commitment(paths: Paths) -> Callable[[dict[str, Any]], None]:
    def commit(record: dict[str, Any]) -> None:
        """The public commitment: hashes, counts and distributions, never a text or a TEST id;
        written once, beside the private directory, for the person to commit and push."""
        freeze_path = paths.test / ho.FREEZE_FILE
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        public = ho.commitment(freeze, bm.sha256_file(freeze_path))
        target = paths.repo / ho.COMMITMENT_PATH
        if target.exists():
            if json.loads(target.read_text(encoding="utf-8")) != public:
                raise ho.HoldoutError(f"{target} exists with other content: it is written once")
            return
        ho.write_json(target, public)

    return commit


def freeze(paths: Paths, confirm: str | None, probes: Probes | None = None) -> dict[str, Any]:
    """The freeze of TEST 1.0.0, with no model: the construction replayed from its state with
    every write refused, the invariants checked, then the export, the seal, the frozen event, a
    verified backup and the public commitment."""
    frozen = preflight(paths, "freeze", confirm, probes)
    world = build_world(paths, frozen, probes)
    docs = bm.load_corpus()
    store = hs.Store.open(paths.test, paths.raw)
    try:
        _compatible(store, world)
        pipeline = _pipeline(paths, frozen, world, store, _NoModels())
        pipeline.replay()
        return pipeline.freeze(_export(paths, frozen, docs, store.amendments()), _commitment(paths))
    finally:
        store.close()
