"""A stand-in for TEST v2's model runs and retrievers, to test the construction with no model
and no TEST content. Every reply is synthetic and depends only on the packet and the role, so
a construction resumed in another process gets the same answers; every session has the shape of
Gate 1's runtime (tests/holdout_fakes.py session), so the real isolation audit runs on it.

The world is small on purpose: the real corpus, DEV, fixtures and anchors, but a chunk layer cut
to the chunks the fake retrievers rank and the anchors own, so that pools and the completeness
screen stay a few batches."""

from __future__ import annotations

import functools
import hashlib
import json
import re
import tempfile
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ingest import benchmark as bm
from ingest import evaluation as ev
from ingest import holdout as ho
from ingest import holdout_isolation as hi
from ingest import holdout_packets as hp
from ingest import holdout_pipeline as hpl
from ingest import holdout_roles as hr
from ingest import holdout_v2 as v2
from ingest import holdout_v2_audit as hau
from ingest import holdout_v2_dedup as dd
from ingest import holdout_v2_engine as en
from ingest import holdout_v2_retrieval as rv
from ingest import holdout_v2_runner as hrun
from ingest import holdout_v2_state as vs
from ingest import psd2_exclusions as px
from tests import holdout_fakes as hf

SALT = b"holdout-v2-test-salt-0123456789ab"
SMALL_POOL = 36  # chunks the fake retrievers rank
SYSTEMS = ("E5", "BGE-M3", "Qwen3-Embedding-0.6B", "BM25", "E")


def _h(*parts: str) -> int:
    return int(hashlib.sha256(":".join(parts).encode()).hexdigest()[:12], 16)


@functools.cache
def base() -> dict[str, Any]:
    """The real frozen inputs (corpus, layer, DEV, fixtures, PSD2 exclusions, legal units)."""
    dev = bm.load_benchmark(bm.DEV_DIR)
    fixtures = bm.load_benchmark(bm.FIXTURES_DIR)
    docs = bm.load_corpus()
    layer = rv.load_layer()
    excluded = px.load_psd2_exclusions(px.PSD2_EXCLUSIONS_PATH).excluded()
    return {
        "dev": dev,
        "fixtures": fixtures,
        "docs": docs,
        "corpus": hp.Corpus(docs),
        "layer": layer,
        "excluded": excluded,
        "slots": v2.slots(dev),
        "units": v2.legal_units(docs, ho.load_areas(), excluded, layer.layer),
    }


def small_bank(universe: int = 6, salt: bytes = SALT) -> v2.Bank:
    b = base()
    return v2.build_bank(
        b["slots"], b["units"], salt, v2.DEVELOPMENT_SEED, v2.DEVELOPMENT_SEED + 1, universe
    )


def small_layer(bank: v2.Bank) -> tuple[ev.ChunkLayer, list[str]]:
    """The real layer cut to SMALL_POOL chunks with nodes to judge, and the anchors' chunks."""
    b = base()
    layer = b["layer"].layer
    texts = b["corpus"].texts
    judgeable = [c for c in sorted(layer.chunks) if v2.judged_nodes(c, layer, texts, b["excluded"])]
    picked = sorted(judgeable, key=lambda c: _h("small", c))[:SMALL_POOL]
    anchors = {layer.owner[a.node_key] for c in bank.candidates for a in c.anchors}
    keep = sorted(set(picked) | anchors)
    refs = [layer.chunks[c] for c in keep]
    return ev.layer_from_refs(refs, layer.max_tokens, layer.parser_version, layer.sources), picked


def fake_rankings(
    queries: Sequence[tuple[str, str]],
    chunks: Sequence[str],
    favour: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Every system ranks the same chunks, each in an order of its own drawn from the system and
    the query; favour puts given chunks first for a system."""
    out: dict[str, dict[str, list[str]]] = {}
    for s in SYSTEMS:
        out[s] = {}
        for qid, _ in queries:
            order = sorted(chunks, key=lambda c: _h(s, qid, c))
            first = [c for c in (favour or {}).get(s, ()) if c in set(chunks)]
            out[s][qid] = first + [c for c in order if c not in set(first)]
    return out


def fake_vectors(texts: Sequence[str]) -> np.ndarray:
    rows = [np.random.default_rng(_h("vec", t)).normal(size=16) for t in texts]
    return np.stack(rows) if rows else np.zeros((0, 16))


def default_grade(node: str, role: str) -> tuple[int, bool]:
    """A synthetic label: most nodes 0, some relevant; B2 differs from B1 on a few, so that C
    has work; the anchors' nodes do not matter to it."""
    h = _h("grade", node) % 20
    grade = 0 if h < 13 else 1 if h < 16 else 2 if h < 18 else 3
    ambiguous = h == 16
    if role == "v2_pool_judge_2" and _h("b2", node) % 9 == 0:
        grade = (grade + 1) % 4
    return grade, ambiguous


_CANDIDATE = re.compile(r"^CANDIDATE (C-[0-9a-f]{12})\n\[([A-Za-z0-9]+:[^\]]+)\]", re.M)
_ITEM = re.compile(r"^ITEM (C-[0-9a-f]{12})\n\[([A-Za-z0-9]+:[^\]]+)\]", re.M)
_ANSWERABILITY_ITEM = re.compile(r"^ITEM (C-[0-9a-f]{12}): the answerability", re.M)
_PASSAGE = re.compile(r"^PASSAGE (P-[0-9a-f]{12})", re.M)
_TOPIC = re.compile(r"topic (T\d{3})")


def topic_of(concept_packet: str) -> str:
    """A concept's synthetic topic: its specification, never its anchors or ids. Candidates of
    one profile share it, so that the fake gatekeeper can find them repeating each other."""
    spec = re.sub(r"R-[0-9a-f]+", "", concept_packet.split("\nFrom ", 1)[0])
    return f"T{_h('topic', spec) % 1000:03d}"


@dataclass
class V2Fake:
    """Replies by role, from the packet alone. fail: job-id prefix -> how many of its next
    executions fail technically; answerability: the first blind judgment's answer;
    dup_topics (or dup_all): statements of the same synthetic topic repeat each other, and the
    gatekeeper names the earlier one it is shown."""

    spool: Path
    grade: Callable[[str, str], tuple[int, bool]] = default_grade
    answerability: str = "complete"
    answerability_2: str = "partial"
    flag_every: int = 7
    fail: dict[str, int] = field(default_factory=dict)
    capacity: dict[str, int] = field(default_factory=dict)  # run-id prefix -> capacity refusals
    die_after: str | None = None
    dup_topics: frozenset[str] = frozenset()
    dup_all: bool = False
    broken: frozenset[str] = frozenset()  # roles whose every answer breaks its contract
    workdir: str = field(default_factory=lambda: str(Path(tempfile.gettempdir()) / "tmp.x"))
    calls: Counter[str] = field(default_factory=Counter)
    packets: list[tuple[str, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def headless(self, role: str, run_id: str, packet: str) -> hpl.Session:
        with self.lock:
            self.calls[role] += 1
            self.packets.append((role, packet))
            for prefix, left in list(self.capacity.items()):
                if run_id.startswith(prefix) and left > 0:
                    self.capacity[prefix] = left - 1
                    raise hpl.TechnicalFailure(en.CAPACITY, "refused for its capacity")  # type: ignore[arg-type]
            for prefix, left in list(self.fail.items()):
                if run_id.startswith(prefix) and left > 0:
                    self.fail[prefix] = left - 1
                    raise RuntimeError("the API answered 400 invalid request")
        reply = "{}" if role in self.broken else self.reply(role, packet)
        session = hf.session([packet], [reply], workdir=self.workdir)
        self.spool.mkdir(parents=True, exist_ok=True)
        (self.spool / f"{hashlib.sha256(run_id.encode()).hexdigest()}.json").write_text(
            json.dumps({"reply": reply, "session": session})
        )
        if self.die_after is not None and run_id.startswith(self.die_after):
            self.die_after = None
            raise hf.Crash(f"killed after {run_id}")
        return hpl.Session(reply, session)

    def recover_headless(self, role: str, run_id: str, packet: str) -> hpl.Session | None:
        path = self.spool / f"{hashlib.sha256(run_id.encode()).hexdigest()}.json"
        if not path.exists():
            return None
        found = json.loads(path.read_text())
        return hpl.Session(found["reply"], found["session"])

    # -----------------------------------------------------------------------

    def reply(self, role: str, packet: str) -> str:
        ref = re.search(r"\b(R-[0-9a-f]+)\b", packet)
        rid = ref.group(1) if ref else "R-0"
        if role == "v2_concept_author":
            topic = topic_of(packet)
            case = _h(packet) % 997
            return json.dumps(
                {
                    "cell_id": rid,
                    "concept": f"A synthetic firm on topic {topic} asks about case {case}.",
                    "facts": [f"The firm works on case {case}."],
                    "atomicity": "One question.",
                    "figure_or_date": "no",
                    "fit": "fits",
                }
            )
        if role == "statement_writer":
            found_topic = _TOPIC.search(packet)
            found_case = re.search(r"case (\d+)", packet)
            text = (
                f"A synthetic firm on topic {found_topic.group(1) if found_topic else 'T000'} "
                f"asks what it must do in case {found_case.group(1) if found_case else '0'}."
            )
            return json.dumps(
                {
                    "proposal_id": rid,
                    "primary": text,
                    "primary_atomicity": "One question.",
                    "alternative": None,
                    "alternative_reason": None,
                }
            )
        if role == "query_writer":
            n = _h(packet) % 9973
            return (
                f"en formal: What must a firm do in synthetic case {n}?\n"
                f"en natural: What do I have to do in case {n}?\n"
                f"es natural: Que debo hacer en el caso {n}?"
            )
        if role == "gatekeeper":
            return json.dumps(self._gate(packet, rid))
        if role in ("v2_pool_judge_1", "v2_pool_judge_2"):
            grades = []
            for item, node in _CANDIDATE.findall(packet):
                grade, ambiguous = self.grade(node, role)
                grades.append(
                    {
                        "item_id": item,
                        "grade": grade,
                        "ambiguous": ambiguous,
                        "scope_basis": [node],
                        "reason": "The text says so.",
                    }
                )
            return json.dumps({"need_id": rid, "grades": grades})
        if role == "judge_c":
            items: list[dict[str, Any]] = []
            for item in _ANSWERABILITY_ITEM.findall(packet):
                items.append(
                    {
                        "item_id": item,
                        "grade": None,
                        "ambiguous": False,
                        "answerability": self.answerability_2,
                        "scope_basis": [],
                        "reason": "The evidence covers part of it.",
                    }
                )
            for item, node in _ITEM.findall(packet):
                grade, _ = self.grade(node, "v2_pool_judge_1")
                items.append(
                    {
                        "item_id": item,
                        "grade": grade,
                        "ambiguous": False,
                        "answerability": None,
                        "scope_basis": [node],
                        "reason": "The text settles it.",
                    }
                )
            return json.dumps({"need_id": rid, "items": items})
        if role in ("v2_answerability", "v2_answerability_2"):
            value = self.answerability if role == "v2_answerability" else self.answerability_2
            return json.dumps(
                {
                    "need_id": rid,
                    "answerability": value,
                    "missing": None if value == "complete" else "a synthetic gap",
                    "reason": "From the grades.",
                }
            )
        if role == "metadata_adjudicator":
            partial = 'partial_out_of_scope", "difficulty' in packet
            return json.dumps(
                {
                    "need_id": rid,
                    "type": "partial_out_of_scope" if partial else "direct",
                    "difficulty": "medium",
                    "drivers": {},
                    "reason": "Synthetic.",
                }
            )
        if role == "v2_screen":
            passages = [
                {"item_id": p, "flag": _h("flag", p) % self.flag_every == 0, "reason": "x"}
                for p in _PASSAGE.findall(packet)
            ]
            return json.dumps({"need_id": rid, "passages": passages})
        raise AssertionError(f"no fake reply for {role}")

    def _gate(self, packet: str, rid: str) -> dict[str, Any]:
        proposal = packet.split("THE PROPOSAL", 1)[1].split("THE NEEDS THAT ALREADY EXIST", 1)
        mine = _TOPIC.search(proposal[0])
        existing = re.findall(r"^- (\S+) \((\w+)\): (.*)$", proposal[1], re.M)
        if mine and (self.dup_all or mine.group(1) in self.dup_topics):
            for shown, _, text in existing:
                other = _TOPIC.search(text)
                if other and other.group(1) == mine.group(1):
                    return {
                        "proposal_id": rid,
                        "verdict": "paraphrase",
                        "closest": shown,
                        "reason": "Same.",
                    }
        return {
            "proposal_id": rid,
            "verdict": "independent",
            "closest": None,
            "reason": "Different.",
        }


@functools.cache
def statements() -> tuple[str, ...]:
    """DEV's and the fixtures' statements (versioned: no corpus needed)."""
    dev, fixtures = bm.load_benchmark(bm.DEV_DIR), bm.load_benchmark(bm.FIXTURES_DIR)
    return tuple(n.statement for n in dev.needs + fixtures.needs)


def audit_context(frozen_templates: Mapping[str, str] | None = None) -> en.AuditContext:
    return en.AuditContext(
        terms=hi.forbidden_terms(None, None, []),
        frozen_templates=dict(frozen_templates or hr.template_hashes()),
        forbidden=list(statements()),
        pin=hf.PIN,
        tmp_root=str(Path(tempfile.gettempdir()).resolve()),
        claude_effective=hf.FAKE_EFFECTIVE,
    )


def references() -> list[dd.Reference]:
    b = base()
    refs = [dd.Reference(n.need_id, "DEV", n.statement) for n in b["dev"].needs]
    refs += [dd.Reference(n.need_id, "FIXTURE", n.statement) for n in b["fixtures"].needs]
    refs += [
        dd.Reference(f"historical:{e.key}", "DEVELOPMENT", e.statement) for e in hf.FAKE_EXCLUSIONS
    ]
    return refs


@dataclass
class Built:
    world: hrun.World
    executor: V2Fake
    store: vs.Store
    runner: hrun.Runner
    pool_chunks: list[str]


def world(
    directory: Path,
    executor: V2Fake | None = None,
    universe: int = 6,
    n_target: int = 4,
    evaluate: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    favour: Mapping[str, Sequence[str]] | None = None,
    workers: int = 3,
    check: Callable[[str], None] = lambda phase: None,
) -> tuple[hrun.World, V2Fake, list[str]]:
    b = base()
    bank = small_bank(universe)
    layer, picked = small_layer(bank)
    texts = {c: b["layer"].chunks[c].text for c in layer.chunks}
    fake = executor or V2Fake(directory.with_name(directory.name + "-spool"))

    def rank(system: str, queries: Sequence[tuple[str, str]], work: Path) -> rv.Run:
        return rv.Run(system, fake_rankings(queries, picked, favour)[system])

    w = hrun.World(
        namespace=directory,
        corpus=b["corpus"],
        layer=layer,
        chunk_text=texts,
        excluded_psd2=b["excluded"],
        forbidden=[n.statement for n in b["dev"].needs + b["fixtures"].needs],
        references=references(),
        bank=bank,
        chains=v2.completeness_chains(bank, SALT),
        audit_chain=hau.chain(bank, SALT),
        salt=SALT,
        params=hrun.Params(workers=workers, backoff_seconds=0.0),
        audit=audit_context(),
        rank=rank,
        write_runs=rv.write_runs,
        read_runs=rv.read_runs,
        vectors=fake_vectors,
        check=check,
        evaluate=evaluate,
        n_target=n_target,
        count_tokens=lambda text: len(text) // 4,
    )
    return w, fake, picked


def raw(directory: Path) -> Path:
    return directory.with_name(directory.name + "-raw")


def runner(w: hrun.World, fake: V2Fake, create: bool = True) -> hrun.Runner:
    if create and not (w.namespace / vs.STATE_DIR / vs.DB_FILE).exists():
        store = vs.Store.create(w.namespace, raw(w.namespace), {"kind": "test"})
    else:
        store = vs.Store.open(w.namespace, raw(w.namespace))
    r = hrun.Runner(w, store, fake)
    r.engine.sleep = lambda seconds: None
    return r
