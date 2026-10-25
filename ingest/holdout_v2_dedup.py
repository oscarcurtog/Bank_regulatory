"""TEST v2's global deduplication (D-067): one deterministic stage after the whole bank exists.

Every generated candidate is compared with the needs that already exist: DEV, the fixtures, the
historical statements of the aborted constructions and the development pilots (DEVELOPMENT),
and the candidates of the bank that come before it in the priority order and were kept (TEST).
The semantic decision is the frozen gate's (overlap_gate.v2), under opaque ids; what changes is
how many references it reads: never all of them at once, but a bounded neighborhood each
candidate gets from a deterministic prefilter.

The prefilter nominates, for each candidate:
- MANDATORY, whatever the bound: every reference with the same normalized statement, every
  reference whose content-word Jaccard similarity reaches MANDATORY_JACCARD, and every earlier
  bank candidate that shares an anchor article;
- the TOP_REFERENCES most similar references and the TOP_BANK most similar earlier bank
  candidates by BGE-M3 cosine similarity, and the TOP_LEXICAL most similar of either by Jaccard.
Of the bank candidates nominated, only those already kept are shown. A neighborhood larger than
MAX_SHOWN is shown in several packets, each a gate call, never cut: a candidate is kept only if
every packet finds it independent.

The order of the bank decides, never the order candidates were generated or arrived in: a
candidate is resolved only after every earlier bank candidate it nominated, so the same answers
of the gate give the same selection whatever order the inputs come in (resolve). The nominated
neighborhoods are recorded before the first gate call, so a restart shows the same packets.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ingest import holdout as ho

MANDATORY_JACCARD = 0.5
TOP_REFERENCES = 6
TOP_BANK = 6
TOP_LEXICAL = 3
MAX_SHOWN = 20
KINDS = ("DEV", "FIXTURE", "DEVELOPMENT", "TEST")


@dataclass(frozen=True)
class Reference:
    key: str  # the reference's own key: a DEV or fixture need id, a historical entry's key
    kind: str  # DEV, FIXTURE or DEVELOPMENT
    statement: str


@dataclass(frozen=True)
class Item:
    """A generated candidate, as the deduplication sees it."""

    candidate_id: str
    rank: int
    statement: str
    articles: tuple[tuple[str, str], ...]  # its anchors' (act, article)


@dataclass(frozen=True)
class Neighborhood:
    references: tuple[str, ...]  # reference keys, mandatory first, then by similarity
    bank: tuple[str, ...]  # earlier bank candidates nominated, mandatory first

    def record(self) -> dict[str, Any]:
        return {"references": list(self.references), "bank": list(self.bank)}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def _cosine(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0, 1, norms)
    return unit @ unit.T


def neighborhoods(
    items: Sequence[Item],
    references: Sequence[Reference],
    vectors: Mapping[str, np.ndarray],
) -> dict[str, Neighborhood]:
    """Every candidate's nominated neighborhood. vectors: candidate id or reference key -> its
    statement's BGE-M3 vector. Deterministic: ties go to the lower key."""
    ordered = sorted(items, key=lambda i: i.rank)
    ref_keys = [r.key for r in references]
    texts = {r.key: r.statement for r in references} | {
        i.candidate_id: i.statement for i in ordered
    }
    keys = ref_keys + [i.candidate_id for i in ordered]
    sim = _cosine(np.stack([np.asarray(vectors[k], dtype=np.float64) for k in keys]))
    index = {k: n for n, k in enumerate(keys)}
    out: dict[str, Neighborhood] = {}
    for pos, item in enumerate(ordered):
        me = index[item.candidate_id]
        earlier = [o.candidate_id for o in ordered[:pos]]
        same_article = {
            o.candidate_id for o in ordered[:pos] if set(o.articles) & set(item.articles)
        }
        lexical = {k: ho.jaccard(item.statement, texts[k]) for k in ref_keys + earlier}
        exact = {
            k for k in ref_keys + earlier if _normalized(texts[k]) == _normalized(item.statement)
        }
        mandatory = exact | {k for k, j in lexical.items() if j >= MANDATORY_JACCARD} | same_article

        def by_cosine(pool: Sequence[str], me: int = me) -> list[str]:
            return sorted(pool, key=lambda x: (-sim[me, index[x]], x))

        def by_jaccard(pool: Sequence[str], lexical: dict[str, float] = lexical) -> list[str]:
            return sorted(pool, key=lambda x: (-lexical[x], x))

        nominated = (
            set(by_cosine(ref_keys)[:TOP_REFERENCES])
            | set(by_cosine(earlier)[:TOP_BANK])
            | set(by_jaccard(ref_keys + earlier)[:TOP_LEXICAL])
            | mandatory
        )
        rank_of = by_cosine(sorted(nominated))
        order = sorted(nominated, key=lambda k: (k not in mandatory, rank_of.index(k)))
        out[item.candidate_id] = Neighborhood(
            tuple(k for k in order if k in set(ref_keys)),
            tuple(k for k in order if k not in set(ref_keys)),
        )
    return out


@dataclass(frozen=True)
class Shown:
    """One gate packet's references: (opaque id, kind, statement), and the id -> key map the
    orchestrator keeps."""

    existing: tuple[tuple[str, str, str], ...]
    keys: dict[str, str]


def packets(
    candidate_id: str,
    neighborhood: Neighborhood,
    kept: Mapping[str, str],
    references: Mapping[str, Reference],
    salt: bytes,
) -> list[Shown]:
    """The gate packets of one candidate: its nominated references, and the nominated earlier
    candidates that were kept (kept: candidate id -> statement), every one under an opaque id
    drawn for this candidate (no reference shows its own key, not even DEV's), with its kind,
    MAX_SHOWN at most each."""
    entries: list[tuple[str, str, str, str]] = []
    for key in neighborhood.references:
        r = references[key]
        shown = ho.opaque_id(salt, f"v2-gate:{candidate_id}", key)
        entries.append((shown, r.kind, r.statement, key))
    for other in neighborhood.bank:
        if other in kept:
            shown = ho.opaque_id(salt, f"v2-gate:{candidate_id}", other)
            entries.append((shown, "TEST", kept[other], other))
    if not entries:
        return []
    out = []
    for start in range(0, len(entries), MAX_SHOWN):
        part = entries[start : start + MAX_SHOWN]
        out.append(
            Shown(
                tuple((s, kind, text) for s, kind, text, _ in part), {s: k for s, _, _, k in part}
            )
        )
    return out


@dataclass(frozen=True)
class Decision:
    candidate_id: str
    kept: bool
    verdicts: tuple[str, ...]  # one per packet
    closest_kind: str | None  # the kind of the reference the first non-independent packet named


Gate = Callable[[str, int, Shown], Mapping[str, Any]]


def resolve(
    items: Sequence[Item],
    hoods: Mapping[str, Neighborhood],
    references: Mapping[str, Reference],
    salt: bytes,
    gate: Gate,
) -> dict[str, Decision]:
    """The deduplication of the whole bank: in priority order, each candidate kept only if every
    packet of its neighborhood is independent. gate(candidate id, packet number, packet) is the
    gatekeeper's admitted verdict. The inputs' order is irrelevant: they are sorted by rank."""
    kept: dict[str, str] = {}
    out: dict[str, Decision] = {}
    for item in sorted(items, key=lambda i: i.rank):
        verdicts, closest_kind = [], None
        for n, shown in enumerate(
            packets(item.candidate_id, hoods[item.candidate_id], kept, references, salt), 1
        ):
            answer = gate(item.candidate_id, n, shown)
            verdicts.append(str(answer["verdict"]))
            if answer["verdict"] != "independent" and closest_kind is None:
                closest = answer.get("closest")
                kinds = {s: kind for s, kind, _ in shown.existing}
                closest_kind = kinds.get(closest) if closest else None
        keep = all(v == "independent" for v in verdicts)
        if keep:
            kept[item.candidate_id] = item.statement
        out[item.candidate_id] = Decision(item.candidate_id, keep, tuple(verdicts), closest_kind)
    return out


def ready(
    pending: Sequence[Item], hoods: Mapping[str, Neighborhood], resolved: set[str]
) -> list[Item]:
    """The candidates whose packets can be built now: every earlier bank candidate they nominate
    is resolved. The runner calls the gate for these in parallel; resolve() then gives the same
    selection as calling them one by one in priority order."""
    return [i for i in pending if all(o in resolved for o in hoods[i.candidate_id].bank)]
