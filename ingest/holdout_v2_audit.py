"""TEST v2's optional human audit (D-067): recommended, never required, never corrective.

A person grades, blind, the passages (chunks) of the E-vs-E5 decisive set of AUDIT_SIZE needs
fixed before any judgment, with DECOYS_PER_NEED other pooled passages of each need mixed in as
controls. The packet shows the need and the passages' text only: no system, no rank, no
formulation, no label of any judge, and an order drawn per need. The person is not a legal
expert, so the audit is a coarse diagnostic of judge bias and nothing more.

Its labels are read back with exact coverage and enter sensitivity SA7 only
(ingest/holdout_v2_evaluation.py): they never replace a primary label, and there is no TEST
1.0.1. Nothing here writes a qrel.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ingest import holdout as ho
from ingest import holdout_v2 as v2

AUDIT_SEED = 20261029
AUDIT_SIZE = 15
DECOYS_PER_NEED = 3
GRADES = (0, 1, 2, 3)
INSTRUCTIONS = """# Blind relevance audit (TEST v2, sensitivity SA7)

For each need, read the question and grade every passage below it, on its own words only:
3 answers the question; 2 is needed support; 1 is related context; 0 is not relevant. Write
`unsure` where the text does not let you decide. Nothing tells you how a passage was found,
and the order of the passages means nothing. Fill `labels.json`: one entry per passage id,
each once.
"""


class AuditError(Exception):
    """The audit's input breaks its contract: nothing is read."""


def chain(bank: v2.Bank, salt: bytes, seed: int = AUDIT_SEED) -> list[str]:
    """Every candidate of the bank in an order drawn before any judgment: the audit's needs
    are the first AUDIT_SIZE of it that end up in the primary sample."""
    return sorted(
        (c.candidate_id for c in bank.candidates), key=lambda i: v2.draw(seed, salt, "audit", i)
    )


def selection(order: Sequence[str], primary_candidates: Sequence[str]) -> list[str]:
    sample = set(primary_candidates)
    return [c for c in order if c in sample][:AUDIT_SIZE]


@dataclass(frozen=True)
class Item:
    item_id: str
    need_id: str
    chunk_id: str
    decoy: bool


def items(
    need_id: str,
    pool: Mapping[str, Any],
    salt: bytes,
    seed: int = AUDIT_SEED,
) -> list[Item]:
    """One need's passages: every decisive chunk, and DECOYS_PER_NEED other pooled chunks
    with a node to judge, drawn; all under opaque ids, in an order drawn for the need."""
    decisive = [c for c in pool["decisive"] if c in pool["judged"]]
    others = sorted(
        (c for c in pool["chunks"] if c in pool["judged"] and c not in set(decisive)),
        key=lambda c: v2.draw(seed, salt, "decoy", need_id, c),
    )[:DECOYS_PER_NEED]
    chosen = [(c, False) for c in decisive] + [(c, True) for c in others]
    ordered = sorted(chosen, key=lambda x: v2.draw(seed, salt, "order", need_id, x[0]))
    return [
        Item(ho.opaque_id(salt, f"v2-human:{need_id}", c), need_id, c, decoy)
        for c, decoy in ordered
    ]


def packet(
    needs: Sequence[tuple[str, str, Sequence[Item]]],
    chunk_text: Mapping[str, str],
    salt: bytes,
) -> tuple[str, dict[str, Any]]:
    """The person's packet (markdown) and the template of the labels. needs: (need id, its
    statement, its items). A need is shown under an opaque reference, never its id."""
    parts = [INSTRUCTIONS]
    template: dict[str, Any] = {"labels": {}}
    for need_id, statement, its in needs:
        ref = ho.opaque_id(salt, "v2-human-need", need_id)
        parts.append(f"\n## Need {ref}\n\n{statement}\n")
        for item in its:
            parts.append(f"\n### Passage {item.item_id}\n\n{chunk_text[item.chunk_id].strip()}\n")
            template["labels"][item.item_id] = None
    return "".join(parts), template


def read_labels(text: str, key: Sequence[Item]) -> list[dict[str, Any]]:
    """The person's labels, every passage of the packet exactly once with a grade 0 to 3 or
    `unsure`: (need, chunk, grade or None, decoy) for SA7."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise AuditError(f"the labels are not JSON: {error}") from None
    labels = data.get("labels") if isinstance(data, Mapping) else None
    if not isinstance(labels, Mapping):
        raise AuditError("no labels object")
    by_id = {i.item_id: i for i in key}
    if set(labels) != set(by_id):
        raise AuditError(
            f"{len(set(by_id) - set(labels))} passages missing, "
            f"{len(set(labels) - set(by_id))} unknown"
        )
    out = []
    for item_id, value in sorted(labels.items()):
        if isinstance(value, bool) or value not in (*GRADES, "unsure"):
            raise AuditError(f"{item_id}: {value!r} is not 0, 1, 2, 3 or unsure")
        item = by_id[item_id]
        out.append(
            {
                "need_id": item.need_id,
                "chunk_id": item.chunk_id,
                "grade": None if value == "unsure" else int(value),
                "decoy": item.decoy,
            }
        )
    return out
