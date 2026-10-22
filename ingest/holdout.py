"""The internal TEST benchmark (S3, D-054): its coverage grid, the overlap checks
against DEV, the guard that no system has touched it, and its freeze.

TEST lives in data/benchmark/test/, outside the public repository (D-019). What is
published is its commitment, data/benchmark/test-commitment.json: hashes, counts and
distributions, and not one text. The commitment is pushed before S4 runs any system,
so the date of that commit proves that the exam was fixed first.

The grid is DEV's composition twice over, as a multiset of profiles (acts, type,
answerability, difficulty, and whether the answer is a figure or a date). Areas are
assigned by rule, not chosen: the protocol's areas of each act in order of how little
DEV covers them, cycled across the three batches. Nothing in this module reads a run,
a ranking or an embedding, except the guard, which only checks that none of them holds
a TEST query.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import cycle
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast, get_args

from pydantic import BaseModel, Field

from ingest import benchmark as bm

HOLDOUT_DIR = bm.BENCHMARK_DIR / "test"
COMMITMENT_PATH = bm.BENCHMARK_DIR / "test-commitment.json"
PROTOCOL_PATH = bm.BENCHMARK_DIR / "PROTOCOL.md"
PROTOCOL_VERSION = "v1.4"
PSD2_EXCLUSIONS = bm.BENCHMARK_DIR / "psd2_excluded_articles.json"
GRID_FILE = "grid.json"
FREEZE_FILE = "freeze.json"

GRID_SCHEMA: Literal["holdout-grid.v1"] = "holdout-grid.v1"
FREEZE_SCHEMA = "holdout-freeze.v1"
COMMITMENT_SCHEMA = "holdout-commitment.v1"
GRID_SEED = 20261019
BATCHES = 3
# TEST need ids start here, so that a TEST id can never be taken for a DEV one (DEV
# uses N004 to N035) and a stray TEST query is recognisable in any file (the guard).
FIRST_NEED_NUMBER = 101
TEST_ID = re.compile(r"\bN1\d\d(?:-(?:en|es)-\d+)?\b")
# The duplicate diagnostic of the DEV calibration: Jaccard of content words, flagged
# from 0.35 (data/benchmark/dev/CALIBRATION.md, batch 2, section 2).
JACCARD_FLAG = 0.35

GateVerdict = Literal[
    "independent", "paraphrase", "same_legal_question", "same_rule_superficial_change", "uncertain"
]


class HoldoutError(Exception):
    """A grid, an overlap record or a freeze that breaks the rules of S3."""


# ---------------------------------------------------------------------------
# areas of the acts (PROTOCOL.md, section 10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Area:
    act: bm.Act
    name: str
    first: int
    last: int

    def holds(self, path: str) -> bool:
        return self.first <= article_number(path) <= self.last


def article_number(path: str) -> int:
    """35 for "35", "35a" and "35a.2.sub_1": the number the area ranges use."""
    match = re.match(r"\d+", path)
    if match is None:
        raise HoldoutError(f"{path!r} does not start with an article number")
    return int(match.group(0))


_AREA_ROW = re.compile(r"^\| (DORA|MiCA|PSD2) \| ([^|]+?) \| (\d+)(?: to (\d+))? \|$", re.M)


def load_areas(protocol: Path = PROTOCOL_PATH) -> list[Area]:
    """The areas of section 10, in the protocol's order, read from the protocol itself so
    that the grid and the protocol cannot drift apart."""
    text = protocol.read_text(encoding="utf-8")
    section = re.search(r"^## 10\. .*?(?=^## \d+\. |\Z)", text, re.S | re.M)
    if section is None:
        raise HoldoutError(f"{protocol}: no section 10")
    areas = [
        Area(cast(bm.Act, act), name.strip(), int(first), int(last or first))
        for act, name, first, last in _AREA_ROW.findall(section.group(0))
    ]
    for act in bm.ACTS:
        ranges = sorted((a.first, a.last) for a in areas if a.act == act)
        if not ranges:
            raise HoldoutError(f"{protocol}: section 10 lists no area of {act}")
        if any(
            lo > hi or lo <= prev_hi
            for (_, prev_hi), (lo, hi) in zip(ranges, ranges[1:], strict=False)
        ):
            raise HoldoutError(f"{protocol}: the areas of {act} overlap")
    return areas


def area_of(areas: Sequence[Area], act: str, path: str) -> Area | None:
    """The area of a node, or None for an article after the table: the table of section 10
    stops before the final and transitional provisions (DORA 57 on, MiCA 139 on, PSD2
    104 on). DEV has one need there, N027, planned as "no row of the grid"."""
    found = [a for a in areas if a.act == act and a.holds(path)]
    if len(found) > 1:
        raise HoldoutError(f"{act}:{path} falls in {len(found)} areas of section 10")
    return found[0] if found else None


def need_areas(qrels: Sequence[bm.Qrel], areas: Sequence[Area]) -> set[Area | None]:
    """Where the answer of a need lives: the areas of its grade-3 nodes, or of its grade-2
    nodes when it has no grade 3 (a partial need). Section 10 reads the area again from
    the evidence after the search; this is that reading, done by code."""
    judged = [r for r in qrels if not r.ambiguous]
    top = [r for r in judged if r.grade == 3] or [r for r in judged if r.grade == 2]
    return {area_of(areas, r.act, r.path) for r in top}


def dev_coverage(dev: bm.Benchmark, areas: Sequence[Area]) -> Counter[Area]:
    """How many DEV needs have their answer in each area."""
    covered: Counter[Area] = Counter({a: 0 for a in areas})
    for need in dev.needs:
        covered.update(a for a in need_areas(dev.qrels_of(need.need_id), areas) if a is not None)
    return covered


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------


class Profile(NamedTuple):
    """What the grid takes from a DEV need: everything but its topic."""

    acts: tuple[bm.Act, ...]
    type: bm.NeedType
    answerability: bm.Answerability
    difficulty: bm.Difficulty
    number_or_date: bool

    def order(self) -> tuple[Any, ...]:
        return (
            tuple(bm.ACTS.index(a) for a in self.acts),
            get_args(bm.NeedType).index(self.type),
            get_args(bm.Answerability).index(self.answerability),
            get_args(bm.Difficulty).index(self.difficulty),
            self.number_or_date,
        )


def profile_of(need: bm.Need) -> Profile:
    return Profile(
        acts=tuple(sorted(need.acts, key=bm.ACTS.index)),
        type=need.type,
        answerability=need.answerability,
        difficulty=need.difficulty,
        number_or_date=bool({"numeric", "date"} & set(need.drivers)),
    )


class Cell(BaseModel):
    """One planned need of TEST, fixed before anyone writes a word of it."""

    model_config = bm.STRICT

    cell_id: str = Field(pattern=r"^G\d{2}$")
    # how the grid was dealt, frozen with it; the construction follows the frozen order
    # (cell_order), whose blocks of 20 are operational checkpoints and nothing else
    batch: int = Field(ge=1, le=BATCHES)
    acts: list[bm.Act] = Field(min_length=1)
    # None for a cell with two acts or with answerability "none": its question is not
    # tied to one area (DEV's grids left those open too).
    area: str | None
    type: bm.NeedType
    answerability: bm.Answerability
    difficulty: bm.Difficulty
    number_or_date: bool

    def profile(self) -> Profile:
        return Profile(
            tuple(self.acts), self.type, self.answerability, self.difficulty, self.number_or_date
        )


class Grid(BaseModel):
    model_config = bm.STRICT

    schema_version: Literal["holdout-grid.v1"]
    seed: int
    protocol_sha256: bm.Sha256
    dev_benchmark_version: str
    dev_needs_sha256: bm.Sha256
    dev_qrels_sha256: bm.Sha256
    # per act, the areas in the order they were dealt: least covered by DEV first
    area_order: dict[str, list[str]]
    cells: list[Cell]


def _needs_area(profile: Profile) -> bool:
    return len(profile.acts) == 1 and profile.answerability != "none"


def imbalance(batches: Sequence[Sequence[Profile]]) -> float:
    """Sum over batches, attributes and values of the squared distance between the count
    in the batch and a third of the count in TEST: 0 means every batch is TEST in small."""
    everything = [p for b in batches for p in b]
    total = 0.0
    for k in range(len(Profile._fields)):
        whole = Counter(p[k] for p in everything)
        for batch in batches:
            here = Counter(p[k] for p in batch)
            total += sum((here[v] - n / len(batches)) ** 2 for v, n in whole.items())
    return total


def deal_batches(profiles: Sequence[Profile]) -> list[list[Profile]]:
    """Each profile, in order, goes to the batch where the squared distance of every
    attribute value from a third of its total grows least; batches stay equal in size and
    ties go to the lowest batch. Then swaps between batches, tried in a fixed order, are
    kept while they lower the imbalance. Deterministic: no seed, nothing to choose."""
    size = len(profiles) // BATCHES
    totals = [Counter(p[k] for p in profiles) for k in range(len(Profile._fields))]
    counts = [[Counter[Any]() for _ in Profile._fields] for _ in range(BATCHES)]
    batches: list[list[Profile]] = [[] for _ in range(BATCHES)]

    def cost(b: int, p: Profile) -> float:
        return sum(
            (counts[b][k][p[k]] + 1 - totals[k][p[k]] / BATCHES) ** 2
            - (counts[b][k][p[k]] - totals[k][p[k]] / BATCHES) ** 2
            for k in range(len(Profile._fields))
        )

    for p in profiles:
        best = min((b for b in range(BATCHES) if len(batches[b]) < size), key=lambda b: cost(b, p))
        batches[best].append(p)
        for k in range(len(Profile._fields)):
            counts[best][k][p[k]] += 1
    improved = True
    while improved:
        improved = False
        for a in range(BATCHES):
            for b in range(a + 1, BATCHES):
                for i in range(len(batches[a])):
                    for j in range(len(batches[b])):
                        if batches[a][i] == batches[b][j]:
                            continue
                        before = imbalance(batches)
                        batches[a][i], batches[b][j] = batches[b][j], batches[a][i]
                        if imbalance(batches) < before - 1e-9:
                            improved = True
                        else:
                            batches[a][i], batches[b][j] = batches[b][j], batches[a][i]
    return [sorted(b, key=Profile.order) for b in batches]


def build_grid(
    dev: bm.Benchmark, areas: Sequence[Area], protocol: Path = PROTOCOL_PATH, seed: int = GRID_SEED
) -> Grid:
    """DEV's profiles twice over, dealt into three batches and given areas by rule.

    1. Sort the 2 x len(DEV) profiles (act, type, answerability, difficulty, figure) and
       place each in the batch that keeps every attribute closest to a third of its
       total (deal_batches): each batch is a small TEST with DEV's mix.
    2. For each act, order its areas by DEV coverage, least covered first (section 10:
       prefer what the benchmark has least of), ties in the protocol's order.
    3. Cycle through that order across the batches. Within a batch the cells of an act
       are shuffled with a fixed seed before they take the next areas, so no type is
       tied to the least covered areas. A batch never repeats an area of an act while
       another is unused, and over TEST the areas of an act differ by at most one cell.
    """
    profiles = sorted((profile_of(n) for n in dev.needs for _ in range(2)), key=Profile.order)
    batches = deal_batches(profiles)
    covered = dev_coverage(dev, areas)
    order: dict[str, list[str]] = {
        act: [
            a.name
            for a in sorted(
                (a for a in areas if a.act == act), key=lambda a: (covered[a], areas.index(a))
            )
        ]
        for act in bm.ACTS
    }
    dealer = {act: cycle(order[act]) for act in bm.ACTS}
    cells: list[Cell] = []
    for number, batch in enumerate(batches, start=1):
        assigned: dict[int, str] = {}
        for act in bm.ACTS:
            slots = [i for i, p in enumerate(batch) if _needs_area(p) and p.acts == (act,)]
            random.Random(f"{seed}:{act}:{number}").shuffle(slots)
            for i in slots:
                assigned[i] = next(dealer[act])
        for i, p in enumerate(batch):
            cells.append(
                Cell(
                    cell_id=f"G{len(cells) + 1:02d}",
                    batch=number,
                    acts=list(p.acts),
                    area=assigned.get(i),
                    type=p.type,
                    answerability=p.answerability,
                    difficulty=p.difficulty,
                    number_or_date=p.number_or_date,
                )
            )
    return Grid(
        schema_version=GRID_SCHEMA,
        seed=seed,
        protocol_sha256=bm.sha256_file(protocol),
        dev_benchmark_version=dev.manifest.benchmark_version,
        dev_needs_sha256=dev.manifest.files["needs.jsonl"],
        dev_qrels_sha256=dev.manifest.files["qrels.jsonl"],
        area_order=order,
        cells=cells,
    )


def grid_problems(grid: Grid, dev: bm.Benchmark, areas: Sequence[Area]) -> list[str]:
    """Every rule of the grid, checked again on the file, whoever wrote it."""
    problems: list[str] = []
    cells = grid.cells
    if len({c.cell_id for c in cells}) != len(cells):
        problems.append("a cell id appears twice")
    expected = Counter(profile_of(n) for n in dev.needs for _ in range(2))
    if Counter(c.profile() for c in cells) != expected:
        problems.append("the profiles are not DEV's twice over")
    size = len(cells) // BATCHES
    sizes = Counter(c.batch for c in cells)
    if len(cells) % BATCHES or any(sizes[b] != size for b in range(1, BATCHES + 1)):
        problems.append(f"batches {dict(sorted(sizes.items()))}, not {BATCHES} of {size}")
    names = {(a.act, a.name) for a in areas}
    for c in cells:
        if _needs_area(c.profile()) != (c.area is not None):
            problems.append(f"{c.cell_id}: an area exactly when the cell has one act and an answer")
        if c.area is not None and (c.acts[0], c.area) not in names:
            problems.append(f"{c.cell_id}: {c.area!r} is not an area of {c.acts[0]}")
    for act in bm.ACTS:
        n_areas = sum(a.act == act for a in areas)
        per_area = Counter(c.area for c in cells if c.area and c.acts == [act])
        counts = [per_area[a.name] for a in areas if a.act == act]
        if max(counts) - min(counts) > 1:
            problems.append(f"{act}: areas used {min(counts)} to {max(counts)} times")
        for b in range(1, BATCHES + 1):
            used = [c.area for c in cells if c.batch == b and c.area and c.acts == [act]]
            if len(used) != len(set(used)) and len(set(used)) < n_areas:
                problems.append(f"batch {b}: an area of {act} repeats while another is unused")
    for b in range(1, BATCHES + 1):
        figures = sum(c.number_or_date for c in cells if c.batch == b)
        if figures > size / 2:
            problems.append(f"batch {b}: {figures} cells expect a figure or a date, over half")
    return problems


def load_grid(directory: Path = HOLDOUT_DIR) -> Grid:
    return Grid.model_validate_json((directory / GRID_FILE).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# a structural diagnostic: how often each area states a figure or a date
# ---------------------------------------------------------------------------

_NUMBER_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|thirty"
_UNITS = r"(?:working |business |calendar )?(?:days?|weeks?|months?|years?|hours?)"
FIGURE_OR_DATE = re.compile(
    rf"\b\d+(?:[.,]\d+)*\s*(?:%|per ?cent|EUR|euros?|{_UNITS}|times)(?![\w-])"
    r"|\bEUR ?\d"
    rf"|\b(?:{_NUMBER_WORDS}) {_UNITS}\b"
    r"|\b\d{1,2} (?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December) \d{4}\b",
    re.I,
)


def figure_density(
    areas: Sequence[Area], texts: Mapping[bm.NodeKey, str], excluded_psd2: frozenset[str]
) -> dict[str, dict[str, float]]:
    """For each area, the share of its nodes whose own text states an amount, a
    percentage, a period, a frequency or a calendar date, by a fixed pattern. A diagnostic
    recorded before authoring, to read the figure cells that were not realized; it never
    changes a cell and uses no retrieval."""
    out: dict[str, dict[str, float]] = {}
    for area in areas:
        nodes = [
            text
            for (act, path), text in texts.items()
            if act == area.act
            and text
            and area.holds(path)
            and not (act == "PSD2" and path.split(".")[0] in excluded_psd2)
        ]
        hits = sum(bool(FIGURE_OR_DATE.search(t)) for t in nodes)
        out[f"{area.act}: {area.name}"] = {
            "nodes": len(nodes),
            "with_figure_or_date": hits,
            "share": round(hits / len(nodes), 3) if nodes else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# overlap with DEV: statements before the lock, evidence after the qrels
# ---------------------------------------------------------------------------


def content_words(text: str) -> frozenset[str]:
    """The words the DEV calibration compared statements on."""
    return frozenset(
        w for w in re.findall(r"[a-z]+", text.lower()) if w not in bm._STOPWORDS and len(w) > 2
    )


def jaccard(a: str, b: str) -> float:
    wa, wb = content_words(a), content_words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


class Closest(NamedTuple):
    other: str
    jaccard: float


def closest_statements(
    statements: Mapping[str, str], reference: Mapping[str, str], top: int = 3
) -> dict[str, list[Closest]]:
    """For each statement, the most similar ones of the reference (DEV and the fixtures,
    or the rest of TEST). A diagnostic for the record: the gatekeeper decides, not this."""
    out: dict[str, list[Closest]] = {}
    for key, text in statements.items():
        scored = sorted(
            (
                Closest(other, jaccard(text, ref))
                for other, ref in reference.items()
                if other != key
            ),
            key=lambda c: (-c.jaccard, c.other),
        )
        out[key] = scored[:top]
    return out


def grade3_nodes(qrels: Iterable[bm.Qrel]) -> dict[str, set[bm.NodeKey]]:
    out: dict[str, set[bm.NodeKey]] = {}
    for r in qrels:
        if r.grade == 3 and not r.ambiguous:
            out.setdefault(r.need_id, set()).add(r.node)
    return out


class EvidenceOverlap(NamedTuple):
    test_need: str
    dev_need: str
    shared_nodes: tuple[str, ...]
    shared_articles: tuple[str, ...]


def evidence_overlap(
    test_qrels: Iterable[bm.Qrel], dev_qrels: Iterable[bm.Qrel]
) -> list[EvidenceOverlap]:
    """Pairs of a TEST need and a DEV need that share a grade-3 article. A shared node
    sends the pair to the gatekeeper again, with the evidence (the protocol's duplicate:
    the same intent and the same grade-3 evidence). A shared article alone is legitimate,
    since one article can hold several rules: it only marks the S4 stratum."""
    test, dev = grade3_nodes(test_qrels), grade3_nodes(dev_qrels)
    pairs = []
    for t, t_nodes in sorted(test.items()):
        t_articles = {(act, path.split(".")[0]) for act, path in t_nodes}
        for d, d_nodes in sorted(dev.items()):
            articles = t_articles & {(act, path.split(".")[0]) for act, path in d_nodes}
            if articles:
                pairs.append(
                    EvidenceOverlap(
                        t,
                        d,
                        tuple(sorted(f"{a}:{p}" for a, p in t_nodes & d_nodes)),
                        tuple(sorted(f"{a}:{p}" for a, p in articles)),
                    )
                )
    return pairs


def evidence_strata(test_needs: Iterable[str], pairs: Iterable[EvidenceOverlap]) -> dict[str, str]:
    """The pre-registered S4 stratum of every TEST need."""
    shared = {p.test_need for p in pairs}
    return {
        n: "shared_grade3_article" if n in shared else "no_shared_grade3"
        for n in sorted(test_needs)
    }


# ---------------------------------------------------------------------------
# isolation: no system has seen a TEST query
# ---------------------------------------------------------------------------

SYSTEM_DIRS = ("data/runs", "data/embeddings", "data/results")
_TEST_NAME = re.compile(r"(?:^|[-_.])test(?:[-_.]|$)", re.I)
_TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".csv", ".tsv", ".txt"}


def isolation_problems(root: Path = Path("."), test_ids: Iterable[str] = ()) -> list[str]:
    """Every run, embedding and result on disk is checked for a sign of TEST: a path
    named after it, a TEST need or query id (the N1xx range, or any id in test_ids),
    in any text file. The run, embedding and result files of this project are text
    (JSON Lines, JSON, Markdown); vectors sit next to a meta.json that lists their ids."""
    ids = set(test_ids)
    problems: list[str] = []
    for base in SYSTEM_DIRS:
        top = root / base
        if not top.exists():
            continue
        for path in sorted(top.rglob("*")):
            rel = path.relative_to(root)
            if any(_TEST_NAME.search(part) for part in rel.parts):
                problems.append(f"{rel}: named after TEST")
            if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
                continue
            with path.open(encoding="utf-8", errors="replace") as handle:
                for number, line in enumerate(handle, start=1):
                    found = set(TEST_ID.findall(line)) | {i for i in ids if i in line}
                    if found:
                        problems.append(f"{rel}:{number}: holds TEST ids {sorted(found)[:3]}")
                        break
    return problems


# ---------------------------------------------------------------------------
# the freeze and the public commitment
# ---------------------------------------------------------------------------


def tree_sha256(directory: Path, exclude: tuple[str, ...] = (".git",)) -> str:
    """One hash for a directory: the SHA-256 of its sorted 'path sha' lines, leaving out
    every file with a part that matches a pattern of exclude (by default the git history,
    which has its own hashes)."""

    def kept(p: Path) -> bool:
        parts = p.relative_to(directory).parts
        return not any(fnmatch.fnmatchcase(part, x) for part in parts for x in exclude)

    lines = [
        f"{p.relative_to(directory).as_posix()} {bm.sha256_file(p)}"
        for p in sorted(directory.rglob("*"))
        if p.is_file() and kept(p)
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": median,
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 3),
    }


def _words(text: str) -> int:
    return len(bm._words(text))


def lexical_profile(benchmark: bm.Benchmark, docs: Mapping[str, Any]) -> dict[str, Any]:
    """The basic lexical features that S4 compares between DEV and TEST, fixed before TEST
    exists: statement and query length in words, and the overlap of each query with the
    text of its evidence (bm.lexical_overlap: share of content words, longest verbatim
    run, POSSIBLE COPY). Evaluable needs only, since a none need has no evidence."""
    evaluable = {n.need_id for n in benchmark.needs if n.answerability != "none"}
    by_variant: dict[str, list[bm.Overlap]] = {}
    lengths: dict[str, list[float]] = {}
    for q in benchmark.queries:
        variant = f"{q.lang}:{q.style}"
        lengths.setdefault(variant, []).append(_words(q.text))
        if q.need_id in evaluable:
            evidence = bm.evidence_texts(benchmark, q.need_id, docs)
            by_variant.setdefault(variant, []).append(bm.lexical_overlap(q.text, evidence))
    return {
        "statement_words": _summary([_words(n.statement) for n in benchmark.needs]),
        "query_words": {v: _summary(x) for v, x in sorted(lengths.items())},
        "query_evidence_share": {
            v: _summary([round(o.share, 3) for o in x]) for v, x in sorted(by_variant.items())
        },
        "query_evidence_longest_run": {
            v: _summary([o.longest_run for o in x]) for v, x in sorted(by_variant.items())
        },
        "possible_copies": sum(o.possible_copy for x in by_variant.values() for o in x),
    }


def distributions(benchmark: bm.Benchmark, areas: Sequence[Area]) -> dict[str, Any]:
    """Counts only, no text: what the commitment publishes and what S4 compares with DEV."""
    needs = benchmark.needs
    evaluable = [n for n in needs if n.answerability != "none"]
    relevant = {
        n.need_id: [
            r
            for r in benchmark.qrels_of(n.need_id)
            if r.grade >= bm.RELEVANT_GRADE and not r.ambiguous
        ]
        for n in needs
    }
    area_counts: Counter[str] = Counter()
    for n in needs:
        area_counts.update(
            f"{a.act}: {a.name}" if a else "after the area table"
            for a in need_areas(benchmark.qrels_of(n.need_id), areas)
        )
    return {
        "needs": len(needs),
        "evaluable_needs": len(evaluable),
        "queries": len(benchmark.queries),
        "qrels": len(benchmark.qrels),
        "type": dict(sorted(Counter(n.type for n in needs).items())),
        "answerability": dict(sorted(Counter(n.answerability for n in needs).items())),
        "acts": dict(sorted(Counter("+".join(n.acts) for n in needs).items())),
        "difficulty": dict(sorted(Counter(n.difficulty for n in needs).items())),
        "drivers": dict(sorted(Counter(d for n in needs for d in n.drivers).items())),
        "areas_of_the_answer": dict(sorted(area_counts.items())),
        "queries_by_language_and_style": dict(
            sorted(Counter(f"{q.lang}:{q.style}" for q in benchmark.queries).items())
        ),
        "queries_by_origin": dict(sorted(Counter(q.origin for q in benchmark.queries).items())),
        "grades": {str(g): c for g, c in sorted(Counter(r.grade for r in benchmark.qrels).items())},
        "ambiguous_qrels": sum(r.ambiguous for r in benchmark.qrels),
        "qrels_per_need": _summary([len(benchmark.qrels_of(n.need_id)) for n in needs]),
        "relevant_nodes_per_evaluable_need": _summary(
            [len(relevant[n.need_id]) for n in evaluable]
        ),
        "relevant_articles_per_evaluable_need": _summary(
            [len({(r.act, r.path.split(".")[0]) for r in relevant[n.need_id]}) for n in evaluable]
        ),
    }


FREEZE_COUNTS = (
    "overlap",
    "ledger",
    "plan_against_realized",
    "diagnostic",
    "omission_audit",
    "rejections",
    "technical",
)
FREEZE_VERSION = "1.0.0"


def freeze_record(
    directory: Path,
    areas: Sequence[Area],
    plan: Path,
    counts: Mapping[str, Mapping[str, Any]],
    isolation: Sequence[str],
    templates: Path,
    pre_authoring: Mapping[str, Any],
    state: Mapping[str, Any],
    frozen_at: str,
) -> dict[str, Any]:
    """Everything that fixes TEST, in one record. pre_authoring names the contract and the
    public commit that dated it; state, the durable state's last event when it was frozen;
    frozen_at, the time of the frozen event, so that the record can be written again equal.
    It refuses to exist if the benchmark is not sealed with role test at 1.0.0 or later, if
    any need is not ready, if the guard found any sign that a system touched TEST, or if a
    count it must carry is missing: the overlap verdicts, the attempt ledger, the plan
    against the realized set, the diagnostic judge with judge D, the omission audit, the
    rejections and the technical executions."""
    benchmark = bm.load_benchmark(directory)
    manifest = benchmark.manifest
    problems = []
    if manifest.role != "test":
        problems.append(f"role {manifest.role!r}, not 'test'")
    if bm._version_tuple(manifest.benchmark_version) < (1, 0, 0):
        problems.append(f"version {manifest.benchmark_version} is a build seal, not a freeze")
    problems += [f"need {n.need_id}: {n.status}" for n in benchmark.needs if n.status != "ready"]
    problems += [f"isolation: {p}" for p in isolation]
    problems += [f"no {k} counts" for k in FREEZE_COUNTS if k not in counts]
    if problems:
        raise HoldoutError("cannot freeze TEST:\n  " + "\n  ".join(problems))
    # the batches of 20 are operational checkpoints, never a calibration (the person's
    # decision of 2026-10-19): there is no per-batch record to require
    required = ("provenance", "overlap")
    missing = [n for n in required if not (directory / n).exists()]
    if missing:
        raise HoldoutError(f"cannot freeze TEST: {directory} has no {', '.join(missing)}")
    grid = load_grid(directory)
    files = {
        name: bm.sha256_file(directory / name)
        for name in (
            bm.MANIFEST_FILE,
            *bm.DATA_FILES,
            bm.ADJUDICATIONS_FILE,
            GRID_FILE,
        )
    }
    trees = {name: tree_sha256(directory / name) for name in ("provenance", "overlap")}
    return {
        "schema_version": FREEZE_SCHEMA,
        "benchmark_version": manifest.benchmark_version,
        "frozen_at": frozen_at,
        "pre_authoring": dict(pre_authoring),
        "state": dict(state),
        "protocol": {"version": PROTOCOL_VERSION, "sha256": bm.sha256_file(PROTOCOL_PATH)},
        "corpus": {
            "parser_version": manifest.parser_version,
            "acts": {act: src.model_dump() for act, src in sorted(manifest.corpus.items())},
            "psd2_exclusions_sha256": bm.sha256_file(PSD2_EXCLUSIONS),
        },
        "query_templates": dict(sorted(manifest.templates.items())),
        "holdout_templates_sha256": tree_sha256(templates),
        "analysis_plan": {"path": plan.as_posix(), "sha256": bm.sha256_file(plan)},
        "grid": {"seed": grid.seed, "cells": len(grid.cells)},
        "files_sha256": files,
        "trees_sha256": trees,
        "distributions": distributions(benchmark, areas),
        **{k: dict(counts[k]) for k in FREEZE_COUNTS},
        "isolation": {"checked": list(SYSTEM_DIRS), "problems": list(isolation)},
    }


# What the public commitment carries: hashes, counts and distributions. Never a
# statement, a query, a node, a rationale or a need id.
COMMITTED_KEYS = (
    "benchmark_version",
    "frozen_at",
    "protocol",
    "corpus",
    "query_templates",
    "holdout_templates_sha256",
    "analysis_plan",
    "pre_authoring",
    "state",
    "grid",
    "files_sha256",
    "trees_sha256",
    "distributions",
    *FREEZE_COUNTS,
    "isolation",
)


def commitment(freeze: Mapping[str, Any], freeze_sha256: str) -> dict[str, Any]:
    """The public record. freeze_sha256 is the hash of the private freeze.json as written,
    so the commitment also fixes every detail it does not publish."""
    record: dict[str, Any] = {"schema_version": COMMITMENT_SCHEMA, "freeze_sha256": freeze_sha256}
    record |= {k: freeze[k] for k in COMMITTED_KEYS}
    text = json.dumps(record, ensure_ascii=False)
    leaked = TEST_ID.findall(text)
    if leaked:
        raise HoldoutError(f"the commitment would publish TEST ids: {sorted(set(leaked))[:5]}")
    return record


# ---------------------------------------------------------------------------
# the backup of the private history
# ---------------------------------------------------------------------------

BACKUP_DIR = Path.home() / ".fundamento-private" / "holdout-backups"
# The construction's state (ingest/holdout_state.py) keeps beside its database a lock and,
# for an instant, temporary files and SQLite's journal: never recorded, never compared.
STATE_LOCK_FILE = "holdout-state.lock"
STATE_UNTRACKED = (STATE_LOCK_FILE, "*.sqlite3-journal", ".tmp-*")
PRIVATE_AUTHOR = ("Oscar Curto", "195488773+oscarcurtog@users.noreply.github.com")
NO_PUSH_HOOK = "#!/bin/sh\necho 'this history is private: it is never pushed' >&2\nexit 1\n"


def git(directory: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(directory), *args], check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def guard_private_history(directory: Path) -> None:
    """The private history's author and the hook that refuses every push: set at init, and
    again on a copy restored from a bundle, which carries neither."""
    git(directory, "config", "user.name", PRIVATE_AUTHOR[0])
    git(directory, "config", "user.email", PRIVATE_AUTHOR[1])
    hook = directory / ".git" / "hooks" / "pre-push"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(NO_PUSH_HOOK, encoding="utf-8")
    hook.chmod(0o755)


def private_folder(folder: Path, repo: Path) -> Path:
    """A folder outside the repository, readable by this user only: the backup folder and
    only the parents this call creates are restricted."""
    folder = folder.resolve()
    if folder.is_relative_to(repo.resolve()):
        raise HoldoutError(f"{folder} is inside the repository: a backup lives outside")
    created = [p for p in (folder, *folder.parents) if not p.exists()]
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    for private in {folder, *created}:
        private.chmod(0o700)
    return folder


def backup(
    directory: Path,
    repo: Path,
    folder: Path = BACKUP_DIR,
    check: Callable[[Path], Any] | None = None,
) -> Path:
    """A git bundle of the whole private history, outside the repository, verified and
    then restored into a temporary clone that must match the private directory file by
    file; check, if given, also runs on the clone (the construction's state). The history
    must be clean first: what is not recorded is not backed up."""
    if folder.resolve().is_relative_to(repo.resolve()):
        raise HoldoutError(f"{folder} is inside the repository: a backup lives outside")
    if git(directory, "status", "--porcelain"):
        raise HoldoutError(f"{directory} has changes not recorded: record them first")
    folder = private_folder(folder, repo)
    head = git(directory, "rev-parse", "HEAD")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bundle = folder / f"test-{stamp}-{head[:12]}.bundle"
    if bundle.exists():
        raise HoldoutError(f"{bundle} exists: a bundle is never overwritten")
    git(directory, "bundle", "create", str(bundle), "--all")
    git(directory, "bundle", "verify", str(bundle))
    restored = restore_check(directory, bundle, check)
    entry = {
        "bundle": bundle.name,
        "created": stamp,
        "head": head,
        "sha256": bm.sha256_file(bundle),
        "verified": True,
        "restored_tree_sha256": restored,
        "state_checked": check is not None,
    }
    with (folder / "backups.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry) + "\n")
    return bundle


def restore_check(directory: Path, bundle: Path, check: Callable[[Path], Any] | None = None) -> str:
    """Clone the bundle into a new temporary directory and compare it with the private
    directory: the same HEAD and the same recorded files; then check, if given, on the
    clone. The clone is removed by its literal path."""
    scratch = Path(tempfile.mkdtemp(prefix="restore."))
    clone = scratch / "test"
    try:
        subprocess.run(
            ["git", "clone", "-q", str(bundle), str(clone)], check=True, capture_output=True
        )
        if git(clone, "rev-parse", "HEAD") != git(directory, "rev-parse", "HEAD"):
            raise HoldoutError(f"{bundle}: the restored HEAD differs")
        untracked = (".git", *STATE_UNTRACKED)
        restored, original = tree_sha256(clone, untracked), tree_sha256(directory, untracked)
        if restored != original:
            raise HoldoutError(f"{bundle}: the restored files differ from {directory}")
        if check is not None:
            check(clone)
        return restored
    finally:
        if scratch.parent.resolve() != Path(tempfile.gettempdir()).resolve():
            raise HoldoutError(f"refusing to remove {scratch}: not in the temporary folder")
        shutil.rmtree(scratch)


# ---------------------------------------------------------------------------
# fixed before authoring: the order of the cells, the omission audit, the opaque ids
# (Codex Gate 2, P1.2 and P1.4)
# ---------------------------------------------------------------------------

ORDER_SEED = 20261020
ORDER_ALGORITHM = (
    "holdout-order.v1: the cells sorted by the hex SHA-256 of '<seed>:<cell_id>', ascending; "
    "no other input"
)
OMISSION_SEED = 20261021
OMISSION_SIZE = 12
OMISSION_ALGORITHM = (
    "holdout-omission.v1: strata (planned type, planned acts); each gets round(12 x its share) "
    "by largest remainder, ties by the hex SHA-256 of '<seed>:<stratum>'; within a stratum the "
    "cells with the lowest hex SHA-256 of '<seed>:<cell_id>'"
)


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def cell_order(grid: Grid, seed: int = ORDER_SEED) -> list[str]:
    """The order in which TEST's cells are built, fixed before any concept is written. It
    depends only on the seed and the cell ids: never on a rejection, a result, the
    difficulty, the evidence or a system. It decides which of two needs that turn out to be
    duplicates comes first."""
    return sorted((c.cell_id for c in grid.cells), key=lambda cid: _rank(seed, cid))


def omission_cells(grid: Grid, seed: int = OMISSION_SEED, size: int = OMISSION_SIZE) -> list[str]:
    """The cells whose needs get a second, blind evidence search, chosen before authoring
    and stratified by planned type and acts: no content of a future need is used."""
    strata: dict[str, list[str]] = {}
    for c in grid.cells:
        strata.setdefault(f"{c.type}|{'+'.join(c.acts)}", []).append(c.cell_id)
    total = len(grid.cells)
    exact = {s: size * len(cells) / total for s, cells in strata.items()}
    quota = {s: int(v) for s, v in exact.items()}
    left = size - sum(quota.values())
    by_remainder = sorted(strata, key=lambda s: (-(exact[s] - quota[s]), _rank(seed, s)))
    for s in by_remainder[:left]:
        quota[s] += 1
    chosen = [
        cid
        for s, cells in sorted(strata.items())
        for cid in sorted(cells, key=lambda cid: _rank(seed, cid))[: quota[s]]
    ]
    return sorted(chosen, key=lambda cid: _rank(ORDER_SEED, cid))


def plan_record(grid: Grid) -> dict[str, Any]:
    """Order and omission audit, with their algorithms and hashes, for the private history and
    the contract: frozen before any concept is written."""
    order = cell_order(grid)
    omission = omission_cells(grid)
    return {
        "order": {
            "algorithm": ORDER_ALGORITHM,
            "seed": ORDER_SEED,
            "cells": order,
            "sha256": hashlib.sha256("\n".join(order).encode()).hexdigest(),
        },
        "omission_audit": {
            "algorithm": OMISSION_ALGORITHM,
            "seed": OMISSION_SEED,
            "cells": omission,
            "sha256": hashlib.sha256("\n".join(omission).encode()).hexdigest(),
        },
    }


def item_id(salt: bytes, scope: str, target: str) -> str:
    """The opaque id of one thing a judge grades, a candidate node or the answerability, under
    one stage of one attempt (scope). The orchestrator owns the mapping from id to node and is
    the only one that ever resolves it: the judge receives the id with the node's text and
    answers by it, and never writes the node's identity itself."""
    import hmac

    return "C-" + hmac.new(salt, f"item:{scope}:{target}".encode(), hashlib.sha256).hexdigest()[:12]


def opaque_id(salt: bytes, kind: str, key: str) -> str:
    """An identifier a context can see without learning anything from it: not the cell, the
    attempt, the order or whether it is a retry. HMAC-SHA-256 under a private salt that never
    leaves the TEST directory; the ledger maps it back."""
    import hmac

    return "R-" + hmac.new(salt, f"{kind}:{key}".encode(), hashlib.sha256).hexdigest()[:12]
