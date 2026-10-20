"""The TEST coverage grid built from the real DEV files (S3, D-054). DEV is versioned and
needs no corpus to load, so these run everywhere, CI included."""

from __future__ import annotations

import hashlib
import re
from collections import Counter

import pytest

from ingest import benchmark as bm
from ingest import holdout as ho


@pytest.fixture(scope="module")
def dev() -> bm.Benchmark:
    return bm.load_benchmark(bm.DEV_DIR)


@pytest.fixture(scope="module")
def areas() -> list[ho.Area]:
    return ho.load_areas()


@pytest.fixture(scope="module")
def grid(dev, areas) -> ho.Grid:
    return ho.build_grid(dev, areas)


def test_the_grid_keeps_its_rules(grid, dev, areas):
    assert ho.grid_problems(grid, dev, areas) == []


def test_the_grid_is_dev_twice_over(grid, dev):
    assert len(grid.cells) == 2 * len(dev.needs) == 60
    assert Counter(c.type for c in grid.cells) == {
        "direct": 22,
        "exception_confusion": 14,
        "multi_provision": 12,
        "partial_out_of_scope": 12,
    }
    assert Counter("+".join(c.acts) for c in grid.cells) == {
        "MiCA": 22,
        "DORA": 18,
        "PSD2": 16,
        "DORA+PSD2": 4,
    }
    assert Counter(c.answerability for c in grid.cells) == {"complete": 48, "partial": 6, "none": 6}
    assert Counter(c.difficulty for c in grid.cells) == {"easy": 14, "medium": 30, "hard": 16}


def test_every_batch_is_test_in_small(grid):
    for b in range(1, ho.BATCHES + 1):
        cells = [c for c in grid.cells if c.batch == b]
        types = Counter(c.type for c in cells)
        assert len(cells) == 20
        assert types["multi_provision"] == 4 and types["partial_out_of_scope"] == 4
        assert sum(c.number_or_date for c in cells) == 8
        assert 4 <= Counter(c.difficulty for c in cells)["hard"] <= 6


def test_the_grid_is_deterministic(grid, dev, areas):
    assert ho.build_grid(dev, areas) == grid


def test_the_least_covered_areas_come_first(grid):
    assert grid.area_order["MiCA"][0] == "Scope and definitions"
    assert grid.area_order["PSD2"][:2] == [
        "Scope and definitions",
        "Data protection, operational and security risks, authentication",
    ]


def test_a_tampered_grid_is_refused(grid, dev, areas):
    cells = [c.model_copy() for c in grid.cells]
    first, second = [c for c in cells if c.batch == 1 and c.acts == ["DORA"] and c.area][:2]
    second.area = first.area
    problems = ho.grid_problems(grid.model_copy(update={"cells": cells}), dev, areas)
    assert any("repeats while another is unused" in p for p in problems)
    cells[0].difficulty = "hard" if cells[0].difficulty != "hard" else "easy"
    problems = ho.grid_problems(grid.model_copy(update={"cells": cells}), dev, areas)
    assert "the profiles are not DEV's twice over" in problems


def test_the_dev_calibration_jaccard_is_reproduced(dev):
    fixtures = bm.load_benchmark(bm.FIXTURES_DIR)
    statements = {n.need_id: n.statement for n in dev.needs + fixtures.needs}
    # CALIBRATION.md, batch 2, section 2: "N034 with the test case N003 (Jaccard of
    # content words 0.30)"
    assert round(ho.jaccard(statements["N034"], statements["N003"]), 2) == 0.30


def test_the_cell_order_is_frozen_reproducible_and_not_the_grids(grid):
    order = ho.cell_order(grid)
    assert order == ho.cell_order(grid) and sorted(order) == sorted(c.cell_id for c in grid.cells)
    assert order != [c.cell_id for c in grid.cells]
    assert ho.cell_order(grid, seed=1) != order
    record = ho.plan_record(grid)
    assert record["order"]["sha256"] == hashlib.sha256("\n".join(order).encode()).hexdigest()


def test_the_omission_audit_is_12_cells_stratified_before_authoring(grid):
    cells = {c.cell_id: c for c in grid.cells}
    chosen = ho.omission_cells(grid)
    assert len(chosen) == 12 == len(set(chosen)) and chosen == ho.omission_cells(grid)
    assert Counter(cells[c].type for c in chosen) == {
        "direct": 4,
        "exception_confusion": 3,
        "multi_provision": 3,
        "partial_out_of_scope": 2,
    }
    acts = Counter("+".join(cells[c].acts) for c in chosen)
    assert acts == {"MiCA": 4, "DORA": 4, "PSD2": 3, "DORA+PSD2": 1}


def test_opaque_ids_say_nothing_of_cell_attempt_or_order():
    salt = b"s"
    a1, a2 = (
        ho.opaque_id(salt, "ref:concept_author", "G01-a1"),
        ho.opaque_id(salt, "ref:concept_author", "G01-a2"),
    )
    assert a1 != a2 and re.fullmatch(r"R-[0-9a-f]{12}", a1)
    assert "G" not in a1  # never the cell id
    ids = {
        ho.opaque_id(salt, "ref:concept_author", f"G{c:02d}-a{n}")
        for c in range(1, 61)
        for n in (1, 2, 3)
    }
    assert len(ids) == 180  # one per cell and attempt, none repeated
    assert ho.opaque_id(b"other", "ref:concept_author", "G01-a1") != a1
