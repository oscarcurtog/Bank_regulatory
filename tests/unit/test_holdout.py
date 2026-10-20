"""The TEST machinery that needs no benchmark on disk (S3, D-054): areas, the content-word
Jaccard, the evidence overlap, the isolation guard and the commitment."""

from __future__ import annotations

import json

import pytest

from ingest import benchmark as bm
from ingest import holdout as ho

SHA = "a" * 64


def _qrel(need: str, act: str, path: str, grade: int, ambiguous: bool = False) -> bm.Qrel:
    return bm.Qrel(
        need_id=need,
        act=act,  # type: ignore[arg-type]
        path=path,
        grade=grade,
        rationale="r",
        text_sha256=SHA,
        ambiguous=ambiguous,
    )


def test_the_areas_are_read_from_the_protocol():
    areas = ho.load_areas()
    counts = {act: sum(a.act == act for a in areas) for act in bm.ACTS}
    assert counts == {"DORA": 8, "MiCA": 10, "PSD2": 7}
    assert ho.area_of(areas, "DORA", "45.1").name == "Information sharing"  # type: ignore[union-attr]
    assert ho.area_of(areas, "PSD2", "35a.2.sub_1").name.startswith("Payment institutions")  # type: ignore[union-attr]


def test_an_article_after_the_area_table_has_no_area():
    areas = ho.load_areas()
    assert ho.area_of(areas, "MiCA", "143.3") is None
    assert ho.need_areas([_qrel("N101", "MiCA", "143.3", 3)], areas) == {None}


def test_the_area_of_a_need_comes_from_grade_3_then_grade_2():
    areas = ho.load_areas()
    qrels = [
        _qrel("N101", "DORA", "19.1", 3),
        _qrel("N101", "DORA", "3.8", 2),
        _qrel("N101", "DORA", "28.1", 3, ambiguous=True),
    ]
    assert {a.name for a in ho.need_areas(qrels, areas) if a} == {
        "ICT-related incidents: management, classification, reporting"
    }
    partial = [_qrel("N102", "DORA", "3.8", 2), _qrel("N102", "DORA", "28.1", 1)]
    assert {a.name for a in ho.need_areas(partial, areas) if a} == {
        "Scope, definitions, proportionality"
    }


def test_article_numbers_ignore_letters_and_paths():
    assert ho.article_number("35a.2.sub_1") == 35
    with pytest.raises(ho.HoldoutError):
        ho.article_number("sub_1")


def test_jaccard_uses_the_content_words_of_the_dev_calibration():
    assert ho.content_words("The bank, its ICT and an IT vendor") == {"bank", "ict", "vendor"}
    assert ho.jaccard("bank vendor contract", "bank vendor exit") == pytest.approx(0.5)
    assert ho.jaccard("", "") == 0.0


def test_closest_statements_skip_the_statement_itself():
    texts = {"N101": "bank vendor contract", "N102": "bank vendor exit", "N103": "crypto token"}
    closest = ho.closest_statements(texts, texts, top=1)
    assert closest["N101"][0].other == "N102"
    assert closest["N101"][0].jaccard == pytest.approx(0.5)
    assert closest["N103"][0].jaccard == 0.0


def test_evidence_overlap_separates_shared_nodes_from_shared_articles():
    test = [_qrel("N101", "DORA", "28.7.a", 3), _qrel("N102", "MiCA", "75.5", 3)]
    dev = [
        _qrel("N014", "DORA", "28.7.b", 3),
        _qrel("N008", "MiCA", "75.5", 3),
        _qrel("N009", "MiCA", "75.5", 2),
    ]
    pairs = ho.evidence_overlap(test, dev)
    assert pairs == [
        ho.EvidenceOverlap("N101", "N014", (), ("DORA:28",)),
        ho.EvidenceOverlap("N102", "N008", ("MiCA:75.5",), ("MiCA:75",)),
    ]
    strata = ho.evidence_strata(["N101", "N102", "N103"], pairs)
    assert strata == {
        "N101": "shared_grade3_article",
        "N102": "shared_grade3_article",
        "N103": "no_shared_grade3",
    }


def test_the_isolation_guard_finds_a_test_query_in_a_run(tmp_path):
    run = tmp_path / "data/runs/e5-layer-500-dev-0.10.0/run.jsonl"
    run.parent.mkdir(parents=True)
    run.write_text(json.dumps({"query_id": "N004-en-1"}) + "\n", encoding="utf-8")
    assert ho.isolation_problems(tmp_path) == []
    run.write_text(run.read_text() + json.dumps({"query_id": "N101-es-3"}) + "\n")
    assert ho.isolation_problems(tmp_path) == [
        "data/runs/e5-layer-500-dev-0.10.0/run.jsonl:2: holds TEST ids ['N101-es-3']"
    ]


def test_the_isolation_guard_finds_a_directory_named_after_test(tmp_path):
    (tmp_path / "data/embeddings/e5/queries-test-1.0.0").mkdir(parents=True)
    assert ho.isolation_problems(tmp_path) == [
        "data/embeddings/e5/queries-test-1.0.0: named after TEST"
    ]
    (tmp_path / "data/results/latest").mkdir(parents=True)
    assert len(ho.isolation_problems(tmp_path)) == 1


def test_the_isolation_guard_finds_an_id_outside_the_range(tmp_path):
    meta = tmp_path / "data/embeddings/e5/queries-0.11.0/meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"item_ids": ["N250-en-1"]}), encoding="utf-8")
    assert ho.isolation_problems(tmp_path) == []
    assert ho.isolation_problems(tmp_path, {"N250-en-1"}) == [
        "data/embeddings/e5/queries-0.11.0/meta.json:1: holds TEST ids ['N250-en-1']"
    ]


def test_the_commitment_never_publishes_a_test_id():
    freeze: dict[str, object] = {k: {} for k in ho.COMMITTED_KEYS}
    record = ho.commitment(freeze, SHA)
    assert record["schema_version"] == ho.COMMITMENT_SCHEMA
    assert record["freeze_sha256"] == SHA
    freeze["overlap"] = {"closest": "N101"}
    with pytest.raises(ho.HoldoutError, match="TEST ids"):
        ho.commitment(freeze, SHA)


def test_tree_hash_changes_with_any_file(tmp_path):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    first = ho.tree_sha256(tmp_path)
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    assert ho.tree_sha256(tmp_path) != first


def test_the_figure_pattern_finds_figures_and_dates_but_not_references():
    found = ["within 30 days", "EUR 10 000", "2 % of", "ten working days", "17 January 2025"]
    ignored = ["Article 30", "paragraph 2", "Regulation (EU) 2022/2554", "point (b)"]
    assert all(ho.FIGURE_OR_DATE.search(s) for s in found)
    assert not any(ho.FIGURE_OR_DATE.search(s) for s in ignored)


def test_figure_density_skips_excluded_psd2_articles():
    areas = [a for a in ho.load_areas() if a.act == "PSD2" and a.first == 5]
    texts = {("PSD2", "7.1"): "EUR 125 000", ("PSD2", "5.1"): "within 3 months", ("PSD2", "8"): "x"}
    density = ho.figure_density(areas, texts, frozenset({"5"}))
    (record,) = density.values()
    assert record == {"nodes": 2, "with_figure_or_date": 1, "share": 0.5}


def _repo(path):
    path.mkdir()
    for args in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "t@t"],
        ["config", "user.name", "t"],
    ):
        ho.git(path, *args)
    (path / "grid.json").write_text("{}", encoding="utf-8")
    ho.git(path, "add", "-A")
    ho.git(path, "commit", "-q", "-m", "one")
    return path


def test_a_backup_is_verified_and_restored(tmp_path):
    private = _repo(tmp_path / "test")
    folder = tmp_path / "backups"
    bundle = ho.backup(private, tmp_path / "elsewhere", folder)
    assert bundle.exists() and (folder / "backups.jsonl").exists()
    entry = json.loads((folder / "backups.jsonl").read_text().splitlines()[0])
    assert entry["head"] == ho.git(private, "rev-parse", "HEAD") and entry["verified"]
    assert oct(folder.stat().st_mode)[-3:] == "700"


def test_a_backup_refuses_unrecorded_changes_and_a_folder_in_the_repo(tmp_path):
    private = _repo(tmp_path / "test")
    (private / "needs.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ho.HoldoutError, match="not recorded"):
        ho.backup(private, tmp_path / "elsewhere", tmp_path / "backups")
    with pytest.raises(ho.HoldoutError, match="inside the repository"):
        ho.backup(private, tmp_path, tmp_path / "backups")


def test_a_restore_that_does_not_match_is_refused(tmp_path):
    private = _repo(tmp_path / "test")
    bundle = ho.backup(private, tmp_path / "elsewhere", tmp_path / "backups")
    (private / "grid.json").write_text('{"changed": 1}', encoding="utf-8")
    ho.git(private, "commit", "-q", "-am", "two")
    with pytest.raises(ho.HoldoutError, match="HEAD differs"):
        ho.restore_check(private, bundle)
