"""Unit tests for the external benchmark's adapter (T4, D-053), on synthetic rows: the rules
that keep the interface adapted and the algorithm untouched. The pinned dataset itself is
never needed here."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingest import external as ex

INPUTS = {"corpus/corpus.parquet": ex.FileRecord(bytes=1, sha256="0" * 64)}
EXPECTED = {
    "corpus_rows": 5,
    "root_nodes": 2,
    "articles": 3,
    "questions_train": 2,
    "questions_validation": 5,
    "aml_train": 1,
    "aml_validation": 1,
}


def _corpus() -> list[dict]:
    return [
        {"id": 10, "file_name": "§akn§it§act§a", "path": "/", "content": ex.ROOT_CONTENT},
        {"id": 2, "file_name": "§akn§it§act§a", "path": "art_1", "content": "Articolo 1 testo"},
        {"id": 11, "file_name": "§akn§eu§act§b", "path": "/", "content": ex.ROOT_CONTENT},
        {"id": 12, "file_name": "§akn§eu§act§b", "path": "art_2", "content": "Articolo 2 testo"},
        {"id": 3, "file_name": "§akn§eu§act§b", "path": "art_3", "content": "Articolo 3 testo"},
    ]


def _question(text: str, area: str, relevant: list[int]) -> dict:
    return {
        "question": text,
        "topics": "a topic",
        "normative_area": area,
        "relevant_doc_ids": relevant,
    }


def _questions() -> dict[str, list[dict]]:
    return {
        "train": [_question("q t0", "Privacy", [2]), _question("q t1", ex.AREA, [12, 3, 3])],
        "validation": [
            _question("q v0", ex.ESTIMATE_AREA, [2]),
            _question("q v1", ex.AREA, [3, 10]),
            _question("q v2", ex.ESTIMATE_AREA, [3]),
            _question("q v3", ex.ESTIMATE_AREA, [12]),
            _question("q v4", ex.ESTIMATE_AREA, [12]),
        ],
    }


def _adapt(out: Path) -> ex.AdaptManifest:
    return ex.adapt_rows(_corpus(), _questions(), out, INPUTS, EXPECTED)


def test_root_nodes_leave_the_corpus_and_articles_keep_id_order(tmp_path):
    manifest = _adapt(tmp_path)
    assert manifest.counts["root_nodes"] == 2 and manifest.counts["articles"] == 3
    assert [d for d, _ in ex.load_corpus(tmp_path)] == ["2", "3", "12"]
    assert ex.load_documents(tmp_path)["12"] == ("§akn§eu§act§b", "art_2")


def test_a_root_path_row_that_is_not_a_root_node_is_refused(tmp_path):
    corpus = _corpus()
    corpus[0] = {**corpus[0], "content": "real text"}
    with pytest.raises(ex.ExternalError, match="not a synthetic root node"):
        ex.adapt_rows(corpus, _questions(), tmp_path, INPUTS, EXPECTED)


def test_counts_other_than_the_published_ones_are_refused(tmp_path):
    with pytest.raises(ex.ExternalError, match="root_nodes 2"):
        ex.adapt_rows(_corpus(), _questions(), tmp_path, INPUTS, {**EXPECTED, "root_nodes": 159})


def test_queries_hold_only_id_and_text_and_qrels_live_apart(tmp_path):
    _adapt(tmp_path)
    assert ex.load_queries(tmp_path, "aml-validation") == [("validation-0001", "q v1")]
    assert ex.load_queries(tmp_path, "aml-train") == [("train-0001", "q t1")]
    for name in ex.QUERY_SETS:
        for line in (tmp_path / f"queries/{name}.jsonl").read_text(encoding="utf-8").splitlines():
            assert set(json.loads(line)) == {"query_id", "text"}
    # the qrels keep a relevant id the corpus does not hold (a root node): never corrected
    assert ex.load_qrels(tmp_path, "aml-validation") == {"validation-0001": frozenset({"3", "10"})}
    assert ex.load_qrels(tmp_path, "aml-train") == {"train-0001": frozenset({"3", "12"})}


def test_the_estimate_questions_are_another_area_and_have_no_qrels(tmp_path):
    _adapt(tmp_path)
    estimate = ex.load_queries(tmp_path, "estimate")
    assert [q for q, _ in estimate] == ["validation-0000", "validation-0002", "validation-0003"]
    assert not (tmp_path / "qrels/estimate.jsonl").exists()
    with pytest.raises(ex.ExternalError, match="unknown qrels set"):
        ex.load_qrels(tmp_path, "estimate")


def test_the_adapter_is_deterministic(tmp_path):
    first = _adapt(tmp_path / "a")
    second = _adapt(tmp_path / "b")
    assert first.outputs == second.outputs
    assert (tmp_path / "a" / ex.MANIFEST).read_bytes() == (
        tmp_path / "b" / ex.MANIFEST
    ).read_bytes()


def test_a_changed_output_file_is_refused_on_load(tmp_path):
    _adapt(tmp_path)
    path = tmp_path / "queries/aml-validation.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("q v1", "q v1 rewritten"))
    with pytest.raises(ex.ExternalError, match="not the file the adapter wrote"):
        ex.load_queries(tmp_path, "aml-validation")


def test_raw_files_other_than_the_pinned_ones_are_refused(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "corpus"
    raw.mkdir(parents=True)
    (raw / "corpus.parquet").write_bytes(b"x")
    monkeypatch.setattr(ex, "FILES", {"corpus/corpus.parquet": (1, "0" * 64)})
    with pytest.raises(ex.ExternalError, match="not the file pinned"):
        ex.verify_inputs(tmp_path / "raw")


def test_parquet_rows_are_read_as_written(tmp_path):
    # pyarrow is in requirements-dev.lock.txt and in the test image: never a skip
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = _corpus()
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "c.parquet")
    assert ex.read_rows(tmp_path / "c.parquet") == rows
