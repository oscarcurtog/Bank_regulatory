"""Integration tests for the retrieval benchmark boundaries (D-019): persistence,
the dataset rules, the corpus check, and the explicit seal.

They run on a small synthetic processed layer and benchmark in tmp_path, so they
need no corpus and run in CI. The real fixtures are checked against the real
corpus in test_benchmark_fixtures.py.
"""

from __future__ import annotations

import json

import pytest

from ingest import benchmark as bm
from ingest import hierarchy as h
from ingest import processed as pr
from ingest import psd2_exclusions as px

SHAS = {"DORA": "1" * 64, "MiCA": "2" * 64, "PSD2": "3" * 64}
CELEX = {"DORA": "32022R2554", "MiCA": "02023R1114-20240109", "PSD2": "02015L2366-20151223"}
ENTRY_INTO_FORCE = "This Regulation shall enter into force on the twentieth day."
TEXTS = {
    ("DORA", "19.1"): "Financial entities shall report major ICT-related incidents.",
    ("DORA", "19.2"): "Financial entities may notify significant cyber threats.",
    ("DORA", "64"): ENTRY_INTO_FORCE,
    ("MiCA", "149.1"): ENTRY_INTO_FORCE,
    ("MiCA", "149.2"): "This Regulation shall apply from 30 December 2024.",
    ("PSD2", "96.1"): "Payment service providers shall notify major incidents.",
}


def _article(act: str, number: str, texts: dict) -> h.Article:
    paths = sorted(p for a, p in texts if a == act and p.split(".")[0] == number)
    container = (
        [] if paths == [number] else [h.Node(path=number, level=1, marker=number, kind="articulo")]
    )
    nodes = container + [
        h.Node(
            path=p,
            level=len(p.split(".")),
            marker=p.split(".")[-1],
            kind="articulo" if p == number else "apartado",
            own_text=texts[(act, p)],
        )
        for p in paths
    ]
    return h.Article(
        article_id=f"art_{number}",
        path=number,
        title="Title",
        text=" ".join(n.own_text for n in nodes if n.own_text),
        max_depth=max(n.level for n in nodes),
        nodes=nodes,
    )


def _docs(parser_version: str = h.PARSER_VERSION, texts_override: dict | None = None) -> dict:
    texts = TEXTS | (texts_override or {})
    articles = {
        "DORA": [_article("DORA", "19", texts), _article("DORA", "64", texts)],
        "MiCA": [_article("MiCA", "149", texts)],
        "PSD2": [_article("PSD2", "96", texts)],
    }
    return {
        act: pr.ProcessedDocument(
            source_celex=CELEX[act],
            source_role="original" if act == "DORA" else "consolidated",
            source_sha256=SHAS[act],
            parser_version=parser_version,
            family="doue" if act == "DORA" else "consolidado",
            articles=articles[act],
        )
        for act in bm.ACTS
    }


def _need(**overrides) -> bm.Need:
    need = {
        "need_id": "N001",
        "split": "dev",
        "statement": "From what date does MiCA apply?",
        "type": "direct",
        "drivers": ["date", "lexical_twin"],
        "difficulty": "easy",
        "acts": ["MiCA"],
        "answerability": "complete",
        "status": "ready",
    }
    return bm.Need.model_validate(need | overrides)


def _query(query_id: str = "N001-en-1", **overrides) -> bm.Query:
    lang = query_id.split("-")[1]
    query = {
        "query_id": query_id,
        "need_id": query_id.split("-")[0],
        "lang": lang,
        "style": "formal" if lang == "en" else "natural",
        "origin": "human",
        "text": f"Text of {query_id}",
    }
    return bm.Query.model_validate(query | overrides)


def _qrel(act: str, path: str, grade: int, need_id: str = "N001", **overrides) -> bm.Qrel:
    qrel = {
        "need_id": need_id,
        "act": act,
        "path": path,
        "grade": grade,
        "rationale": f"Grade {grade} for {act} {path}.",
        "text_sha256": bm.sha256_text(TEXTS.get((act, path), "")),
    }
    return bm.Qrel.model_validate(qrel | overrides)


def _records() -> tuple[list, list, list]:
    needs = [_need()]
    queries = [_query("N001-en-1"), _query("N001-es-1")]
    qrels = [_qrel("MiCA", "149.2", 3), _qrel("MiCA", "149.1", 1), _qrel("DORA", "64", 0)]
    return needs, queries, qrels


def _write(directory, needs, queries, qrels) -> None:
    bm.write_jsonl(directory / bm.NEEDS_FILE, needs)
    bm.write_jsonl(directory / bm.QUERIES_FILE, queries)
    bm.write_jsonl(directory / bm.QRELS_FILE, qrels)


def _exclusions(base_sha: str = SHAS["PSD2"]) -> px.Psd2Exclusions:
    return px.Psd2Exclusions(
        schema_version="1",
        base=px.Document(celex=CELEX["PSD2"], sha256=base_sha),
        reference=px.Document(celex="02015L2366-20250117", sha256="4" * 64),
        parser_version=h.PARSER_VERSION,
        articles=[px.ExcludedArticle(article="96", change="modified", amended_by=["32022L2556"])],
    )


@pytest.fixture(autouse=True)
def synthetic_exclusions(tmp_path, monkeypatch):
    """A PSD2 exclusion list built on the synthetic PSD2 snapshot of this module."""
    path = tmp_path / "psd2_excluded_articles.json"
    px.write_psd2_exclusions(_exclusions(), path)
    monkeypatch.setattr(px, "PSD2_EXCLUSIONS_PATH", path)
    return path


@pytest.fixture
def sealed(tmp_path):
    directory = tmp_path / "dev"
    _write(directory, *_records())
    bm.seal(directory, "0.1.0", "dev", _docs(), templates_dir=tmp_path / "templates")
    return directory


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def test_write_then_read_is_byte_stable(tmp_path):
    needs, queries, qrels = _records()
    path = tmp_path / "qrels.jsonl"
    bm.write_jsonl(path, qrels)
    first = path.read_bytes()
    bm.write_jsonl(path, bm.read_jsonl(path, bm.Qrel))
    assert path.read_bytes() == first
    assert bm.read_jsonl(path, bm.Qrel) == qrels


def test_an_empty_file_is_an_empty_list(tmp_path):
    path = tmp_path / "needs.jsonl"
    path.write_text("", encoding="utf-8")
    assert bm.read_jsonl(path, bm.Need) == []


def test_a_bad_line_is_reported_with_its_number(tmp_path):
    path = tmp_path / "needs.jsonl"
    bm.write_jsonl(path, [_need()])
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")  # a blank line 2
    with pytest.raises(bm.BenchmarkError, match=r"needs\.jsonl:2"):
        bm.read_jsonl(path, bm.Need)


# ---------------------------------------------------------------------------
# seal, load and check: the happy path
# ---------------------------------------------------------------------------


def test_seal_load_and_check_corpus(sealed, tmp_path):
    manifest = json.loads((sealed / bm.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["role"] == "dev" and manifest["parser_version"] == h.PARSER_VERSION
    assert manifest["corpus"]["PSD2"] == {
        "source_celex": CELEX["PSD2"],
        "source_sha256": SHAS["PSD2"],
    }
    assert manifest["files"] == {n: bm.sha256_file(sealed / n) for n in bm.DATA_FILES}
    assert manifest["templates"] == {}
    benchmark = bm.load_benchmark(sealed, templates_dir=tmp_path / "templates")
    assert [n.need_id for n in benchmark.needs] == ["N001"]
    assert len(benchmark.queries_of("N001")) == 2 and len(benchmark.qrels_of("N001")) == 3
    bm.check_corpus(benchmark, _docs())


def test_a_draft_need_may_have_no_qrels_yet(tmp_path):
    """Queries are locked before the evidence is searched, so a draft is valid
    without qrels: the evidence rules apply from 'reviewed' on."""
    directory = tmp_path / "dev"
    _write(directory, [_need(status="draft")], [_query("N001-en-1")], [])
    bm.seal(directory, "0.1.0", "dev", _docs())


# ---------------------------------------------------------------------------
# the corpus boundary: a stale qrel is reported, never fixed
# ---------------------------------------------------------------------------


def test_seal_names_every_qrel_that_does_not_match_the_corpus(tmp_path):
    needs, queries, qrels = _records()
    qrels += [
        _qrel("MiCA", "150.1", 1, text_sha256="0" * 64),  # no such node
        _qrel("DORA", "19", 1, text_sha256=bm.sha256_text("")),  # the empty article node
        _qrel("DORA", "19.1", 1, text_sha256="0" * 64),  # judged on another text
        _qrel("DORA", "19.2", 1, span={"start": 0, "end": 999}),  # span past the end
    ]
    directory = tmp_path / "dev"
    _write(directory, needs, queries, qrels)
    with pytest.raises(bm.BenchmarkError) as raised:
        bm.seal(directory, "0.1.0", "dev", _docs())
    message = str(raised.value)
    assert "need N001 MiCA:150.1: no such node" in message
    assert "need N001 DORA:19: the node has no own text" in message
    assert "need N001 DORA:19.1: the node text changed since it was judged" in message
    assert "need N001 DORA:19.2: span ends at 999" in message
    assert not (directory / bm.MANIFEST_FILE).exists()


def test_a_corpus_change_is_stale_and_names_the_qrels_it_broke(sealed, tmp_path):
    benchmark = bm.load_benchmark(sealed, templates_dir=tmp_path / "templates")
    changed = _docs(
        texts_override={("MiCA", "149.2"): "This Regulation shall apply from 1 January 2025."}
    )
    changed["MiCA"] = changed["MiCA"].model_copy(update={"source_sha256": "9" * 64})
    with pytest.raises(bm.StaleBenchmarkError) as raised:
        bm.check_corpus(benchmark, changed)
    message = str(raised.value)
    assert "MiCA: sealed against" in message
    assert "need N001 MiCA:149.2: the node text changed since it was judged" in message


def test_a_parser_change_that_breaks_nothing_still_asks_for_a_new_seal(sealed, tmp_path):
    benchmark = bm.load_benchmark(sealed, templates_dir=tmp_path / "templates")
    with pytest.raises(bm.StaleBenchmarkError, match="every qrel still matches"):
        bm.check_corpus(benchmark, _docs(parser_version="9.9.9"))


# ---------------------------------------------------------------------------
# the dataset boundary
# ---------------------------------------------------------------------------


def _with(records, need=None, queries=None, qrels=None):
    needs, qs, rs = records
    return (
        need if need is not None else needs,
        queries if queries is not None else qs,
        qrels if qrels is not None else rs,
    )


DATASET_CASES = {
    "duplicate need": lambda r: _with(r, need=r[0] + [_need()]),
    "query of an unknown need": lambda r: _with(r, queries=r[1] + [_query("N009-en-1")]),
    "qrel of an unknown need": lambda r: _with(
        r, qrels=r[2] + [_qrel("DORA", "19.2", 1, need_id="N009")]
    ),
    "duplicate qrel": lambda r: _with(r, qrels=r[2] + [_qrel("MiCA", "149.2", 2)]),
    "duplicate query text": lambda r: _with(
        r, queries=r[1] + [_query("N001-en-2", text="text of N001-EN-1")]
    ),
    "split test in a dev benchmark": lambda r: _with(r, need=[_need(split="test")]),
    "evidence_aware query outside a fixture": lambda r: _with(
        r, queries=[_query("N001-en-1", origin="evidence_aware"), _query("N001-es-1")]
    ),
    "ready need without a Spanish query": lambda r: _with(r, queries=[_query("N001-en-1")]),
    "complete need without evidence": lambda r: _with(r, qrels=[_qrel("MiCA", "149.1", 1)]),
    "evidence that is only ambiguous": lambda r: _with(
        r, qrels=[_qrel("MiCA", "149.2", 3, ambiguous=True), _qrel("MiCA", "149.1", 1)]
    ),
    "complete need whose evidence is only support": lambda r: _with(
        r, qrels=[_qrel("MiCA", "149.2", 2), _qrel("MiCA", "149.1", 1)]
    ),
    "complete need whose only answer is ambiguous": lambda r: _with(
        r, qrels=[_qrel("MiCA", "149.2", 3, ambiguous=True), _qrel("MiCA", "149.1", 2)]
    ),
    "acts that do not match the evidence": lambda r: _with(r, need=[_need(acts=["DORA"])]),
    "none need with evidence": lambda r: _with(
        r,
        need=[_need(type="partial_out_of_scope", answerability="none", gap_note="Not in corpus.")],
    ),
}
DATASET_MESSAGES = {
    "duplicate need": "need N001 appears 2 times",
    "query of an unknown need": "query N009-en-1: unknown need N009",
    "qrel of an unknown need": "qrel N009 DORA:19.2: unknown need",
    "duplicate qrel": "appears 2 times",
    "duplicate query text": "the same text as another query",
    "split test in a dev benchmark": "split 'test' in a benchmark with role 'dev'",
    "evidence_aware query outside a fixture": "only allowed in a fixture",
    "ready need without a Spanish query": "without a query in 'es'",
    "complete need without evidence": "without any non-ambiguous qrel of grade >= 2",
    "evidence that is only ambiguous": "without any non-ambiguous qrel of grade >= 2",
    "complete need whose evidence is only support": "without a non-ambiguous qrel of grade 3",
    "complete need whose only answer is ambiguous": "without a non-ambiguous qrel of grade 3",
    "acts that do not match the evidence": "acts ['DORA'] but its evidence",
    "none need with evidence": "answerability 'none' but it has qrels of grade >= 2",
}


@pytest.mark.parametrize("case", list(DATASET_CASES))
def test_seal_refuses_a_broken_dataset(tmp_path, case):
    directory = tmp_path / "dev"
    _write(directory, *DATASET_CASES[case](_records()))
    with pytest.raises(bm.BenchmarkError) as raised:
        bm.seal(directory, "0.1.0", "dev", _docs())
    assert DATASET_MESSAGES[case] in str(raised.value)
    assert not (directory / bm.MANIFEST_FILE).exists()


def test_a_partial_need_may_rest_on_support_alone(tmp_path):
    """Only a complete need must have a grade-3 answer: when the answer itself lives
    outside the corpus, grade 2 support is all the corpus can give."""
    directory = tmp_path / "dev"
    _, queries, _ = _records()
    partial = _need(type="partial_out_of_scope", answerability="partial", gap_note="Elsewhere.")
    _write(directory, [partial], queries, [_qrel("MiCA", "149.2", 2), _qrel("MiCA", "149.1", 1)])
    bm.seal(directory, "0.1.0", "dev", _docs())


def test_an_ambiguous_need_cannot_enter_test(tmp_path):
    directory = tmp_path / "test"
    needs, queries, qrels = _records()
    qrels.append(_qrel("DORA", "19.1", 2, ambiguous=True))
    _write(directory, [_need(split="test")], queries, qrels)
    with pytest.raises(bm.BenchmarkError, match="an ambiguous need cannot be in TEST"):
        bm.seal(directory, "0.1.0", "test", _docs())


# ---------------------------------------------------------------------------
# the adjudication record (PROTOCOL.md, section 12): outside the seal, but it must
# agree with the sealed qrels label by label
# ---------------------------------------------------------------------------


def _judged(grade: int, scope: list[str] | None = None) -> dict:
    return {"grade": grade, "rationale": f"Grade {grade}."} | (
        {"scope_basis": scope} if scope else {}
    )


def _adjudication(target: str, a: dict, b: dict, c: dict | None = None) -> bm.Adjudication:
    data = {"need_id": "N001", "target": target, "a": a, "b": b}
    return bm.Adjudication.model_validate(data | ({"c": c} if c else {}))


def _adjudications() -> list[bm.Adjudication]:
    """How the labels of _records() were decided: MiCA 149.2 went to judge C."""
    answerability = {"answerability": "complete", "rationale": "149.2 states the date."}
    return [
        _adjudication("MiCA:149.2", _judged(2), _judged(3), c=_judged(3)),
        _adjudication("MiCA:149.1", _judged(1), _judged(1)),
        _adjudication("DORA:64", _judged(0, ["DORA:64"]), _judged(0, ["DORA:64"])),
        _adjudication("answerability", answerability, answerability),
    ]


def _with_adjudications(tmp_path, adjudications):
    directory = tmp_path / "dev"
    _write(directory, *_records())
    bm.write_jsonl(directory / bm.ADJUDICATIONS_FILE, adjudications)
    return directory


def test_the_adjudication_record_is_read_with_the_benchmark(tmp_path):
    directory = _with_adjudications(tmp_path, _adjudications())
    bm.seal(directory, "0.1.0", "dev", _docs())
    benchmark = bm.load_benchmark(directory)
    assert [a.target for a in benchmark.adjudications if a.c is not None] == ["MiCA:149.2"]
    bm.check_corpus(benchmark, _docs())


def test_seal_refuses_a_record_that_disagrees_with_the_qrels(tmp_path):
    adjudications = _adjudications()
    adjudications[0] = _adjudication("MiCA:149.2", _judged(2), _judged(2))  # the qrel says 3
    with pytest.raises(bm.BenchmarkError, match="the qrel has 3"):
        bm.seal(_with_adjudications(tmp_path, adjudications), "0.1.0", "dev", _docs())


def test_with_a_record_no_ready_need_skips_adjudication(tmp_path):
    directory = _with_adjudications(tmp_path, _adjudications()[:-1])
    with pytest.raises(bm.BenchmarkError, match="no adjudication of its answerability"):
        bm.seal(directory, "0.1.0", "dev", _docs())


def test_a_record_edited_after_the_seal_is_refused_on_load(tmp_path):
    directory = _with_adjudications(tmp_path, _adjudications())
    bm.seal(directory, "0.1.0", "dev", _docs())
    adjudications = _adjudications()
    adjudications[1] = _adjudication("MiCA:149.1", _judged(1), _judged(2), c=_judged(2))
    bm.write_jsonl(directory / bm.ADJUDICATIONS_FILE, adjudications)
    with pytest.raises(bm.BenchmarkError, match="the adjudication record does not match"):
        bm.load_benchmark(directory)


def test_a_scope_check_names_nodes_that_exist(tmp_path):
    adjudications = _adjudications()
    adjudications[2] = _adjudication(
        "DORA:64", _judged(0, ["DORA:64", "DORA:65"]), _judged(0, ["DORA:64"])
    )
    with pytest.raises(bm.BenchmarkError, match="no node DORA:65"):
        bm.seal(_with_adjudications(tmp_path, adjudications), "0.1.0", "dev", _docs())


# ---------------------------------------------------------------------------
# the manifest boundary
# ---------------------------------------------------------------------------


def test_a_file_changed_after_the_seal_is_refused(sealed, tmp_path):
    path = sealed / bm.QUERIES_FILE
    path.write_text(
        path.read_text(encoding="utf-8").replace("Text of N001-es-1", "Another text"),
        encoding="utf-8",
    )
    with pytest.raises(bm.StaleBenchmarkError, match="queries.jsonl: changed since the seal"):
        bm.load_benchmark(sealed, templates_dir=tmp_path / "templates")


def test_a_template_is_sealed_with_the_queries_that_use_it(tmp_path):
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "query_variants.v1.md").write_text(
        "Write variants of: {statement}\n", encoding="utf-8"
    )
    directory = tmp_path / "dev"
    needs, _, qrels = _records()
    queries = [
        _query("N001-en-1", origin="llm_from_need", template="query_variants.v1"),
        _query("N001-es-1"),
    ]
    _write(directory, needs, queries, qrels)
    manifest = bm.seal(directory, "0.1.0", "dev", _docs(), templates_dir=templates)
    assert manifest.templates == {
        "query_variants.v1": bm.sha256_file(templates / "query_variants.v1.md")
    }
    bm.load_benchmark(directory, templates_dir=templates)

    (templates / "query_variants.v1.md").write_text("Edited.\n", encoding="utf-8")
    with pytest.raises(
        bm.StaleBenchmarkError, match="template query_variants.v1: changed since the seal"
    ):
        bm.load_benchmark(directory, templates_dir=templates)
    (templates / "query_variants.v1.md").unlink()
    with pytest.raises(bm.StaleBenchmarkError, match="query_variants.v1.md does not exist"):
        bm.load_benchmark(directory, templates_dir=templates)


def test_the_schema_version_is_compared_before_the_schema(sealed):
    path = sealed / bm.MANIFEST_FILE
    path.write_text(json.dumps({"schema_version": "2", "anything": "else"}), encoding="utf-8")
    with pytest.raises(bm.StaleBenchmarkError, match="schema_version 2"):
        bm.load_benchmark(sealed)


def test_a_missing_manifest_says_what_to_do(tmp_path):
    _write(tmp_path, *_records())
    with pytest.raises(bm.BenchmarkError, match="seal the benchmark first"):
        bm.load_benchmark(tmp_path)


# ---------------------------------------------------------------------------
# the explicit reseal
# ---------------------------------------------------------------------------


def test_a_reseal_without_changes_writes_nothing(sealed):
    before = (sealed / bm.MANIFEST_FILE).read_bytes()
    current = bm.seal(sealed, "0.2.0", "dev", _docs())
    assert current.benchmark_version == "0.1.0"
    assert (sealed / bm.MANIFEST_FILE).read_bytes() == before


def test_a_reseal_after_a_change_needs_a_greater_version(sealed):
    needs, queries, qrels = _records()
    _write(sealed, needs, queries, qrels + [_qrel("DORA", "19.1", 0)])
    with pytest.raises(bm.BenchmarkError, match="must be greater than 0.1.0"):
        bm.seal(sealed, "0.1.0", "dev", _docs())
    assert bm.seal(sealed, "0.2.0", "dev", _docs()).benchmark_version == "0.2.0"
    assert len(bm.load_benchmark(sealed).qrels) == 4


def test_a_reseal_never_changes_the_role(sealed):
    with pytest.raises(bm.BenchmarkError, match="a role never changes"):
        bm.seal(sealed, "0.2.0", "fixture", _docs())


def test_a_reseal_refuses_a_malformed_version(sealed):
    with pytest.raises(bm.BenchmarkError, match="is not x.y.z"):
        bm.seal(sealed, "v2", "dev", _docs())


# ---------------------------------------------------------------------------
# the PSD2 exclusion list
# ---------------------------------------------------------------------------


def test_a_qrel_on_an_amended_psd2_article_is_refused(tmp_path):
    needs, queries, qrels = _records()
    _write(tmp_path / "dev", needs, queries, qrels + [_qrel("PSD2", "96.1", 0)])
    with pytest.raises(
        bm.BenchmarkError, match="PSD2:96.1: Article 96 was amended after the corpus snapshot"
    ):
        bm.seal(tmp_path / "dev", "0.1.0", "dev", _docs())


def test_an_exclusion_list_built_on_another_psd2_snapshot_is_stale(sealed, synthetic_exclusions):
    px.write_psd2_exclusions(_exclusions(base_sha="5" * 64), synthetic_exclusions)
    with pytest.raises(
        bm.StaleBenchmarkError, match="rebuild it with scripts/build_psd2_exclusions.py"
    ):
        bm.load_benchmark(sealed)
    with pytest.raises(bm.StaleBenchmarkError, match="the PSD2 exclusion list was built on"):
        bm.seal(sealed, "0.2.0", "dev", _docs())
