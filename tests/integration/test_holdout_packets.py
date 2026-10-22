"""The packets of TEST (S3, D-054), built from the real protocol, templates, grid and DEV
statements. No corpus: the judges' corpus is a small stand-in."""

from __future__ import annotations

import json
import re

import pytest

from ingest import benchmark as bm
from ingest import holdout as ho
from ingest import holdout_packets as hp


@pytest.fixture(scope="module")
def grid() -> ho.Grid:
    return ho.build_grid(bm.load_benchmark(bm.DEV_DIR), ho.load_areas())


@pytest.fixture(scope="module")
def forbidden() -> list[str]:
    needs = bm.load_benchmark(bm.DEV_DIR).needs + bm.load_benchmark(bm.FIXTURES_DIR).needs
    return [n.statement for n in needs]


def test_judges_b_and_c_get_the_rules_without_devs_examples():
    rules = hp.judge_bc_rules()
    for heading in ("## 2. Grades", "## 3. Answerability", "### Legal snapshot", "## 12. "):
        assert heading in rules
    assert "worked example" not in rules and "| Node | Need |" not in rules


def test_judge_a_of_dev_got_the_protocol_verbatim_and_test_without_devs_examples():
    text = ho.PROTOCOL_PATH.read_text(encoding="utf-8")
    verbatim = hp.judge_a_rules_dev()
    for n in (1, 2, 3, 4, 5, 9, 13):
        assert hp.section(text, n) in verbatim
    sanitized = hp.judge_a_rules()
    assert not [b for b in hp.BANNED_FOR_JUDGE_A if b in sanitized]
    assert [b for b in hp.BANNED_FOR_JUDGE_A if b in verbatim]  # DEV's had them
    for rule in (
        "## 2. Grades",
        "### Applicability that depends on unstated facts",
        "### Legal snapshot",
        "The size of the relevant set is\na symptom, not the criterion",
        '"Article N", "paragraph N"',
    ):
        assert rule in sanitized
    assert "## 12. " not in sanitized


def test_every_cell_gives_a_clean_concept_packet(grid, forbidden):
    for cell in grid.cells:
        packet = hp.concept_packet(cell, forbidden)
        assert "$" not in packet and cell.cell_id in packet
        assert "EXISTING NEEDS" not in packet and "ALREADY USED" not in packet


def test_the_writer_gets_the_concept_and_its_facts(grid, forbidden):
    cell = grid.cells[0]
    packet = hp.writer_packet(cell, "P001", "A bank wants to know X.", ["fact one"], forbidden)
    assert "- Concept: A bank wants to know X." in packet and "- fact one" in packet
    assert "$" not in packet


@pytest.mark.parametrize(
    ("text", "leak"),
    [
        ("Financial entities shall report", "shall"),
        ("see DORA:19.2", "DORA:1"),
        ("under Article 19(4)", "Article 19("),
        ("like N016 does", "N016"),
        ("as 28(7) says", "28(7)"),
    ],
)
def test_author_leaks_are_found(text, leak):
    assert any(leak in found for found in hp.author_leaks(text, []))


def test_a_run_of_eight_words_from_dev_is_a_leak(forbidden):
    copied = " ".join(re.findall(r"\w+", forbidden[0])[:8])
    assert hp.author_leaks(f"A concept that says {copied} here.", forbidden)
    with pytest.raises(hp.PacketError):
        hp.writer_packet(
            ho.Cell(
                cell_id="G01",
                batch=1,
                acts=["DORA"],
                area="Information sharing",
                type="direct",
                answerability="complete",
                difficulty="easy",
                number_or_date=False,
            ),
            "P001",
            copied,
            [],
            forbidden,
        )


def test_the_gatekeeper_sees_every_existing_need_with_its_source():
    packet = hp.gate_packet(
        "P007",
        "A founder asks X.",
        [hp.Existing("N004", "DEV", "dev statement"), hp.Existing("N101", "TEST", "test one")],
    )
    assert "- N004 (DEV): dev statement" in packet and "- N101 (TEST): test one" in packet
    assert "$" not in packet


def _corpus() -> hp.Corpus:
    corpus = hp.Corpus.__new__(hp.Corpus)
    corpus.texts = {("DORA", "19"): "", ("DORA", "19.1"): "report", ("DORA", "19.2"): "notify"}
    corpus.titles = {("DORA", "19"): "Reporting"}
    corpus.order = {("DORA", "19"): 0, ("DORA", "19.1"): 1, ("DORA", "19.2"): 2}
    return corpus


def _need(need_id: str = "N101") -> bm.Need:
    return bm.Need(
        need_id=need_id,
        split="test",
        statement="A bank detected a threat. Must it report it?",
        type="direct",
        drivers=[],
        difficulty="easy",
        acts=["DORA"],
        answerability="complete",
        status="draft",
    )


def ids_for(*targets: str) -> dict[str, str]:
    """An orchestrator's mapping for one stage: target -> opaque id."""
    return {t: ho.item_id(b"test-salt", "N101:stage", t) for t in targets}


def test_judge_b_never_sees_a_grade():
    ids = ids_for("DORA:19.2", "DORA:19.1")
    packet = hp.judge_b_packet(_corpus(), _need(), ["DORA:19.2", "DORA:19.1"], {}, ids)
    candidates = packet.split("CANDIDATES (document order)")[1]
    assert candidates.index("DORA:19.1") < candidates.index("DORA:19.2")
    assert not re.search(r"grade [0-3]|rationale", candidates)
    # every candidate under its opaque id, which the answer must carry
    for node, opaque in ids.items():
        assert f"CANDIDATE {opaque}\n[{node}]" in candidates
    assert '"item_id": "C-..."' in packet and '"node": "ACT:path"' not in packet


def test_judge_c_draws_the_order_per_need():
    a = {
        "DORA:19.1": {"grade": 3, "rationale": "A says"},
        "answerability": {"answerability": "complete", "rationale": "A"},
    }
    b = {
        "DORA:19.1": {"grade": 2, "rationale": "B says"},
        "answerability": {"answerability": "partial", "rationale": "B"},
    }

    ids = ids_for("DORA:19.1", "answerability")

    def build() -> tuple[str, list[hp.CItem]]:
        return hp.judge_c_packet(
            _corpus(), _need(), ["DORA:19.1"], {}, a, b, ["DORA:19.1"], True, ids
        )

    text, items = build()
    again, same = build()
    assert (text, items) == (again, same)
    assert [i.target for i in items] == ["DORA:19.1", "answerability"]
    assert [i.item_id for i in items] == [ids["DORA:19.1"], ids["answerability"]]
    assert f"ITEM {ids['DORA:19.1']}\n[DORA:19.1]" in text
    assert f"ITEM {ids['answerability']}: the answerability" in text
    assert '"item_id": "C-... exactly as given"' in text and '"target"' not in text
    first = items[0].rationale_a_is_judge
    shown = "A says" if first == "A" else "B says"
    assert f"Rationale A: grade {3 if first == 'A' else 2}. {shown}" in text


def test_a_planned_figure_is_asked_for_softly(grid, forbidden):
    figure = next(c for c in grid.cells if c.number_or_date)
    rule = next(c for c in grid.cells if not c.number_or_date)
    assert "if none is natural, ask about the rule instead" in hp.concept_packet(figure, forbidden)
    assert "a rule or a decision, not a figure" in hp.concept_packet(rule, forbidden)
    assert '"figure_or_date"' in hp.concept_packet(rule, forbidden)


def test_the_author_packets_open_for_their_channel(grid, forbidden):
    cell = grid.cells[0]
    headless = hp.concept_packet(cell, forbidden)
    assert headless.startswith(hp.HEADLESS_OPENING) and "this file" not in headless
    on_file = hp.writer_packet(cell, "G01-a1", "A concept.", ["a fact"], forbidden, hp.FILE_OPENING)
    assert on_file.startswith(hp.FILE_OPENING)


def test_judge_d_sees_every_candidate_and_no_label():
    ids = ids_for("DORA:19.1", "DORA:19.2", "answerability")
    packet = hp.judge_d_packet(
        _corpus(), _need(), ["DORA:19.1", "DORA:19.2"], {}, ["DORA:19.2", "answerability"], ids
    )
    items = packet.split("\nITEMS TO GRADE\n")[1].split("ALL THE CANDIDATES")[0]
    assert f"ITEM {ids['DORA:19.2']}: the candidate {ids['DORA:19.2']} below" in items
    assert f"ITEM {ids['answerability']}: the answerability" in items
    assert "DORA:19" not in items  # an item names its candidate by id only
    candidates = packet.split("ALL THE CANDIDATES (document order)")[1].split("OUTPUT")[0]
    assert f"CANDIDATE {ids['DORA:19.1']}\n[DORA:19.1]" in candidates
    assert f"CANDIDATE {ids['DORA:19.2']}\n[DORA:19.2]" in candidates
    assert not re.search(r"grade [0-3]|Rationale|judge [ABC]\b", candidates)
    assert "$" not in packet and packet.startswith(hp.HEADLESS_OPENING)


def test_the_diagnostic_judge_gets_judge_bs_packet():
    ids = ids_for("DORA:19.1")
    messages = hp.diagnostic_messages(_corpus(), _need(), ["DORA:19.1"], {}, ids)
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].startswith(hp.DIAGNOSTIC_OPENING)
    same = hp.judge_b_packet(
        _corpus(), _need(), ["DORA:19.1"], {}, ids, opening=hp.DIAGNOSTIC_OPENING
    )
    assert messages[0]["content"] == same


def test_every_attempt_gets_frame_a_and_only_its_ref_changes(grid, forbidden):
    cell = grid.cells[0]
    packets = [
        hp.concept_packet(cell, forbidden, ref) for ref in ("R-aaaaaaaaaaaa", "R-bbbbbbbbbbbb")
    ]
    assert hp.attempt_frame() in packets[0] and hp.attempt_frame() in packets[1]
    assert packets[0].replace("R-aaaaaaaaaaaa", "R-bbbbbbbbbbbb") == packets[1]
    assert cell.cell_id not in packets[0]


def test_only_frame_a_is_part_of_the_protocol(tmp_path):
    frames = json.loads(hp.ATTEMPT_FRAMES.read_text(encoding="utf-8"))["frames"]
    assert sorted(frames) == ["A"]
    banned = re.compile(
        r"fail|reject|drop|previous|earlier|again|overlap|DEV|retriev|rank|embed|benchmark|E5",
        re.I,
    )
    assert not banned.search(frames["A"])
    old = hp.TEMPLATES / "attempt_frames.v1.json"
    with pytest.raises(hp.PacketError, match="only frame A"):
        hp.attempt_frame(old)


def test_judge_a_v2_reads_through_the_broker_and_keeps_devs_text(grid):
    from ingest import holdout_broker as hb

    need = _need()
    packet = hp.judge_a_broker_packet(need, grid.cells[0])
    assert "$" not in packet and packet.startswith(hp.HEADLESS_OPENING)
    assert "Bash" not in packet and "Write tool" not in packet and "search_log" not in packet
    assert f"at most {hb.MAX_REQUESTS} requests" in packet and "8,000 characters" in packet
    v1 = hp.judge_a_packet(need, grid.cells[0], hp.Path("/OUT/N101.json"), hp.Path("/REPO"))
    keep = packet.split("HOW TO SEARCH")[1].split("THE RULES")[0]
    assert keep == v1.split("HOW TO SEARCH")[1].split("THE RULES")[0]
    assert hp.judge_a_rules() in packet and hp.judge_a_rules_dev() in v1
