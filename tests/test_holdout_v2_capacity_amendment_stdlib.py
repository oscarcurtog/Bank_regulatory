"""Chained approval tests with source-only synthetic repositories, no private TEST."""

from __future__ import annotations

import copy
import unittest
from unittest import mock

from tests import test_holdout_v2_operational_amendment_stdlib as fixtures

amendment = fixtures.amendment


class CapacityApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = fixtures.ApprovalTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.repo = self.fixture.repo
        self.parent = copy.deepcopy(self.fixture.manifest)
        self.parent_raw = amendment.canonical(self.parent)
        self.parent_sha = fixtures.digest(self.parent_raw)
        self.fixture.write(amendment.PREDECESSOR_FILE, self.parent_raw)
        patcher = mock.patch.object(amendment, "PREDECESSOR_SHA256", self.parent_sha)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.v1_bytes = {
            p: (self.repo / p).read_bytes() for p in amendment.REPLACEMENTS | amendment.NEW_FILES
        }
        self.value = copy.deepcopy(self.parent)
        self.value.update(
            schema=amendment.SCHEMA_V2,
            purpose=amendment.PURPOSE_V2,
            scope=amendment.SCOPE_V2,
            predecessor={"file": amendment.PREDECESSOR_FILE, "sha256": self.parent_sha},
        )
        engine = "ingest/holdout_v2_engine.py"
        loader = "ingest/holdout_v2_operational_amendment.py"
        self.value["replacements"][engine]["after_sha256"] = self.fixture.write(
            engine, b"synthetic corrected capacity classifier\n"
        )
        self.value["new_files"][loader] = self.fixture.write(
            loader, b"synthetic chained approval loader\n"
        )
        self.path = self.repo / "capacity-approval.json"
        self.refresh()

    def refresh(self) -> None:
        self.path.write_bytes(amendment.canonical(self.value))
        self.sha = fixtures.digest(self.path.read_bytes())

    def load(self):
        return amendment.load_authorized(self.path, self.sha, self.fixture.base_sha, self.repo)

    def test_exact_chain_composes_base_and_keeps_parent_bytes(self) -> None:
        loaded = self.load()
        self.assertEqual(loaded.predecessor_sha256, self.parent_sha)
        loaded.validate(self.repo, self.fixture.contract)
        self.assertEqual((self.repo / amendment.PREDECESSOR_FILE).read_bytes(), self.parent_raw)
        for path, pair in self.value["replacements"].items():
            self.assertTrue(
                loaded.permitted_change(path, pair["before_sha256"], pair["after_sha256"])
            )

    def test_missing_or_modified_predecessor_fails(self) -> None:
        path = self.repo / amendment.PREDECESSOR_FILE
        path.write_bytes(self.parent_raw + b" ")
        with self.assertRaises(amendment.AmendmentError):
            self.load()
        path.unlink()
        with self.assertRaises(amendment.AmendmentError):
            self.load()

    def test_predecessor_rechecked_at_each_preflight(self) -> None:
        loaded = self.load()
        (self.repo / amendment.PREDECESSOR_FILE).write_bytes(self.parent_raw + b" ")
        with self.assertRaises(amendment.AmendmentError):
            loaded.validate(self.repo, self.fixture.contract)

    def test_other_parent_identity_or_path_rejected(self) -> None:
        for field, value in (("sha256", "0" * 64), ("file", "another.json")):
            old = self.value["predecessor"][field]
            self.value["predecessor"][field] = value
            self.refresh()
            with self.subTest(field=field), self.assertRaises(amendment.AmendmentError):
                self.load()
            self.value["predecessor"][field] = old

    def test_inherited_sources_and_operations_cannot_be_reauthorized(self) -> None:
        for path in amendment.REPLACEMENTS - {"ingest/holdout_v2_engine.py"}:
            old = self.value["replacements"][path]["after_sha256"]
            self.value["replacements"][path]["after_sha256"] = self.fixture.write(path, b"changed")
            self.refresh()
            with self.subTest(path=path), self.assertRaises(amendment.AmendmentError):
                self.load()
            self.value["replacements"][path]["after_sha256"] = old
            (self.repo / path).write_bytes(self.v1_bytes[path])
        path = "ingest/holdout_v2_operations.py"
        self.value["new_files"][path] = self.fixture.write(path, b"changed")
        self.refresh()
        with self.assertRaises(amendment.AmendmentError):
            self.load()

    def test_original_engine_identity_cannot_change(self) -> None:
        self.value["replacements"]["ingest/holdout_v2_engine.py"]["before_sha256"] = "0" * 64
        self.refresh()
        with self.assertRaises(amendment.AmendmentError):
            self.load()

    def test_both_new_replacements_are_required(self) -> None:
        for key, path in (
            ("replacements", "ingest/holdout_v2_engine.py"),
            ("new_files", "ingest/holdout_v2_operational_amendment.py"),
        ):
            old = self.value[key][path]
            self.value[key][path] = copy.deepcopy(self.parent[key][path])
            self.refresh()
            with self.subTest(path=path), self.assertRaises(amendment.AmendmentError):
                self.load()
            self.value[key][path] = old

    def test_v1_still_works_only_for_its_original_exact_bytes(self) -> None:
        with self.assertRaises(amendment.AmendmentError):
            amendment.load_authorized(
                self.repo / amendment.PREDECESSOR_FILE,
                self.parent_sha,
                self.fixture.base_sha,
                self.repo,
            )
        for path, raw in self.v1_bytes.items():
            (self.repo / path).write_bytes(raw)
        loaded = amendment.load_authorized(
            self.repo / amendment.PREDECESSOR_FILE,
            self.parent_sha,
            self.fixture.base_sha,
            self.repo,
        )
        self.assertIsNone(loaded.predecessor_sha256)

    def test_exact_hash_and_scope_remain_mandatory(self) -> None:
        with self.assertRaises(amendment.AmendmentError):
            amendment.load_authorized(self.path, "0" * 64, self.fixture.base_sha, self.repo)
        self.value["scope"] = "retroactively forgive model_api failures"
        self.refresh()
        with self.assertRaises(amendment.AmendmentError):
            self.load()

    def test_other_frozen_input_still_checked(self) -> None:
        self.fixture.write("data/input.json", b"changed synthetic input")
        with self.assertRaises(amendment.AmendmentError):
            self.load()


if __name__ == "__main__":
    unittest.main()
