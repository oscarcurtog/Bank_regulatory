"""Source-only operational approval tests; stdlib, no model or private TEST data.

Run directly with python -m unittest tests.test_holdout_v2_operational_amendment_stdlib.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ingest import holdout_v2_operational_amendment as amendment


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="fundamento-ops-approval-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.original = {}
        for path in sorted(amendment.REPLACEMENTS):
            self.original[path] = self.write(path, f"original {path}\n".encode())
        self.other = self.write("ingest/scientific.py", b"unchanged scientific code\n")
        protocol = self.write("config/holdout/test-v2-protocol.md", b"frozen protocol\n")
        plan = self.write("config/holdout/plan.md", b"frozen analysis\n")
        template = self.write("data/templates/judge.md", b"frozen judge prompt\n")
        dense = self.write("config/dense.json", b"frozen dense config\n")
        pool = self.write("config/pool.json", b"frozen pool config\n")
        inputs = self.write("data/input.json", b"frozen input\n")
        self.contract: dict[str, Any] = {
            "schema": "holdout-v2-contract.v1",
            "code": {
                "files_sha256": {**self.original, "ingest/scientific.py": self.other},
                "inputs_sha256": {"data/input.json": inputs},
                "protocol": {
                    "file": "config/holdout/test-v2-protocol.md",
                    "sha256": protocol,
                },
            },
            "evaluation": {"plan_file": "config/holdout/plan.md", "plan_sha256": plan},
            "retrieval": {
                "code_sha256": {"ingest/scientific.py": self.other},
                "pool_only": {"pool": {"config_file": "config/pool.json", "config_sha256": pool}},
                "systems": {
                    "dense": {
                        "nested": {
                            "config_file": "config/dense.json",
                            "config_sha256": dense,
                        }
                    }
                },
            },
            "roles": {
                "templates_sha256": {"judge.v1": template},
                "template_files": {"judge.v1": "data/templates/judge.md"},
            },
        }
        raw = json.dumps(self.contract).encode()
        self.base = self.repo / amendment.BASE_FILE
        self.base_sha = self.write(amendment.BASE_FILE, raw)
        for path in amendment.REPLACEMENTS:
            self.write(path, f"reviewed operational replacement {path}\n".encode())
        for path in amendment.NEW_FILES:
            self.write(path, f"reviewed new module {path}\n".encode())
        self.manifest = amendment.build_manifest(self.repo, self.base)
        self.path = self.repo / "reviewed-operations.json"
        self.approval = amendment.write_exclusive(self.path, self.manifest)

    def write(self, path: str, data: bytes) -> str:
        output = self.repo / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        return digest(data)

    def load(self) -> amendment.Amendment:
        return amendment.load_authorized(self.path, self.approval, self.base_sha, self.repo)

    def replace_manifest(self, manifest: dict) -> None:
        self.path.write_bytes(amendment.canonical(manifest))
        self.approval = digest(self.path.read_bytes())

    def test_exact_approval_accepts_only_named_changes_and_preserves_base(self) -> None:
        before = self.base.read_bytes()
        approved = self.load()
        approved.validate(self.repo, self.contract)
        for path, hashes in self.manifest["replacements"].items():
            self.assertTrue(
                approved.permitted_change(path, hashes["before_sha256"], hashes["after_sha256"])
            )
        self.assertFalse(approved.permitted_change("ingest/scientific.py", self.other, "a" * 64))
        self.assertEqual(self.base.read_bytes(), before)

    def test_context_is_explicit_and_resets_even_after_an_exception(self) -> None:
        self.assertIsNone(amendment.get_current())
        approved = self.load()
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with amendment.authorized(approved):
                self.assertIs(amendment.get_current(), approved)
                raise RuntimeError("synthetic")
        self.assertIsNone(amendment.get_current())

    def test_manifest_presence_is_not_approval(self) -> None:
        for value in ("", "a" * 63, "a" * 64):
            with self.subTest(value=value), self.assertRaises(amendment.AmendmentError):
                amendment.load_authorized(self.path, value, self.base_sha, self.repo)

    def test_missing_or_different_base_pin_is_refused(self) -> None:
        for value in ("", "0" * 64):
            with self.subTest(value=value), self.assertRaises(amendment.AmendmentError):
                amendment.load_authorized(self.path, self.approval, value, self.repo)

    def test_changed_manifest_does_not_reuse_old_approval(self) -> None:
        self.path.write_bytes(self.path.read_bytes() + b" ")
        with self.assertRaises(amendment.AmendmentError):
            self.load()

    def test_no_existing_manifest_is_overwritten(self) -> None:
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            amendment.write_exclusive(self.path, self.manifest)
        self.assertEqual(self.path.read_bytes(), before)

    def test_unexpected_fields_and_sources_are_refused(self) -> None:
        for change in ("field", "source", "new_module", "path_traversal"):
            value = json.loads(json.dumps(self.manifest))
            if change == "field":
                value["allow_anything"] = True
            elif change == "source":
                value["replacements"]["ingest/scientific.py"] = {
                    "before_sha256": self.other,
                    "after_sha256": self.other,
                }
            elif change == "new_module":
                value["new_files"]["ingest/other.py"] = "0" * 64
            else:
                first = next(iter(value["replacements"]))
                value["replacements"]["../escape.py"] = value["replacements"].pop(first)
            self.replace_manifest(value)
            with self.subTest(change=change), self.assertRaises(amendment.AmendmentError):
                self.load()

    def test_each_new_module_is_rechecked_after_initial_approval(self) -> None:
        for path in amendment.NEW_FILES:
            with self.subTest(path=path):
                before = (self.repo / path).read_bytes()
                approved = self.load()
                (self.repo / path).write_bytes(before + b"changed")
                with self.assertRaises(amendment.AmendmentError):
                    approved.validate(self.repo, self.contract)
                (self.repo / path).write_bytes(before)

    def test_frozen_nonoperational_code_and_inputs_stay_exact(self) -> None:
        for path in (
            "ingest/scientific.py",
            "config/holdout/test-v2-protocol.md",
            "config/holdout/plan.md",
            "data/templates/judge.md",
            "config/dense.json",
            "config/pool.json",
            "data/input.json",
        ):
            with self.subTest(path=path):
                before = (self.repo / path).read_bytes()
                (self.repo / path).write_bytes(before + b"changed")
                with self.assertRaises(amendment.AmendmentError):
                    self.load()
                (self.repo / path).write_bytes(before)

    def test_changed_base_is_refused_even_with_same_semantics(self) -> None:
        self.base.write_bytes(self.base.read_bytes() + b"\n")
        with self.assertRaises(amendment.AmendmentError):
            self.load()

    def test_false_old_source_hash_is_refused(self) -> None:
        value = json.loads(json.dumps(self.manifest))
        path = next(iter(value["replacements"]))
        value["replacements"][path]["before_sha256"] = "0" * 64
        self.replace_manifest(value)
        with self.assertRaises(amendment.AmendmentError):
            self.load()

    def test_missing_source_pin_cannot_be_prepared(self) -> None:
        del self.contract["code"]["files_sha256"][next(iter(amendment.REPLACEMENTS))]
        self.base.write_text(json.dumps(self.contract))
        with self.assertRaises(amendment.AmendmentError):
            amendment.build_manifest(self.repo, self.base)

    def test_missing_frozen_input_is_refused(self) -> None:
        (self.repo / "data/input.json").unlink()
        with self.assertRaises(amendment.AmendmentError):
            self.load()

    def test_conflicting_base_hashes_are_refused(self) -> None:
        self.contract["retrieval"]["code_sha256"]["ingest/scientific.py"] = "0" * 64
        self.base.write_text(json.dumps(self.contract))
        self.base_sha = digest(self.base.read_bytes())
        value = json.loads(json.dumps(self.manifest))
        value["base_contract_sha256"] = self.base_sha
        self.replace_manifest(value)
        with self.assertRaises(amendment.AmendmentError):
            self.load()

    def test_symlink_outside_repository_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "source.py"
            path = self.repo / next(iter(amendment.REPLACEMENTS))
            target.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(target)
            with self.assertRaises(amendment.AmendmentError):
                self.load()

    def test_duplicate_json_keys_are_not_silently_discarded(self) -> None:
        raw = self.path.read_bytes()
        raw = raw[:-2] + b',"schema":"holdout-v2-operational-amendment.v1"}\n'
        self.path.write_bytes(raw)
        self.approval = digest(raw)
        with self.assertRaises(amendment.AmendmentError):
            self.load()


if __name__ == "__main__":
    unittest.main()
