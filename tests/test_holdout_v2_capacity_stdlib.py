"""Provider capacity classification against synthetic CLI events and real engine AST.

No repository runtime, model, scientific packet or private TEST state is imported.
The existing operational tests provide an in-memory event store and inert audit
boundary; classification, executor dispatch, retries and ledger replay stay real.
"""

from __future__ import annotations
import __future__

import ast
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "capacity_engine_test_boundaries", ROOT / "tests/test_holdout_v2_engine_ops_stdlib.py"
)
assert spec is not None and spec.loader is not None
boundary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = boundary
spec.loader.exec_module(boundary)
en: Any = boundary.en


def result(text: str = "", **fields: Any) -> str:
    return json.dumps({"type": "result", "is_error": True, "result": text, **fields})


SIGNATURES = (
    "You've hit your session limit · resets 1:00am (Europe/Madrid)",
    "You've hit your session limit · resets 1:00pm (Europe/Madrid)",
    "You've hit your weekly limit · resets Oct 31 at 1:00pm (Europe/Madrid)",
    "You've hit your Opus limit · resets Oct 31 at 1:00pm (Europe/Madrid)",
    "You've hit your Sonnet limit · resets Oct 31 at 1:00pm (Europe/Madrid)",
    "You've hit your Fable limit · resets Oct 31 at 1:00pm (Europe/Madrid)",
    "Usage credit limit reached",
    "You've hit your limit",
    "You're out of usage credits",
    "You're out of extra usage",
    "Your org is out of usage · add funds to continue",
    "Your org is out of usage · contact your admin",
    "You've hit your monthly spend limit",
    "You've hit your individual spend limit",
    "You've hit your org's monthly spend limit",
    "You've hit your team's shared budget",
    "You've hit your channel's monthly spend limit",
    "You've hit your individual monthly usage limit",
    "You've hit your org's monthly usage limit",
    "You've hit your channel's monthly usage limit",
    "Claude AI usage limit reached",
    "API Error: 429 rate_limit_error",
    "Rate limit exceeded",
    "API Error: 529 overloaded_error",
    "Overloaded",
    "Opus is experiencing high load, please use /model to switch to Sonnet",
    "Fable is experiencing high load, please use /model to switch to Sonnet",
    "Server is temporarily limiting requests (not your usage limit)",
    "Request rejected (429)",
    "The API is at capacity",
)

NON_CAPACITY = (
    "Your seat does not have extra usage enabled",
    "Your seat has no credits allocated",
    "Additional credits are required to use this model",
    "Fast mode has been disabled",
    "SDK max_budget_usd exceeded",
)


class BaseExecutor:
    """Only the certified executor boundary, never an executable CLI."""

    def __init__(self) -> None:
        self.stream = ""
        self.stderr = ""
        self.delegations: list[Any] = []

    def _answer(self, role: str, run: Any) -> Any:
        self.delegations.append((role, run))
        if run.returncode != 0 or run.answer is None:
            raise RuntimeError("synthetic ordinary API failure")
        return boundary.Session(json.dumps({"value": "accepted"}), {})

    def _spool(self, run_id: str) -> Any:
        return types.SimpleNamespace(exists=lambda: True)


def load_executor() -> Any:
    source = ROOT / "scripts/holdout_v2.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "V2RealExecutor"
    ]
    if len(nodes) != 1:
        raise AssertionError("V2RealExecutor definition is unavailable")
    module = types.ModuleType("capacity_executor_real_ast")
    module.__dict__.update(
        hx=types.SimpleNamespace(RealExecutor=BaseExecutor),
        en=en,
        hi=types.SimpleNamespace(),
        hpl=types.SimpleNamespace(TechnicalFailure=boundary.TechnicalFailure),
        hsp=types.SimpleNamespace(UNOBSERVED=-999),
    )
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        module.__dict__,
    )
    return module


executor_module = load_executor()


def run_event(stream: str, stderr: str = "", *, rc: int = 1, answer: Any = None) -> Any:
    return types.SimpleNamespace(stream=stream, stderr=stderr, returncode=rc, answer=answer)


class ClassificationTests(unittest.TestCase):
    def test_each_provider_text_signature_in_final_error(self) -> None:
        for text in SIGNATURES:
            with self.subTest(text=text):
                self.assertTrue(en.capacity_refusal(result(text, subtype="success"), ""))

    def test_structured_429_and_529_accept_integer_or_exact_string(self) -> None:
        for status in (429, "429", 529, "529"):
            with self.subTest(status=status):
                self.assertTrue(en.capacity_refusal(result(api_error_status=status), ""))

    def test_ordinary_errors_are_not_capacity(self) -> None:
        for status in (400, "400", 500, "500", 401, 403, None):
            with self.subTest(status=status):
                self.assertFalse(
                    en.capacity_refusal(result("ordinary API failure", api_error_status=status), "")
                )

    def test_permissions_allocations_and_local_budget_are_not_capacity_without_status(self) -> None:
        for text in NON_CAPACITY:
            with self.subTest(text=text):
                self.assertFalse(en.capacity_refusal(result(text), ""))
                self.assertFalse(en.capacity_refusal("", text))
                self.assertTrue(en.capacity_refusal(result(text, api_error_status=429), ""))

    def test_structured_status_requires_error_true_boolean(self) -> None:
        for is_error in (False, None, "true", 1, [], {}):
            with self.subTest(is_error=is_error):
                self.assertFalse(
                    en.capacity_refusal(
                        result(SIGNATURES[0], is_error=is_error, api_error_status=429), ""
                    )
                )

    def test_status_has_no_coercion_from_other_types_or_arbitrary_digits(self) -> None:
        for status in (429.0, 529.0, True, "429 rejected", " 429", "0529", [429], {"code": 529}):
            with self.subTest(status=status):
                self.assertFalse(en.capacity_refusal(result(api_error_status=status), ""))
        self.assertFalse(en.capacity_refusal(result("ordinary error identifier 429 or 529"), ""))
        self.assertFalse(en.capacity_refusal("", "ordinary identifier 429 or 529"))

    def test_packet_reply_request_and_non_result_events_do_not_classify(self) -> None:
        for event_type in ("assistant", "user", "request", "packet", "system", "metadata"):
            with self.subTest(event_type=event_type):
                event = {
                    "type": event_type,
                    "is_error": True,
                    "api_error_status": 429,
                    "result": SIGNATURES[0],
                    "content": SIGNATURES[-1],
                }
                self.assertFalse(en.capacity_refusal(json.dumps(event), ""))

    def test_success_result_and_unselected_metadata_do_not_classify(self) -> None:
        self.assertFalse(
            en.capacity_refusal(result(SIGNATURES[0], is_error=False, api_error_status=429), "")
        )
        self.assertFalse(
            en.capacity_refusal(
                result("ordinary error", metadata={"api_error_status": 429, "text": SIGNATURES[0]}),
                "",
            )
        )

    def test_only_final_result_decides_not_earlier_error(self) -> None:
        prior = result(SIGNATURES[0], api_error_status=429)
        for final in (result("ok", is_error=False), result("ordinary error", api_error_status=400)):
            with self.subTest(final=final):
                self.assertFalse(en.capacity_refusal(prior + "\n" + final, ""))

    def test_final_error_can_classify_after_earlier_unrelated_result(self) -> None:
        self.assertTrue(
            en.capacity_refusal(
                result("ordinary error", api_error_status=400)
                + "\n"
                + result(api_error_status=529),
                "",
            )
        )

    def test_error_fields_are_read_and_case_is_insensitive(self) -> None:
        for field in ("result", "error", "subtype"):
            with self.subTest(field=field):
                self.assertTrue(en.capacity_refusal(result(**{field: "SESSION LIMIT"}), ""))

    def test_only_last_nonblank_stderr_line_can_classify(self) -> None:
        for text in SIGNATURES:
            with self.subTest(text=text):
                self.assertFalse(en.capacity_refusal("", text + "\nordinary 400 failure\n\n"))
                self.assertTrue(en.capacity_refusal("", "earlier notice\n" + text + "\n \n"))

    def test_malformed_and_non_object_json_do_not_crash_or_contaminate(self) -> None:
        noise = "\n".join(("not JSON", "{broken", "[]", "null", '"session limit"', '{"type":'))
        self.assertFalse(en.capacity_refusal(noise, ""))
        self.assertTrue(en.capacity_refusal(noise + "\n" + result(api_error_status=429), ""))
        self.assertFalse(
            en.capacity_refusal(noise + "\n" + result("ordinary error", api_error_status=500), "")
        )


class ExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = executor_module.V2RealExecutor()
        executor_module.hsp.text = lambda spool, name: getattr(
            self.executor, "stream" if name == "stdout" else "stderr"
        )

    def test_normal_failure_uses_classifier_for_each_signature_and_status(self) -> None:
        streams = [result(text) for text in SIGNATURES] + [
            result(api_error_status=s) for s in (429, "429", 529, "529")
        ]
        for stream in streams:
            with self.subTest(stream=stream):
                with self.assertRaises(boundary.TechnicalFailure) as raised:
                    self.executor._answer("synthetic_role", run_event(stream))
                self.assertEqual(raised.exception.reason, en.CAPACITY)
        self.assertFalse(self.executor.delegations)

    def test_ordinary_failure_delegates_and_remains_an_error(self) -> None:
        for status in (400, 500):
            with self.subTest(status=status):
                with self.assertRaises(RuntimeError):
                    self.executor._answer(
                        "synthetic_role", run_event(result(api_error_status=status))
                    )
        self.assertEqual(len(self.executor.delegations), 2)

    def test_successful_call_with_limit_words_in_reply_stays_successful(self) -> None:
        answer = self.executor._answer(
            "synthetic_role", run_event(result(SIGNATURES[0], is_error=False), rc=0, answer="reply")
        )
        self.assertEqual(json.loads(answer.reply), {"value": "accepted"})
        self.assertEqual(len(self.executor.delegations), 1)

    def test_recover_failure_uses_same_classifier(self) -> None:
        for stream in [result(t) for t in SIGNATURES] + [
            result(api_error_status=s) for s in (429, "429", 529, "529")
        ]:
            with self.subTest(stream=stream):
                self.executor.stream = stream
                self.executor.stderr = ""
                self.assertEqual(
                    self.executor.recover_failure("synthetic_role", "synthetic#1"), en.CAPACITY
                )
        self.executor.stream = result("ordinary error", api_error_status=500)
        self.assertIsNone(self.executor.recover_failure("synthetic_role", "synthetic#1"))
        self.executor.stream = ""
        self.executor.stderr = "earlier notice\n" + SIGNATURES[0]
        self.assertEqual(
            self.executor.recover_failure("synthetic_role", "synthetic#1"), en.CAPACITY
        )

    def test_recovery_without_spool_does_not_invent_capacity(self) -> None:
        self.executor._spool = lambda run_id: types.SimpleNamespace(exists=lambda: False)
        self.assertIsNone(self.executor.recover_failure("synthetic_role", "synthetic#1"))


class EngineLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = boundary.Clock()
        self.store = boundary.Store(self.clock)
        self.job = boundary.job(1)

    def engine(self, executor: Any) -> Any:
        return boundary.Engine(
            self.store,
            executor,
            types.SimpleNamespace(claude_effective=None),
            workers=1,
            sleep=self.clock.sleep,
        )

    def executor(self, streams: list[str]) -> Any:
        adapter = executor_module.V2RealExecutor()
        adapter.calls = []
        pending = list(streams)

        def headless(role: str, run_id: str, packet: str) -> Any:
            adapter.calls.append(run_id)
            if pending:
                return adapter._answer(role, run_event(pending.pop(0)))
            return adapter._answer(
                role, run_event(result("ok", is_error=False), rc=0, answer="answer")
            )

        adapter.headless = headless
        adapter.recover_headless = lambda role, run_id, packet: None
        return adapter

    def test_capacity_refusals_beyond_max_runs_do_not_consume_attempts(self) -> None:
        self.assertEqual(en.MAX_RUNS, 3)
        adapter = self.executor([result(api_error_status=429)] * (en.MAX_RUNS + 1))
        engine = self.engine(adapter)
        outcome = engine.run([self.job])[self.job.job_id]
        self.assertFalse(outcome.exhausted)
        self.assertEqual(engine.ledger.failures(self.job.job_id), 0)
        self.assertEqual(len(adapter.calls), en.MAX_RUNS + 2)
        failures = [e for e in self.store.log if e.kind == "run_failed"]
        self.assertEqual([e.payload["reason"] for e in failures], [en.CAPACITY] * (en.MAX_RUNS + 1))
        self.assertFalse(engine.ledger.exhausted)

    def test_ordinary_400_and_500_errors_each_exhaust_at_three(self) -> None:
        for status in (400, 500):
            with self.subTest(status=status):
                self.store = boundary.Store(self.clock)
                adapter = self.executor([result("ordinary error", api_error_status=status)] * 4)
                engine = self.engine(adapter)
                outcome = engine.run([self.job])[self.job.job_id]
                self.assertTrue(outcome.exhausted)
                self.assertEqual(len(adapter.calls), 3)
                self.assertEqual(engine.ledger.failures(self.job.job_id), 3)
                self.assertEqual(
                    {e.payload["reason"] for e in self.store.log if e.kind == "run_failed"},
                    {"model_api"},
                )

    def test_historical_model_api_stays_counted_and_resume_keeps_two_attempts(self) -> None:
        initial = self.engine(self.executor([]))
        old_id = initial._start(self.job)
        initial._failed(old_id, "model_api", SIGNATURES[0])
        historical = tuple(self.store.log)
        adapter = self.executor([result("ordinary error", api_error_status=500)] * 3)
        resumed = self.engine(adapter)
        self.assertEqual(resumed.ledger.failures(self.job.job_id), 1)
        outcome = resumed.run([self.job])[self.job.job_id]
        self.assertTrue(outcome.exhausted)
        self.assertEqual(len(adapter.calls), 2)
        self.assertEqual(resumed.ledger.failures(self.job.job_id), 3)
        self.assertEqual(tuple(self.store.log[: len(historical)]), historical)
        folded = en.Ledger.fold(self.store.events())
        self.assertEqual(folded.failures(self.job.job_id), resumed.ledger.failures(self.job.job_id))
        self.assertEqual(folded.exhausted, resumed.ledger.exhausted)
        last_count = len(self.store.log)
        replay_adapter = self.executor([])
        self.assertTrue(self.engine(replay_adapter).run([self.job])[self.job.job_id].exhausted)
        self.assertFalse(replay_adapter.calls)
        self.assertEqual(len(self.store.log), last_count)

    def test_recovered_capacity_is_not_an_interruption_or_counted_failure(self) -> None:
        initial = self.engine(self.executor([]))
        initial._start(self.job)
        adapter = self.executor([])
        adapter.stream = result(api_error_status=529)
        adapter.stderr = ""
        executor_module.hsp.text = lambda spool, name: getattr(
            adapter, "stream" if name == "stdout" else "stderr"
        )
        resumed = self.engine(adapter)
        self.assertFalse(resumed.run([self.job])[self.job.job_id].exhausted)
        self.assertEqual(resumed.ledger.failures(self.job.job_id), 0)
        failures = [e.payload["reason"] for e in self.store.log if e.kind == "run_failed"]
        self.assertEqual(failures, [en.CAPACITY])
        count = len(self.store.log)
        replay_adapter = self.executor([])
        self.assertFalse(self.engine(replay_adapter).run([self.job])[self.job.job_id].exhausted)
        self.assertFalse(replay_adapter.calls)
        self.assertEqual(len(self.store.log), count)


if __name__ == "__main__":
    unittest.main()
