# tests/DispatcherTester.py
import math
import unittest
from typing import Any, Dict

from src.Dispatcher import Dispatcher
from src.Metrics import Metric


class _StubMetric(Metric):
    """Simple metric stub that records its inputs."""

    def __init__(self, name: str, key: str, result: float) -> None:
        self.name = name
        self.key = key
        self.result = result
        self.calls: list[Dict[str, Any]] = []

    def compute(self, inputs: Dict[str, Any], **_: Any) -> float:
        self.calls.append(inputs)
        return self.result


class _FailingMetric(Metric):
    name = "Fail"
    key = "fail_metric"

    def compute(self, inputs: Dict[str, Any], **kwargs: Any) -> float:
        raise ValueError("explode")


class TestDispatcher(unittest.TestCase):
    """Test that the Dispatcher runs metrics and aggregates results."""

    def test_dispatch_no_metrics_returns_empty_results(self) -> None:
        dispatcher = Dispatcher()

        results = dispatcher.dispatch({})

        self.assertEqual(results, [])

    def test_dispatch_runs_all_metrics_and_preserves_order(self) -> None:
        inputs = {"model_url": "https://example.com/model"}
        metric_a = _StubMetric("A", "metric_a", 0.3)
        metric_b = _StubMetric("B", "metric_b", 0.7)
        dispatcher = Dispatcher([metric_a, metric_b])

        results = dispatcher.dispatch(inputs)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].metric, "A")
        self.assertEqual(results[0].key, "metric_a")
        self.assertEqual(results[0].value, 0.3)
        self.assertIsNone(results[0].error)
        self.assertGreaterEqual(results[0].latency_ms, 0.0)

        self.assertEqual(results[1].metric, "B")
        self.assertEqual(results[1].key, "metric_b")
        self.assertEqual(results[1].value, 0.7)
        self.assertIsNone(results[1].error)
        self.assertGreaterEqual(results[1].latency_ms, 0.0)

        self.assertEqual(metric_a.calls[0], inputs)
        self.assertEqual(metric_b.calls[0], inputs)

    def test_dispatch_handles_metric_exception(self) -> None:
        dispatcher = Dispatcher([_FailingMetric()])

        results = dispatcher.dispatch({})

        self.assertEqual(len(results), 1)
        self.assertTrue(math.isnan(results[0].value))
        self.assertIn("ValueError", results[0].error or "")
        self.assertGreaterEqual(results[0].latency_ms, 0.0)

    def test_add_clear_and_metrics_property(self) -> None:
        metric = _StubMetric("Name", "key", 1.0)
        dispatcher = Dispatcher()

        dispatcher.add_metric(metric)
        registered = dispatcher.metrics

        self.assertEqual(len(registered), 1)
        self.assertIs(registered[0], metric)

        registered.append(_StubMetric("Other", "other", 0.5))
        self.assertEqual(len(dispatcher.metrics), 1)

        dispatcher.clear_metrics()
        self.assertEqual(dispatcher.metrics, [])


if __name__ == "__main__":
    unittest.main()
