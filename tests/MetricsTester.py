import unittest
from dataclasses import FrozenInstanceError

from src.Metrics import Metric, MetricResult


class TestMetricResult(unittest.TestCase):
    """
    Test that the MetricResult class works properly in terms of
    construction and immutablility
    """
    def test_construction_and_defaults(self) -> None:
        res = MetricResult(
            metric="License Check",
            key="license",
            value=True,
            latency_ms=12.3,
        )
        self.assertEqual(res.metric, "License Check")
        self.assertEqual(res.key, "license")
        self.assertTrue(res.value)
        self.assertEqual(res.latency_ms, 12.3)
        self.assertIsNone(res.details)
        self.assertIsNone(res.error)

    def test_frozen_immutability(self) -> None:
        res = MetricResult(metric="m", key="k", value=1, latency_ms=0.1)
        with self.assertRaises(FrozenInstanceError):
            res.value = 2  # type: ignore[misc]

    def test_equality_and_hash(self) -> None:
        a = MetricResult(metric="m", key="k", value=1, latency_ms=1.0)
        b = MetricResult(metric="m", key="k", value=1, latency_ms=1.0)
        c = MetricResult(metric="m", key="k", value=2, latency_ms=1.0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        # Hashable since frozen=True
        s = {a, b, c}
        self.assertEqual(len(s), 2)


class TestMetricABC(unittest.TestCase):
    """
    Test that the Metric abstract class works properly
    """
    def test_metric_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            Metric()  # type: ignore[abstract]

    def test_concrete_metric_subclass(self) -> None:
        class HalfMetric(Metric):
            def compute(self, inputs: dict[str, object], **_: object) -> float:
                # Return 0.5 regardless of inputs
                return 0.5

        m = HalfMetric()
        score = m.compute({})
        self.assertIsInstance(score, float)
        self.assertEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
