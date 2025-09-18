# tests/MetricsTester.py
import math
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

from src.Metrics import (LicenseMetric, Metric, MetricResult, RampUpTime,
                         SizeMetric)


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


class TestSizeMetric(unittest.TestCase):
    """Test the SizeMetric implementation without hitting the HF API."""

    @patch("src.Metrics.HFClient")
    def test_compute_requires_model_url(
        self,
        mock_hf_client_cls: MagicMock,
    ) -> None:
        metric = SizeMetric()
        with self.assertRaises(ValueError):
            metric.compute({})

    @patch("src.Metrics.browse_hf_repo")
    @patch("src.Metrics.HFClient")
    def test_compute_uses_safetensors_parameters(
        self,
        mock_hf_client_cls: MagicMock,
        mock_browse: MagicMock,
    ) -> None:
        mock_client = mock_hf_client_cls.return_value
        mock_client.request.return_value = {
            "safetensors": {
                "parameters": {
                    "float16": 50,
                    "float32": 100,
                }
            }
        }

        metric = SizeMetric()
        inputs = {"model_url": "https://huggingface.co/acme/model"}
        score = metric.compute(inputs)

        self.assertEqual(score, 1.0)
        mock_client.request.assert_called_once_with(
            "GET",
            "/api/models/acme/model",
        )
        mock_browse.assert_not_called()

    @patch("src.Metrics.browse_hf_repo")
    @patch("src.Metrics.HFClient")
    def test_compute_falls_back_to_repo_when_no_safetensors(
        self,
        mock_hf_client_cls: MagicMock,
        mock_browse: MagicMock,
    ) -> None:
        mock_client = mock_hf_client_cls.return_value
        mock_client.request.return_value = {}
        mock_browse.return_value = []

        metric = SizeMetric()
        inputs = {"model_url": "https://huggingface.co/acme/empty"}
        score = metric.compute(inputs)

        self.assertEqual(score, 0.0)
        mock_client.request.assert_called_once_with(
            "GET",
            "/api/models/acme/empty",
        )
        mock_browse.assert_called_once_with(
            mock_client,
            "acme/empty",
            repo_type="model",
            revision="main",
            recursive=True,
        )

    @patch("src.Metrics.browse_hf_repo")
    @patch("src.Metrics.HFClient")
    def test_compute_averages_model_file_sizes(
        self,
        mock_hf_client_cls: MagicMock,
        mock_browse: MagicMock,
    ) -> None:
        mock_client = mock_hf_client_cls.return_value
        mock_client.request.return_value = {}
        mock_browse.return_value = [
            ("weights/model.bin", 1000),
            ("weights/model.pt", 3000),
            ("README.md", 10),
        ]

        metric = SizeMetric()
        inputs = {"model_url": "https://huggingface.co/acme/medium"}
        score = metric.compute(inputs)

        expected_bits = 8 * (1000 + 3000) / 2
        denom = 1 - math.log(1.2e10 / SizeMetric.maxModelBits)
        expected_score = 1 - math.log(expected_bits / SizeMetric.maxModelBits)
        expected_score /= denom
        expected_score = min(max(expected_score, 0.0), 1.0)

        self.assertAlmostEqual(score, expected_score)
        mock_client.request.assert_called_once_with(
            "GET",
            "/api/models/acme/medium",
        )
        mock_browse.assert_called_once()


class TestRampUpTimeMetric(unittest.TestCase):
    """Test the RampUpTime metric without calling external services."""

    @patch("src.Metrics.GrokClient")
    @patch("src.Metrics.HFClient")
    def test_requires_model_url(
        self,
        mock_hf_client_cls: MagicMock,
        mock_grok_client_cls: MagicMock,
    ) -> None:
        metric = RampUpTime()
        with self.assertRaises(ValueError):
            metric.compute({})

    @patch("src.Metrics.injectHFBrowser")
    @patch.object(RampUpTime, "_extract_usage_section")
    @patch("src.Metrics.GrokClient")
    @patch("src.Metrics.HFClient")
    def test_compute_uses_usage_excerpt(
        self,
        _mock_hf_client_cls: MagicMock,
        _mock_grok_client_cls: MagicMock,
        mock_extract: MagicMock,
        mock_inject: MagicMock,
    ) -> None:
        metric = RampUpTime()
        mock_inject.return_value = "full page text"
        mock_extract.return_value = "Example usage instructions"

        url = "https://huggingface.co/acme/model"
        score = metric.compute({"model_url": url})

        mock_inject.assert_called_once_with(url)
        mock_extract.assert_called_once_with("full page text")

        char_count = len("Example usage instructions")
        expected = 1.0 / (1.0 + math.log1p(char_count / 500))
        expected = max(0.0, min(expected, 1.0))
        self.assertAlmostEqual(score, expected)

    @patch("builtins.print")
    @patch("src.Metrics.injectHFBrowser")
    @patch.object(RampUpTime, "_extract_usage_section")
    @patch("src.Metrics.GrokClient")
    @patch("src.Metrics.HFClient")
    def test_compute_handles_missing_usage_excerpt(
        self,
        _mock_hf_client_cls: MagicMock,
        _mock_grok_client_cls: MagicMock,
        mock_extract: MagicMock,
        mock_inject: MagicMock,
        mock_print: MagicMock,
    ) -> None:
        metric = RampUpTime()
        mock_inject.return_value = "full page text"
        mock_extract.return_value = None

        input = {"model_url": "https://huggingface.co/acme/model"}
        score = metric.compute(input)

        mock_inject.assert_called_once()
        mock_extract.assert_called_once_with("full page text")
        mock_print.assert_called()
        self.assertEqual(score, 1.0)


class TestLicenseMetric(unittest.TestCase):
    """Test the LicenseMetric behavior under different license sources."""

    @patch("src.Metrics.GrokClient")
    @patch("src.Metrics.HFClient")
    def test_requires_model_url(
        self,
        mock_hf_client_cls: MagicMock,
        mock_grok_client_cls: MagicMock,
    ) -> None:
        metric = LicenseMetric()
        with self.assertRaises(ValueError):
            metric.compute({})

    @patch("src.Metrics.GrokClient")
    @patch("src.Metrics.HFClient")
    def test_known_license_uses_lookup(
        self,
        mock_hf_client_cls: MagicMock,
        mock_grok_client_cls: MagicMock,
    ) -> None:
        mock_client = mock_hf_client_cls.return_value
        mock_client.request.return_value = {
            "cardData": {"license": "apache-2.0"},
        }

        metric = LicenseMetric()
        inputs = {"model_url": "https://huggingface.co/acme/model"}
        score = metric.compute(inputs)

        self.assertEqual(score, 1.0)
        mock_client.request.assert_called_once_with(
            "GET",
            "/api/models/acme/model",
        )
        mock_grok_client_cls.return_value.llm.assert_not_called()

    @patch("src.Metrics.GrokClient")
    @patch("src.Metrics.HFClient")
    def test_missing_license_defaults_to_zero(
        self,
        mock_hf_client_cls: MagicMock,
        mock_grok_client_cls: MagicMock,
    ) -> None:
        mock_client = mock_hf_client_cls.return_value
        mock_client.request.return_value = {"cardData": {}}

        metric = LicenseMetric()
        inputs = {"model_url": "https://huggingface.co/acme/model"}
        score = metric.compute(inputs)

        self.assertEqual(score, 0.0)
        mock_grok_client_cls.return_value.llm.assert_not_called()

    @patch("src.Metrics.GrokClient")
    @patch("src.Metrics.HFClient")
    def test_unknown_license_asks_grok(
        self,
        mock_hf_client_cls: MagicMock,
        mock_grok_client_cls: MagicMock,
    ) -> None:
        mock_client = mock_hf_client_cls.return_value
        mock_client.request.return_value = {
            "cardData": {"license": "mystery-license"},
        }

        mock_grok = mock_grok_client_cls.return_value
        mock_grok.llm.side_effect = [
            "The model seems moderately permissive, score 0.75",
            "0.75",
        ]

        metric = LicenseMetric()
        inputs = {"model_url": "https://huggingface.co/acme/model"}
        score = metric.compute(inputs)

        self.assertEqual(score, 0.75)
        self.assertEqual(mock_grok.llm.call_count, 2)


if __name__ == "__main__":
    unittest.main()
