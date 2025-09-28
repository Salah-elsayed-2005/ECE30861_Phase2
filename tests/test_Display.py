import json
import unittest
from typing import cast
from unittest.mock import patch

from src.Display import (_extract_model_name, _get_value_latency,
                         _results_by_key, build_output_object, print_results)
from src.Metrics import MetricResult


class TestDisplayHelpers(unittest.TestCase):
    def test_extract_model_name(self) -> None:
        url = "https://huggingface.co/org/model/tree/main"
        self.assertEqual(_extract_model_name(url), "model")

    def test_extract_model_name_handles_root_path(self) -> None:
        self.assertEqual(_extract_model_name("https://huggingface.co/"), "")
        self.assertEqual(
            _extract_model_name("https://huggingface.co/model"),
            "model",
        )

    def test_get_value_latency_handles_scalar(self) -> None:
        res = MetricResult(metric="Ramp", key="ramp", value=1.5,
                           latency_ms=12)
        value, latency = _get_value_latency({"ramp": res}, "ramp")
        self.assertEqual(value, 1.0)  # clamped
        self.assertEqual(latency, 12)

    def test_get_value_latency_defaults_when_missing(self) -> None:
        value, latency = _get_value_latency({}, "unknown")
        self.assertEqual(value, 0.0)
        self.assertEqual(latency, 0)

    def test_build_output_object_aggregates_metrics(self) -> None:
        results = [
            MetricResult("Ramp", "ramp_up_time", 0.8, 10),
            MetricResult("License", "license_metric", 0.9, 5),
            MetricResult(
                "Size",
                "size_metric",
                {
                    "raspberry_pi": 0.4,
                    "jetson_nano": 0.5,
                    "desktop_pc": 0.6,
                    "aws_server": 0.7,
                },
                8,
            ),
            MetricResult("Availability", "availability_metric", 0.7, 4),
            MetricResult("Dataset", "dataset_quality", 0.6, 6),
            MetricResult("Code", "code_quality", 0.5, 7),
            MetricResult("Performance", "performance_claims", 0.4, 9),
            MetricResult("Bus", "bus_factor", 0.3, 3),
        ]
        group = {"model_url": "https://huggingface.co/org/model"}

        out = build_output_object(group, results)

        self.assertEqual(out["name"], "model")
        size_avg = (0.4 + 0.5 + 0.6 + 0.7) / 4
        expected_net = sum([
            0.2 * 0.8,
            0.15 * 0.9,
            0.1 * size_avg,
            0.1 * 0.7,
            0.1 * 0.6,
            0.15 * 0.5,
            0.1 * 0.4,
            0.1 * 0.3,
        ])
        self.assertAlmostEqual(out["net_score"], expected_net)
        expected_latency = max([10, 5, 8, 4, 6, 7, 9, 3])
        self.assertEqual(out["net_score_latency"], expected_latency)
        self.assertEqual(out["ramp_up_time"], 0.8)
        self.assertEqual(out["size_score"]["jetson_nano"], 0.5)

    def test_print_results_emits_json(self) -> None:
        results = [MetricResult("Ramp", "ramp_up_time", 0.5, 1)]
        group = {"model_url": "https://huggingface.co/demo/model"}

        with patch("builtins.print") as mock_print:
            print_results(group, results)

        mock_print.assert_called_once()
        payload = json.loads(mock_print.call_args.args[0])
        self.assertEqual(payload["category"], "MODEL")

    def test_results_by_key_maps_results(self) -> None:
        results = [
            MetricResult("Ramp", "ramp", 0.4, 1),
            MetricResult("Size", "size", 0.5, 2),
        ]

        mapping = _results_by_key(results)

        self.assertEqual(mapping["ramp"].metric, "Ramp")
        self.assertEqual(mapping["size"].value, 0.5)

    def test_get_value_latency_normalizes_mapping_and_latency(self) -> None:
        res = MetricResult(
            "Size",
            "size_metric",
            {
                "raspberry_pi": -0.5,
                "jetson_nano": float("nan"),
                "desktop_pc": 1.2,
                "aws_server": "0.7",
            },
            cast(float, "15"),
        )

        values, latency = _get_value_latency(
            {"size_metric": res},
            "size_metric",
        )

        self.assertEqual(latency, 15)
        self.assertEqual(
            values,
            {
                "raspberry_pi": 0.0,
                "jetson_nano": 0.0,
                "desktop_pc": 1.0,
                "aws_server": 0.7,
            },
        )

    def test_build_output_object_handles_scalar_size_metric(self) -> None:
        results = [
            MetricResult("Ramp", "ramp_up_time", 0.9, 10),
            MetricResult("Size", "size_metric", 0.42, cast(float, "11")),
        ]

        out = build_output_object({}, results)

        self.assertEqual(out["name"], "")
        self.assertEqual(
            out["size_score"],
            {
                "raspberry_pi": 0.42,
                "jetson_nano": 0.42,
                "desktop_pc": 0.42,
                "aws_server": 0.42,
            },
        )
        self.assertEqual(out["size_score_latency"], 11)
        self.assertEqual(out["net_score_latency"], 11)
        expected_net = (0.2 * 0.9) + (0.1 * 0.42)
        self.assertAlmostEqual(out["net_score"], expected_net)
        self.assertEqual(out["bus_factor"], 0.0)
        self.assertEqual(out["performance_claims"], 0.0)


if __name__ == "__main__":
    unittest.main()
