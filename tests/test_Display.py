import json
import unittest
from unittest.mock import patch

from src.Display import (_extract_model_name, _get_value_latency,
                         build_output_object, print_results)
from src.Metrics import MetricResult


class TestDisplayHelpers(unittest.TestCase):
    def test_extract_model_name(self) -> None:
        url = "https://huggingface.co/org/model/tree/main"
        self.assertEqual(_extract_model_name(url), "model")

    def test_get_value_latency_handles_scalar(self) -> None:
        res = MetricResult(metric="Ramp", key="ramp", value=1.5,
                           latency_ms=12)
        value, latency = _get_value_latency({"ramp": res}, "ramp")
        self.assertEqual(value, 1.0)  # clamped
        self.assertEqual(latency, 12)

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
        expected_net = sum(
            [0.8, 0.9, size_avg, 0.7, 0.6, 0.5, 0.4, 0.3]
        ) / 8
        self.assertAlmostEqual(out["net_score"], expected_net)
        expected_latency = 10 + 5 + 8 + 4 + 6 + 7 + 9 + 3
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


if __name__ == "__main__":
    unittest.main()
