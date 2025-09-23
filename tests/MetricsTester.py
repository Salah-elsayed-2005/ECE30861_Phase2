# tests/MetricsTester.py
import math
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

from src.Metrics import (AvailabilityMetric, CodeQuality, DatasetQuality,
                         LicenseMetric, Metric, MetricResult, RampUpTime,
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

    @patch("src.Metrics.PurdueClient")
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
    @patch("src.Metrics.PurdueClient")
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

    @patch("src.Metrics.injectHFBrowser")
    @patch.object(RampUpTime, "_extract_usage_section")
    @patch("src.Metrics.PurdueClient")
    @patch("src.Metrics.HFClient")
    def test_compute_handles_missing_usage_excerpt(
        self,
        _mock_hf_client_cls: MagicMock,
        _mock_grok_client_cls: MagicMock,
        mock_extract: MagicMock,
        mock_inject: MagicMock,
    ) -> None:
        metric = RampUpTime()
        mock_inject.return_value = "full page text"
        mock_extract.return_value = None

        input = {"model_url": "https://huggingface.co/acme/model"}
        score = metric.compute(input)
        mock_inject.assert_called_once()
        mock_extract.assert_called_once_with("full page text")
        self.assertEqual(score, 0.0)


class TestLicenseMetric(unittest.TestCase):
    """Test the LicenseMetric behavior under different license sources."""

    @patch("src.Metrics.PurdueClient")
    @patch("src.Metrics.HFClient")
    def test_requires_model_url(
        self,
        mock_hf_client_cls: MagicMock,
        mock_grok_client_cls: MagicMock,
    ) -> None:
        metric = LicenseMetric()
        with self.assertRaises(ValueError):
            metric.compute({})

    @patch("src.Metrics.PurdueClient")
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

    @patch("src.Metrics.PurdueClient")
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

    @patch("src.Metrics.PurdueClient")
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


class TestDatasetQualityMetric(unittest.TestCase):
    """
    Test the DatasetQuality metric without touching real Hugging Face APIs.
    """

    @patch("src.Metrics.HFClient")
    def test_compute_returns_zero_when_dataset_missing(
        self,
        _mock_hf_client_cls: MagicMock,
    ) -> None:
        metric = DatasetQuality()

        score = metric.compute({})

        self.assertEqual(score, 0.0)
        self.assertEqual(metric.last_details, {"reason": "dataset_not_found"})

    @patch.object(DatasetQuality, "count_models_for_dataset")
    @patch.object(DatasetQuality, "_fetch_dataset", return_value=None)
    @patch("src.Metrics.HFClient")
    def test_compute_handles_dataset_fetch_failure(
        self,
        _mock_hf_client_cls: MagicMock,
        mock_fetch_dataset: MagicMock,
        mock_count_models: MagicMock,
    ) -> None:
        metric = DatasetQuality()

        inputs = {"dataset_url": "https://huggingface.co/datasets/acme/data"}
        score = metric.compute(inputs)

        self.assertEqual(score, 0.0)
        self.assertEqual(
            metric.last_details,
            {"dataset_id": "acme/data", "reason": "hf_api_error"},
        )
        mock_fetch_dataset.assert_called_once_with("acme/data")
        mock_count_models.assert_not_called()

    @patch.object(DatasetQuality, "count_models_for_dataset", return_value=25)
    @patch.object(DatasetQuality, "_fetch_dataset")
    @patch("src.Metrics.HFClient")
    def test_compute_blends_reuse_and_likes(
        self,
        _mock_hf_client_cls: MagicMock,
        mock_fetch_dataset: MagicMock,
        mock_count_models: MagicMock,
    ) -> None:
        likes = 100
        mock_fetch_dataset.return_value = {"likes": likes}

        metric = DatasetQuality()
        score = metric.compute(
            {"dataset_url": "https://huggingface.co/datasets/acme/data"}
        )

        expected_likes = min(1.0, math.log1p(likes) / math.log1p(250))
        expected_use = min(1.0, math.log1p(25) / math.log1p(40))
        expected_score = 0.6 * expected_use + 0.4 * expected_likes
        expected_score = max(0.0, min(expected_score, 1.0))

        self.assertAlmostEqual(score, expected_score)

        details = metric.last_details
        self.assertEqual(details["dataset_id"], "acme/data")
        self.assertEqual(details["likes"], likes)
        self.assertEqual(details["use_count"], 25)
        self.assertAlmostEqual(details["likes_score"], expected_likes)
        self.assertAlmostEqual(details["use_score"], expected_use)

        mock_fetch_dataset.assert_called_once_with("acme/data")
        args, kwargs = mock_count_models.call_args
        self.assertEqual(args[0], "acme/data")
        self.assertEqual(kwargs, {})

    @patch.object(DatasetQuality, "_fetch_model")
    @patch("src.Metrics.HFClient")
    def test_extract_dataset_id_from_model_metadata(
        self,
        _mock_hf_client_cls: MagicMock,
        mock_fetch_model: MagicMock,
    ) -> None:
        mock_fetch_model.return_value = {
            "datasets": ["owner/data", "owner/other"],
        }

        metric = DatasetQuality()
        dataset_id = metric._extract_dataset_id(
            {"model_url": "https://huggingface.co/acme/model"}
        )

        self.assertEqual(dataset_id, "owner/data")
        mock_fetch_model.assert_called_once_with("acme/model")

    @patch("src.Metrics.HFClient")
    def test_count_models_for_dataset_dedupes_results(
        self,
        mock_hf_client_cls: MagicMock,
    ) -> None:
        metric = DatasetQuality()
        mock_client = mock_hf_client_cls.return_value
        mock_client.request.side_effect = [
            [{"modelId": "org/model-a"}, {"modelId": "org/model-b"}],
            [{"modelId": "org/model-b"}, {"modelId": None}],
            [{"modelId": "org/model-c"}],
            [{"modelId": "org/model-a"}],
        ]

        total = metric.count_models_for_dataset("owner/data", limit=10)

        self.assertEqual(total, 3)
        self.assertEqual(mock_client.request.call_count, 4)

        filters = {
            call.kwargs["params"]["filter"]
            for call in mock_client.request.call_args_list
        }
        self.assertSetEqual(
            filters,
            {"data", "dataset:data", "owner/data", "dataset:owner/data"},
        )
        for call in mock_client.request.call_args_list:
            self.assertEqual(call.args[:2], ("GET", "/api/models"))
            self.assertEqual(call.kwargs["params"]["limit"], 10)


class TestCodeQualityMetric(unittest.TestCase):
    """Test the CodeQuality metric using controlled doubles."""

    def test_compute_returns_average_of_subscores(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())

        with ExitStack() as stack:
            mock_load = stack.enter_context(
                patch.object(
                    CodeQuality,
                    "_load_code",
                    return_value=({"a.py": "print('hi')"}, "github", "card"),
                )
            )
            mock_lint = stack.enter_context(
                patch.object(CodeQuality, "_lint_score", return_value=0.9)
            )
            mock_typing = stack.enter_context(
                patch.object(CodeQuality, "_typing_score", return_value=0.6)
            )
            mock_llm = stack.enter_context(
                patch.object(CodeQuality, "_llm_code_rating", return_value=0.3)
            )
            score = metric.compute({})

        self.assertAlmostEqual(score, (0.9 + 0.6 + 0.3) / 3)
        self.assertEqual(
            metric.last_details,
            {
                "origin": "github",
                "lint_score": 0.9,
                "typing_score": 0.6,
                "llm_score": 0.3,
                "file_count": 1,
            },
        )
        mock_load.assert_called_once_with({})
        mock_lint.assert_called_once()
        mock_typing.assert_called_once()
        mock_llm.assert_called_once()

    def test_compute_falls_back_to_model_card(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())

        inputs = {"model_url": "https://huggingface.co/acme/model"}

        with ExitStack() as stack:
            mock_load = stack.enter_context(
                patch.object(
                    CodeQuality,
                    "_load_code",
                    return_value=({}, "", ""),
                )
            )
            mock_card = stack.enter_context(
                patch.object(
                    CodeQuality,
                    "_model_card_text",
                    return_value="card text",
                )
            )
            mock_llm = stack.enter_context(
                patch.object(
                    CodeQuality,
                    "_llm_card_rating",
                    return_value=0.42,
                )
            )
            score = metric.compute(inputs)

        self.assertEqual(score, 0.42)
        self.assertEqual(
            metric.last_details,
            {
                "origin": "model_card",
                "llm_score": 0.42,
                "card_available": True,
            },
        )
        mock_load.assert_called_once_with(inputs)
        mock_card.assert_called_once_with(inputs["model_url"])
        mock_llm.assert_called_once_with("card text")

    def test_load_code_prefers_git_url(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())

        with ExitStack() as stack:
            mock_loader = stack.enter_context(
                patch.object(
                    CodeQuality,
                    "_load_from_github",
                    return_value={"src/main.py": "print('hi')"},
                )
            )
            mock_card = stack.enter_context(
                patch.object(CodeQuality, "_model_card_text")
            )
            files, origin, card_text = metric._load_code(
                {
                    "git_url": "https://github.com/acme/repo",
                    "model_url": "https://huggingface.co/acme/model",
                }
            )

        self.assertEqual(files, {"src/main.py": "print('hi')"})
        self.assertEqual(origin, "github")
        self.assertEqual(card_text, "")
        mock_loader.assert_called_once_with("https://github.com/acme/repo")
        mock_card.assert_not_called()

    def test_load_code_uses_model_card_github_link(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())

        with ExitStack() as stack:
            mock_card = stack.enter_context(
                patch.object(
                    CodeQuality,
                    "_model_card_text",
                    return_value="Some text",
                )
            )
            mock_extract = stack.enter_context(
                patch.object(
                    CodeQuality,
                    "_github_from_card",
                    return_value="https://github.com/acme/repo",
                )
            )
            mock_loader = stack.enter_context(
                patch.object(
                    CodeQuality,
                    "_load_from_github",
                    return_value={"main.py": "print('hi')"},
                )
            )
            files, origin, card_text = metric._load_code(
                {"model_url": "https://huggingface.co/acme/model"}
            )

        self.assertEqual(files, {"main.py": "print('hi')"})
        self.assertEqual(origin, "github_from_card")
        self.assertEqual(card_text, "Some text")
        mock_card.assert_called_once_with("https://huggingface.co/acme/model")
        mock_extract.assert_called_once_with("Some text")
        mock_loader.assert_called_once_with("https://github.com/acme/repo")

    def test_parse_llm_score_handles_malformed_output(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())

        self.assertEqual(metric._parse_llm_score("not a number"), 0.5)
        self.assertEqual(metric._parse_llm_score(None), 0.5)
        self.assertEqual(metric._parse_llm_score("0.8 confident"), 0.8)

    def test_load_from_github_returns_empty_when_clone_fails(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())

        with ExitStack() as stack:
            mock_clone = stack.enter_context(
                patch.object(
                    CodeQuality,
                    "_clone_repo",
                    return_value=False,
                )
            )
            mock_read = stack.enter_context(
                patch.object(CodeQuality, "_read_python_files")
            )
            files = metric._load_from_github("https://github.com/acme/repo")

        self.assertEqual(files, {})
        mock_clone.assert_called_once()
        mock_read.assert_not_called()

    def test_load_from_github_invokes_read_when_clone_succeeds(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())

        with ExitStack() as stack:
            mock_clone = stack.enter_context(
                patch.object(
                    CodeQuality,
                    "_clone_repo",
                    return_value=True,
                )
            )
            mock_read = stack.enter_context(
                patch.object(
                    CodeQuality,
                    "_read_python_files",
                    return_value={"app.py": "print('ok')"},
                )
            )
            files = metric._load_from_github("https://github.com/acme/repo")

        self.assertEqual(files, {"app.py": "print('ok')"})
        mock_clone.assert_called_once()
        mock_read.assert_called_once()
        # First positional argument after binding is the URL.
        self.assertEqual(
            mock_clone.call_args[0][0],
            "https://github.com/acme/repo",
        )

    @patch("src.Metrics.GitClient")
    def test_read_python_files_scans_filesystem_when_gitclient_fails(
        self,
        mock_git_client_cls: MagicMock,
    ) -> None:
        mock_git_client_cls.side_effect = RuntimeError("git unavailable")

        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "keep.py").write_text("print('hello')\n", encoding="utf-8")
            (root / "skip.txt").write_text("ignore", encoding="utf-8")
            (root / "empty.py").write_text("\n\n", encoding="utf-8")
            sub = root / "pkg"
            sub.mkdir()
            (sub / "other.py").write_text(
                "def foo():\n    pass\n",
                encoding="utf-8",
            )

            files = metric._read_python_files(root, limit=2)

        self.assertEqual(len(files), 2)
        self.assertIn("keep.py", files)
        self.assertNotIn("empty.py", files)
        self.assertTrue(any(path.endswith("other.py") for path in files))
        mock_git_client_cls.assert_called_once()

    def test_code_snippet_enforces_character_budget(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())
        code = {
            "main.py": "print('hello world')\n" * 5,
            "utils.py": "def helper():\n    return 42\n",
        }

        snippet = metric._code_snippet(code, limit=80)

        self.assertIn("# File: main.py", snippet)
        self.assertIn("print('hello world')", snippet)
        # Second file should be truncated or omitted depending on the
        # remaining budget.
        self.assertLessEqual(len(snippet), 82)  # allow for join newlines

    def test_llm_code_rating_handles_empty_snippet(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())
        grok_mock = cast(MagicMock, metric.grok)
        llm_mock = grok_mock.llm

        with patch.object(CodeQuality, "_code_snippet", return_value=""):
            score = metric._llm_code_rating({})

        self.assertEqual(score, 0.5)
        llm_mock.assert_not_called()

    def test_llm_code_rating_handles_llm_failure(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())
        grok_mock = cast(MagicMock, metric.grok)
        grok_mock.llm.side_effect = RuntimeError("llm down")

        with patch.object(
            CodeQuality,
            "_code_snippet",
            return_value="print('hi')",
        ):
            score = metric._llm_code_rating({"a.py": "print('hi')"})

        self.assertEqual(score, 0.5)
        grok_mock.llm.assert_called_once()

    def test_llm_code_rating_uses_parse_helper(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())
        grok_mock = cast(MagicMock, metric.grok)
        grok_mock.llm.return_value = "0.75"

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    CodeQuality,
                    "_code_snippet",
                    return_value="print('hi')",
                )
            )
            mock_parse = stack.enter_context(
                patch.object(
                    CodeQuality,
                    "_parse_llm_score",
                    return_value=0.75,
                )
            )
            score = metric._llm_code_rating({"a.py": "print('hi')"})

        self.assertEqual(score, 0.75)
        mock_parse.assert_called_once_with("0.75")

    def test_llm_card_rating_short_circuit_on_missing_text(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())
        grok_mock = cast(MagicMock, metric.grok)
        llm_mock = grok_mock.llm

        score = metric._llm_card_rating("")

        self.assertEqual(score, 0.3)
        llm_mock.assert_not_called()

    def test_llm_card_rating_uses_parse_helper(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())
        grok_mock = cast(MagicMock, metric.grok)
        grok_mock.llm.return_value = "0.6"

        with patch.object(
            CodeQuality,
            "_parse_llm_score",
            return_value=0.6,
        ) as mock_parse:
            score = metric._llm_card_rating("Some card text")

        self.assertEqual(score, 0.6)
        mock_parse.assert_called_once_with("0.6")

    def test_llm_card_rating_returns_fallback_on_exception(self) -> None:
        metric = CodeQuality(hf_client=MagicMock(), grok_client=MagicMock())
        grok_mock = cast(MagicMock, metric.grok)
        grok_mock.llm.side_effect = RuntimeError("llm down")

        score = metric._llm_card_rating("card text")

        self.assertEqual(score, 0.3)
        grok_mock.llm.assert_called_once()

    def test_model_card_text_prefers_hf_api_bytes(self) -> None:
        hf_client = MagicMock()
        hf_client.request.return_value = b"# Title\nBody"
        metric = CodeQuality(hf_client=hf_client, grok_client=MagicMock())

        with patch.object(
            DatasetQuality,
            "_model_id_from_url",
            return_value="owner/model",
        ):
            text = metric._model_card_text(
                "https://huggingface.co/owner/model"
            )

        self.assertIn("# Title", text)
        hf_client.request.assert_called_once_with(
            "GET",
            "/owner/model/resolve/main/README.md",
        )

    def test_model_card_text_falls_back_to_browser(self) -> None:
        hf_client = MagicMock()
        hf_client.request.side_effect = RuntimeError("hf down")
        metric = CodeQuality(hf_client=hf_client, grok_client=MagicMock())

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    DatasetQuality,
                    "_model_id_from_url",
                    return_value="owner/model",
                )
            )
            mock_inject = stack.enter_context(
                patch(
                    "src.Metrics.injectHFBrowser",
                    return_value="rendered card",
                )
            )
            text = metric._model_card_text(
                "https://huggingface.co/owner/model"
            )

        self.assertEqual(text, "rendered card")
        mock_inject.assert_called_once_with(
            "https://huggingface.co/owner/model"
        )


class TestAvailabilityMetric(unittest.TestCase):
    """
    Test the AvailabilityMetric logic around dataset and code discovery.
    """

    @patch("src.Metrics.injectHFBrowser")
    @patch("src.Metrics.PurdueClient")
    def test_requires_model_url(
        self,
        _mock_grok_cls: MagicMock,
        _mock_inject: MagicMock,
    ) -> None:
        metric = AvailabilityMetric()
        with self.assertRaises(ValueError):
            metric.compute({})

    @patch.object(AvailabilityMetric, "_llm_detect_availability")
    @patch("src.Metrics.injectHFBrowser")
    @patch("src.Metrics.PurdueClient")
    def test_compute_scores_partial_availability(
        self,
        _mock_grok_cls: MagicMock,
        mock_inject: MagicMock,
        mock_detect: MagicMock,
    ) -> None:
        metric = AvailabilityMetric()
        url = "https://huggingface.co/org/model"
        mock_inject.return_value = "rendered page text"
        mock_detect.return_value = (True, False, "dataset link", "")

        score = metric.compute({"model_url": url})

        self.assertEqual(score, 0.5)
        mock_inject.assert_called_once_with(url)
        mock_detect.assert_called_once_with("rendered page text")
        self.assertEqual(
            metric.last_details,
            {
                "dataset_available": True,
                "codebase_available": False,
                "dataset_evidence": "dataset link",
                "codebase_evidence": "",
            },
        )

    @patch.object(AvailabilityMetric, "_llm_detect_availability")
    @patch("src.Metrics.injectHFBrowser")
    @patch("src.Metrics.PurdueClient")
    def test_compute_scores_full_availability(
        self,
        _mock_grok_cls: MagicMock,
        mock_inject: MagicMock,
        mock_detect: MagicMock,
    ) -> None:
        metric = AvailabilityMetric()
        url = "https://huggingface.co/org/model"
        mock_inject.return_value = "full model card"
        mock_detect.return_value = (True, True, "data url", "code repo")

        score = metric.compute({"model_url": url})

        self.assertEqual(score, 1.0)
        mock_inject.assert_called_once_with(url)
        mock_detect.assert_called_once_with("full model card")
        self.assertEqual(
            metric.last_details,
            {
                "dataset_available": True,
                "codebase_available": True,
                "dataset_evidence": "data url",
                "codebase_evidence": "code repo",
            },
        )

    @patch("src.Metrics.PurdueClient")
    def test_llm_detect_availability_parses_llm_json(
        self,
        mock_grok_cls: MagicMock,
    ) -> None:
        metric = AvailabilityMetric()
        grok_mock = mock_grok_cls.return_value
        grok_mock.llm.return_value = (
            "```json\n"
            "{\n"
            '  "dataset_available": true,\n'
            '  "codebase_available": false,\n'
            '  "dataset_evidence": "https://huggingface.co/datasets/demo",\n'
            '  "codebase_evidence": ""\n'
            "}\n"
            "```"
        )

        dataset, codebase, dataset_ev, codebase_ev = (
            metric._llm_detect_availability("page text")
        )

        self.assertTrue(dataset)
        self.assertFalse(codebase)
        self.assertEqual(dataset_ev, "https://huggingface.co/datasets/demo")
        self.assertEqual(codebase_ev, "")
        grok_mock.llm.assert_called_once()

    @patch("src.Metrics.PurdueClient")
    def test_llm_detect_availability_handles_empty_text(
        self,
        _mock_grok_cls: MagicMock,
    ) -> None:
        metric = AvailabilityMetric()

        dataset, codebase, dataset_ev, codebase_ev = (
            metric._llm_detect_availability("")
        )

        self.assertFalse(dataset)
        self.assertFalse(codebase)
        self.assertEqual(dataset_ev, "")
        self.assertEqual(codebase_ev, "")

    @patch("src.Metrics.PurdueClient")
    def test_llm_detection_fallback_uses_heuristics(
        self,
        mock_grok_cls: MagicMock,
    ) -> None:
        metric = AvailabilityMetric()
        mock_grok = mock_grok_cls.return_value
        mock_grok.llm.side_effect = ValueError("LLM unavailable")

        snippet = (
            "Trained on https://huggingface.co/datasets/acme/data "
            "with source code at https://github.com/acme/model"
        )

        dataset, codebase, dataset_ev, codebase_ev = (
            metric._llm_detect_availability(snippet)
        )

        self.assertTrue(dataset)
        self.assertTrue(codebase)
        self.assertIn("huggingface.co/datasets/acme/data", dataset_ev)
        self.assertIn("github.com/acme/model", codebase_ev)


if __name__ == "__main__":
    unittest.main()
