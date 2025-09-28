import unittest
from unittest.mock import ANY, MagicMock, patch

import requests

from src.Client import HFClient
from src.utils import _normalize_hf_text, browse_hf_repo, injectHFBrowser


class TestBrowseHFRepo(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock(spec=HFClient)

    def test_builds_dataset_url_and_params(self) -> None:
        self.client.request.return_value = []

        browse_hf_repo(
            self.client,
            repo_id="owner/dataset",
            repo_type="dataset",
            revision="dev",
            recursive=False,
        )

        self.client.request.assert_called_once_with(
            "GET",
            "/api/datasets/owner/dataset/tree/dev",
            params={},
        )

    def test_filters_directories_and_missing_sizes(self) -> None:
        payload = [
            {"path": "weights.bin", "type": "file", "size": 123},
            {"path": "docs", "type": "directory"},
            {"path": "README.md", "type": "file"},
        ]
        self.client.request.return_value = payload

        result = browse_hf_repo(self.client, repo_id="owner/model")

        self.assertEqual(result, [("weights.bin", 123), ("README.md", -1)])
        self.client.request.assert_called_once_with(
            "GET",
            "/api/models/owner/model/tree/main",
            params={"recursive": 1},
        )

    def test_accepts_tree_payload_from_dict(self) -> None:
        payload = {
            "tree": [
                {"path": "app.py", "type": "file", "size": 88},
                {"path": "assets", "type": "directory"},
            ]
        }
        self.client.request.return_value = payload

        result = browse_hf_repo(
            self.client,
            repo_id="owner/space",
            repo_type="space",
        )

        self.assertEqual(result, [("app.py", 88)])
        self.client.request.assert_called_once_with(
            "GET",
            "/api/spaces/owner/space/tree/main",
            params={"recursive": 1},
        )

    def test_returns_empty_list_for_unexpected_payload(self) -> None:
        self.client.request.return_value = {"unexpected": True}

        result = browse_hf_repo(self.client, repo_id="owner/model")

        self.assertEqual(result, [])


class TestInjectHFBrowser(unittest.TestCase):
    """Verify HTTP-based Hugging Face page fetching helper."""

    @patch("src.utils.requests.get")
    def test_returns_main_text(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.text = """
            <html>
              <main>
                <h1>Title</h1>
                <p>Line one.</p>
                <p>Line two.</p>
              </main>
            </html>
        """
        mock_get.return_value = response

        url = "https://huggingface.co/owner/model"
        output = injectHFBrowser(url)

        response.raise_for_status.assert_called_once()
        mock_get.assert_called_once_with(
            url,
            headers=ANY,
            timeout=20.0,
        )
        self.assertIn("Title", output)
        self.assertIn("Line one.", output)
        self.assertIn("Line two.", output)

    @patch("src.utils.requests.get")
    def test_raises_runtime_error_when_request_fails(
        self,
        mock_get: MagicMock,
    ) -> None:
        mock_get.side_effect = requests.RequestException("boom")

        with self.assertRaises(RuntimeError):
            injectHFBrowser("https://huggingface.co/owner/broken")


class TestNormalizeHFText(unittest.TestCase):
    def test_collapses_extra_blank_lines_and_trailing_space(self) -> None:
        raw = "Line 1  \r\n\r\n\r\nLine 2   \n\n\n  "
        self.assertEqual(_normalize_hf_text(raw), "Line 1\n\nLine 2")

    def test_handles_empty_string(self) -> None:
        self.assertEqual(_normalize_hf_text(""), "")


if __name__ == "__main__":
    unittest.main()
