import unittest
from unittest.mock import MagicMock, patch

import src.utils as utils
from src.Client import HFClient
from src.utils import browse_hf_repo, injectHFBrowser


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
    """Verify Selenium-driven Hugging Face page scraping helper."""

    @patch("src.utils.WebDriverWait")
    @patch("src.utils.webdriver.Chrome")
    def test_returns_body_text_and_cleans_up(
        self,
        mock_chrome_cls: MagicMock,
        mock_wait_cls: MagicMock,
    ) -> None:
        driver = MagicMock()
        mock_chrome_cls.return_value = driver

        wait_instance = MagicMock()
        mock_wait_cls.return_value = wait_instance

        body_element = MagicMock()
        body_element.text = "Visible content"
        driver.find_element.return_value = body_element

        url = "https://huggingface.co/owner/model"
        output = injectHFBrowser(url)

        self.assertEqual(output, "Visible content")
        mock_chrome_cls.assert_called_once_with()
        driver.get.assert_called_once_with(url)
        self.assertEqual(wait_instance.until.call_count, 2)
        driver.find_element.assert_called_once()
        args, _ = driver.find_element.call_args
        self.assertEqual(args, (utils.By.TAG_NAME, "body"))
        driver.quit.assert_called_once()

    @patch("src.utils.WebDriverWait")
    @patch("src.utils.webdriver.Chrome")
    def test_quits_driver_even_on_failure(
        self,
        mock_chrome_cls: MagicMock,
        mock_wait_cls: MagicMock,
    ) -> None:
        driver = MagicMock()
        mock_chrome_cls.return_value = driver

        wait_instance = MagicMock()
        mock_wait_cls.return_value = wait_instance

        driver.find_element.side_effect = RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            injectHFBrowser("https://huggingface.co/owner/broken")

        driver.quit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
