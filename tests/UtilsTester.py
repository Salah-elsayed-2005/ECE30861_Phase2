import unittest
from unittest.mock import MagicMock

from src.Client import HFClient
from src.utils import browse_hf_repo


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


if __name__ == "__main__":
    unittest.main()
