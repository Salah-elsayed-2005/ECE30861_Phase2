# tests/ParserTester.py

import tempfile
import unittest

from src.Parser import Parser


def makeTestFile(urls):
    """
    Helper to create a temporary file containing the given URLs.

    Parameters
    ----------
    urls : list[str]
        List of URLs to write into the file.

    Returns
    -------
    str
        Path to the temporary file containing the URLs.
    """
    tmp = tempfile.NamedTemporaryFile(
        delete=False, mode="w+", encoding="utf-8")
    tmp.write("\n".join(urls))
    tmp.flush()
    return tmp.name


class TestParser(unittest.TestCase):
    """
    Unit tests for the Parser class to ensure URLs
    are categorized correctly.
    """

    def test_model_url(self):
        """Test that a Hugging Face model URL is
        categorized into 'model_url'"""
        path = makeTestFile(["https://huggingface.co/bert-base-uncased"])
        parser = Parser(path)
        groups = parser.getGroups()
        self.assertIn("https://huggingface.co/bert-base-uncased",
                      groups["model_url"])

    def test_dataset_url(self):
        """Test that a Hugging Face dataset URL is
        categorized into 'dataset_url'"""
        path = makeTestFile(["https://huggingface.co/datasets/imdb"])
        parser = Parser(path)
        groups = parser.getGroups()
        self.assertIn("https://huggingface.co/datasets/imdb",
                      groups["dataset_url"])

    def test_git_url(self):
        """Test that a GitHub repo URL is categorized into 'git_url'"""
        path = makeTestFile(["https://github.com/user/repo"])
        parser = Parser(path)
        groups = parser.getGroups()
        self.assertIn("https://github.com/user/repo", groups["git_url"])

    def test_multiple_urls(self):
        """Test that multiple URLs are categorized into correct groups"""
        model = "https://huggingface.co/google/gemma-3-270m/tree/main"
        path = makeTestFile([
            model,
            "https://huggingface.co/datasets/xlangai/AgentNet",
            "https://github.com/SkyworkAI/Matrix-Game",
        ])
        parser = Parser(path)
        groups = parser.getGroups()
        self.assertEqual(groups["model_url"], model)
        self.assertEqual(groups["dataset_url"],
                         "https://huggingface.co/datasets/xlangai/AgentNet")
        self.assertEqual(groups["git_url"],
                         "https://github.com/SkyworkAI/Matrix-Game")


if __name__ == '__main__':
    unittest.main()
