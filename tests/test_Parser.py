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
        """A model URL in the 3rd column populates 'model_url'."""
        url = "https://huggingface.co/bert-base-uncased"
        path = makeTestFile([f",,{url}"])
        parser = Parser(path)
        groups = parser.getGroups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["model_url"], url)
        self.assertEqual(groups[0]["git_url"], "")
        self.assertEqual(groups[0]["dataset_url"], "")

    def test_dataset_url(self):
        """A dataset URL in the 2nd column populates 'dataset_url'."""
        url = "https://huggingface.co/datasets/imdb"
        path = makeTestFile([f",{url},"])
        parser = Parser(path)
        groups = parser.getGroups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["dataset_url"], url)
        self.assertEqual(groups[0]["git_url"], "")
        self.assertEqual(groups[0]["model_url"], "")

    def test_git_url(self):
        """A GitHub URL in the 1st column populates 'git_url'."""
        url = "https://github.com/user/repo"
        path = makeTestFile([f"{url},,"])
        parser = Parser(path)
        groups = parser.getGroups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["git_url"], url)
        self.assertEqual(groups[0]["dataset_url"], "")
        self.assertEqual(groups[0]["model_url"], "")

    def test_multiple_urls(self):
        """All three URLs on one line map to their columns."""
        git = "https://github.com/SkyworkAI/Matrix-Game"
        dataset = "https://huggingface.co/datasets/xlangai/AgentNet"
        model = "https://huggingface.co/google/gemma-3-270m/tree/main"
        line = f"{git},{dataset},{model}"
        path = makeTestFile([line])
        parser = Parser(path)
        groups = parser.getGroups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["git_url"], git)
        self.assertEqual(groups[0]["dataset_url"], dataset)
        self.assertEqual(groups[0]["model_url"], model)


if __name__ == '__main__':
    unittest.main()
