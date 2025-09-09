# tests/ParserTester.py

import tempfile

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


def test_model_url():
    """
    Test that a valid Hugging Face model URL is categorized
    into the 'model_url' group.
    """
    path = makeTestFile(["https://huggingface.co/bert-base-uncased"])
    parser = Parser(path)
    groups = parser.getGroups()
    assert "https://huggingface.co/bert-base-uncased" in groups["model_url"]


def test_dataset_url():
    """
    Test that a valid Hugging Face dataset URL is categorized
    into the 'dataset_url' group.
    """
    path = makeTestFile(["https://huggingface.co/datasets/imdb"])
    parser = Parser(path)
    groups = parser.getGroups()
    assert "https://huggingface.co/datasets/imdb" in groups["dataset_url"]


def test_git_url():
    """
    Test that a valid GitHub repository URL is categorized
    into the 'git_url' group.
    """
    path = makeTestFile(["https://github.com/user/repo"])
    parser = Parser(path)
    groups = parser.getGroups()
    assert "https://github.com/user/repo" in groups["git_url"]


def test_multiple_urls():
    """
    Test that multiple input URLs are correctly categorized
    into their respective groups.
    """
    path = makeTestFile([
        "https://huggingface.co/google/gemma-3-270m/tree/main",
        "https://huggingface.co/datasets/xlangai/AgentNet",
        "https://github.com/SkyworkAI/Matrix-Game",
    ])
    parser = Parser(path)
    groups = parser.getGroups()
    assert len(groups["model_url"]) == 1
    assert len(groups["dataset_url"]) == 1
    assert len(groups["git_url"]) == 1


def test_unknown_url():
    """
    Test that a random website URL not matching
    known patterns is categorized as 'unknown'.
    """
    path = makeTestFile(["https://example.com/somepage"])
    parser = Parser(path)
    groups = parser.getGroups()
    assert "https://example.com/somepage" in groups["unknown"]


def test_malformed_urls():
    """
    Test that malformed URLs are not matched
    by any category and are categorized as 'unknown'.
    """
    path = makeTestFile([
        "https://hugingface.co/bert-base-uncased",
        "https://hugingface.co/datasets/imdb",
        "https://gihub.com/user/repo"
    ])
    parser = Parser(path)
    groups = parser.getGroups()
    assert len(groups["unknown"]) == 3
