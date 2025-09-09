# src/Parser.py
# THIS CODE WILL HANDLE THE PARSER OBJECT.
# THIS OBJECT WILL TAKE A DICT OF REGEXES
# AND WILL EXTRACT THE STUFF FROM TEXT USING
# THE REGEX INTO A DICT WHERE THE KEYS WILL
# LINE UP BETWEEN THE REGEX AND THE OUTPUT

import re
from typing import Dict, List


class Parser:
    """
    URL Parser to categorize input URLs into the following groups:
    - model_url : Hugging Face model links
    - dataset_url : Hugging Face dataset links
    - git_url : GitHub repository links
    - unknown : Any URL that does not match a known category
    """

    def __init__(self, filepath: str):
        """
        Initialize the Parser object.

        Parameters
        ----------
        filepath : str
            Path to the file containing newline-delimited URLs.

        Notes
        -----
        The constructor compiles regular expressions for each
        category and immediately categorizes the provided URLs.
        """
        # Regex dictionary keyed by category
        self.filepath = filepath
        self.urls: List[str] = self._loadUrls()
        self.regex_dict: Dict[str, re.Pattern] = {
            "dataset_url": re.compile(
                r"https?://huggingface\.co/datasets/[A-Za-z0-9_.-]\
                +(/[\w\-\.]+)?"),
            "model_url": re.compile(
                r"https?://huggingface\.co/(?!datasets/)[A-Za-z0-9_.-]\
                +(/[\w\-\.]+)?"),
            "git_url": re.compile(
                r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"),
        }
        self.groups: Dict[str, List[str]] = self._categorize()

    def _loadUrls(self) -> List[str]:
        """
        Read URLs from the provided text file.

        Returns
        -------
        list[str]
            List of non-empty, stripped URLs read from the file.
        """
        with open(self.filepath, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _categorize(self) -> Dict[str, List[str]]:
        """
        Categorize URLs into groups based on regex patterns.

        Returns
        -------
        dict[str, list[str]]
            Dictionary mapping category names to lists of URLs.

        Notes
        -----
        Any URL not matched by a known pattern is added
        to the 'unknown' category.
        """
        result: Dict[str, List[str]] = {key: [] for key in self.regex_dict}

        for url in self.urls:
            matched = False
            for category, pattern in self.regex_dict.items():
                if pattern.match(url):
                    result[category].append(url)
                    matched = True
                    break
            if not matched:
                # Unknown URL default category
                result.setdefault("unknown", []).append(url)

        return result

    def getGroups(self) -> Dict[str, List[str]]:
        """
        Return categorized URLs.

        Returns
        -------
        dict[str, list[str]]
            Dictionary of categorized URLs by type.
        """
        return (self.groups)
