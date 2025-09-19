# src/utils.py
# Store any helper functions here

from typing import List, Tuple

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from Client import HFClient


def browse_hf_repo(
    client: HFClient,
    repo_id: str,
    repo_type: str = "model",
    revision: str = "main",
    recursive: bool = True,
) -> List[Tuple[str, int]]:
    """
    Browse files of a Hugging Face repo using HFClient.

    Parameters
    ----------
    client : HFClient
        An instance of your HFClient (already configured with token).
    repo_id : str
        Repo identifier, e.g. "facebook/opt-125m".
    repo_type : str
        One of "model", "dataset", or "space".
    revision : str
        Branch, tag, or commit SHA (default: "main").
    recursive : bool
        Whether to traverse all subfolders.

    Returns
    -------
    List[Tuple[str, int]]
        A list of (file_path, size_in_bytes). Directories are skipped.
    """
    plural = {"model": "models",
              "dataset": "datasets",
              "space": "spaces"}[repo_type]
    path = f"/api/{plural}/{repo_id}/tree/{revision}"
    params = {"recursive": 1} if recursive else {}

    data = client.request("GET", path, params=params)

    # Handle response format
    if isinstance(data, dict) and "tree" in data:
        entries = data["tree"]
    elif isinstance(data, list):
        entries = data
    else:
        return []

    return [
        (e["path"], e.get("size", -1))
        for e in entries
        if e.get("type") != "directory"
    ]


def injectHFBrowser(model: str, headless: bool = True) -> str:
    """
    Retrieve the rendered Hugging Face model page via Selenium.

    Parameters
    ----------
    model : str
        Fully-qualified URL for the target Hugging Face model repository.
    headless : bool
        When True (default), runs Chrome in headless mode so no window appears.

    Returns
    -------
    str
        Visible text contained within the page `<body>` element.

    Notes
    -----
    A new Chrome WebDriver instance is created per call to avoid leaking
    session state between runs. The host must have a compatible chromedriver
    installation available on the PATH.
    """
    # Use an ephemeral Chrome session so each call starts from a clean slate.
    # Configure Chrome to run headless by default to avoid opening a window.
    chrome_options = webdriver.ChromeOptions()
    if headless:
        # Use the modern headless mode if available
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1280,800")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-infobars")

    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(model)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "main"))
        )
        # Wait until the body content is fully rendered
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Get *all* visible text on the page
        return driver.find_element(By.TAG_NAME, "body").text

    finally:
        driver.quit()
