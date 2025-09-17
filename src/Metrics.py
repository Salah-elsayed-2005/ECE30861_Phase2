# src/Metrics.py
# THIS CODE WILL HANDLE THE METRIC OBJECTS
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from src.Client import HFClient, GrokClient
import time


@dataclass(frozen=True)
class MetricResult:
    """
    Canonical result object returned by all metrics.

    Attributes
    ----------
    metric : str
        Human-friendly metric name (e.g., "License Check").
    key : str
        Stable identifier/slug for the metric (e.g., "license").
    value : Any
        The primary result produced by the metric (bool, str, dict, etc.).
    latency_ms : float
        How long the metric took to execute (milliseconds).
    details : Optional[Mapping[str, Any]]
        Optional extra information for display or debugging.
    error : Optional[str]
        If the metric failed, put a concise error message here
        and set `value` as appropriate.
    """
    metric: str
    key: str
    value: Any
    latency_ms: float
    details: Optional[Mapping[str, Any]] = None
    error: Optional[str] = None


class Metric(ABC):
    """
    Abstract base class for metrics.

    Subclasses must implement ``compute()`` to perform the actual work.
    """

    @abstractmethod
    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Compute the metric score from parsed inputs.

        Parameters
        ----------
        inputs : dict[str, Any]
            Parsed inputs required by the metric.
        **kwargs : Any
            Optional per-metric tuning parameters.

        Returns
        -------
        float
            A score between 0.0 and 1.0.
        """
        raise NotImplementedError


class LicenseMetric(Metric):
    """
    Metric that find the model license and assigns it
    a score from 0 to 1 using a lookup table
    or LLM if not found.
    """

    # The below lookup table assigns scores based on
    # how much each license allows for:
    # linking, distribution, modification,
    # patent grant, private use and sublicensing.
    # Licenses that allow 5 or 6 of these are given a score of 1.0,
    # those that allow 3 or 4 are given a score of 0.75,
    # others are given a score of 0.5.

    license_scores: dict[str, float] = {
        # Permissive (1.0)
        "apache-2.0": 1.0,
        "mit": 1.0,
        "afl-3.0": 1.0,
        "bsd": 1.0,
        "bsd-2-clause": 1.0,
        "bsd-3-clause": 1.0,
        "bsd-3-clause-clear": 1.0,
        "isc": 1.0,
        "zlib": 1.0,
        "cdla-permissive-1.0": 1.0,
        "cdla-permissive-2.0": 1.0,
        "ms-pl": 1.0,
        "postgresql": 1.0,
        "osl-3.0": 1.0,
        "apple-ascl": 1.0,
        "mpl-2.0": 1.0,
        "pddl": 1.0,
        "unlicense": 1.0,
        "cc0-1.0": 1.0,
        "wtfpl": 1.0,
        "intel-research": 1.0,
        "ofl-1.1": 1.0,
        "lppl-1.3c": 1.0,
        "ncsa": 1.0,
        "etalab-2.0": 1.0,

        # Less permissive (0.75)
        "gpl-3.0": 0.75,
        "gpl-2.0": 0.75,
        "gpl": 0.75,
        "agpl-3.0": 0.75,
        "lgpl-3.0": 0.75,
        "lgpl-2.1": 0.75,
        "lgpl": 0.75,
        "lgpl-lr": 0.75,
        "epl-2.0": 0.75,
        "epl-1.0": 0.75,
        "ecl-2.0": 0.75,
        "eupl-1.1": 0.75,
        "eupl-1.2": 0.75,
        "artistic-2.0": 0.75,
        "cdla-sharing-1.0": 0.75,
        "cc-by-4.0": 0.75,
        "cc-by-3.0": 0.75,
        "cc-by-2.0": 0.75,
        "cc-by-2.5": 0.75,
        "cc-by-sa-4.0": 0.75,
        "cc-by-sa-3.0": 0.75,
        "odc-by": 0.75,
        "bsl-1.0": 0.75,
        "odbl": 0.75,
        "gfdl": 0.75,

        # Restrictive (0.5)
        "cc-by-nc-4.0": 0.5,
        "cc-by-nc-2.0": 0.5,
        "cc-by-nc-3.0": 0.5,
        "cc-by-nc-nd-4.0": 0.5,
        "cc-by-nc-nd-3.0": 0.5,
        "cc-by-nc-sa-4.0": 0.5,
        "cc-by-nc-sa-3.0": 0.5,
        "cc-by-nc-sa-2.0": 0.5,
        "cc-by-nd-4.0": 0.5,
        "fair-noncommercial-research-license": 0.5,

        # Model-specific / special licenses
        "openrail": 0.75,
        "creativeml-openrail-m": 0.75,
        "openrail++": 0.75,
        "gemma": 0.75,
        "llama2": 0.75,
        "llama3": 0.75,
        "llama3.1": 0.75,
        "llama3.2": 0.75,
        "llama3.3": 0.75,
        "llama4": 0.75,
        "bigscience-bloom-rail-1.0": 0.75,
        "bigscience-openrail-m": 0.75,
        "bigcode-openrail-m": 0.75,
        "open-mdw": 1,  # open data / model weights, permissive
        "h-research": 0.5,  # research-only, restrictive
        "c-uda": 0.5,  # non-commercial, restrictive
        "apple-amlr": 0.5,  # apple-specific, restrictive
        "deepfloyd-if-license": 0.5,  # non-commercial research only
        "cc": 0.75,  # attribution-required

        "other": 0.5  # catch-all
    }

    def __init__(self):
        self.hf_client = HFClient(max_requests=10)
        self.grok_client = GrokClient(max_requests=10)

    def compute(self, inputs: dict[str, Any], **kwargs) -> float:
        """
        Compute the license score from parsed inputs.

        Parameters
        ----------
        inputs : dict[str, Any]
            Parsed inputs required by the metric. Must include a key called
            'model_url' with its corresponding correct link

        **kwargs : Any
            Optional per-metric tuning parameters.

        Returns
        -------
        float
            A score between 0.0 and 1.0.

        Raises
        ------
        RuntimeError
            If no valid HF model URL is found in the dict
        """
        # model_url must be in the dict
        if "model_url" not in inputs.keys():
            raise ValueError("Model link not found in input dictionary")

        # start latency timer and extract model_id from URL
        start = time.time()
        model_id = inputs['model_url'].split("https://huggingface.co/")[-1]
        error = None

        # try to get license from HFClient and assign a score
        try:
            model_info = self.hf_client.request("GET",
                                                f"/api/models/{model_id}")
            card_data = model_info["cardData"]
            license_type = card_data["license"]
            if license_type is None:
                score = 0.0
                error = "Liscense not in card data"
            elif license_type not in self.license_scores:
                # this grok call will likely never happen because all
                # licenses not given on HF's license filter are labeled "other"
                prompt = (f"Rank the this HuggingFace model: {model_id}. "
                          "Give it of three scores, 1, 0.75, or 0.5. "
                          "base your ranking on the rakings of this "
                          "dictionary of licenses and their scores: "
                          f"{str(self.license_scores)}")
                score = self.grok_client.llm(prompt)
            else:
                score = self.license_scores[license_type]
        except Exception as e:
            license_type = None
            score = 0.0
            error = str(e)

        # end latency timer
        latency_ms = (time.time() - start) * 1000

        # put all relevant info into MetricResult object
        self.result = MetricResult(
            metric="License Check",
            key="license",
            value=score,
            latency_ms=latency_ms,
            details={"license_type": license_type, "model_id": model_id},
            error=error
        )

        return score
