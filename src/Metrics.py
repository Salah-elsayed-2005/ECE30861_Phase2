# src/Metrics.py
# THIS CODE WILL HANDLE THE METRIC OBJECTS
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .Client import HFClient


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


class RampUpTime(Metric):
    """
    Metric estimating ramp-up time based on the length of the
    'Use this model' section in a model's README on HuggingFace.

    A shorter section implies quicker ramp-up, yielding a higher score.
    """

    def __init__(self, client: HFClient):
        self.client = client

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Compute the ramp-up time score for a given HuggingFace model.

        Parameters
        ----------
        inputs : dict[str, Any]
            Must contain 'model_url' with a valid HuggingFace model URL.

        Returns
        -------
        float
            Score in [0.0, 1.0] where 1.0 means very quick to ramp up.
        """
        start = time.perf_counter()

        url = inputs.get("model_url")
        if not url:
            raise ValueError("Missing required input: model_url")

        url = url[0]
        # Extract repo_id like "bert-base-uncased"
        repo_id = url.replace("https://huggingface.co/", "").strip("/")

        # Fetch README text directly
        readme_text = self.client.request(
            "GET", f"/{repo_id}/raw/main/README.md"
        )
        # Extract the "Use this model" section
        section = re.search(
            r"(?is)(##+\s*(?:Use this model|How to use).*?)(?=\n##+|\Z)",
            readme_text)

        if section:
            char_count = len(section.group(1))
        else:
            print("Could not find 'Use this model' section.")
            char_count = 0

        # Convert character count into a normalized score [0,1]
        # <500 chars → 1.0 ; >2000 chars → 0.0
        if char_count <= 500:
            score = 1.0
        elif char_count >= 2000:
            score = 0.0
        else:
            score = 1.0 - ((char_count - 500) / (2000 - 500))

        latency = (time.perf_counter() - start) * 1000

        result = MetricResult(
            metric="Ramp Up Time",
            key="ramp_up_time",
            value=score,
            latency_ms=latency,
            details={"char_count": char_count, "repo_id": repo_id},
        )

        return result.value
