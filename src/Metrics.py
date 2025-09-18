# src/Metrics.py
# THIS CODE WILL HANDLE THE METRIC OBJECTS
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from src.utils import injectHFBrowser

from .Client import HFClient, GrokClient


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
    'Use this model' / 'Usage' section in a model's README on HuggingFace.

    A shorter section implies quicker ramp-up, yielding a higher score.
    """

    def __init__(self):
        self.client = HFClient(max_requests=3)
        self.grok = GrokClient(max_requests=100)

    def _extract_usage_section(self, text: str) -> str | None:
        """
        Use the Grok LLM to extract all usage-related text from the input text.
        """
        if not text:
            return None

        prompt = f"""
        You are an AI assistant. Extract and return ONLY the sections
        that explain how to use the model, code examples, or instructions
        to get started. Ignore unrelated sections.
        
        Text:
        {text}
        
        Extract usage text verbatim.
        """
        try:
        # Assuming your GrokClient has an llm() method
            response = self.grok.llm(prompt)
            return response.strip() if response else None
        except Exception as e:
            print(f"Grok LLM extraction failed: {e}")
            return None

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Compute the ramp-up time score for a given HuggingFace model.
        """
        start = time.perf_counter()
        url = inputs.get("model_url")
        if not url:
            raise ValueError("Missing required input: model_url")

        # Parse repo_id like "google/gemma-1.1-7b-it"
        repo_id = url.replace("https://huggingface.co/", "").strip("/")

        full_page_text = injectHFBrowser(url)
        usage_text = self._extract_usage_section(full_page_text)

        if usage_text:
            char_count = len(usage_text)
        else:
            print("Could not find any usage examples")
            char_count = 0
        print(char_count)
        # 4) Convert character count into a normalized score [0,1]
        # <500 chars → 1.0 ; >2000 chars → 0.0
        score = 1.0 / (1.0 + math.log1p(char_count / 500))
        score = max(0.0, min(score, 1.0))

        latency = (time.perf_counter() - start) * 1000

        result = MetricResult(
            metric="Ramp Up Time",
            key="ramp_up_time",
            value=score,
            latency_ms=latency,
            details={"char_count": char_count, "repo_id": repo_id},
        )

        return result.value

