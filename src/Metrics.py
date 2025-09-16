# src/Metrics.py
# THIS CODE WILL HANDLE THE METRIC OBJECTS
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from src.Client import HFClient
from src.utils import browse_hf_repo
import math


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


class SizeMetric(Metric):
    """
    Metric that finds the model size in bits and converts it
    to a score from 0 to 1 using a lookup table
    """
    lookupTable: dict[int, float] = {
        3: 1.0,
        1000: 0.9,
        1e10: 0.5,
        1e12: 0,
    }
    maxModelBits = 603e9*16
    # Found from : https://huggingface.co/giannisan/llama3.1-405B-upscaled-603B
    commonModelFileEndings = [
        ".bin",
        ".safetensors",
        ".h5",
        ".ckpt",
        ".onnx",
        ".tflite",
        ".pb",
        ".mlmodel",
        ".gguf",
        ".ggml",
        ".ggjt",
        ".pt",
    ]

    def __init__(self):
        self.hf_client = HFClient(max_requests=100)

    def extract_bits_from_saftensor(self,
                                    safeTensorDict: dict[str, int]) -> int:
        """
        Extract the nunber of bits by infering from the key
        """
        bits = []
        for precision in safeTensorDict.keys():
            param_size = int(''.join(ch for ch in precision if ch.isdigit()))
            n_params = param_size * safeTensorDict[precision]
            bits.append(n_params)
        return min(bits)

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Compute the metric score from parsed inputs.

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

        # Extract the model id and get model info using API
        model_id = inputs['model_url'].split("https://huggingface.co/")[-1]
        card_data = self.hf_client.request("GET", f"/api/models/{model_id}")

        bits = None
        # If we have access to safetensors, use that
        if 'safetensors' in card_data.keys() and 'parameters' in card_data['safetensors'].keys():
            bits = self.extract_bits_from_saftensor(card_data['safetensors']['parameters'])
        # If not we will need to browse the repo
        else:
            files = browse_hf_repo(self.hf_client, model_id, repo_type="model", revision="main", recursive=True)
            files_filtered = [f for f in files if any([f[0].endswith(ext) for ext in SizeMetric.commonModelFileEndings])]
            if len(files_filtered) == 0:
                bits = -1
            else:
                bits = min(files_filtered, key = lambda x: x[1])[1]

        # Now that we have the bits, let's assign our score. This will be based
        # on a log scale between the nubmer of bits of the model and the number of
        # bits in HFs biggest model. The log is there to smooth things out and we divide
        # by a factor that will make the score approach 1 as the size fits into
        # a jetson nano. Fianlly, we will clip between 0 and 1 in case of any
        # extra crazy values. For the size of the jetson nano paremeters we will
        # just want the model file to be under 4GB (how much VRAM the nano has)
        if bits <= 0:
          return 0
        score = (1-math.log(bits / SizeMetric.maxModelBits)) / (1-math.log(4*8e9 / SizeMetric.maxModelBits))
        score = min(score, 1)
        score = max(score, 0)
        return score

if __name__ == "__main__":
    met = SizeMetric()
    out = met.compute(dict({'model_url': "https://huggingface.co/google/gemma-7b",}))
    print(out)
