# src/Metrics.py
# THIS CODE WILL HANDLE THE METRIC OBJECTS
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from src.Client import GrokClient, HFClient
from src.utils import browse_hf_repo, injectHFBrowser


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

    name: str  # Human-friendly metric name (e.g., "License Check").
    key: str  # Identifier/slug for the metric (e.g., "license").

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
    name = "Ramp-Up Time"
    key = "ramp_up_time"

    def __init__(self):
        self.client = HFClient(max_requests=3)
        self.grok = GrokClient(max_requests=100)

    def _extract_usage_section(self, text: str) -> str | None:
        """
        Ask the Grok LLM to isolate usage instructions from the README text.

        Parameters
        ----------
        text : str
            Raw page text harvested from the Hugging Face model page.

        Returns
        -------
        str | None
            Cleaned usage-focused excerpt, or ``None`` if no guidance is found
            or the LLM request fails.
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
        Score how quickly a developer can ramp up on a Hugging Face model.

        Parameters
        ----------
        inputs : dict[str, Any]
            Must include the key ``"model_url"`` pointing at the model page.
        **kwargs : Any
            Present for interface compatibility; unused.

        Returns
        -------
        float
            Ramp-up score between 0.0 (hard to learn) and 1.0 (fast to learn).
        """
        url = inputs.get("model_url")
        if not url:
            raise ValueError("Missing required input: model_url")

        # Fetch the full page text using Selenium
        full_page_text = injectHFBrowser(url)
        usage_text = self._extract_usage_section(full_page_text)

        if usage_text:
            char_count = len(usage_text)
        else:
            print("Could not find any usage examples")
            char_count = 0

        # Weight long instructions logarithmically so modest increases in
        # length do not crater the score, while extremely long sections still
        # reduce it meaningfully.
        score = 1.0 / (1.0 + math.log1p(char_count / 500))
        score = max(0.0, min(score, 1.0))

        return score


class LicenseMetric(Metric):
    """
    Metric that find the model license and assigns it
    a score from 0 to 1 using a lookup table
    or LLM if not found.
    """

    name = "License Permissiveness"
    key = "license_metric"

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
        self.hf_client = HFClient(max_requests=100)
        self.grok_client = GrokClient(max_requests=100)

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

        # extract model_id from URL
        model_id = inputs['model_url'].split("https://huggingface.co/")[-1]

        # try to get license from HFClient and assign a score
        model_info = self.hf_client.request("GET",
                                            f"/api/models/{model_id}")
        card_data = model_info["cardData"]
        license_type = card_data.get("license", None)
        if license_type is None:
            score = 0.0
        elif license_type not in self.license_scores:
            # this grok call will likely never happen because all
            # licenses not given on HF's license filter are labeled "other"
            prompt = (f"Rank the this HuggingFace model: {model_id}. "
                      "Give it of three scores, 1, 0.75, or 0.5. "
                      "base your ranking on the rakings of this "
                      "dictionary of licenses and their scores: "
                      f"{str(self.license_scores)}")
            response = self.grok_client.llm(prompt)
            new_prompt = ("Given this LLM output, please extract"
                          "the score it assigned and only output"
                          "the number and that's it. Here is the"
                          f"response: {response}")
            score = float(self.grok_client.llm(new_prompt))
        else:
            score = self.license_scores[license_type]

        return score


class SizeMetric(Metric):
    """
    Metric that finds the model size in bits and converts it
    to a score from 0 to 1 using a lookup table
    """
    name = "Model Size"
    key = "size_metric"
    maxModelBits = 8e11  # Decieded on 100GB being too big

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
        Estimate the model footprint using safetensor metadata.

        Parameters
        ----------
        safeTensorDict : dict[str, int]
            Mapping from precision label (e.g., ``"float16"``) to the number
            of tensors stored at that precision. The precision text is
            expected to contain the bit-width as digits.

        Returns
        -------
        int
            Total number of bits implied by the smallest precision entry.
        """
        bits = []
        for precision in safeTensorDict.keys():
            # Keys look like "float16" or "bfloat16"; pull out the digits to
            # determine the bit-width represented by that entry.
            param_size = int(''.join(ch for ch in precision if ch.isdigit()))
            n_params = param_size * safeTensorDict[precision]
            bits.append(n_params)
        # Pick the smallest bit count so we do not overestimate the footprint
        # when multiple precisions are present.
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
        have_safetensrs = 'safetensors' in card_data.keys()
        if have_safetensrs and 'parameters' in card_data['safetensors'].keys():
            params = card_data['safetensors']['parameters']
            bits = self.extract_bits_from_saftensor(params)
        # If not we will need to browse the repo
        else:
            files = browse_hf_repo(self.hf_client,
                                   model_id,
                                   repo_type="model",
                                   revision="main",
                                   recursive=True)
            files_filtered = []
            for f in files:
                if any(f[0].endswith(ext)
                       for ext in SizeMetric.commonModelFileEndings):
                    files_filtered.append(f)

            # No model files means we have no bits
            if len(files_filtered) == 0:
                bits = -1
            # Average the file sizes
            else:
                all_bits = [f[1] for f in files_filtered]
                bits = 8 * int(sum(all_bits) / len(all_bits))

        # Now that we have the bits, let's assign our score.
        # This will be based on a log scale between the number of bits of
        # the model and the number of bits in HF's biggest model.
        # The log is there to smooth things out and we divide by a factor
        # that will make the score approach 1 as the size fits into a
        # Jetson Nano.
        # Finally, we will clip between 0 and 1 in case of any extreme values.
        # For the Jetson Nano parameters we just want the model file to be
        # under 1.5GB (1.2e10 bits).
        if bits <= 0:
            return 0
        score_raw = (1-math.log(bits / SizeMetric.maxModelBits))
        score = score_raw / (1-math.log(1.2e10 / SizeMetric.maxModelBits))
        score = min(score, 1)
        score = max(score, 0)
        return score
