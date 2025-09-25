# src/Metrics.py
# THIS CODE WILL HANDLE THE METRIC OBJECTS
from __future__ import annotations

import ast
import math
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import quote, urlparse

from src.Client import GitClient, HFClient, PurdueClient
from src.logging_utils import get_logger
from src.utils import browse_hf_repo, injectHFBrowser

logger = get_logger(__name__)


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
    def compute(self, inputs: dict[str, Any],
                **kwargs: Any) -> float | dict[str, float]:
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
        float | dict[str, float]
            Either a scalar score between 0.0 and 1.0 or a mapping of
            device/platform identifiers to scores in that range.
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
        self.grok = PurdueClient(max_requests=100)

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
            logger.debug("Requesting usage extraction for text length %d",
                         len(text))
            response = self.grok.llm(prompt)
            return response.strip() if response else None
        except Exception:
            logger.info("Usage extraction failed via LLM", exc_info=True)
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

        logger.info("Computing ramp-up score for %s", url)
        # Fetch the full page text using Selenium
        full_page_text = injectHFBrowser(url)
        usage_text = self._extract_usage_section(full_page_text)

        if usage_text:
            char_count = len(usage_text)
            logger.debug("Usage text length for %s: %d", url, char_count)
        else:
            logger.info("No usage guidance found for %s", url)
            return 0.0

        # Weight long instructions logarithmically so modest increases in
        # length do not crater the score, while extremely long sections still
        # reduce it meaningfully.
        score = 1.0 / (1.0 + math.log1p(char_count / 500))
        score = max(0.0, min(score, 1.0))

        logger.info("Ramp-up score for %s: %.3f", url, score)
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
        self.grok_client = PurdueClient(max_requests=100)

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
        logger.info("Computing license score for %s", model_id)

        # try to get license from HFClient and assign a score
        model_info = self.hf_client.request("GET",
                                            f"/api/models/{model_id}")
        card_data = model_info["cardData"]
        license_type = card_data.get("license", None)
        if license_type is None:
            logger.debug("License not specified for %s", model_id)
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
            logger.debug("Derived license score %.2f for %s via LLM",
                         score, model_id)
        else:
            score = self.license_scores[license_type]
            logger.debug("Found license %s with score %.2f for %s",
                         license_type, score, model_id)

        logger.info("License score for %s: %.2f", model_id, score)
        return score


class SizeMetric(Metric):
    """
    Metric that estimates model footprint in bits and reports how well the
    model fits on a set of representative devices.
    """
    name = "Model Size"
    key = "size_metric"
    device_profiles: dict[str, tuple[float, float]] = {
        # (memory in GB, relative throughput multiplier)
        "raspberry_pi": (4.0, 0.35),
        "jetson_nano": (8.0, 0.85),
        "desktop_pc": (32.0, 1.0),
        "aws_server": (128.0, 1.0),
    }
    device_capacity_bits: dict[str, int] = {
        name: int(memory_gb * 1024**3 * 8 * perf_multiplier)
        for name, (memory_gb, perf_multiplier) in device_profiles.items()
    }

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

    def compute(self, inputs: dict[str, Any], **kwargs: Any) \
            -> dict[str, float]:
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
        dict[str, float]
            Mapping of deployment target to a score between 0.0 and 1.0 that
            captures how well the model fits the device's capacity.

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
        logger.info("Computing size score for %s", model_id)
        card_data = self.hf_client.request("GET", f"/api/models/{model_id}")

        bits = None
        # If we have access to safetensors, use that
        keys: Iterable[str] = []
        if isinstance(card_data, dict):
            keys = list(card_data.keys())
        have_safetensrs = 'safetensors' in keys
        if have_safetensrs and 'parameters' in card_data['safetensors'].keys():
            params = card_data['safetensors']['parameters']
            bits = self.extract_bits_from_saftensor(params)
            logger.debug("Using safetensor metadata for %s -> %s bits",
                         model_id, bits)
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
                logger.debug("No model files found for %s", model_id)
            # Average the file sizes
            else:
                all_bits = [f[1] for f in files_filtered]
                bits = 8 * int(sum(all_bits) / len(all_bits))
                logger.debug("Estimated bits for %s from %d file(s): %s",
                             model_id, len(files_filtered), bits)

        # Translate the bit-count into per-device deployability scores by
        # comparing the footprint against each device capacity (memory adjusted
        # by a throughput multiplier to reflect hardware speed) and clamping
        # the resulting ratio to the [0, 1] range.
        if bits is None or bits <= 0:
            logger.info("Size scores for %s default to 0 (no size info)",
                        model_id)
            return {device: 0.0 for device in self.device_capacity_bits}

        scores: dict[str, float] = {}
        for device, capacity_bits in self.device_capacity_bits.items():
            raw_score = capacity_bits / bits
            clamped = max(0.0, min(raw_score, 1.0))
            scores[device] = clamped
            logger.debug(
                "Size score for %s on %s: %.3f " +
                "(capacity_bits=%s, model_bits=%s)",
                model_id,
                device,
                clamped,
                capacity_bits,
                bits,
            )

        logger.info("Size scores for %s: %s", model_id, scores)
        return scores


class AvailabilityMetric(Metric):
    """
    Metric that checks whether a model has an associated dataset and codebase
    available. Awards 0.5 for each item found via the model card.

    This metric uses Selenium (via ``injectHFBrowser``) to retrieve the full
    rendered model page text and a Grok LLM to identify mentions/links to an
    available dataset and an available code repository.
    """

    name = "Availability"
    key = "availability_metric"

    def __init__(self) -> None:
        self.grok = PurdueClient(max_requests=100)
        self.last_details: dict[str, Any] = {}

    def _llm_detect_availability(self, page_text: str) \
            -> tuple[bool, bool, str, str]:
        """
        Use the Grok LLM to determine whether the page text indicates a
        dataset and/or a codebase are available.

        Parameters
        ----------
        page_text : str
            Visible text of the Hugging Face model page.

        Returns
        -------
        tuple[bool, bool, str, str]
            (dataset_available, codebase_available,
            dataset_evidence, codebase_evidence).
        """
        logger.debug("Running availability LLM check with text length %d",
                     len(page_text or ""))
        if not page_text:
            return (False, False, "", "")

        prompt = f"""
        You will be given the visible text of a Hugging Face model page.
        Determine if BOTH of the following are PRESENT AND AVAILABLE to users:

        1) A dataset: a specific dataset link/name indicating training or
        evaluation data,
           or a clear pointer to a dataset page (e.g.,
           huggingface.co/datasets/...,
           Kaggle dataset, etc.).
           If URL, just the URL is sufficient for evidence.
        2) A codebase: a concrete link to source code repository
        (e.g., GitHub/GitLab/Bitbucket) or an
           installable package with a repository reference. If just a reference
           to a repository, but no link, not sufficient. If URL, just the
           URL is sufficient for evidence.

        Respond STRICTLY in compact JSON with four fields:
        {{"dataset_available": <true|false>, "codebase_available":
        <true|false>,
        "dataset_evidence": "<short snippet or URL>",
        "codebase_evidence": "<short snippet or URL>"}}

        Text:
        {page_text}
        """

        try:
            raw = self.grok.llm(prompt)
            import json
            text = (raw or "").strip()
            if text.startswith("```"):
                text = text.strip('`')
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end+1]
            obj = json.loads(text)
            dataset = bool(obj.get("dataset_available", False))
            codebase = bool(obj.get("codebase_available", False))
            dataset_ev = str(obj.get("dataset_evidence", ""))[:500]
            codebase_ev = str(obj.get("codebase_evidence", ""))[:500]
            logger.debug("LLM availability result dataset=%s code=%s",
                         dataset, codebase)
            return (dataset, codebase, dataset_ev, codebase_ev)
        except Exception:
            logger.info("LLM parsing failed; falling back to heuristics",
                        exc_info=True)
            lower = page_text.lower()
            dataset_hits = any(
                kw in lower for kw in [
                    "huggingface.co/datasets/", " datasets/", "dataset:",
                    "trained on", "training data:", "evaluation dataset",
                    "kaggle.com/datasets", "dataset card"
                ]
            )
            codebase_hits = any(
                kw in lower for kw in [
                    "github.com/", "gitlab.com/", "bitbucket.org/",
                    "source code", "repository", "codebase"
                ]
            )
            # Heuristic evidence snippets
            dataset_ev = ""
            codebase_ev = ""
            if dataset_hits:
                for kw in [
                    "huggingface.co/datasets/", "kaggle.com/datasets",
                    "dataset card", "evaluation dataset"
                ]:
                    idx = lower.find(kw)
                    if idx != -1:
                        dataset_ev = page_text[max(0, idx-40): idx+120]
                        break
            if codebase_hits:
                for kw in [
                    "github.com/", "gitlab.com/", "bitbucket.org/",
                    "source code", "repository"
                ]:
                    idx = lower.find(kw)
                    if idx != -1:
                        codebase_ev = page_text[max(0, idx-40): idx+120]
                        break
            logger.debug("Heuristic availability result dataset=%s code=%s",
                         dataset_hits, codebase_hits)
            return (dataset_hits, codebase_hits, dataset_ev, codebase_ev)

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Compute the availability score based on dataset/codebase presence.

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
            A score between 0.0 and 1.0. Dataset availability contributes 0.5,
            and codebase availability contributes 0.5.

        Raises
        ------
        ValueError
            If 'model_url' is missing from inputs.
        """

        model_url = inputs.get("model_url")
        if not isinstance(model_url, str) or not model_url.strip():
            raise ValueError("Model link not found in input dictionary")

        logger.info("Computing availability score for %s", model_url)

        def _nonempty_url(v: Any) -> bool:
            return isinstance(v, str) and v.strip().startswith("http")

        explicit_dataset = inputs.get("dataset_url")
        explicit_git = inputs.get("git_url")

        has_dataset = _nonempty_url(explicit_dataset)
        has_code = _nonempty_url(explicit_git)

        dataset_ev = explicit_dataset if has_dataset else ""
        code_ev = explicit_git if has_code else ""

        if not (has_dataset and has_code):
            try:
                page_text = injectHFBrowser(model_url)
            except Exception:
                page_text = ""
            d_avail, c_avail, d_ev, c_ev = \
                self._llm_detect_availability(page_text)
            if d_avail and not has_dataset:
                has_dataset = True
                dataset_ev = d_ev
            if c_avail and not has_code:
                has_code = True
                code_ev = c_ev

        self.last_details = {
            "dataset_available": has_dataset,
            "codebase_available": has_code,
            "dataset_evidence": dataset_ev,
            "codebase_evidence": code_ev,
        }
        # print(self.last_details)
        score = (0.5 if has_dataset else 0.0) + (0.5 if has_code else 0.0)
        logger.info("Availability score for %s: %.2f", model_url, score)
        return score


class PerformanceClaimsMetric(Metric):
    """
    Metric that inspects the model card/README to detect
    reported benchmarks and performance claims.
    """
    name = "Performance Claims"
    key = "performance_claims"

    def __init__(self):
        self.hf_client = HFClient(max_requests=100)
        self.grok_client = PurdueClient(max_requests=100)

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Compute the metric score from parsed inputs.
            Parsed inputs required by the metric. Must include a key called
            'model_url' with its corresponding correct link

        **kwargs : Any
            Optional per-metric tuning parameters.
            A score between 0.0 and 1.0.

        Raises
        ------
        RuntimeError
            If no valid HF model URL is found in the dict
        """
        # appropriate URL must be in the dict
        if "model_url" not in inputs.keys():
            raise ValueError("No good link found in input dictionary")

        model_url = inputs["model_url"]
        model_id = model_url.split("https://huggingface.co/")[-1]
        logger.info("Computing performance claims score for %s", model_id)

        try:
            logger.debug("Fetching README via API for %s", model_id)
            card_data = self.hf_client.request(
                "GET",
                f"/{model_id}/resolve/main/README.md",
            ).splitlines()
            source = "api"
        except Exception:
            logger.info("Falling back to rendered page for %s", model_id)
            rendered = injectHFBrowser(model_url)
            card_data = rendered.splitlines()
            source = "rendered_page"
        logger.debug("Performance claims text source=%s lines=%d for %s",
                     source,
                     len(card_data),
                     model_id)

        # We can only put so much into llm input
        # so we need to try to find text only relating benchmarks
        ranges = []
        for i, line in enumerate(card_data):
            words = ["benchmark", "performance", "accuracy", "eval"]
            if any(word in line.lower() for word in words):
                start = max(0, i - 5)
                end = min(len(card_data), i + 5 + 1)
                ranges.append((start, end))

        logger.debug("Identified %d candidate snippet range(s) for %s",
                     len(ranges),
                     model_id)
        if not ranges:
            logger.info("No benchmark keywords detected for %s; using fallback prompt context",
                        model_id)

        # Merge overlapping ranges
        merged: list[list[int]] = []
        for start, end in sorted(ranges):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        # Aggregate results into one string
        results = ""
        for start, end in merged:
            results += "\n".join(card_data[start:end])
            # just in case my original limits still
            # capture too much text:
            if len(results) > 6000:
                break

        logger.debug("Prepared %d characters of performance evidence for %s",
                     len(results),
                     model_id)

        # Prompt the LLM to give score
        prompt = (
            f"given the following snippets from a readme: {results}"
            "output a 1.0 if the readme contains performance claims "
            "with some form of benchmark test results. "
            "output a 0.0 if there are no performance claims or "
            "benchmark results. "
        )
        logger.info("Querying LLM for performance claims evaluation on %s",
                    model_id)
        response = self.grok_client.llm(prompt)
        new_prompt = ("Given this LLM output, please extract"
                      "the score it assigned and only output"
                      "the number and that's it. Here is the"
                      f"response: {response}"
                      "the score should be either 0.0 or 1.0")
        score = float(self.grok_client.llm(new_prompt))

        logger.info("Performance claims score for %s: %.1f",
                    model_id,
                    score)

        return score


class DatasetQuality(Metric):
    """
    Evaluate dataset quality by combining reuse and community engagement.

    The metric inspects Hugging Face metadata to determine how often a
    dataset is reused across models and how many likes it has accrued,
    yielding a bounded score that favors broad adoption.
    """

    name = "Dataset Quality"
    key = "dataset_quality"

    def __init__(self, hf_client: Optional[HFClient] = None) -> None:
        self.hf_client = HFClient(max_requests=100)
        self.last_details: dict[str, Any] = {}

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Score a dataset by blending reuse counts with community likes.

        Parameters
        ----------
        inputs : dict[str, Any]

            Parser output that may contain dataset and/or model URLs.
        **kwargs : Any
            Unused placeholder for interface compatibility.

        Returns
        -------
        float
            Weighted dataset quality score clamped to ``[0.0, 1.0]``.
        """

        # Step 1: resolve which dataset we should inspect
        dataset_id = self._extract_dataset_id(inputs)
        if not dataset_id:
            self.last_details = {"reason": "dataset_not_found"}
            logger.info("Dataset quality: no dataset found")
            return 0.0

        logger.info("Computing dataset quality for %s", dataset_id)

        # Step 2: pull the dataset metadata from Hugging Face
        payload = self._fetch_dataset(dataset_id)
        if payload is None:
            self.last_details = {
                "dataset_id": dataset_id,
                "reason": "hf_api_error",
            }
            logger.info("Dataset quality: API error for %s", dataset_id)
            return 0.0

        likes = self._safe_int(payload.get("likes"))
        use_count = self.count_models_for_dataset(dataset_id)
        logger.debug("Dataset %s likes=%d use_count=%d",
                     dataset_id, likes, use_count)

        # Step 3: convert engagement numbers into bounded scores
        likes_score = self._squash_score(likes, scale=250)
        use_score = self._squash_score(use_count, scale=40)

        score = (0.6 * use_score) + (0.4 * likes_score)
        score = max(0.0, min(1.0, score))

        self.last_details = {
            "dataset_id": dataset_id,
            "likes": likes,
            "use_count": use_count,
            "likes_score": likes_score,
            "use_score": use_score,
        }
        logger.info("Dataset quality score for %s: %.2f", dataset_id, score)
        return score

    def _extract_dataset_id(self, inputs: Mapping[str, Any]) -> Optional[str]:
        """
        Resolve the primary dataset slug from parser output or model metadata.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Parsed artifacts describing the model and any referenced datasets.

        Returns
        -------
        Optional[str]
            Normalized ``owner/name`` dataset slug, or ``None`` when absent.
        """
        # Prefer explicit dataset URLs that the parser already categorized.
        dataset_url = inputs.get("dataset_url")
        if isinstance(dataset_url, str):
            slug = self._dataset_slug_from_url(dataset_url)
            if slug:
                return slug

        # Fall back to inspecting the referenced model's metadata.
        model_url = inputs.get("model_url")
        if not isinstance(model_url, str):
            return None

        model_id = self._model_id_from_url(model_url)
        if not model_id:
            return None

        model_info = self._fetch_model(model_id)
        if not isinstance(model_info, Mapping):
            return None

        candidates: list[Any] = []
        # Older model cards advertise datasets under these top-level keys.
        candidates.extend([model_info.get("dataset"),
                           model_info.get("datasets")])
        card = model_info.get("cardData")
        if isinstance(card, Mapping):
            # Newer model cards store richer data inside ``cardData``.
            candidates.extend([card.get("dataset"), card.get("datasets")])
        for candidate in candidates:
            slug = self._first_dataset_slug(candidate)
            if slug:
                return slug

        tags = model_info.get("tags")
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("datasets:"):
                    # Tags occasionally encode dataset information,
                    # e.g. "datasets:mnist".
                    ref = tag.split(":", 1)[1]
                    slug = self._normalize_dataset_reference(ref)
                    if slug:
                        return slug

        return None

    def _fetch_dataset(self, dataset_id: str) -> Optional[Mapping[str, Any]]:
        """
        Fetch dataset metadata for ``dataset_id`` via the Hugging Face API.

        Parameters
        ----------
        dataset_id : str
            Normalized dataset slug (``owner/name``) to request from the API.

        Returns
        -------
        Optional[Mapping[str, Any]]
            Parsed dataset payload, or ``None`` if the request fails.
        """
        try:
            logger.debug("Fetching dataset metadata for %s", dataset_id)
            data = self.hf_client.request(
                "GET",
                f"/api/datasets/{quote(dataset_id, safe='/@.-')}",
            )
        except Exception:
            logger.info("Failed to fetch dataset %s",
                        dataset_id,
                        exc_info=True)
            return None
        return data if isinstance(data, Mapping) else None

    def _fetch_model(self, model_id: str) -> Optional[Mapping[str, Any]]:
        """
        Fetch model metadata so we can inspect the declared datasets.

        Parameters
        ----------
        model_id : str
            Normalized model slug (``owner/name``) to request from the API.

        Returns
        -------
        Optional[Mapping[str, Any]]
            Parsed model payload, or ``None`` when the request fails.
        """
        try:
            logger.debug("Fetching model metadata for %s", model_id)
            data = self.hf_client.request(
                "GET",
                f"/api/models/{quote(model_id, safe='/@.-')}",
            )
        except Exception:
            logger.info("Failed to fetch model %s", model_id, exc_info=True)
            return None
        return data if isinstance(data, Mapping) else None

    @staticmethod
    def _safe_int(value: Any) -> int:
        """
        Convert ``value`` to a non-negative integer, defaulting to ``0``.

        Parameters
        ----------
        value : Any
            Raw value retrieved from Hugging Face metadata.

        Returns
        -------
        int
            Parsed integer or ``0`` if conversion fails.
        """
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _squash_score(value: int, *, scale: int) -> float:
        """
        Compress counts into the ``[0.0, 1.0]`` interval with log roll-off.

        Parameters
        ----------
        value : int
            Raw count to transform.
        scale : int
            Reference value that maps to a score near ``1.0``.

        Returns
        -------
        float
            Log-scaled score bounded to ``[0.0, 1.0]``.
        """
        if value <= 0 or scale <= 0:
            return 0.0
        return min(1.0, math.log1p(value) / math.log1p(scale))

    def _first_dataset_slug(self, value: Any) -> Optional[str]:
        """
        Return the first normalized dataset slug contained in ``value``.

        Parameters
        ----------
        value : Any
            String or list originating from model metadata.

        Returns
        -------
        Optional[str]
            Normalized ``owner/name`` slug, or ``None`` if none are found.
        """
        if isinstance(value, str):
            return self._normalize_dataset_reference(value)
        if isinstance(value, list):
            for item in value:
                slug = self._normalize_dataset_reference(item)
                if slug:
                    return slug
        return None

    def _normalize_dataset_reference(self, reference: Any) -> Optional[str]:
        """
        Normalize free-form dataset references into ``owner/name`` format.

        Parameters
        ----------
        reference : Any
            Arbitrary dataset hint collected from metadata or tags.

        Returns
        -------
        Optional[str]
            Normalized dataset slug suitable for API requests.
        """
        if not isinstance(reference, str):
            return None
        text = reference.strip()
        if not text:
            return None
        if text.startswith("datasets:"):
            text = text.split(":", 1)[1]
        if text.startswith("http://") or text.startswith("https://"):
            return self._dataset_slug_from_url(text)
        if text.startswith("huggingface.co/"):
            return self._dataset_slug_from_url(f"https://{text}")
        if "/" in text:
            parts = [p for p in text.split("/") if p]
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            if parts:
                return parts[0]
        return None

    @staticmethod
    def _dataset_slug_from_url(url: str) -> Optional[str]:
        """
        Extract ``owner/name`` slug from a Hugging Face dataset URL.

        Parameters
        ----------
        url : str
            Dataset URL copied from the Hub.

        Returns
        -------
        Optional[str]
            Normalized slug when extraction succeeds, else ``None``.
        """
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            return None
        if path.startswith("datasets/"):
            path = path[len("datasets/"):]
        segments = [
            seg for seg in path.split("/")
            if seg not in {"blob", "tree", "resolve", "main", "raw", "viewer"}
        ]
        if not segments:
            return None
        if len(segments) >= 2:
            return f"{segments[0]}/{segments[1]}"
        return segments[0]

    @staticmethod
    def _model_id_from_url(url: str) -> Optional[str]:
        """
        Extract ``owner/name`` slug from a Hugging Face model URL.

        Parameters
        ----------
        url : str
            Model URL copied from the Hub.

        Returns
        -------
        Optional[str]
            Normalized slug when extraction succeeds, else ``None``.
        """
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path or path.startswith("datasets/"):
            return None
        parts = [part for part in path.split("/") if part]
        if not parts:
            return None
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return parts[0]

    def count_models_for_dataset(self, dataset_id: str,
                                 limit: int = 1000) -> int:
        """
        Count models on the Hub that self-report using ``dataset_id``.

        Parameters
        ----------
        dataset_id : str
            Normalized dataset slug to pass through the Hub filters.
        limit : int, optional
            Maximum number of results to request per API call,
            by default ``1000``.

        Returns
        -------
        int
            Unique model count associated with the dataset.
        """
        slug = dataset_id.split("/")[-1]

        # Common ways authors tag datasets in model cards
        filters: Iterable[str] = {
            slug,                          # e.g. "imagenet-1k"
            f"dataset:{slug}",             # e.g. "dataset:imagenet-1k"
            dataset_id,                    # e.g. "ILSVRC/imagenet-1k"
            f"dataset:{dataset_id}",       # e.g. "dataset:ILSVRC/imagenet-1k"
        }

        seen: set[str] = set()
        for f in filters:
            try:
                models: list[dict[str, Any]] = self.hf_client.request(
                    "GET",
                    "/api/models",
                    params={"filter": f, "limit": limit}
                )
            except Exception:
                continue
            if not isinstance(models, list):
                continue
            for m in models:
                mid = m.get("modelId")
                if mid:
                    # Track models uniquely to avoid
                    # double-counting across filters.
                    seen.add(mid)

        count = len(seen)
        logger.debug("Dataset %s associated with %d model(s)",
                     dataset_id,
                     count)
        return count


class CodeQuality(Metric):
    """
    Assess codebases by combining lint heuristics, typing coverage,
    and LLM judgment.

    The metric fetches Python sources linked from a model card or explicit
    repository URL, runs lightweight static checks, and asks an LLM to
    estimate engineering quality. When no code is available, it falls back
    to interpreting the model card alone.
    """

    name = "Code Quality"
    key = "code_quality"

    def __init__(
        self,
        hf_client: Optional[HFClient] = None,
        grok_client: Optional[PurdueClient] = None,
    ) -> None:
        self.hf_client = hf_client or HFClient(max_requests=10)
        self.grok = grok_client or PurdueClient(max_requests=100)
        self.last_details: dict[str, Any] = {}

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Resolve source code, analyse it, and optionally fall back to the
        model card.

        Parameters
        ----------
        inputs : dict[str, Any]
            Parser output that may include ``git_url`` or ``model_url``
            pointing to the codebase or model card.
        **kwargs : Any
            Placeholder for interface compatibility; unused.

        Returns
        -------
        float
            Aggregate quality score bounded to ``[0.0, 1.0]``.
        """

        logger.info("Computing code quality")
        code_files, origin, card_text = self._load_code(inputs)
        logger.debug("Code load origin=%s file_count=%d",
                     origin, len(code_files))
        if code_files:
            lint_score = self._lint_score(code_files)
            typing_score = self._typing_score(code_files)
            llm_score = self._llm_code_rating(code_files)
            score = (lint_score + typing_score + llm_score) / 3.0
            score = max(0.0, min(1.0, score))

            self.last_details = {
                "origin": origin,
                "lint_score": lint_score,
                "typing_score": typing_score,
                "llm_score": llm_score,
                "file_count": len(code_files),
            }
            logger.info("Code quality score from codebase: %.2f", score)
            return score

        model_url = inputs.get("model_url")
        if not card_text and isinstance(model_url, str):
            card_text = self._model_card_text(model_url)
        fallback_score = self._llm_card_rating(card_text)
        self.last_details = {
            "origin": "model_card",
            "llm_score": fallback_score,
            "card_available": bool(card_text),
        }
        logger.info("Code quality fallback score: %.2f", fallback_score)
        return fallback_score

    # ------------------------------------------------------------------
    # Source resolution helpers
    # ------------------------------------------------------------------
    def _load_code(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[dict[str, str], str, str]:
        """
        Locate Python sources and return them alongside provenance
        metadata.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Parser artefacts containing potential ``git_url`` or
            ``model_url`` keys.

        Returns
        -------
        tuple[dict[str, str], str, str]
            Mapping of relative paths to file contents, the origin label,
            and any cached model card text for reuse when code is
            unavailable.
        """

        card_text = ""
        git_url = inputs.get("git_url")
        if isinstance(git_url, str):
            logger.debug("Attempting to load code from explicit git_url %s",
                         git_url)
            files = self._load_from_github(git_url)
            if files:
                logger.debug("Loaded %d file(s) from %s", len(files), git_url)
                return files, "github", card_text

        model_url = inputs.get("model_url")
        if isinstance(model_url, str):
            logger.debug("Inspecting model card for %s", model_url)
            card_text = self._model_card_text(model_url)
            gh_url = self._github_from_card(card_text)
            if gh_url:
                logger.debug("Found GitHub URL %s via model card", gh_url)
                files = self._load_from_github(gh_url)
                if files:
                    logger.debug("Loaded %d file(s) from %s",
                                 len(files),
                                 gh_url)
                    return files, "github_from_card", card_text

        return {}, "", card_text

    def _load_from_github(
        self,
        url: str,
        *,
        limit: int = 20,
    ) -> dict[str, str]:
        """
        Clone a GitHub repository and read a bounded set of Python files.

        Parameters
        ----------
        url : str
            HTTPS URL to the GitHub repository.
        limit : int, optional
            Maximum number of Python files to ingest, defaults to ``20``.

        Returns
        -------
        dict[str, str]
            Mapping of relative file paths to their source text.
        """

        logger.debug("Loading code from GitHub repo %s", url)
        with tempfile.TemporaryDirectory(prefix="code-metric-") as tmpdir:
            dest = Path(tmpdir) / "repo"
            if not self._clone_repo(url, dest):
                logger.info("Failed to clone %s", url)
                return {}
            files = self._read_python_files(dest, limit=limit)
            logger.debug("Read %d Python file(s) from %s", len(files), url)
            return files

    def _clone_repo(self, url: str, dest: Path) -> bool:
        """
        Clone ``url`` into ``dest`` and signal success.

        Parameters
        ----------
        url : str
            Git repository URL to clone.
        dest : Path
            Filesystem path where the shallow clone should be created.

        Returns
        -------
        bool
            ``True`` when the clone completes, otherwise ``False``.
        """

        try:
            logger.debug("Cloning repo %s", url)
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=45,
            )
        except (subprocess.SubprocessError, OSError):
            logger.info("Clone failed for %s", url, exc_info=True)
            return False
        logger.debug("Clone succeeded for %s", url)
        return True

    def _read_python_files(self, root: Path, *, limit: int) -> dict[str, str]:
        """
        Collect up to ``limit`` tracked Python files from ``root``.

        Parameters
        ----------
        root : Path
            Directory containing the cloned repository.
        limit : int
            Maximum number of files to return.

        Returns
        -------
        dict[str, str]
            Mapping of relative file paths to their corresponding source text.
        """

        candidates: list[str] = []
        try:
            git_client = GitClient(max_requests=100, repo_path=str(root))
            candidates = [
                path
                for path in git_client.list_files()
                if path.endswith(".py")
            ]
        except Exception:
            logger.debug("git ls-files failed in %s", root, exc_info=True)
            candidates = []

        if not candidates:
            candidates = [
                str(path.relative_to(root))
                for path in sorted(root.rglob("*.py"))
            ]
        logger.debug("Found %d Python candidate(s) in %s",
                     len(candidates), root)

        results: dict[str, str] = {}
        for rel_path in candidates:
            if len(results) >= limit:
                break
            path = root / rel_path
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if text.strip():
                results[rel_path] = text
        logger.debug("Returning %d Python file(s) from %s",
                     len(results), root)
        return results

    def _github_from_card(self, card_text: str) -> Optional[str]:
        """
        Extract a GitHub repository link from rendered model card text.

        Parameters
        ----------
        card_text : str
            Full model card contents sourced from Hugging Face.

        Returns
        -------
        Optional[str]
            First matching GitHub URL if present, else ``None``.
        """

        if not card_text:
            logger.debug("No card text provided for GitHub extraction")
            return None
        match = re.search(
            r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
            card_text,
        )
        if match:
            url = match.group(0)
            logger.debug("Extracted GitHub URL %s from card", url)
            return url
        logger.debug("No GitHub URL found in card text")
        return None

    # ------------------------------------------------------------------
    # Static analysis helpers
    # ------------------------------------------------------------------
    def _lint_score(self, code_files: Mapping[str, str]) -> float:
        """
        Approximate lint quality using lightweight style heuristics.

        Parameters
        ----------
        code_files : Mapping[str, str]
            Mapping of file paths to Python source text.

        Returns
        -------
        float
            Heuristic lint compliance score within ``[0.0, 1.0]``.
        """

        logger.debug("Computing lint score for %d file(s)", len(code_files))
        total = 0
        issues = 0

        for text in code_files.values():
            for raw_line in text.splitlines():
                stripped = raw_line.rstrip("\n")
                if not stripped.strip():
                    continue

                total += 1
                failure = False

                if len(stripped) > 100:
                    failure = True
                if stripped.rstrip() != stripped:
                    failure = True
                if "\t" in stripped:
                    failure = True

                leading_spaces = len(stripped) - len(stripped.lstrip(" "))
                if leading_spaces and leading_spaces % 4 != 0:
                    failure = True

                if failure:
                    issues += 1

        if total == 0:
            logger.debug("No lines seen for lint score; defaulting to 0.5")
            return 0.5
        compliant_ratio = 1.0 - (issues / total)
        # Anything below 90% compliant is treated as a failure, and
        # 100% compliant receives full credit. Linearly scale in-between.
        score = max(0.0, min(1.0, (compliant_ratio - 0.9) / 0.1))
        logger.debug("Lint compliance ratio=%.3f score=%.3f",
                     compliant_ratio, score)
        return score

    def _typing_score(self, code_files: Mapping[str, str]) -> float:
        """
        Measure the proportion of functions that provide complete type hints.

        Parameters
        ----------
        code_files : Mapping[str, str]
            Mapping of file paths to Python source text.

        Returns
        -------
        float
            Fraction of typed functions; defaults to ``0.5`` when none found.
        """

        logger.debug("Computing typing score for %d file(s)", len(code_files))
        total_funcs = 0
        typed_funcs = 0

        for text in code_files.values():
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_funcs += 1
                    if self._function_is_typed(node):
                        typed_funcs += 1

        if total_funcs == 0:
            logger.debug("No functions found; typing score defaults to 0.5")
            return 0.5
        score = typed_funcs / total_funcs
        logger.debug("Typing score %.3f (%d/%d)",
                     score, typed_funcs, total_funcs)
        return score

    def _function_is_typed(self, node: ast.AST) -> bool:
        """
        Determine whether a function annotates all parameters and the
        return value.

        Parameters
        ----------
        node : ast.AST
            Function definition node extracted from the AST.

        Returns
        -------
        bool
            ``True`` when every parameter and the return value are
            annotated.
        """

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False

        params = list(node.args.posonlyargs) + list(node.args.args)
        if params and params[0].arg in {"self", "cls"}:
            params = params[1:]

        params += list(node.args.kwonlyargs)

        for param in params:
            if param.annotation is None:
                return False

        if node.args.vararg and node.args.vararg.annotation is None:
            return False
        if node.args.kwarg and node.args.kwarg.annotation is None:
            return False

        return node.returns is not None

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def _llm_code_rating(self, code_files: Mapping[str, str]) -> float:
        """
        Ask the Grok LLM to assess engineering quality of the provided code.

        Parameters
        ----------
        code_files : Mapping[str, str]
            Mapping of file paths to Python source text.

        Returns
        -------
        float
            Parsed LLM rating normalized to ``[0.0, 1.0]``; defaults to
            ``0.5`` when the snippet is empty or the request fails.
        """

        snippet = self._code_snippet(code_files)
        if not snippet:
            logger.debug("No snippet available for LLM code rating")
            return 0.5

        prompt = (
            "Rate the following Python code's engineering quality on a "
            "scale from 0 to 1, where 0 is extremely poor and 1 is "
            "excellent. Consider readability, structure, tests, and "
            "maintainability. Respond with only the numeric rating.\n\n"
            f"```python\n{snippet}\n```"
        )

        try:
            logger.debug("Requesting LLM code rating (snippet length %d)",
                         len(snippet))
            raw = self.grok.llm(prompt)
            score = self._parse_llm_score(raw)
            logger.debug("LLM code rating %.3f", score)
            return score
        except Exception:
            logger.info("LLM code rating failed", exc_info=True)
            return 0.5

    def _llm_card_rating(self, card_text: str) -> float:
        """
        Generate a fallback LLM-based quality score from model card text.

        Parameters
        ----------
        card_text : str
            Rendered model card contents.

        Returns
        -------
        float
            Parsed LLM rating in ``[0.0, 1.0]``; defaults to ``0.3`` on
            failure.
        """

        if not card_text:
            logger.debug("No card text; defaulting LLM card rating to 0.3")
            return 0.3

        prompt = (
            "Based on this Hugging Face model card, estimate the quality "
            "of the associated codebase. Return a number between 0 and 1 "
            "(0=very poor, 1=excellent). Respond with only the numeric "
            "rating.\n\n"
            f"{shorten(card_text, width=3500, placeholder='...')}"
        )

        try:
            logger.debug("Requesting LLM card rating (text length %d)",
                         len(card_text))
            raw = self.grok.llm(prompt)
            score = self._parse_llm_score(raw)
            logger.debug("LLM card rating %.3f", score)
            return score
        except Exception:
            logger.info("LLM card rating failed", exc_info=True)
            return 0.3

    def _code_snippet(
        self,
        code_files: Mapping[str, str],
        *,
        limit: int = 3500,
    ) -> str:
        """
        Concatenate a bounded sample of code to feed into the LLM prompt.

        Parameters
        ----------
        code_files : Mapping[str, str]
            Mapping of file paths to Python source text.
        limit : int, optional
            Maximum number of characters to include, defaults to ``3500``.

        Returns
        -------
        str
            Truncated multi-file snippet formatted for LLM consumption.
        """

        pieces: list[str] = []
        remaining = limit

        for path, text in code_files.items():
            header = f"# File: {path}\n"
            budget = max(0, remaining - len(header))
            if budget <= 0:
                break
            body = text.strip()
            snippet = body[:budget]
            pieces.append(header + snippet)
            remaining -= len(header) + len(snippet)
            if remaining <= 0:
                break

        snippet = "\n\n".join(pieces)
        logger.debug("Constructed code snippet of length %d", len(snippet))
        return snippet

    def _parse_llm_score(self, raw: Any) -> float:
        """
        Parse a numeric score from the LLM response text.

        Parameters
        ----------
        raw : Any
            Value returned by the LLM client.

        Returns
        -------
        float
            Clamped numeric score with ``0.5`` as the fallback when
            parsing fails.
        """

        if raw is None:
            logger.debug("LLM score parsing fallback (None)")
            return 0.5
        text = str(raw).strip()
        try:
            value = float(text.split()[0])
        except (ValueError, IndexError):
            logger.debug("LLM score parsing fallback (invalid response)")
            return 0.5
        score = max(0.0, min(1.0, value))
        logger.debug("Parsed LLM score %.3f from response", score)
        return score

    def _model_card_text(self, url: Optional[str]) -> str:
        """
        Retrieve model card text via the Hugging Face API or browser helper.

        Parameters
        ----------
        url : Optional[str]
            Hugging Face model URL from which to fetch the card.

        Returns
        -------
        str
            Markdown or HTML card contents, or an empty string when
            unavailable.
        """

        if not url:
            logger.debug("No URL provided for model card fetch")
            return ""
        model_id = DatasetQuality._model_id_from_url(url)
        if model_id:
            try:
                logger.debug("Fetching README for %s", model_id)
                data = self.hf_client.request(
                    "GET",
                    f"/{model_id}/resolve/main/README.md",
                )
                if isinstance(data, bytes):
                    data = data.decode("utf-8", errors="ignore")
                if isinstance(data, str) and data.strip():
                    logger.debug("Retrieved README (%d chars) for %s",
                                 len(data), model_id)
                    return data
            except Exception:
                logger.info("Failed to fetch README for %s", model_id,
                            exc_info=True)
        try:
            logger.debug("Falling back to browser fetch for %s", url)
            text = injectHFBrowser(url)
            logger.debug("Browser fetch returned %d chars", len(text))
            return text
        except Exception:
            logger.info("Browser fetch failed for %s", url, exc_info=True)
            return ""
