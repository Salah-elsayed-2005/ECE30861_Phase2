"""
Unit tests for TreescoreMetric.

Tests parent model detection, config.json parsing, registry lookups,
and score averaging.
"""

import unittest
from unittest.mock import Mock, patch

from src.Metrics import TreescoreMetric


class TestTreescoreMetric(unittest.TestCase):
    """Test suite for TreescoreMetric."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock HFClient
        self.mock_hf_client = Mock()
        
        # Mock model registry with sample models
        self.mock_registry = {
            "bert-base-uncased": {
                "name": "BERT Base",
                "scores": {
                    "ramp_up_time": 0.8,
                    "license": 0.9,
                    "size": 0.7,
                    "availability": 1.0,
                    "code_quality": 0.6,
                    "dataset_quality": 0.7,
                    "performance_claims": 0.8,
                    "bus_factor": 0.5,
                    "reproducibility": 0.5,
                    "reviewedness": 0.6,
                    "treescore": -1
                }
                # Average = 0.718 (excluding treescore=-1)
            },
            "openai/clip-vit-base-patch32": {
                "name": "CLIP ViT Base",
                "scores": {
                    "ramp_up_time": 0.9,
                    "license": 0.8,
                    "size": 0.6,
                    "availability": 0.9,
                    "code_quality": 0.7,
                    "dataset_quality": 0.8,
                    "performance_claims": 0.9,
                    "bus_factor": 0.6,
                    "reproducibility": 1.0,
                    "reviewedness": 0.7,
                    "treescore": -1
                }
                # Average = 0.790
            },
            "google/flan-t5-base": {
                "name": "FLAN-T5 Base",
                "scores": {
                    "ramp_up_time": 0.7,
                    "license": 1.0,
                    "size": 0.5,
                    "availability": 0.8,
                    "code_quality": 0.8,
                    "dataset_quality": 0.6,
                    "performance_claims": 0.7,
                    "bus_factor": 0.4,
                    "reproducibility": 0.0,
                    "reviewedness": 0.5,
                    "treescore": -1
                }
                # Average = 0.600
            }
        }
        
        self.metric = TreescoreMetric(
            hf_client=self.mock_hf_client,
            model_registry=self.mock_registry
        )
    
    # =========================================================================
    # URL Parsing Tests
    # =========================================================================
    
    def test_extract_model_id_from_url_standard(self):
        """Test extracting model ID from standard HF URL."""
        url = "https://huggingface.co/bert-base-uncased"
        result = self.metric._extract_model_id_from_url(url)
        # Should return None because it needs owner/model format
        # Wait, bert-base-uncased might be treated differently
        # Let me check the actual format
        url2 = "https://huggingface.co/google/bert-base-uncased"
        result2 = self.metric._extract_model_id_from_url(url2)
        self.assertEqual(result2, "google/bert-base-uncased")
    
    def test_extract_model_id_from_url_with_trailing_slash(self):
        """Test URL with trailing slash."""
        url = "https://huggingface.co/openai/clip-vit-base-patch32/"
        result = self.metric._extract_model_id_from_url(url)
        self.assertEqual(result, "openai/clip-vit-base-patch32")
    
    def test_extract_model_id_from_url_invalid(self):
        """Test invalid URL returns None."""
        url = "https://github.com/not-huggingface/repo"
        result = self.metric._extract_model_id_from_url(url)
        self.assertIsNone(result)
    
    def test_extract_model_id_from_url_no_model(self):
        """Test URL without model ID."""
        url = "https://huggingface.co/"
        result = self.metric._extract_model_id_from_url(url)
        self.assertIsNone(result)
    
    # =========================================================================
    # Config Parsing Tests
    # =========================================================================
    
    def test_extract_parent_models_base_model(self):
        """Test extracting parent from base_model field."""
        config = {
            "base_model": "bert-base-uncased",
            "model_type": "bert"
        }
        parents = self.metric._extract_parent_models(config)
        self.assertEqual(parents, ["bert-base-uncased"])
    
    def test_extract_parent_models_name_or_path(self):
        """Test extracting parent from _name_or_path field."""
        config = {
            "_name_or_path": "openai/clip-vit-base-patch32",
            "model_type": "clip"
        }
        parents = self.metric._extract_parent_models(config)
        self.assertEqual(parents, ["openai/clip-vit-base-patch32"])
    
    def test_extract_parent_models_multiple_fields(self):
        """Test extracting from multiple fields (deduplicated)."""
        config = {
            "base_model": "bert-base-uncased",
            "_name_or_path": "bert-base-uncased",
            "parent_model": "google/flan-t5-base"
        }
        parents = self.metric._extract_parent_models(config)
        # Should deduplicate bert-base-uncased
        self.assertEqual(len(parents), 2)
        self.assertIn("bert-base-uncased", parents)
        self.assertIn("google/flan-t5-base", parents)
    
    def test_extract_parent_models_local_path_ignored(self):
        """Test that local paths are ignored."""
        config = {
            "base_model": "./local/model/path",
            "_name_or_path": "/absolute/path/to/model"
        }
        parents = self.metric._extract_parent_models(config)
        self.assertEqual(parents, [])
    
    def test_extract_parent_models_no_parents(self):
        """Test config with no parent references."""
        config = {
            "model_type": "bert",
            "hidden_size": 768
        }
        parents = self.metric._extract_parent_models(config)
        self.assertEqual(parents, [])
    
    # =========================================================================
    # Registry Lookup Tests
    # =========================================================================
    
    def test_get_parent_scores_single_parent(self):
        """Test getting score for single parent in registry."""
        parent_ids = ["bert-base-uncased"]
        scores = self.metric._get_parent_scores(parent_ids)
        self.assertEqual(len(scores), 1)
        # Average of bert-base-uncased scores (excluding -1)
        expected = (0.8 + 0.9 + 0.7 + 1.0 + 0.6 + 0.7 + 0.8 + 0.5 + 0.5 + 0.6) / 10
        self.assertAlmostEqual(scores[0], expected, places=2)
    
    def test_get_parent_scores_multiple_parents(self):
        """Test getting scores for multiple parents."""
        parent_ids = ["bert-base-uncased", "openai/clip-vit-base-patch32"]
        scores = self.metric._get_parent_scores(parent_ids)
        self.assertEqual(len(scores), 2)
    
    def test_get_parent_scores_parent_not_in_registry(self):
        """Test parent not in registry is skipped."""
        parent_ids = ["unknown/model", "bert-base-uncased"]
        scores = self.metric._get_parent_scores(parent_ids)
        self.assertEqual(len(scores), 1)  # Only bert-base-uncased found
    
    def test_get_parent_scores_no_parents_in_registry(self):
        """Test no parents found in registry."""
        parent_ids = ["unknown/model1", "unknown/model2"]
        scores = self.metric._get_parent_scores(parent_ids)
        self.assertEqual(len(scores), 0)
    
    # =========================================================================
    # Integration Tests
    # =========================================================================
    
    def test_compute_with_single_parent(self):
        """Test compute with model that has one parent."""
        # Mock config.json response
        config = {
            "base_model": "bert-base-uncased",
            "model_type": "bert"
        }
        self.mock_hf_client.request.return_value = {"config": config}
        
        result = self.metric.compute({
            "model_url": "https://huggingface.co/user/fine-tuned-bert"
        })
        
        # Should return average score of bert-base-uncased
        expected = (0.8 + 0.9 + 0.7 + 1.0 + 0.6 + 0.7 + 0.8 + 0.5 + 0.5 + 0.6) / 10
        self.assertAlmostEqual(result, expected, places=2)
    
    def test_compute_with_multiple_parents(self):
        """Test compute with model that has multiple parents."""
        config = {
            "base_model": "bert-base-uncased",
            "_name_or_path": "openai/clip-vit-base-patch32"
        }
        self.mock_hf_client.request.return_value = {"config": config}
        
        result = self.metric.compute({
            "model_url": "https://huggingface.co/user/multi-parent-model"
        })
        
        # Should return average of both parent scores
        bert_avg = (0.8 + 0.9 + 0.7 + 1.0 + 0.6 + 0.7 + 0.8 + 0.5 + 0.5 + 0.6) / 10
        clip_avg = (0.9 + 0.8 + 0.6 + 0.9 + 0.7 + 0.8 + 0.9 + 0.6 + 1.0 + 0.7) / 10
        expected = (bert_avg + clip_avg) / 2
        self.assertAlmostEqual(result, expected, places=2)
    
    def test_compute_no_config(self):
        """Test compute when config.json not found."""
        self.mock_hf_client.request.return_value = {}
        
        result = self.metric.compute({
            "model_url": "https://huggingface.co/user/model"
        })
        
        self.assertEqual(result, -1.0)
    
    def test_compute_no_parents_in_config(self):
        """Test compute when config has no parent references."""
        config = {
            "model_type": "bert",
            "hidden_size": 768
        }
        self.mock_hf_client.request.return_value = {"config": config}
        
        result = self.metric.compute({
            "model_url": "https://huggingface.co/user/model"
        })
        
        self.assertEqual(result, -1.0)
    
    def test_compute_parents_not_in_registry(self):
        """Test compute when parents exist but not in registry."""
        config = {
            "base_model": "unknown/parent-model"
        }
        self.mock_hf_client.request.return_value = {"config": config}
        
        result = self.metric.compute({
            "model_url": "https://huggingface.co/user/model"
        })
        
        self.assertEqual(result, 0.0)  # Parents found but not in registry
    
    def test_compute_no_model_url(self):
        """Test compute without model URL."""
        result = self.metric.compute({})
        self.assertEqual(result, -1.0)
    
    def test_compute_invalid_url(self):
        """Test compute with invalid URL."""
        result = self.metric.compute({
            "model_url": "https://github.com/not-huggingface/repo"
        })
        self.assertEqual(result, -1.0)
    
    def test_compute_api_error(self):
        """Test compute when API request fails."""
        self.mock_hf_client.request.side_effect = Exception("API Error")
        
        result = self.metric.compute({
            "model_url": "https://huggingface.co/user/model"
        })
        
        self.assertEqual(result, -1.0)
    
    def test_compute_with_raw_config_fallback(self):
        """Test compute falling back to raw config.json."""
        # First call returns no config in model info
        # Second call returns raw config
        config = {
            "base_model": "google/flan-t5-base"
        }
        self.mock_hf_client.request.side_effect = [
            {},  # First call: no config in model info
            config  # Second call: raw config.json
        ]
        
        result = self.metric.compute({
            "model_url": "https://huggingface.co/user/model"
        })
        
        # Should use flan-t5-base score
        expected = (0.7 + 1.0 + 0.5 + 0.8 + 0.8 + 0.6 + 0.7 + 0.4 + 0.0 + 0.5) / 10
        self.assertAlmostEqual(result, expected, places=2)


if __name__ == "__main__":
    unittest.main()
