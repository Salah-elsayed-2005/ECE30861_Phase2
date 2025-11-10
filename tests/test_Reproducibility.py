"""
Unit tests for the Reproducibility metric.
"""
import unittest
from unittest.mock import MagicMock, patch, Mock
from src.Metrics import ReproducibilityMetric


class TestReproducibilityMetric(unittest.TestCase):
    """Test cases for Reproducibility metric."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the clients to avoid API key requirements
        with patch('src.Metrics.HFClient'), patch('src.Metrics.PurdueClient'):
            self.metric = ReproducibilityMetric()
            # Replace clients with mocks
            self.metric.hf_client = MagicMock()
            self.metric.llm_agent = MagicMock()
    
    def test_extract_demo_code_with_python_marker(self):
        """Test extraction of Python code blocks with ```python marker."""
        readme = """
        # Model Card
        
        Here's how to use it:
        
        ```python
        import torch
        model = torch.load('model.pt')
        print('Hello')
        ```
        """
        
        code_blocks = self.metric._extract_demo_code(readme)
        
        self.assertEqual(len(code_blocks), 1)
        self.assertIn("import torch", code_blocks[0])
        self.assertIn("print('Hello')", code_blocks[0])
    
    def test_extract_demo_code_with_py_marker(self):
        """Test extraction with ```py marker."""
        readme = """
        ```py
        x = 1 + 1
        print(x)
        ```
        """
        
        code_blocks = self.metric._extract_demo_code(readme)
        
        self.assertEqual(len(code_blocks), 1)
        self.assertIn("x = 1 + 1", code_blocks[0])
    
    def test_extract_demo_code_no_language_marker(self):
        """Test extraction from generic code blocks."""
        readme = """
        ```
        import numpy as np
        x = np.array([1, 2, 3])
        ```
        """
        
        code_blocks = self.metric._extract_demo_code(readme)
        
        # Should find it because it contains 'import'
        self.assertEqual(len(code_blocks), 1)
        self.assertIn("import numpy", code_blocks[0])
    
    def test_extract_demo_code_no_code(self):
        """Test with README containing no code blocks."""
        readme = "Just a description with no code examples."
        
        code_blocks = self.metric._extract_demo_code(readme)
        
        self.assertEqual(len(code_blocks), 0)
    
    def test_extract_demo_code_multiple_blocks(self):
        """Test extraction of multiple code blocks."""
        readme = """
        ```python
        import torch
        ```
        
        Some text.
        
        ```python
        model = torch.load('file')
        ```
        """
        
        code_blocks = self.metric._extract_demo_code(readme)
        
        self.assertEqual(len(code_blocks), 2)
    
    def test_execute_code_safely_success(self):
        """Test successful code execution."""
        code = "x = 1 + 1\nprint('success')"
        
        success, output = self.metric._execute_code_safely(code, timeout=5)
        
        self.assertTrue(success)
        self.assertIn("success", output)
    
    def test_execute_code_safely_syntax_error(self):
        """Test code with syntax error."""
        code = "print('hello'\n"  # Missing closing paren
        
        success, output = self.metric._execute_code_safely(code, timeout=5)
        
        self.assertFalse(success)
        self.assertIn("Exit code", output)
    
    def test_execute_code_safely_runtime_error(self):
        """Test code with runtime error."""
        code = "x = 1 / 0"
        
        success, output = self.metric._execute_code_safely(code, timeout=5)
        
        self.assertFalse(success)
    
    def test_execute_code_safely_timeout(self):
        """Test code that times out."""
        code = "import time\ntime.sleep(100)"
        
        success, output = self.metric._execute_code_safely(code, timeout=1)
        
        self.assertFalse(success)
        self.assertIn("timeout", output.lower())
    
    @patch('src.Metrics.injectHFBrowser')
    def test_compute_no_model_url(self, mock_browser):
        """Test compute with no model URL."""
        inputs = {}
        
        score = self.metric.compute(inputs)
        
        self.assertEqual(score, 0.0)
        mock_browser.assert_not_called()
    
    @patch('src.Metrics.injectHFBrowser')
    def test_compute_no_readme(self, mock_browser):
        """Test compute when README cannot be fetched."""
        mock_browser.return_value = None
        inputs = {"model_url": "https://huggingface.co/test/model"}
        
        score = self.metric.compute(inputs)
        
        self.assertEqual(score, 0.0)
    
    @patch('src.Metrics.injectHFBrowser')
    def test_compute_no_code_blocks(self, mock_browser):
        """Test compute when README has no code blocks."""
        mock_browser.return_value = "Just a description, no code."
        inputs = {"model_url": "https://huggingface.co/test/model"}
        
        score = self.metric.compute(inputs)
        
        self.assertEqual(score, 0.0)
    
    @patch('src.Metrics.injectHFBrowser')
    def test_compute_code_runs_immediately(self, mock_browser):
        """Test compute when code runs successfully without changes."""
        mock_browser.return_value = """
        # Model Card
        
        ```python
        print('Hello World')
        ```
        """
        inputs = {"model_url": "https://huggingface.co/test/model"}
        
        score = self.metric.compute(inputs, use_agent=False)
        
        self.assertEqual(score, 1.0)
    
    @patch('src.Metrics.injectHFBrowser')
    def test_compute_code_fails(self, mock_browser):
        """Test compute when code fails and no agent available."""
        mock_browser.return_value = """
        ```python
        import nonexistent_module
        ```
        """
        inputs = {"model_url": "https://huggingface.co/test/model"}
        
        score = self.metric.compute(inputs, use_agent=False)
        
        self.assertEqual(score, 0.0)
    
    @patch('src.Metrics.injectHFBrowser')
    def test_compute_code_runs_with_debugging(self, mock_browser):
        """Test compute when code runs after LLM debugging."""
        mock_browser.return_value = """
        ```python
        # This will fail initially
        undefined_variable
        ```
        """
        inputs = {"model_url": "https://huggingface.co/test/model"}
        
        # Mock the LLM agent to return working code
        self.metric.llm_agent = MagicMock()
        self.metric.llm_agent.chat.return_value = """
        Here's the fixed code:
        ```python
        print('Fixed!')
        ```
        """
        
        score = self.metric.compute(inputs, use_agent=True)
        
        # Should return 0.5 (runs with debugging)
        self.assertEqual(score, 0.5)
    
    def test_try_debug_with_agent_success(self):
        """Test LLM agent successfully fixes code."""
        code = "undefined_var"
        error = "NameError: name 'undefined_var' is not defined"
        
        # Mock the LLM agent
        self.metric.llm_agent = MagicMock()
        self.metric.llm_agent.chat.return_value = """
        ```python
        print('Fixed code')
        ```
        """
        
        success, output = self.metric._try_debug_with_agent(code, error)
        
        self.assertTrue(success)
        self.assertIn("Fixed code", output)
    
    def test_try_debug_with_agent_no_code_in_response(self):
        """Test LLM agent returns response without code block."""
        code = "bad code"
        error = "SyntaxError"
        
        self.metric.llm_agent = MagicMock()
        self.metric.llm_agent.chat.return_value = "I don't know how to fix this."
        
        success, output = self.metric._try_debug_with_agent(code, error)
        
        self.assertFalse(success)
        self.assertIn("did not return valid code", output)
    
    def test_try_debug_with_agent_llm_error(self):
        """Test LLM agent raises an exception."""
        code = "bad code"
        error = "Error"
        
        self.metric.llm_agent = MagicMock()
        self.metric.llm_agent.chat.side_effect = Exception("LLM API error")
        
        success, output = self.metric._try_debug_with_agent(code, error)
        
        self.assertFalse(success)
        self.assertIn("LLM error", output)


if __name__ == "__main__":
    unittest.main()
