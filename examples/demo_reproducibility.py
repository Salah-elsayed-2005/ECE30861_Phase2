"""
Demo script for the Reproducibility Metric

This script demonstrates how to use the ReproducibilityMetric class
to assess whether model demonstration code can be executed successfully.

Usage:
    python demo_reproducibility.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.Metrics import ReproducibilityMetric
from src.logging_utils import get_logger

logger = get_logger(__name__)


def demo_basic_usage():
    """Demonstrate basic usage of the Reproducibility metric."""
    print("=" * 70)
    print("Reproducibility Metric Demo")
    print("=" * 70)
    print()
    
    # Initialize the metric
    metric = ReproducibilityMetric()
    
    # Example 1: A simple model with working demo code
    print("Example 1: Model with working demo code")
    print("-" * 70)
    
    # Simulating a model URL (in real usage, this would be an actual HuggingFace URL)
    test_url = "https://huggingface.co/gpt2"
    
    try:
        score = metric.compute(
            inputs={"model_url": test_url},
            use_agent=True,  # Enable LLM debugging if needed
            timeout=30
        )
        
        print(f"Model URL: {test_url}")
        print(f"Reproducibility Score: {score}")
        print()
        
        # Interpret the score
        if score == 1.0:
            print("✅ EXCELLENT: Demo code runs without any modifications!")
        elif score == 0.5:
            print("⚠️  GOOD: Demo code runs after automated debugging/fixes")
        elif score == 0.0:
            print("❌ POOR: No demo code found or code cannot be executed")
        
    except Exception as e:
        print(f"Error computing reproducibility: {e}")
    
    print()


def demo_code_extraction():
    """Demonstrate code extraction from README text."""
    print("=" * 70)
    print("Code Extraction Demo")
    print("=" * 70)
    print()
    
    metric = ReproducibilityMetric()
    
    # Sample README with code blocks
    sample_readme = """
    # My Awesome Model
    
    This model is great! Here's how to use it:
    
    ```python
    import torch
    from transformers import AutoModel
    
    model = AutoModel.from_pretrained("my-model")
    print("Model loaded!")
    ```
    
    You can also do this:
    
    ```python
    # Another example
    x = model.encode("Hello world")
    print(x.shape)
    ```
    
    And some text without code.
    """
    
    code_blocks = metric._extract_demo_code(sample_readme)
    
    print(f"Found {len(code_blocks)} code blocks in README:")
    print()
    
    for idx, code in enumerate(code_blocks, 1):
        print(f"Code Block {idx}:")
        print("-" * 40)
        print(code)
        print("-" * 40)
        print()


def demo_code_execution():
    """Demonstrate safe code execution."""
    print("=" * 70)
    print("Safe Code Execution Demo")
    print("=" * 70)
    print()
    
    metric = ReproducibilityMetric()
    
    # Test 1: Working code
    print("Test 1: Working Python code")
    code1 = """
x = 1 + 1
y = x * 2
print(f"Result: {y}")
"""
    success1, output1 = metric._execute_code_safely(code1, timeout=5)
    print(f"  Success: {success1}")
    print(f"  Output: {output1.strip()}")
    print()
    
    # Test 2: Code with error
    print("Test 2: Code with syntax error")
    code2 = """
print("Hello"
"""  # Missing closing parenthesis
    success2, output2 = metric._execute_code_safely(code2, timeout=5)
    print(f"  Success: {success2}")
    print(f"  Error: {output2[:100]}...")
    print()
    
    # Test 3: Code with runtime error
    print("Test 3: Code with runtime error")
    code3 = """
x = 1 / 0
"""
    success3, output3 = metric._execute_code_safely(code3, timeout=5)
    print(f"  Success: {success3}")
    print(f"  Error: {output3[:100]}...")
    print()


def demo_scoring_interpretation():
    """Explain the scoring system."""
    print("=" * 70)
    print("Reproducibility Scoring System")
    print("=" * 70)
    print()
    
    print("Score: 0.0 (No Code / Doesn't Run)")
    print("  • No demonstration code found in model card")
    print("  • Code exists but has fatal errors that cannot be fixed")
    print("  • Code requires external resources that aren't available")
    print()
    
    print("Score: 0.5 (Runs With Debugging)")
    print("  • Code initially fails but can be fixed by LLM agent")
    print("  • Minor issues like missing imports or variable names")
    print("  • Requires some automated intervention to work")
    print()
    
    print("Score: 1.0 (Runs Immediately)")
    print("  • Code executes successfully on first try")
    print("  • No modifications or debugging needed")
    print("  • Demonstrates excellent documentation quality")
    print()


def main():
    """Run all demos."""
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 15 + "REPRODUCIBILITY METRIC DEMO" + " " * 27 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print("\n")
    
    # Run demos
    demo_scoring_interpretation()
    print("\n")
    
    demo_code_extraction()
    print("\n")
    
    demo_code_execution()
    print("\n")
    
    # Only run the actual metric if user confirms (requires API keys)
    response = input("Run actual HuggingFace model test? (requires API keys) [y/N]: ")
    if response.lower() == 'y':
        demo_basic_usage()
    else:
        print("\nSkipping live model test. Set up API keys and try again to test with real models!")
    
    print("\n")
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Set up your API keys (HF_TOKEN, GEN_AI_STUDIO_API_KEY)")
    print("  2. Test with real HuggingFace models")
    print("  3. Integrate into your model evaluation pipeline")
    print()


if __name__ == "__main__":
    main()
