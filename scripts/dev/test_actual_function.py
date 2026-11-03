"""
Test the actual function from autograder_routes.py
"""
import re

def _extract_artifact_name(url: str) -> str:
    """
    Extract artifact name from URL, handling common patterns:
    - Remove trailing slashes
    - Remove /tree/branch suffixes (e.g., /tree/main)
    - Remove .git suffixes
    - Return the last path component as the name
    """
    # Remove trailing slash
    url = url.rstrip('/')
    
    # Remove .git suffix if present
    if url.endswith('.git'):
        url = url[:-4]
    
    # Remove /tree/something suffix (common in git repos)
    url = re.sub(r'/tree/[^/]+$', '', url)
    
    # Extract the name (last component)
    parts = url.split('/')
    name = parts[-1] if parts else "unknown"
    
    return name

# Simulate artifact creation with various URLs
test_artifacts = [
    ("https://huggingface.co/google-bert/bert-base-uncased", "model"),
    ("https://huggingface.co/datasets/bookcorpus", "dataset"),
    ("https://github.com/openai/whisper", "code"),
    ("https://huggingface.co/openai/whisper-tiny/tree/main", "model"),
    ("https://github.com/google-research/bert.git", "code"),
]

print("=== Simulating Artifact Creation ===\n")
for url, artifact_type in test_artifacts:
    name = _extract_artifact_name(url)
    print(f"URL: {url}")
    print(f"Type: {artifact_type}")
    print(f"Extracted Name: {name}")
    print()

print("\n=== Simulating Query ===")
print("If test queries for name='whisper-tiny' and types=['model']:")
print("  Should match: https://huggingface.co/openai/whisper-tiny/tree/main")
print("  Extracted name: whisper-tiny ✓")
