"""
Improved artifact name extraction
"""
import re

def extract_name_improved(url: str) -> str:
    """
    Extract artifact name from URL, handling common patterns:
    - Remove trailing slashes
    - Remove /tree/branch or /tree/main suffixes
    - Remove .git suffixes
    - Handle HuggingFace model/dataset URLs
    - Handle GitHub repo URLs
    """
    # Remove trailing slash
    url = url.rstrip('/')
    
    # Remove .git suffix if present
    if url.endswith('.git'):
        url = url[:-4]
    
    # Remove /tree/something suffix (common in git repos)
    # Match /tree/anything at the end
    url = re.sub(r'/tree/[^/]+$', '', url)
    
    # Now extract the name (last component)
    parts = url.split('/')
    name = parts[-1] if parts else "unknown"
    
    return name

# Test with all edge cases
test_cases = [
    # HuggingFace models
    ("https://huggingface.co/google-bert/bert-base-uncased", "bert-base-uncased"),
    ("https://huggingface.co/openai/whisper-tiny", "whisper-tiny"),
    ("https://huggingface.co/microsoft/resnet-50", "resnet-50"),
    
    # HuggingFace datasets  
    ("https://huggingface.co/datasets/bookcorpus", "bookcorpus"),
    ("https://huggingface.co/datasets/squad", "squad"),
    
    # GitHub repos
    ("https://github.com/openai/whisper", "whisper"),
    ("https://github.com/google-research/bert", "bert"),
    
    # Edge cases
    ("https://huggingface.co/openai/whisper-tiny/tree/main", "whisper-tiny"),
    ("https://github.com/openai/whisper.git", "whisper"),
    ("https://github.com/openai/whisper/", "whisper"),
    ("https://github.com/openai/whisper/tree/master", "whisper"),
]

print("=== Testing Improved Name Extraction ===\n")
all_pass = True
for url, expected in test_cases:
    extracted = extract_name_improved(url)
    status = "✓" if extracted == expected else "✗"
    if extracted != expected:
        all_pass = False
    print(f"{status} {url}")
    print(f"  Expected: {expected}")
    print(f"  Got: {extracted}")
    if extracted != expected:
        print(f"  MISMATCH!")
    print()

if all_pass:
    print("\n✓ All tests passed!")
else:
    print("\n✗ Some tests failed")
