"""
Test artifact name extraction from URLs
"""

def extract_name_from_url(url: str) -> str:
    """Extract artifact name from URL - same logic as in autograder_routes.py"""
    parts = url.rstrip('/').split('/')
    name = parts[-1] if parts else "unknown"
    return name

# Test with typical URLs
test_urls = [
    ("https://huggingface.co/google-bert/bert-base-uncased", "bert-base-uncased"),
    ("https://huggingface.co/datasets/bookcorpus", "bookcorpus"),
    ("https://github.com/openai/whisper", "whisper"),
    ("https://github.com/openai/whisper/", "whisper"),  # trailing slash
    ("https://huggingface.co/google-bert/bert-base-uncased/tree/main", "main"),  # has /tree/main
]

print("=== Testing Name Extraction ===\n")
for url, expected in test_urls:
    extracted = extract_name_from_url(url)
    status = "✓" if extracted == expected else "✗"
    print(f"{status} URL: {url}")
    print(f"  Expected: {expected}")
    print(f"  Got: {extracted}")
    if extracted != expected:
        print(f"  ERROR: Mismatch!")
    print()

print("\n=== Issue Found ===")
print("URLs with /tree/main or other path suffixes will extract the wrong name!")
print("We need to be smarter about name extraction.")
