"""
Test what names would be generated from realistic test URLs
"""

def extract_name(url):
    """Current implementation"""
    parts = url.rstrip('/').split('/')
    return parts[-1] if parts else "unknown"

# Realistic test URLs that might be used
test_urls = [
    # HuggingFace models
    "https://huggingface.co/google-bert/bert-base-uncased",
    "https://huggingface.co/openai/whisper-tiny",
    "https://huggingface.co/microsoft/resnet-50",
    
    # HuggingFace datasets  
    "https://huggingface.co/datasets/bookcorpus",
    "https://huggingface.co/datasets/squad",
    "https://huggingface.co/datasets/wikipedia",
    
    # GitHub repos
    "https://github.com/openai/whisper",
    "https://github.com/google-research/bert",
    "https://github.com/facebookresearch/pytorch",
    
    # Edge cases
    "https://huggingface.co/openai/whisper-tiny/tree/main",
    "https://github.com/openai/whisper.git",
]

print("=== Artifact Names ===\n")
for url in test_urls:
    name = extract_name(url)
    print(f"{url}")
    print(f"  → {name}\n")
