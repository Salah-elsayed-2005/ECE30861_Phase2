"""Test to identify URL encoding issues in artifact names"""
from urllib.parse import unquote, quote

# Common HuggingFace model/dataset/code names that might have special chars
test_names = [
    "bert-base-uncased",
    "gpt2-medium",
    "facebook/opt-350m",  # Contains /
    "google/flan-t5-base",  # Contains /
    "sentence-transformers/all-MiniLM-L6-v2",  # Contains / and -
    "openai/whisper-tiny",
    "meta-llama/Llama-2-7b-hf",  # Contains / and -
]

print("Testing URL encoding:")
for name in test_names:
    encoded = quote(name, safe='')
    decoded = unquote(encoded)
    print(f"Original: {name}")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {decoded}")
    print(f"  Match: {name == decoded}")
    print()
