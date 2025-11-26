"""
Test wildcard query logic
"""

# Mock data - what might be in the database after uploading one of each type
mock_artifacts = [
    ("id1", {"name": "bert-base-uncased", "type": "model"}),
    ("id2", {"name": "bookcorpus", "type": "dataset"}),
    ("id3", {"name": "whisper", "type": "code"}),
]

def test_wildcard_query_with_type_filter(artifact_type):
    """Simulate wildcard query with type filter"""
    print(f"\n=== Querying for type={artifact_type} ===")
    query = {"name": "*", "types": [artifact_type]}
    
    results = []
    for artifact_id, artifact in mock_artifacts:
        # Wildcard query with type filter
        if query["types"] is None or artifact["type"] in query["types"]:
            results.append({
                "name": artifact["name"],
                "id": artifact_id,
                "type": artifact["type"]
            })
    
    print(f"Query: {query}")
    print(f"Results: {results}")
    print(f"Count: {len(results)}")
    return results

# Test each type
model_results = test_wildcard_query_with_type_filter("model")
dataset_results = test_wildcard_query_with_type_filter("dataset")
code_results = test_wildcard_query_with_type_filter("code")

print("\n=== Analysis ===")
print(f"Models found: {len(model_results)} (expected 1)")
print(f"Datasets found: {len(dataset_results)} (expected 1)")
print(f"Code found: {len(code_results)} (expected 1)")

if len(model_results) == 1 and len(dataset_results) == 1 and len(code_results) == 1:
    print("\n✓ Logic appears correct!")
else:
    print("\n✗ Logic has issues!")
