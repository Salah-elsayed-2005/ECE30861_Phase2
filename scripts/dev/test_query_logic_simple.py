"""Test to understand why model queries work but dataset/code fail"""

# Simulated _list_artifacts output after pagination
all_artifacts = [
    ("id1", {"name": "bert-base-uncased", "type": "model"}),
    ("id2", {"name": "bookcorpus", "type": "dataset"}),
    ("id3", {"name": "transformers", "type": "code"}),
]

# Test query logic
def test_query(queries):
    results = []
    
    # Handle wildcard query  
    if len(queries) == 1 and queries[0]["name"] == "*":
        print(f"Wildcard query detected: {queries[0]}")
        for artifact_id, artifact in all_artifacts:
            # Filter by types if specified
            if queries[0]["types"] is None or artifact["type"] in queries[0]["types"]:
                results.append({
                    "name": artifact["name"],
                    "id": artifact_id,
                    "type": artifact["type"]
                })
    return results

# Test 1: Model query (PASSES in autograder)
print("=== Test 1: Model Query ===")
query1 = [{"name": "*", "types": ["model"]}]
result1 = test_query(query1)
print(f"Query: {query1}")
print(f"Results: {result1}")
print(f"Expected 1 model: {'PASS' if len(result1) == 1 else 'FAIL'}")

# Test 2: Dataset query (FAILS in autograder)
print("\n=== Test 2: Dataset Query ===")
query2 = [{"name": "*", "types": ["dataset"]}]
result2 = test_query(query2)
print(f"Query: {query2}")
print(f"Results: {result2}")
print(f"Expected 1 dataset: {'PASS' if len(result2) == 1 else 'FAIL'}")

# Test 3: Code query (FAILS in autograder)
print("\n=== Test 3: Code Query ===")
query3 = [{"name": "*", "types": ["code"]}]
result3 = test_query(query3)
print(f"Query: {query3}")
print(f"Results: {result3}")
print(f"Expected 1 code: {'PASS' if len(result3) == 1 else 'FAIL'}")

print("\n=== Analysis ===")
print("All three tests pass with this logic!")
print("So the issue must be:")
print("1. The artifact type is not being stored correctly in DynamoDB")
print("2. The query.types array has the wrong format/casing")
print("3. The artifacts aren't being retrieved at all (pagination issue)")
