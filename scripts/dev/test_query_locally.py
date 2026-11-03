"""
Local test script to debug query issues
"""
import json

# Simulating the query logic from autograder_routes.py

# Mock artifacts (simulating what's in the database after upload tests)
mock_artifacts = [
    ("id1", {"name": "model1", "type": "model"}),
    ("id2", {"name": "dataset1", "type": "dataset"}),
    ("id3", {"name": "code1", "type": "code"}),
    ("id4", {"name": "model2", "type": "model"}),
]

def test_single_dataset_query():
    """Test what happens with Single Dataset Query"""
    print("\n=== Testing Single Dataset Query ===")
    query = {"name": "*", "types": ["dataset"]}
    
    results = []
    for artifact_id, artifact in mock_artifacts:
        # Check if types filter applies
        if query["types"] is None or artifact["type"] in query["types"]:
            results.append({
                "name": artifact["name"],
                "id": artifact_id,
                "type": artifact["type"]
            })
    
    print(f"Query: {query}")
    print(f"Results: {json.dumps(results, indent=2)}")
    print(f"Expected: Should only return datasets")
    
def test_single_code_query():
    """Test what happens with Single Code Query"""
    print("\n=== Testing Single Code Query ===")
    query = {"name": "*", "types": ["code"]}
    
    results = []
    for artifact_id, artifact in mock_artifacts:
        # Check if types filter applies
        if query["types"] is None or artifact["type"] in query["types"]:
            results.append({
                "name": artifact["name"],
                "id": artifact_id,
                "type": artifact["type"]
            })
    
    print(f"Query: {query}")
    print(f"Results: {json.dumps(results, indent=2)}")
    print(f"Expected: Should only return code")

def test_all_artifacts_query():
    """Test what happens with All Artifacts Query"""
    print("\n=== Testing All Artifacts Query ===")
    query = {"name": "*", "types": None}
    
    results = []
    for artifact_id, artifact in mock_artifacts:
        # Check if types filter applies
        if query["types"] is None or artifact["type"] in query["types"]:
            results.append({
                "name": artifact["name"],
                "id": artifact_id,
                "type": artifact["type"]
            })
    
    print(f"Query: {query}")
    print(f"Results: {json.dumps(results, indent=2)}")
    print(f"Expected: Should return all artifacts")

def test_by_name():
    """Test get artifact by name"""
    print("\n=== Testing Get Artifact By Name ===")
    target_name = "model1"
    
    results = []
    for artifact_id, artifact in mock_artifacts:
        if artifact["name"] == target_name:
            results.append({
                "name": artifact["name"],
                "id": artifact_id,
                "type": artifact["type"]
            })
    
    print(f"Name: {target_name}")
    print(f"Results: {json.dumps(results, indent=2)}")
    print(f"Expected: Should return model1")

if __name__ == "__main__":
    test_single_dataset_query()
    test_single_code_query()
    test_all_artifacts_query()
    test_by_name()
    
    print("\n\n=== Analysis ===")
    print("The logic looks correct. The issue might be:")
    print("1. The query payload format from the test")
    print("2. The response format doesn't match expectations")
    print("3. Artifacts aren't being stored properly")
