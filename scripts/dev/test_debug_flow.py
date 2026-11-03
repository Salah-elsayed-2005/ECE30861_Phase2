"""
Debug test - simulate exact autograder flow
"""

# Simulate what might be happening
def simulate_uploads_and_queries():
    # Simulated database
    artifacts_db = []
    
    # Upload model 1
    model1_name = "bert-base-uncased"
    model1_id = "id1"
    artifacts_db.append({"id": model1_id, "name": model1_name, "type": "model"})
    print(f"✓ Uploaded model 1: {model1_name}")
    
    # Query for model 1 using wildcard + type filter
    query = {"name": "*", "types": ["model"]}
    results = [a for a in artifacts_db if query["types"] is None or a["type"] in query["types"]]
    if len(results) == 1 and results[0]["id"] == model1_id:
        print(f"✓ Single Model Query Test PASSED - Found {len(results)} result(s)")
    else:
        print(f"✗ Single Model Query Test FAILED - Found {len(results)} result(s), expected 1")
    
    # Upload dataset 1
    dataset1_name = "bookcorpus"
    dataset1_id = "id2"
    artifacts_db.append({"id": dataset1_id, "name": dataset1_name, "type": "dataset"})
    print(f"\n✓ Uploaded dataset 1: {dataset1_name}")
    
    # Query for dataset 1 using wildcard + type filter
    query = {"name": "*", "types": ["dataset"]}
    results = [a for a in artifacts_db if query["types"] is None or a["type"] in query["types"]]
    print(f"All artifacts in DB: {artifacts_db}")
    print(f"Query: {query}")
    print(f"Results: {results}")
    
    # The test might be expecting ONLY the newly uploaded dataset
    if len(results) == 1 and results[0]["id"] == dataset1_id:
        print(f"✓ Single Dataset Query Test WOULD PASS")
    else:
        print(f"✗ Single Dataset Query Test WOULD FAIL")
        print(f"  Expected: 1 dataset with id={dataset1_id}")
        print(f"  Got: {len(results)} dataset(s)")
    
    # OR maybe the test is querying by NAME, not by wildcard?
    query_by_name = {"name": dataset1_name, "types": ["dataset"]}
    results_by_name = [a for a in artifacts_db if a["name"] == query_by_name["name"] and (query_by_name["types"] is None or a["type"] in query_by_name["types"])]
    print(f"\nIf querying by NAME instead of wildcard:")
    print(f"Query: {query_by_name}")
    print(f"Results: {results_by_name}")
    if len(results_by_name) == 1:
        print(f"✓ Query by name WOULD PASS")
    else:
        print(f"✗ Query by name WOULD FAIL")

if __name__ == "__main__":
    print("=== Simulating Autograder Upload/Query Flow ===\n")
    simulate_uploads_and_queries()
    
    print("\n\n=== Key Insight ===")
    print("The test names suggest: 'Single Model Query', 'Single Dataset Query', 'Single Code Query'")
    print("This implies the test is checking if it can query for JUST that specific type.")
    print("Both wildcard+filter and name-specific queries should work.")
    print("The fact that model works but dataset/code don't suggests an issue specific to those types.")
