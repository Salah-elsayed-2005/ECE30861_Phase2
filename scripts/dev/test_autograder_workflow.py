"""
Simulate what autograder tests might be doing
"""
import requests
import json

# Assuming the API URL
API_URL = "https://your-api-url.com"  # Replace with actual

def test_workflow():
    # 1. Login
    auth_response = requests.put(f"{API_URL}/authenticate", json={
        "User": {
            "name": "ece30861defaultadminuser",
            "isAdmin": True
        },
        "Secret": {
            "password": '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''
        }
    })
    token = auth_response.json()
    print(f"Auth token: {token[:50]}...")
    
    # 2. Upload a model
    model_response = requests.post(
        f"{API_URL}/artifact/model",
        headers={"X-Authorization": token},
        json={"url": "https://huggingface.co/google-bert/bert-base-uncased"}
    )
    model_data = model_response.json()
    print(f"\nUploaded model:")
    print(json.dumps(model_data, indent=2))
    model_name = model_data["metadata"]["name"]
    model_id = model_data["metadata"]["id"]
    
    # 3. Query for that specific model using POST /artifacts
    query_response = requests.post(
        f"{API_URL}/artifacts",
        headers={"X-Authorization": token},
        json=[{"name": model_name, "types": ["model"]}]
    )
    query_results = query_response.json()
    print(f"\nQuery results for name='{model_name}':")
    print(json.dumps(query_results, indent=2))
    
    # Check if the model is in the results
    found = any(r["id"] == model_id for r in query_results)
    print(f"\nModel found in query results: {found}")
    
    # 4. Get artifact by name using GET /artifact/byName/{name}
    byname_response = requests.get(
        f"{API_URL}/artifact/byName/{model_name}",
        headers={"X-Authorization": token}
    )
    byname_results = byname_response.json()
    print(f"\nGet by name results:")
    print(json.dumps(byname_results, indent=2))
    
if __name__ == "__main__":
    print("This is a simulation - would need actual API URL to run")
    print("\nExpected workflow:")
    print("1. Upload artifact -> get back {metadata: {name, id, type}, data: {url, download_url}}")
    print("2. Query POST /artifacts with [{name: <artifact_name>, types: [<type>]}]")
    print("3. Should return [{name, id, type}] matching the artifact")
    print("4. GET /artifact/byName/<name> should return [{name, id, type}]")
