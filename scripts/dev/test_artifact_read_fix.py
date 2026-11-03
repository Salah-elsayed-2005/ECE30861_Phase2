"""Test artifact read endpoints after fix"""
import requests
import json
from urllib.parse import quote

API_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# 1. Authenticate
print("1. Authenticating...")
auth_response = requests.put(
    f"{API_URL}/authenticate",
    json={
        "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
        "Secret": {"password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;"}
    }
)
print(f"   Status: {auth_response.status_code}")
if auth_response.status_code == 200:
    token = auth_response.text.strip()
    print(f"   Token: {token[:50]}...")
else:
    print(f"   Error: {auth_response.text}")
    exit(1)

headers = {"X-Authorization": token}

# 2. Reset registry
print("\n2. Resetting registry...")
reset_response = requests.delete(f"{API_URL}/reset", headers=headers)
print(f"   Status: {reset_response.status_code}")

# 3. Upload test artifacts with different name patterns
test_artifacts = [
    {"url": "https://github.com/lodash/lodash", "type": "code", "expected_name": "lodash"},
    {"url": "https://huggingface.co/bert-base-uncased", "type": "model", "expected_name": "bert-base-uncased"},
    {"url": "https://huggingface.co/facebook/opt-350m", "type": "model", "expected_name": "opt-350m"},
    {"url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2", "type": "model", "expected_name": "all-MiniLM-L6-v2"},
]

uploaded_artifacts = []

print("\n3. Uploading test artifacts...")
for i, artifact in enumerate(test_artifacts):
    print(f"   {i+1}. Uploading {artifact['type']}: {artifact['url']}")
    response = requests.post(
        f"{API_URL}/artifact/{artifact['type']}",
        headers=headers,
        json={"url": artifact["url"]}
    )
    print(f"      Status: {response.status_code}")
    if response.status_code == 201:
        data = response.json()
        artifact_id = data["metadata"]["id"]
        artifact_name = data["metadata"]["name"]
        artifact_type = data["metadata"]["type"]
        print(f"      ID: {artifact_id}, Name: {artifact_name}")
        uploaded_artifacts.append({
            "id": artifact_id,
            "name": artifact_name,
            "type": artifact_type,
            "expected_name": artifact["expected_name"]
        })
    else:
        print(f"      Error: {response.text}")

# 4. Test GET by name
print("\n4. Testing GET by name...")
for artifact in uploaded_artifacts:
    name = artifact["name"]
    # Try with URL encoding
    encoded_name = quote(name, safe='')
    print(f"   Testing name: {name}")
    print(f"   URL-encoded: {encoded_name}")
    
    response = requests.get(
        f"{API_URL}/artifact/byName/{encoded_name}",
        headers=headers
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        print(f"   Found {len(results)} artifact(s)")
        for result in results:
            print(f"     - {result['name']} (ID: {result['id']}, Type: {result['type']})")
    else:
        print(f"   Error: {response.text}")

# 5. Test GET by ID
print("\n5. Testing GET by ID...")
for artifact in uploaded_artifacts:
    response = requests.get(
        f"{API_URL}/artifacts/{artifact['type']}/{artifact['id']}",
        headers=headers
    )
    print(f"   ID {artifact['id']}: Status {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"     Name: {data['metadata']['name']}, Type: {data['metadata']['type']}")
    else:
        print(f"     Error: {response.text}")

print("\n✓ Test complete!")
