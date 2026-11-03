import requests
import json

API_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"
PASSWORD = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''

# Authenticate
auth_response = requests.put(f"{API_URL}/authenticate", json={
    "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
    "Secret": {"password": PASSWORD}
})

if auth_response.status_code != 200:
    print(f"Auth failed: {auth_response.status_code}")
    exit(1)

token = auth_response.text.strip()
headers = {"X-Authorization": token}

print("=" * 80)
print("TESTING ARTIFACT QUERIES")
print("=" * 80)

# Upload a test artifact first
print("\n1. Uploading test artifact...")
test_artifact = {
    "metadata": {
        "Name": "test-model",
        "Version": "1.0.0", 
        "ID": "test-123"
    },
    "data": {
        "URL": "https://github.com/test/repo"
    }
}

upload_resp = requests.post(f"{API_URL}/package", json=test_artifact, headers=headers)
print(f"   Upload status: {upload_resp.status_code}")
if upload_resp.status_code in [200, 201]:
    artifact_data = upload_resp.json()
    print(f"   Artifact ID: {artifact_data.get('metadata', {}).get('ID')}")
    print(f"   Artifact Name: {artifact_data.get('metadata', {}).get('Name')}")

# Test 1: Query all artifacts
print("\n2. Testing query all artifacts (Name: *)...")
query_resp = requests.post(f"{API_URL}/packages", json=[{"Name": "*"}], headers=headers)
print(f"   Status: {query_resp.status_code}")
if query_resp.status_code == 200:
    packages = query_resp.json()
    print(f"   Found {len(packages)} packages")
    for i, pkg in enumerate(packages[:3]):
        print(f"   - {i}: {pkg}")
else:
    print(f"   Error: {query_resp.text}")

# Test 2: Query by exact name
print("\n3. Testing query by exact name (test-model)...")
query_resp = requests.post(f"{API_URL}/packages", json=[{"Name": "test-model"}], headers=headers)
print(f"   Status: {query_resp.status_code}")
if query_resp.status_code == 200:
    packages = query_resp.json()
    print(f"   Found {len(packages)} packages")
    for pkg in packages:
        print(f"   - {pkg}")
else:
    print(f"   Error: {query_resp.text}")

# Test 3: Get artifact by name (using byName endpoint)
print("\n4. Testing GET /package/byName/test-model...")
get_resp = requests.get(f"{API_URL}/package/byName/test-model", headers=headers)
print(f"   Status: {get_resp.status_code}")
if get_resp.status_code == 200:
    data = get_resp.json()
    print(f"   Response keys: {list(data.keys())}")
    print(f"   Response: {json.dumps(data, indent=2)[:500]}")
else:
    print(f"   Error: {get_resp.text}")

# Test 4: Get artifact by ID
print("\n5. Testing GET /package/{id}...")
if 'artifact_data' in locals():
    artifact_id = artifact_data.get('metadata', {}).get('ID')
    get_resp = requests.get(f"{API_URL}/package/{artifact_id}", headers=headers)
    print(f"   Status: {get_resp.status_code}")
    if get_resp.status_code == 200:
        data = get_resp.json()
        print(f"   Response keys: {list(data.keys())}")
        print(f"   Has metadata: {'metadata' in data}")
        print(f"   Has data: {'data' in data}")
    else:
        print(f"   Error: {get_resp.text}")

print("\n" + "=" * 80)
