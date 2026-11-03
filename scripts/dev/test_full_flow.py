import requests
import json

API_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"
PASSWORD = 'correcthorsebatterystaple123(!)'

# Authenticate
auth_response = requests.put(f"{API_URL}/authenticate", json={
    "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
    "Secret": {"password": PASSWORD}
})

if auth_response.status_code != 200:
    print(f"Auth failed: {auth_response.status_code} - {auth_response.text}")
    exit(1)

token = auth_response.text.strip()
headers = {"X-Authorization": token}

print("✓ Authentication successful")

# Test package upload
print("\nTesting package upload...")
test_package = {
    "metadata": {
        "Name": "test-pkg",
        "Version": "1.0.0",
        "ID": "test-123"
    },
    "data": {
        "Content": "UEsDBBQAAAAIAA=="  # minimal zip content
    }
}

upload_response = requests.post(f"{API_URL}/package", json=test_package, headers=headers)
print(f"Upload status: {upload_response.status_code}")
if upload_response.status_code not in [200, 201]:
    print(f"Response: {upload_response.text[:200]}")
else:
    print("✓ Upload successful")

# Test packages query
print("\nTesting packages query...")
query_response = requests.post(f"{API_URL}/packages", json=[{"Name": "*"}], headers=headers)
print(f"Query status: {query_response.status_code}")
if query_response.status_code == 200:
    packages = query_response.json()
    print(f"✓ Found {len(packages)} packages")
else:
    print(f"Response: {query_response.text[:200]}")

# Test reset
print("\nTesting reset...")
reset_response = requests.delete(f"{API_URL}/reset", headers=headers)
print(f"Reset status: {reset_response.status_code}")
if reset_response.status_code == 200:
    print("✓ Reset successful")
