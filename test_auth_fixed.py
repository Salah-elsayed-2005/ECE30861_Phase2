import requests
import json

API_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# Exact password from OpenAPI spec
PASSWORD = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE artifacts;'''

print(f"Testing authentication with password length: {len(PASSWORD)}")
print(f"First 30 chars: {PASSWORD[:30]}...")
print(f"Last 20 chars: ...{PASSWORD[-20:]}")

# Test authentication
auth_payload = {
    "user": {
        "name": "ece30861defaultadminuser",
        "isAdmin": True
    },
    "secret": {
        "password": PASSWORD
    }
}

print("\n1. Testing /authenticate endpoint...")
response = requests.put(f"{API_URL}/authenticate", json=auth_payload)
print(f"   Status: {response.status_code}")

if response.status_code == 200:
    token = response.text.strip()
    print(f"   ✓ Authentication successful!")
    print(f"   Token: {token[:50]}...")
    
    # Test artifact endpoint with token
    print("\n2. Testing /artifacts endpoint with token...")
    headers = {"X-Authorization": token}
    artifacts_response = requests.post(
        f"{API_URL}/artifacts",
        headers=headers,
        json=[{"Name": "*", "Version": "1.0.0"}]
    )
    print(f"   Status: {artifacts_response.status_code}")
    if artifacts_response.status_code == 200:
        print(f"   ✓ Artifacts endpoint working!")
        print(f"   Response: {artifacts_response.json()}")
    else:
        print(f"   ✗ Error: {artifacts_response.text}")
else:
    print(f"   ✗ Authentication failed!")
    print(f"   Response: {response.text}")
