import requests

BASE_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# Auth
auth_resp = requests.put(f"{BASE_URL}/authenticate", json={
    "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
    "Secret": {"password": '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''}
})
token = auth_resp.text.strip('"')

# Test POST /artifacts with specific name (what autograder might be doing for "byName" tests)
print("Testing POST /artifacts with name='test-model'...")
resp = requests.post(
    f"{BASE_URL}/artifacts",
    json=[{"name": "test-model"}],
    headers={"X-Authorization": token}
)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text}")

# Also test GET /artifact/byName/test-model
print("\nTesting GET /artifact/byName/test-model...")
resp2 = requests.get(f"{BASE_URL}/artifact/byName/test-model")
print(f"Status: {resp2.status_code}")
print(f"Response: {resp2.text}")
