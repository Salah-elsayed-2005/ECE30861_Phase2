import requests

API_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"
PASSWORD = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''

print(f"Testing with PACKAGES password (length {len(PASSWORD)})")

auth_payload = {
    "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
    "Secret": {"password": PASSWORD}
}

response = requests.put(f"{API_URL}/authenticate", json=auth_payload)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print(f"✓ SUCCESS! Token: {response.text[:50]}...")
else:
    print(f"✗ FAILED: {response.text}")
