import requests

API_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"
PASSWORD = 'correcthorsebatterystaple123(!)'

auth_payload = {
    "user": {
        "name": "ece30861defaultadminuser",
        "isAdmin": True
    },
    "secret": {
        "password": PASSWORD
    }
}

print(f"Testing with OLD password: {PASSWORD}")
print(f"Length: {len(PASSWORD)}\n")

response = requests.put(f"{API_URL}/authenticate", json=auth_payload)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print(f"✓ SUCCESS! Token: {response.text[:50]}...")
else:
    print(f"✗ FAILED: {response.text}")
