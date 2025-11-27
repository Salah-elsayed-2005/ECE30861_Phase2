import requests
import json

API_BASE = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# Test health first
print("Testing /health...")
response = requests.get(f"{API_BASE}/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.text}\n")

# Test authentication
print("Testing /authenticate...")
password = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''
payload = {
    "User": {
        "name": "ece30861defaultadminuser",
        "isAdmin": True
    },
    "Secret": {
        "password": password
    }
}

response = requests.put(f"{API_BASE}/authenticate", json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code == 200:
    print("\n✅ AUTHENTICATION SUCCESSFUL!")
    token = response.json()
    print(f"Token: {token}")
else:
    print("\n❌ AUTHENTICATION FAILED!")
