import requests

API_BASE = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"
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

print("Testing authentication...")
response = requests.put(f"{API_BASE}/authenticate", json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:200]}")

if response.status_code == 200:
    print("\n✅ AUTHENTICATION WORKING!")
else:
    print(f"\n❌ AUTHENTICATION FAILED!")
