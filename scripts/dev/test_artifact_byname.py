import requests

API_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"
PASSWORD = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''

auth_response = requests.put(f"{API_URL}/authenticate", json={
    "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
    "Secret": {"password": PASSWORD}
})

token = auth_response.text.strip()
headers = {"X-Authorization": token}

# Test the /artifact/byName endpoint
print("Testing /artifact/byName/{name}:")
print("=" * 80)

test_names = ["distillation", "fashion-clip", "flickr2k"]

for name in test_names:
    print(f"\n{name}:")
    resp = requests.get(f"{API_URL}/artifact/byName/{name}", headers=headers)
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Response: {data}")
    else:
        print(f"  Error: {resp.text}")
