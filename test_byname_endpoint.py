import requests

API_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"
PASSWORD = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''

auth_response = requests.put(f"{API_URL}/authenticate", json={
    "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
    "Secret": {"password": PASSWORD}
})

token = auth_response.text.strip()
headers = {"X-Authorization": token}

# Test byName with existing packages
test_names = ["distillation", "fashion-clip", "flickr2k"]

print("Testing GET /package/byName/{name} endpoint:")
print("=" * 80)

for name in test_names:
    print(f"\nTesting: {name}")
    resp = requests.get(f"{API_URL}/package/byName/{name}", headers=headers)
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Response type: {type(data)}")
        if isinstance(data, list):
            print(f"  Number of items: {len(data)}")
            if len(data) > 0:
                print(f"  First item keys: {list(data[0].keys())}")
        else:
            print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    else:
        print(f"  Error: {resp.text}")
