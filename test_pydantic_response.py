import requests
import json

BASE_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# Authenticate
auth_resp = requests.put(
    f"{BASE_URL}/authenticate",
    json={
        "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
        "Secret": {"password": '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''}
    }
)
print(f"Auth: {auth_resp.status_code}")
print(f"Response text: '{auth_resp.text}'")
print(f"Response headers: {auth_resp.headers}")
if auth_resp.status_code != 200:
    exit(1)

token = auth_resp.text.strip('"')  # Token might be plain text
print(f"Token: {token[:50]}...")

# Test byName
byname_resp = requests.get(
    f"{BASE_URL}/artifact/byName/distillation",
    headers={"X-Authorization": token}
)
print(f"\n/artifact/byName/distillation: {byname_resp.status_code}")
if byname_resp.status_code == 200:
    data = byname_resp.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    print(f"Type check: {type(data)}, first item type: {type(data[0]) if data else 'N/A'}")
else:
    print(byname_resp.text)
