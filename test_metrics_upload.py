import requests
import json

API_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"
PASSWORD = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''

# Authenticate
auth_resp = requests.post(
    f"{API_URL}/authenticate",
    json={
        "Username": "ece30861defaultadminuser",
        "Secret": {"password": PASSWORD}
    },
    headers={"X-Authorization": PASSWORD}
)

print(f"Auth status: {auth_resp.status_code}")
token = auth_resp.text.strip()
print(f"Token: {token[:50]}...")

# Upload test artifact
upload_resp = requests.post(
    f"{API_URL}/artifacts",
    json=[{
        "name": "test-metrics",
        "url": "https://huggingface.co/google/gemma-2-2b",
        "debloat": False,
        "jsProgram": "dummy"
    }],
    headers={"X-Authorization": token}
)

print(f"\nUpload status: {upload_resp.status_code}")
print(f"Response: {upload_resp.text[:500]}")
