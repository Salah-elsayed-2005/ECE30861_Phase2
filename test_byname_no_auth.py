import requests

BASE_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# Test byName WITHOUT auth
print("Testing /artifact/byName/distillation without auth...")
byname_resp = requests.get(f"{BASE_URL}/artifact/byName/distillation")
print(f"Status: {byname_resp.status_code}")
print(f"Response: {byname_resp.text}")
