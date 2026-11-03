import requests

BASE_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# Test what artifacts exist
print("1. Testing wildcard query...")
resp = requests.post(f"{BASE_URL}/artifacts", 
    json=[{"name": "*"}],
    headers={"X-Authorization": "bearer dummy"})
print(f"Status: {resp.status_code}")
artifacts = resp.json() if resp.status_code == 200 else []
print(f"Found {len(artifacts)} artifacts")
if artifacts:
    for i, art in enumerate(artifacts[:5]):
        print(f"  [{i}] {art}")

# Test byName with POST (what autograder might be using?)
print("\n2. Testing POST to /artifact/byName/distillation...")
resp = requests.post(f"{BASE_URL}/artifact/byName/distillation")
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:200]}")

# Test byName with GET
print("\n3. Testing GET to /artifact/byName/distillation...")
resp = requests.get(f"{BASE_URL}/artifact/byName/distillation")
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:200]}")

# Test if artifacts have the expected names
if artifacts:
    print(f"\n4. Sample artifact names:")
    for art in artifacts[:10]:
        print(f"  - {art['name']}")
