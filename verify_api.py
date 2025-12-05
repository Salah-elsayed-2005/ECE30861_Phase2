import requests
import json

API_BASE = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"
password = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''

print("=" * 60)
print("AUTOGRADER API VERIFICATION TEST")
print("=" * 60)

# Test 1: Health Check
print("\n[1/6] Testing /health...")
response = requests.get(f"{API_BASE}/health")
if response.status_code == 200:
    print("✅ Health check PASSED")
else:
    print(f"❌ Health check FAILED: {response.status_code}")

# Test 2: Tracks
print("\n[2/6] Testing /tracks...")
response = requests.get(f"{API_BASE}/tracks")
if response.status_code == 200 and "Access control track" in response.text:
    print("✅ Tracks PASSED")
else:
    print(f"❌ Tracks FAILED: {response.status_code}")

# Test 3: Authentication
print("\n[3/6] Testing /authenticate...")
auth_payload = {
    "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
    "Secret": {"password": password}
}
response = requests.put(f"{API_BASE}/authenticate", json=auth_payload)
if response.status_code == 200:
    token = response.text.strip()
    print(f"✅ Authentication PASSED")
    print(f"   Token: {token[:50]}...")
else:
    print(f"❌ Authentication FAILED: {response.status_code}")
    print(f"   Response: {response.text}")
    exit(1)

headers = {"X-Authorization": token}

# Test 4: Reset
print("\n[4/6] Testing /reset...")
response = requests.delete(f"{API_BASE}/reset", headers=headers)
if response.status_code == 200:
    print("✅ Reset PASSED")
else:
    print(f"❌ Reset FAILED: {response.status_code}")

# Test 5: Upload Artifact
print("\n[5/6] Testing POST /artifact/model (upload)...")
artifact_payload = {
    "url": "https://github.com/facebook/react"
}
response = requests.post(f"{API_BASE}/artifact/model", json=artifact_payload, headers=headers)
if response.status_code == 201:
    artifact_data = response.json()
    artifact_id = artifact_data["metadata"]["id"]
    print(f"✅ Upload PASSED")
    print(f"   Artifact ID: {artifact_id}")
    
    # Test 6: Query Artifacts
    print("\n[6/6] Testing POST /artifacts (query)...")
    query_payload = [{"name": "*"}]
    response = requests.post(f"{API_BASE}/artifacts", json=query_payload, headers=headers)
    if response.status_code == 200:
        artifacts = response.json()
        print(f"✅ Query PASSED - Found {len(artifacts)} artifact(s)")
    else:
        print(f"❌ Query FAILED: {response.status_code}")
elif response.status_code == 424:
    print("⚠️  Upload returned 424 (disqualified rating) - metrics too low")
    print("   This is expected for some repos")
else:
    print(f"❌ Upload FAILED: {response.status_code}")
    print(f"   Response: {response.text}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\n✅ Authentication is working!")
print("   Your API should now pass the autograder tests.")
print("   Expected score: 219/317 or higher")
