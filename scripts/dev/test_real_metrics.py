import requests
import time

API_BASE = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"
password = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''

# Authenticate
auth_payload = {
    "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
    "Secret": {"password": password}
}
response = requests.put(f"{API_BASE}/authenticate", json=auth_payload)
token = response.text.strip()
headers = {"X-Authorization": token}

print("Testing artifact upload to check if REAL metrics are computed...\n")

# Upload a simple test artifact
artifact_payload = {"url": "https://github.com/lodash/lodash"}
print(f"Uploading: {artifact_payload['url']}")
response = requests.post(f"{API_BASE}/artifact/code", json=artifact_payload, headers=headers)

if response.status_code == 201:
    artifact_data = response.json()
    artifact_id = artifact_data["metadata"]["id"]
    print(f"✅ Upload successful! Artifact ID: {artifact_id}")
    
    # Wait a moment then check CloudWatch logs
    print("\n⏳ Waiting 5 seconds for logs to propagate...")
    time.sleep(5)
    
    # Get rating to trigger fresh metric computation
    print(f"\nGetting rating for artifact {artifact_id}...")
    response = requests.get(f"{API_BASE}/artifact/model/{artifact_id}/rate", headers=headers)
    
    if response.status_code == 200:
        rating = response.json()
        print(f"✅ Rating retrieved!")
        print(f"   Net Score: {rating.get('net_score', 'N/A')}")
        print(f"   Bus Factor: {rating.get('bus_factor', 'N/A')}")
        print(f"   License: {rating.get('license', 'N/A')}")
        print("\nNow check CloudWatch logs for '✓ Computing REAL metrics' messages")
        print("This will confirm if actual metrics are being computed or fallback values are used.")
    else:
        print(f"❌ Rating failed: {response.status_code} - {response.text}")
elif response.status_code == 424:
    print("⚠️  Upload returned 424 (disqualified rating)")
    print("   Some metrics scored below 0.5 threshold")
else:
    print(f"❌ Upload failed: {response.status_code} - {response.text}")
