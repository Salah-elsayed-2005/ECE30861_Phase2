"""Quick test to verify metrics are computing (not using fallbacks)"""
import requests
import json

BASE_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# Auth
auth = requests.put(f"{BASE_URL}/authenticate", json={
    "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
    "Secret": {"password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;"}
})
token = auth.text.strip()
headers = {"X-Authorization": token}

print("Testing metrics computation...")
print("=" * 60)

# Upload artifact
resp = requests.post(
    f"{BASE_URL}/artifact/model",
    headers=headers,
    json={"url": "https://github.com/lodash/lodash"}
)

if resp.status_code == 201:
    artifact_id = resp.json()["metadata"]["id"]
    print(f"✓ Uploaded artifact: {artifact_id}")
    
    # Get rating
    rating = requests.get(
        f"{BASE_URL}/artifact/model/{artifact_id}/rate",
        headers=headers
    ).json()
    
    print(f"\nMetrics:")
    print(f"  bus_factor: {rating['bus_factor']}")
    print(f"  ramp_up_time: {rating['ramp_up_time']}")
    print(f"  license: {rating['license']}")
    print(f"  code_quality: {rating['code_quality']}")
    print(f"  dataset_quality: {rating['dataset_quality']}")
    
    # Check if ALL fallback
    fallbacks = {
        "bus_factor": 0.5,
        "ramp_up_time": 0.75,
        "license": 0.8,
        "code_quality": 0.7,
        "dataset_quality": 0.6
    }
    
    matches = sum(1 for k, v in fallbacks.items() if abs(rating[k] - v) < 0.001)
    
    print(f"\nResult:")
    if matches == 5:
        print("❌ ALL metrics match fallbacks - still using hardcoded values!")
    elif matches == 0:
        print("✅ NO metrics match fallbacks - using REAL computation!")
    else:
        print(f"⚠️  {matches}/5 metrics match fallbacks - partial computation")
else:
    print(f"❌ Upload failed: {resp.status_code} - {resp.text}")
