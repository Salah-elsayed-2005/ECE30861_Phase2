"""
Final verification that real metrics are computing, not hardcoded fallbacks.
Tests multiple repos with different characteristics to ensure varied scores.
"""
import requests
import json
import time
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# Authenticate
auth_response = requests.put(f"{BASE_URL}/authenticate", json={
    "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
    "Secret": {"password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;"}
})
token = auth_response.text.strip()
headers = {"X-Authorization": token}

print("=" * 80)
print("FINAL METRICS VERIFICATION - Testing Real vs Hardcoded Scores")
print("=" * 80)

# Test repos with VERY different characteristics
test_cases = [
    {
        "name": "tiny-repo",
        "url": "https://github.com/octocat/Hello-World",
        "type": "code",
        "expected": "Small, simple repo - should have different scores than large projects"
    },
    {
        "name": "large-popular",
        "url": "https://github.com/tensorflow/tensorflow",
        "type": "model",
        "expected": "Huge, well-maintained - should have high bus_factor, reviewedness"
    },
    {
        "name": "dataset-repo",
        "url": "https://github.com/huggingface/datasets",
        "type": "dataset",
        "expected": "Dataset-focused - should have high dataset_quality"
    }
]

# Hardcoded fallback values to check against
FALLBACK_VALUES = {
    "bus_factor": 0.5,
    "ramp_up_time": 0.75,
    "license": 0.8,
    "availability": 0.9,
    "code_quality": 0.7,
    "dataset_quality": 0.6,
    "performance_claims": 0.85,
    "reproducibility": 0.6,
    "reviewedness": 0.6,
    "tree_score": 0.7
}

results = []

for i, test in enumerate(test_cases):
    print(f"\n{'='*80}")
    print(f"Test {i+1}/{len(test_cases)}: {test['name']}")
    print(f"URL: {test['url']}")
    print(f"Type: {test['type']}")
    print(f"Expected: {test['expected']}")
    print(f"{'='*80}")
    
    # Upload artifact
    response = requests.post(
        f"{BASE_URL}/artifact/{test['type']}",
        headers=headers,
        json={"url": test['url']}
    )
    
    if response.status_code != 201:
        print(f"❌ Upload failed: {response.status_code} - {response.text}")
        continue
    
    artifact = response.json()
    artifact_id = artifact['metadata']['id']
    print(f"✓ Uploaded: ID={artifact_id}")
    
    # Get rating
    time.sleep(1)  # Small delay
    rating_response = requests.get(
        f"{BASE_URL}/artifact/model/{artifact_id}/rate",
        headers=headers
    )
    
    if rating_response.status_code != 200:
        print(f"❌ Rating failed: {rating_response.status_code}")
        continue
    
    rating = rating_response.json()
    
    # Extract key metrics
    metrics = {
        "net_score": rating.get("net_score", 0),
        "bus_factor": rating.get("bus_factor", 0),
        "ramp_up_time": rating.get("ramp_up_time", 0),
        "license": rating.get("license", 0),
        "availability": rating.get("availability", 0),
        "code_quality": rating.get("code_quality", 0),
        "dataset_quality": rating.get("dataset_quality", 0),
        "performance_claims": rating.get("performance_claims", 0),
        "reproducibility": rating.get("reproducibility", 0),
        "reviewedness": rating.get("reviewedness", 0),
        "tree_score": rating.get("tree_score", 0)
    }
    
    print(f"\nMetrics received:")
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: None")
    
    # Check if ANY metric matches fallback exactly (suspicious)
    exact_matches = []
    for key, value in metrics.items():
        if key == "net_score":
            continue
        if abs(value - FALLBACK_VALUES.get(key, -999)) < 0.001:
            exact_matches.append(f"{key}={value}")
    
    if exact_matches:
        print(f"\n⚠️  WARNING: {len(exact_matches)} metrics match fallback values exactly:")
        for match in exact_matches:
            print(f"     {match}")
    else:
        print(f"\n✅ GOOD: No metrics match fallback values exactly")
    
    results.append({
        "name": test['name'],
        "metrics": metrics,
        "exact_fallback_matches": len(exact_matches)
    })

# Final analysis
print(f"\n{'='*80}")
print("FINAL ANALYSIS")
print(f"{'='*80}")

all_scores = []
for result in results:
    all_scores.extend([v for k, v in result['metrics'].items() if k != 'net_score'])

unique_scores = len(set(all_scores))
print(f"\nUnique score values across all tests: {unique_scores}")
print(f"Total scores measured: {len(all_scores)}")

if unique_scores > 10:
    print(f"✅ PASS: Wide variety of scores ({unique_scores} unique values) - REAL METRICS!")
elif unique_scores <= 10:
    print(f"❌ FAIL: Only {unique_scores} unique values - likely using HARDCODED FALLBACKS!")
    print(f"\nAll unique scores seen: {sorted(set(all_scores))}")
    print(f"Fallback values: {sorted(FALLBACK_VALUES.values())}")

# Check if different repos have different scores
if len(results) >= 2:
    print(f"\nComparing repos:")
    for i in range(len(results) - 1):
        r1 = results[i]
        r2 = results[i + 1]
        
        differences = 0
        for key in ['bus_factor', 'ramp_up_time', 'code_quality']:
            if abs(r1['metrics'][key] - r2['metrics'][key]) > 0.01:
                differences += 1
        
        print(f"  {r1['name']} vs {r2['name']}: {differences}/3 key metrics differ")
        
        if differences == 0:
            print(f"    ❌ SUSPICIOUS: All key metrics identical!")
        elif differences >= 2:
            print(f"    ✅ GOOD: Metrics vary between repos")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")

total_exact_matches = sum(r['exact_fallback_matches'] for r in results)
if total_exact_matches == 0 and unique_scores > 10:
    print("✅ VERIFIED: Using REAL METRICS computation!")
    print("   - No exact matches with fallback values")
    print("   - Wide variety of unique scores")
    print("   - Different repos produce different metrics")
elif total_exact_matches > len(results) * 5:  # Most metrics are fallbacks
    print("❌ WARNING: Likely using HARDCODED FALLBACKS!")
    print(f"   - {total_exact_matches} exact fallback matches found")
else:
    print("⚠️  UNCERTAIN: Some metrics may be using fallbacks")
    print(f"   - {total_exact_matches} exact fallback matches")
    print("   - Check CloudWatch logs to confirm")

print(f"{'='*80}\n")
