"""Test with repos that should produce distinctly different scores"""
import requests

BASE_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# Auth
auth = requests.put(f"{BASE_URL}/authenticate", json={
    "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
    "Secret": {"password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;"}
})
token = auth.text.strip()
headers = {"X-Authorization": token}

print("Testing different repos to verify real metrics...")
print("=" * 70)

test_repos = [
    ("https://github.com/torvalds/linux", "Linux Kernel - HUGE project"),
    ("https://github.com/microsoft/vscode", "VS Code - Large, active"),
    ("https://github.com/octocat/Spoon-Knife", "Tiny tutorial repo")
]

for url, description in test_repos:
    print(f"\n{description}")
    print(f"URL: {url}")
    print("-" * 70)
    
    resp = requests.post(f"{BASE_URL}/artifact/model", headers=headers, json={"url": url})
    
    if resp.status_code == 201:
        aid = resp.json()["metadata"]["id"]
        rating = requests.get(f"{BASE_URL}/artifact/model/{aid}/rate", headers=headers).json()
        
        print(f"  bus_factor: {rating['bus_factor']:.4f}")
        print(f"  ramp_up: {rating['ramp_up_time']:.4f}")
        print(f"  license: {rating['license']:.4f}")
        print(f"  code_quality: {rating['code_quality']:.4f}")
    else:
        print(f"  ❌ Failed: {resp.status_code}")

print("\n" + "=" * 70)
print("If all three repos have IDENTICAL scores, metrics are hardcoded.")
print("If scores VARY between repos, real metrics are computing!")
