from fastapi.testclient import TestClient
import json
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.autograder_routes import app

client = TestClient(app)

ADMIN_USER = "ece30861defaultadminuser"
ADMIN_PASS = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''

def test_fixes():
    print("Testing fixes with TestClient...")
    
    # 1. Authenticate
    print("Authenticating...")
    auth_payload = {
        "User": {"name": ADMIN_USER, "isAdmin": True},
        "Secret": {"password": ADMIN_PASS}
    }
    resp = client.put("/authenticate", json=auth_payload)
    if resp.status_code != 200:
        print(f"Auth failed: {resp.text}")
        return
    token = resp.text.replace("bearer ", "")
    headers = {"X-Authorization": token}
    print("Authenticated.")

    # 2. Create Artifact with Name
    print("Creating artifact with explicit name...")
    artifact_name = "test-artifact-custom-name"
    payload = {
        "url": "https://github.com/test/repo",
        "name": artifact_name
    }
    resp = client.post("/artifact/model", json=payload, headers=headers)
    if resp.status_code != 201:
        print(f"Creation failed: {resp.text}")
        return
    data = resp.json()
    artifact_id = data["metadata"]["id"]
    print(f"Created artifact {artifact_id} with name {data['metadata']['name']}")
    
    if data["metadata"]["name"] != artifact_name:
        print(f"FAIL: Name mismatch. Expected {artifact_name}, got {data['metadata']['name']}")
    else:
        print("PASS: Name preserved.")

    # 3. Get Artifact By Name
    print("Getting artifact by name...")
    resp = client.get(f"/artifact/byName/{artifact_name}", headers=headers)
    if resp.status_code != 200:
        print(f"Get by name failed: {resp.text}")
    else:
        results = resp.json()
        if len(results) > 0 and results[0]["name"] == artifact_name:
            print("PASS: Get by name successful.")
        else:
            print(f"FAIL: Get by name returned unexpected: {results}")

    # 4. Regex Search
    print("Testing Regex search...")
    regex_payload = {"regex": "test-artifact.*"}
    resp = client.post("/artifact/byRegEx", json=regex_payload, headers=headers)
    if resp.status_code != 200:
        print(f"Regex failed: {resp.text}")
    else:
        results = resp.json()
        found = any(r["name"] == artifact_name for r in results)
        if found:
            print("PASS: Regex search successful.")
        else:
            print(f"FAIL: Regex did not find artifact. Results: {results}")

    # 5. Rate Model
    print("Testing Model Rating...")
    resp = client.get(f"/artifact/model/{artifact_id}/rate", headers=headers)
    if resp.status_code != 200:
        print(f"Rating failed: {resp.text}")
    else:
        rating = resp.json()
        print(f"Rating: {json.dumps(rating, indent=2)}")
        if rating["ramp_up_time"] <= 0.5:
            print("PASS: ramp_up_time is low enough.")
        else:
            print(f"FAIL: ramp_up_time too high: {rating['ramp_up_time']}")
            
        if rating["size_score"]["raspberry_pi"] >= 0.9:
            print("PASS: raspberry_pi score is high.")
        else:
            print(f"FAIL: raspberry_pi score too low: {rating['size_score']['raspberry_pi']}")

if __name__ == "__main__":
    try:
        test_fixes()
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
