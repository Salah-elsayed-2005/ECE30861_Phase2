"""
Quick test to verify authenticate and reset endpoints work
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi.testclient import TestClient
from api.autograder_routes import app

client = TestClient(app)

def test_authenticate():
    """Test that default admin can authenticate"""
    response = client.put("/authenticate", json={
        "User": {
            "name": "ece30861defaultadminuser",
            "isAdmin": True
        },
        "Secret": {
            "password": "correcthorsebatterystaple123(!__+@**(A;DROP TABLE packages)"
        }
    })
    print(f"Authenticate status: {response.status_code}")
    print(f"Authenticate response: {response.text}")
    print(f"Authenticate headers: {response.headers}")
    
    # Should return 200 with a token
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "bearer" in response.text.lower(), f"Expected 'bearer' in response, got {response.text}"
    
    return response.text

def test_reset():
    """Test that reset works with valid admin token"""
    # First authenticate
    token = test_authenticate()
    
    # Then reset
    response = client.delete("/reset", headers={"X-Authorization": token})
    print(f"\nReset status: {response.status_code}")
    print(f"Reset response: {response.json()}")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "Registry is reset" in response.json().get("message", ""), f"Unexpected response: {response.json()}"

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    print(f"\nHealth status: {response.status_code}")
    print(f"Health response: {response.json()}")
    
    assert response.status_code == 200

if __name__ == "__main__":
    print("Testing autograder endpoints...")
    print("\n=== Testing /health ===")
    test_health()
    
    print("\n=== Testing /authenticate ===")
    test_authenticate()
    
    print("\n=== Testing /reset ===")
    test_reset()
    
    print("\n✓ All tests passed!")
