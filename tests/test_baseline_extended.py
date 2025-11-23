import pytest
from fastapi.testclient import TestClient
from src.api.autograder_routes import app, _artifacts_store, _store_artifact, _create_user, _users_store
from unittest.mock import patch, MagicMock
import src.api.autograder_routes as routes

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup():
    # Force memory mode
    routes.AWS_AVAILABLE = False
    _artifacts_store.clear()
    _users_store.clear()
    # Seed default admin
    _create_user("ece30861defaultadminuser", "correcthorsebatterystaple123(!)", is_admin=True)

def get_auth_token():
    response = client.put("/authenticate", json={
        "user": {"name": "ece30861defaultadminuser", "isAdmin": True},
        "secret": {"password": "correcthorsebatterystaple123(!)"}
    })
    assert response.status_code == 200
    return response.text.replace("bearer ", "")

def test_lineage_graph():
    token = get_auth_token()
    headers = {"X-Authorization": f"bearer {token}"}
    
    _store_artifact("123", {"name": "my-model", "type": "model", "url": "https://huggingface.co/org/my-model"})
    
    # Mock HFClient
    with patch("src.api.autograder_routes.HFClient") as MockHF:
        mock_instance = MockHF.return_value
        mock_instance.request.return_value = {"_name_or_path": "bert-base-uncased"}
        
        resp = client.get("/artifact/model/123/lineage", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        
        assert len(data["nodes"]) == 2
        assert data["nodes"][1]["name"] == "bert-base-uncased"
        assert data["edges"][0]["relationship"] == "base_model"

def test_artifact_cost():
    token = get_auth_token()
    headers = {"X-Authorization": f"bearer {token}"}
    
    _store_artifact("123", {"name": "my-model", "type": "model", "url": "https://huggingface.co/org/my-model"})
    
    # Mock HFClient for size
    with patch("src.api.autograder_routes.HFClient") as MockHF:
        mock_instance = MockHF.return_value
        mock_instance.request.side_effect = [
            {"siblings": []}, # model info
            [{"size": 1024 * 1024 * 10}] # tree: 10MB
        ]
        
        resp = client.get("/artifact/model/123/cost", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["123"]["total_cost"] == 10.0

def test_license_check():
    token = get_auth_token()
    headers = {"X-Authorization": f"bearer {token}"}
    
    _store_artifact("123", {"name": "my-model", "type": "model", "url": "https://huggingface.co/org/my-model"})
    
    # Mock requests.get for GitHub license
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "MIT License"
        
        # Mock LicenseMetric
        with patch("src.api.autograder_routes.LicenseMetric") as MockMetric:
            MockMetric.return_value.compute.return_value = 1.0 # Permissive
            
            resp = client.post("/artifact/model/123/license-check", 
                             json={"github_url": "https://github.com/org/repo"},
                             headers=headers)
            assert resp.status_code == 200
            assert resp.json() == True
