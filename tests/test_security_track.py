import pytest
from fastapi.testclient import TestClient
from src.api.autograder_routes import app, _sensitive_models, _download_history, _artifacts_store, _users_store, _create_user, _store_artifact
import src.api.autograder_routes as routes
from unittest.mock import patch
import json

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup():
    routes.AWS_AVAILABLE = False
    _sensitive_models.clear()
    _download_history.clear()
    _artifacts_store.clear()
    _users_store.clear()
    _create_user("admin", "password", is_admin=True)
    _create_user("user", "password", is_admin=False)

def get_auth_token(username="admin", password="password"):
    response = client.put("/authenticate", json={
        "user": {"name": username, "isAdmin": username == "admin"},
        "secret": {"password": password}
    })
    return response.text.replace("bearer ", "")

def test_sensitive_model_lifecycle():
    token = get_auth_token()
    headers = {"X-Authorization": f"bearer {token}"}
    
    # Create artifact
    _store_artifact("123", {"name": "test-model", "type": "model", "url": "http://example.com"})
    
    # Mark sensitive
    js_program = "console.log('Hello');"
    resp = client.post("/artifact/model/123/sensitive", json={"js_program": js_program}, headers=headers)
    assert resp.status_code == 200
    
    # Get info
    resp = client.get("/artifact/model/123/sensitive", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["js_program"] == js_program
    
    # Delete flag
    resp = client.delete("/artifact/model/123/sensitive", headers=headers)
    assert resp.status_code == 200
    
    # Get info should fail
    resp = client.get("/artifact/model/123/sensitive", headers=headers)
    assert resp.status_code == 404

def test_download_js_execution():
    token = get_auth_token()
    headers = {"X-Authorization": f"bearer {token}"}
    
    _store_artifact("123", {"name": "test-model", "type": "model", "url": "http://example.com"})
    
    # Mark sensitive with passing JS
    js_program = "console.log('Allowed'); process.exit(0);"
    client.post("/artifact/model/123/sensitive", json={"js_program": js_program}, headers=headers)
    
    # Mock subprocess.run
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Allowed"
        mock_run.return_value.stderr = ""
        
        # Download
        resp = client.get("/download/123", headers=headers)
        assert resp.status_code == 200
        
        # Mark sensitive with failing JS
        js_program = "console.error('Blocked'); process.exit(1);"
        client.post("/artifact/model/123/sensitive", json={"js_program": js_program}, headers=headers)
        
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "Blocked"
        
        # Download should fail
        resp = client.get("/download/123", headers=headers)
        assert resp.status_code == 403
    
    # Check history
    resp = client.get("/artifact/model/123/download-history", headers=headers)
    assert resp.status_code == 200
    history = resp.json()["downloads"]
    assert len(history) == 2
    assert history[0]["success"] == True
    assert history[1]["success"] == False

def test_package_confusion():
    token = get_auth_token()
    headers = {"X-Authorization": f"bearer {token}"}
    
    # Create artifacts
    _store_artifact("1", {"name": "transformers", "type": "model"})
    _store_artifact("2", {"name": "transformers-fake", "type": "model"}) # Similar (>0.8)
    
    resp = client.get("/PackageConfusionAudit", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    
    # Should flag transformers-fake
    flagged = [item["package_name"] for item in data]
    assert "transformers-fake" in flagged
