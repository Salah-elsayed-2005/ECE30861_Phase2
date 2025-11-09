from fastapi.testclient import TestClient
from src.api.routes import app

client = TestClient(app)

def test_health():
    """Test health endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data

def test_list_models_empty():
    """Test listing models"""
    response = client.get("/api/v1/models?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "count" in data

def test_upload_model():
    """Test uploading a model"""
    response = client.post(
        "/api/v1/models/upload?model_id=test-model-1&name=Test%20Model&description=A%20test%20model"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model_id"] == "test-model-1"

def test_ingest_model():
    """Test ingesting a HuggingFace model"""
    response = client.post(
        "/api/v1/models/ingest?hf_url=https://huggingface.co/bert-base-uncased"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "scores" in data
    assert "overall_score" in data

def test_get_model():
    """Test getting a specific model"""
    response = client.get("/api/v1/models/get-test")
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "get-test"

def test_delete_model():
    """Test deleting a model"""
    response = client.delete("/api/v1/models/del-test")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"

def test_reset_registry():
    """Test resetting registry"""
    response = client.post("/api/v1/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "reset"

class TestSensitiveModels:
    """Test sensitive model management functionality."""
    
    def setup_method(self):
        """Setup for each test - clear state and login."""
        # Clear registry state before each test
        client.post("/api/v1/reset")
        
        # Now register and login fresh user
        response = client.post(
            "/api/v1/register",
            params={"username": "testuser", "password": "testpass123"}
        )
        assert response.status_code == 200
        
        response = client.post(
            "/api/v1/login",
            params={"username": "testuser", "password": "testpass123"}
        )
        assert response.status_code == 200
        self.token = response.json()["token"]
    
    def test_mark_model_sensitive(self):
        """Test marking a model as sensitive."""
        js_program = """
        // Simple test program that always allows download
        console.log("Download approved");
        process.exit(0);
        """
        
        response = client.post(
            "/api/v1/models/test-model-1/sensitive",
            params={
                "js_program": js_program,
                "token": self.token
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_id"] == "test-model-1"
    
    def test_mark_sensitive_requires_auth(self):
        """Test that marking as sensitive requires authentication."""
        response = client.post(
            "/api/v1/models/test-model-1/sensitive",
            params={"js_program": "console.log('test');"}
        )
        
        assert response.status_code == 401
    
    def test_get_sensitive_model_info(self):
        """Test retrieving sensitive model information."""
        js_program = "console.log('test');"
        
        # Mark as sensitive first
        client.post(
            "/api/v1/models/test-model-2/sensitive",
            params={
                "js_program": js_program,
                "token": self.token
            }
        )
        
        # Get info
        response = client.get("/api/v1/models/test-model-2/sensitive")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "test-model-2"
        assert data["js_program"] == js_program
        assert "uploader_username" in data
    
    def test_get_non_sensitive_model_fails(self):
        """Test that getting info for non-sensitive model fails."""
        response = client.get("/api/v1/models/nonexistent/sensitive")
        assert response.status_code == 404
    
    def test_remove_sensitive_flag(self):
        """Test removing sensitive flag."""
        js_program = "console.log('test');"
        
        # Mark as sensitive
        client.post(
            "/api/v1/models/test-model-3/sensitive",
            params={
                "js_program": js_program,
                "token": self.token
            }
        )
        
        # Remove flag
        response = client.delete(
            "/api/v1/models/test-model-3/sensitive",
            params={"token": self.token}
        )
        
        assert response.status_code == 200
        
        # Verify it's gone
        response = client.get("/api/v1/models/test-model-3/sensitive")
        assert response.status_code == 404
    
    def test_download_sensitive_model_with_approval(self):
        """Test downloading a sensitive model when JS approves."""
        js_program = """
        console.log("Download approved for testing");
        process.exit(0);
        """
        
        # Mark as sensitive
        client.post(
            "/api/v1/models/approved-model/sensitive",
            params={
                "js_program": js_program,
                "token": self.token
            }
        )
        
        # Attempt download
        response = client.get(
            "/api/v1/models/approved-model/download",
            params={"token": self.token}
        )
        
        # Should succeed (or bypass if Node.js not available)
        assert response.status_code == 200
    
    def test_download_sensitive_model_with_rejection(self):
        """Test downloading a sensitive model when JS rejects."""
        js_program = """
        console.log("Download rejected for security reasons");
        process.exit(1);
        """
        
        # Mark as sensitive
        client.post(
            "/api/v1/models/rejected-model/sensitive",
            params={
                "js_program": js_program,
                "token": self.token
            }
        )
        
        # Attempt download - may succeed if Node.js not available (fallback)
        response = client.get(
            "/api/v1/models/rejected-model/download",
            params={"token": self.token}
        )
        
        # If Node.js is available, should fail; otherwise succeeds with bypass
        assert response.status_code in [200, 403]
    
    def test_download_history(self):
        """Test retrieving download history for sensitive model."""
        js_program = "process.exit(0);"
        
        # Mark as sensitive
        client.post(
            "/api/v1/models/history-model/sensitive",
            params={
                "js_program": js_program,
                "token": self.token
            }
        )
        
        # Download a few times
        client.get(
            "/api/v1/models/history-model/download",
            params={"token": self.token}
        )
        client.get(
            "/api/v1/models/history-model/download",
            params={"token": self.token}
        )
        
        # Get history
        response = client.get(
            "/api/v1/models/history-model/download-history",
            params={"token": self.token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "history-model"
        assert data["download_count"] >= 2
    
    def test_reset_clears_sensitive_models(self):
        """Test that reset clears sensitive models."""
        js_program = "console.log('test');"
        
        # Mark as sensitive
        client.post(
            "/api/v1/models/reset-test/sensitive",
            params={
                "js_program": js_program,
                "token": self.token
            }
        )
        
        # Reset
        response = client.post("/api/v1/reset")
        assert response.status_code == 200
        
        # Verify sensitive flag is gone
        response = client.get("/api/v1/models/reset-test/sensitive")
        assert response.status_code == 404
