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
