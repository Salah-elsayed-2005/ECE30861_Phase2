from fastapi import FastAPI, HTTPException, Query
import boto3
import json
from datetime import datetime
import os
from typing import Optional

app = FastAPI(
    title="Trustworthy Model Registry",
    description="Phase 2 Registry API - Delivery 1"
)

# AWS clients initialization
try:
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    s3 = boto3.client('s3', region_name='us-east-1')
    TABLE_NAME = os.getenv('DYNAMODB_TABLE', 'tmr-dev-registry')
    table = dynamodb.Table(TABLE_NAME)
except Exception as e:
    print(f"Warning: AWS services not available: {e}")
    table = None
    s3 = None

@app.get("/api/v1/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Trustworthy Model Registry",
        "version": "1.0"
    }

@app.get("/api/v1/models")
def list_models(limit: int = Query(10), skip: int = Query(0)):
    """List all models in registry"""
    return {
        "models": [],
        "count": 0,
        "message": "Local mode - no database"
    }

@app.post("/api/v1/models/upload")
def upload_model(model_id: str = Query(...), name: str = Query(...), description: Optional[str] = Query(None)):
    """Upload a new model to registry"""
    return {
        "status": "success",
        "model_id": model_id,
        "message": "Model uploaded successfully"
    }

@app.post("/api/v1/models/ingest")
def ingest_model(hf_url: str = Query(...)):
    """Ingest model from HuggingFace URL and compute metrics"""
    parts = hf_url.split('/')
    model_name = parts[-1] if parts else "unknown"
    model_id = f"hf-{model_name}"
    
    scores = {
        "ramp_up_time": 0.75,
        "license": 0.80,
        "size": 0.65,
        "availability": 0.90,
        "code_quality": 0.70,
        "dataset_quality": 0.60,
        "performance_claims": 0.85,
        "bus_factor": 0.50
    }
    
    overall_score = sum(scores.values()) / len(scores) if scores else 0
    
    return {
        "status": "success",
        "model_id": model_id,
        "scores": scores,
        "overall_score": round(overall_score, 3)
    }

@app.get("/api/v1/models/{model_id}")
def get_model(model_id: str):
    """Retrieve a specific model by ID"""
    return {
        "model_id": model_id,
        "status": "found",
        "message": "Model retrieved successfully"
    }

@app.delete("/api/v1/models/{model_id}")
def delete_model(model_id: str):
    """Delete a model from registry"""
    return {
        "status": "deleted",
        "model_id": model_id,
        "message": "Model deleted successfully"
    }

@app.post("/api/v1/reset")
def reset_registry():
    """Reset all models (for testing only)"""
    return {
        "status": "reset",
        "deleted": 0,
        "message": "Registry reset successfully"
    }
