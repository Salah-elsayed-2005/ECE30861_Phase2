"""
Autograder-compatible routes for Phase 2.
These endpoints match the OpenAPI specification exactly.
"""

from fastapi import FastAPI, HTTPException, Header, Query, Body
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import hashlib
import time
import uuid
import os
import jwt
import boto3
from decimal import Decimal

# Import existing utilities and stores from routes.py
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import metrics bridge
try:
    from src.metric_bridge import compute_artifact_metrics
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Metrics bridge not available: {e}")
    METRICS_AVAILABLE = False

# Import registry helpers
try:
    from src.registry_helpers import (
        extract_lineage_graph,
        check_license_compatibility,
        calculate_artifact_cost_with_dependencies
    )
    HELPERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Registry helpers not available: {e}")
    HELPERS_AVAILABLE = False

app = FastAPI(
    title="ECE 461 - Fall 2025 - Project Phase 2",
    version="3.4.4",
    description="API for ECE 461/Fall 2025/Project Phase 2: A Trustworthy Model Registry"
)

# ==================== AWS Setup ====================
try:
    # DynamoDB setup
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    TABLE_NAME = os.getenv('DYNAMODB_TABLE', 'tmr-dev-registry')
    table = dynamodb.Table(TABLE_NAME)
    
    # S3 setup
    s3_client = boto3.client('s3', region_name='us-east-1')
    S3_BUCKET = os.getenv('S3_BUCKET', 'tmr-dev-models')
    
    AWS_AVAILABLE = True
    print(f"AWS initialized: DynamoDB table={TABLE_NAME}, S3 bucket={S3_BUCKET}")
except Exception as e:
    print(f"Warning: AWS not available: {e}")
    table = None
    s3_client = None
    AWS_AVAILABLE = False

# JWT secret
JWT_SECRET = os.getenv('JWT_SECRET', 'ece461-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'

# API Gateway URL for download links
API_GATEWAY_URL = os.getenv('API_GATEWAY_URL', 'https://example.com')

# ==================== Data Models ====================

class ArtifactMetadata(BaseModel):
    name: str
    id: str
    type: str  # model, dataset, or code

class ArtifactData(BaseModel):
    url: str
    download_url: Optional[str] = None

class Artifact(BaseModel):
    metadata: ArtifactMetadata
    data: ArtifactData

class ArtifactQuery(BaseModel):
    name: str
    types: Optional[List[str]] = None

class ArtifactRegEx(BaseModel):
    regex: str

class User(BaseModel):
    name: str
    is_admin: bool

class UserAuthenticationInfo(BaseModel):
    password: str

class AuthenticationRequest(BaseModel):
    user: User
    secret: UserAuthenticationInfo

class SimpleLicenseCheckRequest(BaseModel):
    github_url: str

class ModelRating(BaseModel):
    name: str
    category: str
    net_score: float
    net_score_latency: float
    ramp_up_time: float
    ramp_up_time_latency: float
    bus_factor: float
    bus_factor_latency: float
    performance_claims: float
    performance_claims_latency: float
    license: float
    license_latency: float
    dataset_and_code_score: float
    dataset_and_code_score_latency: float
    dataset_quality: float
    dataset_quality_latency: float
    code_quality: float
    code_quality_latency: float
    reproducibility: float
    reproducibility_latency: float
    reviewedness: float
    reviewedness_latency: float
    tree_score: float
    tree_score_latency: float
    size_score: Dict[str, float]
    size_score_latency: float

# ==================== In-Memory Storage ====================

# Fall back to memory if DynamoDB not available
_artifacts_store: Dict[str, Dict[str, Any]] = {}
_users_store: Dict[str, Dict[str, str]] = {}

SESSION_TTL_SECONDS = 3600

# Seed default admin
_DEFAULT_ADMIN_USERNAME = 'ece30861defaultadminuser'
_DEFAULT_ADMIN_PASSWORD = "correcthorsebatterystaple123(!__+@**(A;DROP TABLE artifacts;"

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()

def _create_user(username: str, password: str, is_admin: bool = False):
    """Create user in DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            # Check if exists
            response = table.get_item(Key={'model_id': f'USER#{username}'})
            if 'Item' in response:
                raise ValueError("user_exists")
            
            salt = uuid.uuid4().hex
            pw_hash = _hash_password(password, salt)
            
            table.put_item(Item={
                'model_id': f'USER#{username}',
                'password_hash': pw_hash,
                'salt': salt,
                'is_admin': is_admin,
                'created_at': datetime.utcnow().isoformat()
            })
            return
        except Exception as e:
            print(f"DynamoDB error, falling back to memory: {e}")
    
    # Fallback to memory
    if username in _users_store:
        raise ValueError("user_exists")
    salt = uuid.uuid4().hex
    pw_hash = _hash_password(password, salt)
    _users_store[username] = {
        "password_hash": pw_hash,
        "salt": salt,
        "is_admin": is_admin,
        "created_at": datetime.utcnow().isoformat()
    }

def _get_user(username: str) -> Optional[Dict[str, Any]]:
    """Get user from DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            response = table.get_item(Key={'model_id': f'USER#{username}'})
            if 'Item' in response:
                return dict(response['Item'])
        except Exception as e:
            print(f"DynamoDB error: {e}")
    
    return _users_store.get(username)

def _validate_token(token: Optional[str]) -> Optional[str]:
    """Validate JWT token and return username if valid"""
    if not token:
        return None
    
    # Remove 'bearer ' prefix if present
    if token.lower().startswith('bearer '):
        token = token[7:]
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get('sub')
        exp = payload.get('exp')
        
        # Check expiration
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
            return None
        
        return username
    except jwt.InvalidTokenError:
        return None

def _generate_artifact_id() -> str:
    """Generate unique artifact ID"""
    return str(abs(hash(uuid.uuid4().hex + str(time.time()))))[:12]

def _store_artifact(artifact_id: str, artifact_data: Dict[str, Any]):
    """Store artifact in DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            # Convert floats to Decimal for DynamoDB
            item = {'model_id': f'ARTIFACT#{artifact_id}'}
            for key, value in artifact_data.items():
                if isinstance(value, float):
                    item[key] = Decimal(str(value))
                elif isinstance(value, dict):
                    item[key] = {k: Decimal(str(v)) if isinstance(v, float) else v for k, v in value.items()}
                else:
                    item[key] = value
            
            table.put_item(Item=item)
            return
        except Exception as e:
            print(f"DynamoDB error, falling back to memory: {e}")
    
    _artifacts_store[artifact_id] = artifact_data

def _get_artifact(artifact_id: str) -> Optional[Dict[str, Any]]:
    """Get artifact from DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            response = table.get_item(Key={'model_id': f'ARTIFACT#{artifact_id}'})
            if 'Item' in response:
                item = dict(response['Item'])
                # Convert Decimal back to float
                for key, value in item.items():
                    if isinstance(value, Decimal):
                        item[key] = float(value)
                    elif isinstance(value, dict):
                        item[key] = {k: float(v) if isinstance(v, Decimal) else v for k, v in value.items()}
                return item
        except Exception as e:
            print(f"DynamoDB error: {e}")
    
    return _artifacts_store.get(artifact_id)

def _list_artifacts() -> List[Dict[str, Any]]:
    """List all artifacts from DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            response = table.scan(
                FilterExpression='begins_with(model_id, :prefix)',
                ExpressionAttributeValues={':prefix': 'ARTIFACT#'}
            )
            artifacts = []
            for item in response.get('Items', []):
                artifact_id = item['model_id'].replace('ARTIFACT#', '')
                artifact = dict(item)
                # Convert Decimal to float
                for key, value in artifact.items():
                    if isinstance(value, Decimal):
                        artifact[key] = float(value)
                artifacts.append((artifact_id, artifact))
            return artifacts
        except Exception as e:
            print(f"DynamoDB error: {e}")
    
    return list(_artifacts_store.items())

def _delete_artifact(artifact_id: str):
    """Delete artifact from DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            table.delete_item(Key={'model_id': f'ARTIFACT#{artifact_id}'})
            return
        except Exception as e:
            print(f"DynamoDB error: {e}")
    
    _artifacts_store.pop(artifact_id, None)

def _clear_all_artifacts():
    """Clear all artifacts from DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            # Scan and delete all artifacts
            response = table.scan(
                FilterExpression='begins_with(model_id, :prefix)',
                ExpressionAttributeValues={':prefix': 'ARTIFACT#'}
            )
            for item in response.get('Items', []):
                table.delete_item(Key={'model_id': item['model_id']})
            return
        except Exception as e:
            print(f"DynamoDB error: {e}")
    
    _artifacts_store.clear()

# ==================== S3 Helper Functions ====================

def _upload_to_s3(artifact_id: str, content: bytes, artifact_type: str) -> str:
    """Upload artifact content to S3 and return S3 key"""
    if not AWS_AVAILABLE or not s3_client:
        raise HTTPException(status_code=500, detail="S3 not available")
    
    # Create S3 key: artifacts/{type}/{id}/content
    s3_key = f"artifacts/{artifact_type}/{artifact_id}/content"
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=content,
            ContentType='application/octet-stream'
        )
        return s3_key
    except Exception as e:
        print(f"S3 upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {str(e)}")

def _generate_presigned_url(s3_key: str, expiration: int = 3600) -> str:
    """Generate presigned URL for S3 object (1 hour expiry)"""
    if not AWS_AVAILABLE or not s3_client:
        # Fallback for local testing
        return f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
    
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        print(f"Presigned URL error: {e}")
        return f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"

def _get_s3_object_size(s3_key: str) -> int:
    """Get size of S3 object in bytes"""
    if not AWS_AVAILABLE or not s3_client:
        return 0
    
    try:
        response = s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return response.get('ContentLength', 0)
    except Exception as e:
        print(f"S3 head_object error: {e}")
        return 0

def _delete_from_s3(s3_key: str):
    """Delete artifact from S3"""
    if not AWS_AVAILABLE or not s3_client:
        return
    
    try:
        s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception as e:
        print(f"S3 delete error: {e}")

def _download_url_content(url: str) -> bytes:
    """Download content from a URL (for artifact ingestion)"""
    import requests
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download from URL: {str(e)}")

# ==================== End S3 Helpers ====================

# Seed admin user
try:
    _create_user(_DEFAULT_ADMIN_USERNAME, _DEFAULT_ADMIN_PASSWORD, is_admin=True)
    print(f"✓ Seeded admin user: {_DEFAULT_ADMIN_USERNAME}")
except ValueError:
    pass

# ==================== ENDPOINTS ====================

@app.get("/health")
def health_check():
    """Heartbeat check (BASELINE)"""
    return JSONResponse(status_code=200, content={})

@app.get("/tracks")
def get_tracks():
    """Get planned tracks (BASELINE)"""
    return {
        "plannedTracks": ["Access control track"]
    }

@app.delete("/reset")
def reset_registry(x_authorization: Optional[str] = Header(None, alias="X-Authorization")):
    """Reset the registry (BASELINE)"""
    # Validate authentication
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    # Check if user is admin
    user = _get_user(username)
    if not user or not user.get("is_admin"):
        raise HTTPException(status_code=401, detail="You do not have permission to reset the registry.")
    
    # Clear all artifacts
    _clear_all_artifacts()
    
    # Re-seed admin
    try:
        _create_user(_DEFAULT_ADMIN_USERNAME, _DEFAULT_ADMIN_PASSWORD, is_admin=True)
    except:
        pass
    
    return JSONResponse(status_code=200, content={"message": "Registry is reset."})

@app.post("/artifacts")
def list_artifacts_query(
    queries: List[ArtifactQuery] = Body(...),
    offset: Optional[str] = Query(None),
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Get artifacts from registry (BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    results = []
    
    # Get all artifacts
    all_artifacts = _list_artifacts()
    
    # Handle wildcard query
    if len(queries) == 1 and queries[0].name == "*":
        for artifact_id, artifact in all_artifacts:
            results.append({
                "name": artifact["name"],
                "id": artifact_id,
                "type": artifact["type"]
            })
    else:
        # Handle specific queries
        for query in queries:
            for artifact_id, artifact in all_artifacts:
                if artifact["name"] == query.name:
                    if query.types is None or artifact["type"] in query.types:
                        results.append({
                            "name": artifact["name"],
                            "id": artifact_id,
                            "type": artifact["type"]
                        })
    
    # Apply offset for pagination
    start_idx = int(offset) if offset else 0
    page_size = 10
    paginated = results[start_idx:start_idx + page_size]
    
    # Return with offset header
    next_offset = str(start_idx + page_size) if start_idx + page_size < len(results) else None
    
    return JSONResponse(
        status_code=200,
        content=paginated,
        headers={"offset": next_offset} if next_offset else {}
    )

@app.post("/artifact/{artifact_type}")
def create_artifact(
    artifact_type: str,
    artifact_data: ArtifactData = Body(...),
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Register a new artifact (BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    if artifact_type not in ["model", "dataset", "code"]:
        raise HTTPException(status_code=400, detail="Invalid artifact_type.")
    
    if not artifact_data.url:
        raise HTTPException(status_code=400, detail="Missing url in artifact_data.")
    
    # Extract name from URL
    parts = artifact_data.url.rstrip('/').split('/')
    name = parts[-1] if parts else "unknown"
    
    # Generate ID
    artifact_id = _generate_artifact_id()
    
    # Compute metrics using bridge - NO MOCKS, let errors surface
    if not METRICS_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Metrics system not available. Cannot validate artifact."
        )
    
    try:
        print(f"Computing metrics for {artifact_type}: {artifact_data.url}")
        
        # Get all existing artifacts for treescore computation
        all_artifacts_list = _list_artifacts()
        registry = {aid: adata for aid, adata in all_artifacts_list}
        
        # Compute all metrics - this may take time
        metric_results = compute_artifact_metrics(
            artifact_url=artifact_data.url,
            artifact_type=artifact_type,
            artifact_name=name,
            model_registry=registry
        )
        
        print(f"Metrics computed successfully. Net score: {metric_results.get('net_score', 'N/A')}")
        
        # Extract scores (non-latency fields)
        scores = {
            k: v for k, v in metric_results.items()
            if not k.endswith('_latency') and k not in ['net_score', 'size_score', 'dataset_and_code_score']
        }
        net_score = metric_results.get('net_score', 0.0)
        
    except Exception as e:
        print(f"ERROR: Metrics computation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"The artifact rating system encountered an error: {str(e)}"
        )
    
    # Check if artifact meets threshold (all non-negative metrics >= 0.5)
    failing_metrics = [k for k, v in scores.items() if v >= 0 and v < 0.5]
    if failing_metrics:
        raise HTTPException(
            status_code=424,
            detail=f"Artifact is not registered due to the disqualified rating. Failing metrics: {', '.join(failing_metrics)}"
        )
    
    # Download content from URL and upload to S3
    s3_key = None
    download_url = artifact_data.url  # Default fallback
    
    if AWS_AVAILABLE and s3_client:
        try:
            print(f"Downloading artifact from {artifact_data.url}")
            content = _download_url_content(artifact_data.url)
            
            print(f"Uploading to S3 (size: {len(content)} bytes)")
            s3_key = _upload_to_s3(artifact_id, content, artifact_type)
            
            # Generate presigned URL for download
            download_url = _generate_presigned_url(s3_key)
            print(f"✓ Artifact uploaded to S3: {s3_key}")
        except Exception as e:
            print(f"Warning: S3 upload failed, using original URL: {e}")
            # Continue with original URL as fallback
    
    # If no S3, use API Gateway download endpoint
    if not s3_key:
        download_url = f"{API_GATEWAY_URL}/download/{artifact_id}"
    
    # Store artifact with all computed metrics
    artifact = {
        "name": name,
        "type": artifact_type,
        "url": artifact_data.url,
        "s3_key": s3_key,  # Store S3 key for later retrieval
        "download_url": download_url,  # Store download URL
        "scores": scores,
        "metric_results": metric_results,  # Store all metrics with latencies
        "net_score": net_score,
        "created_at": datetime.utcnow().isoformat(),
        "created_by": username
    }
    _store_artifact(artifact_id, artifact)
    
    # Build response
    response = {
        "metadata": {
            "name": name,
            "id": artifact_id,
            "type": artifact_type
        },
        "data": {
            "url": artifact_data.url,
            "download_url": download_url
        }
    }
    
    return JSONResponse(status_code=201, content=response)

@app.get("/artifacts/{artifact_type}/{id}")
def get_artifact(
    artifact_type: str,
    id: str,
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Get artifact by ID (BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    artifact = _get_artifact(id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    if artifact["type"] != artifact_type:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    # Use stored download_url or generate fresh presigned URL if S3 key exists
    download_url = artifact.get("download_url", artifact["url"])  # Fallback to original URL
    
    # If S3 key exists and we have AWS, try to generate fresh presigned URL
    if artifact.get("s3_key") and AWS_AVAILABLE:
        try:
            download_url = _generate_presigned_url(artifact["s3_key"])
        except Exception as e:
            print(f"Warning: Failed to generate presigned URL: {e}")
            # Fall back to stored download_url
    
    return {
        "metadata": {
            "name": artifact["name"],
            "id": id,
            "type": artifact["type"]
        },
        "data": {
            "url": artifact["url"],
            "download_url": download_url
        }
    }

@app.put("/artifacts/{artifact_type}/{id}")
def update_artifact(
    artifact_type: str,
    id: str,
    artifact: Artifact = Body(...),
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Update artifact (BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    stored = _get_artifact(id)
    if not stored:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    # Validate name and id match
    if artifact.metadata.id != id or artifact.metadata.name != stored["name"]:
        raise HTTPException(status_code=400, detail="Name and ID must match existing artifact.")
    
    # Update artifact
    stored["url"] = artifact.data.url
    stored["updated_at"] = datetime.utcnow().isoformat()
    stored["updated_by"] = username
    _store_artifact(id, stored)
    
    return JSONResponse(status_code=200, content={"message": "Artifact is updated."})

@app.delete("/artifacts/{artifact_type}/{id}")
def delete_artifact(
    artifact_type: str,
    id: str,
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Delete artifact (NON-BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    artifact = _get_artifact(id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    # Delete from S3 if exists
    if artifact.get("s3_key"):
        try:
            _delete_from_s3(artifact["s3_key"])
            print(f"✓ Deleted from S3: {artifact['s3_key']}")
        except Exception as e:
            print(f"Warning: S3 deletion failed: {e}")
    
    # Delete from database
    _delete_artifact(id)
    
    return JSONResponse(status_code=200, content={"message": "Artifact is deleted."})

@app.get("/artifact/byName/{name}")
def get_artifact_by_name(
    name: str,
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Get artifacts by name (NON-BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    results = []
    for artifact_id, artifact in _list_artifacts():
        if artifact["name"] == name:
            results.append({
                "name": artifact["name"],
                "id": artifact_id,
                "type": artifact["type"]
            })
    
    if not results:
        raise HTTPException(status_code=404, detail="No such artifact.")
    
    return results

@app.post("/artifact/byRegEx")
def get_artifact_by_regex(
    regex_query: ArtifactRegEx = Body(...),
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Search artifacts by regex (BASELINE)"""
    import re
    
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    try:
        pattern = re.compile(regex_query.regex, re.IGNORECASE)
    except re.error:
        raise HTTPException(status_code=400, detail="Invalid regex pattern.")
    
    results = []
    for artifact_id, artifact in _list_artifacts():
        if pattern.search(artifact["name"]):
            results.append({
                "name": artifact["name"],
                "id": artifact_id,
                "type": artifact["type"]
            })
    
    if not results:
        raise HTTPException(status_code=404, detail="No artifact found under this regex.")
    
    return results

@app.get("/artifact/model/{id}/rate")
def rate_model(
    id: str,
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Get model rating (BASELINE) - FIX ISSUE #2: Return real stored metrics"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    artifact = _get_artifact(id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    if artifact["type"] != "model":
        raise HTTPException(status_code=400, detail="Rating only available for model artifacts.")
    
    # Retrieve stored metric results - NO FALLBACK TO DEFAULTS
    metric_results = artifact.get("metric_results", {})
    
    if not metric_results:
        raise HTTPException(
            status_code=500,
            detail="The artifact rating system encountered an error while computing at least one metric."
        )
    
    # Ensure all required metrics are present
    required_metrics = [
        "net_score", "ramp_up_time", "bus_factor", "performance_claims",
        "license", "dataset_and_code_score", "dataset_quality", "code_quality",
        "reproducibility", "reviewedness", "tree_score", "size_score"
    ]
    
    for metric in required_metrics:
        if metric not in metric_results:
            raise HTTPException(
                status_code=500,
                detail=f"The artifact rating system encountered an error: missing {metric}."
            )
    
    # Build rating response from stored metrics (no defaults!)
    rating = {
        "name": artifact["name"],
        "category": artifact.get("category", "Machine Learning Model"),
        "net_score": float(metric_results["net_score"]),
        "net_score_latency": float(metric_results.get("net_score_latency", 0.0)),
        "ramp_up_time": float(metric_results["ramp_up_time"]),
        "ramp_up_time_latency": float(metric_results.get("ramp_up_time_latency", 0.0)),
        "bus_factor": float(metric_results["bus_factor"]),
        "bus_factor_latency": float(metric_results.get("bus_factor_latency", 0.0)),
        "performance_claims": float(metric_results["performance_claims"]),
        "performance_claims_latency": float(metric_results.get("performance_claims_latency", 0.0)),
        "license": float(metric_results["license"]),
        "license_latency": float(metric_results.get("license_latency", 0.0)),
        "dataset_and_code_score": float(metric_results["dataset_and_code_score"]),
        "dataset_and_code_score_latency": float(metric_results.get("dataset_and_code_score_latency", 0.0)),
        "dataset_quality": float(metric_results["dataset_quality"]),
        "dataset_quality_latency": float(metric_results.get("dataset_quality_latency", 0.0)),
        "code_quality": float(metric_results["code_quality"]),
        "code_quality_latency": float(metric_results.get("code_quality_latency", 0.0)),
        "reproducibility": float(metric_results["reproducibility"]),
        "reproducibility_latency": float(metric_results.get("reproducibility_latency", 0.0)),
        "reviewedness": float(metric_results["reviewedness"]),
        "reviewedness_latency": float(metric_results.get("reviewedness_latency", 0.0)),
        "tree_score": float(metric_results["tree_score"]),
        "tree_score_latency": float(metric_results.get("tree_score_latency", 0.0)),
        "size_score": metric_results["size_score"],
        "size_score_latency": float(metric_results.get("size_score_latency", 0.0))
    }
    
    return rating

@app.get("/artifact/{artifact_type}/{id}/cost")
def get_artifact_cost(
    artifact_type: str,
    id: str,
    dependency: bool = Query(False),
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Get artifact cost (BASELINE) - FIX ISSUE #5: Real cost calculation with dependencies"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    artifact = _get_artifact(id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    if artifact["type"] != artifact_type:
        raise HTTPException(status_code=400, detail="Artifact type mismatch.")
    
    try:
        if HELPERS_AVAILABLE:
            # Use helper function for cost calculation with dependencies
            all_artifacts_list = _list_artifacts()
            artifacts_store = {aid: adata for aid, adata in all_artifacts_list}
            
            cost_data = calculate_artifact_cost_with_dependencies(
                artifact_id=id,
                artifacts_store=artifacts_store,
                include_dependencies=dependency
            )
            
            return cost_data
        else:
            # Fallback: basic cost calculation
            def get_artifact_size(artifact_data: Dict) -> float:
                """Get artifact size in MB."""
                # First try S3
                if artifact_data.get("s3_key") and AWS_AVAILABLE:
                    try:
                        response = s3_client.head_object(Bucket=S3_BUCKET, Key=artifact_data["s3_key"])
                        size_bytes = response.get('ContentLength', 0)
                        if size_bytes > 0:
                            return round(size_bytes / (1024 * 1024), 2)
                    except Exception as e:
                        print(f"S3 size check failed: {e}")
                
                # Fallback: try HTTP HEAD request
                try:
                    import requests
                    url = artifact_data.get("url", "")
                    response = requests.head(url, allow_redirects=True, timeout=10)
                    size_bytes = int(response.headers.get('content-length', 0))
                    if size_bytes > 0:
                        return round(size_bytes / (1024 * 1024), 2)
                except Exception as e:
                    print(f"HTTP HEAD failed: {e}")
                
                # Fallback: estimate based on type
                artifact_type = artifact_data.get("type", "model")
                if artifact_type == "model":
                    return 412.5
                elif artifact_type == "dataset":
                    return 562.5
                else:
                    return 280.0
            
            base_cost = get_artifact_size(artifact)
            
            if not dependency:
                return {
                    id: {
                        "total_cost": base_cost
                    }
                }
            else:
                return {
                    id: {
                        "standalone_cost": base_cost,
                        "total_cost": base_cost
                    }
                }
    
    except Exception as e:
        print(f"Cost calculation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="The artifact cost calculator encountered an error."
        )

@app.get("/artifact/model/{id}/lineage")
def get_model_lineage(
    id: str,
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Get model lineage graph (BASELINE) - FIX ISSUE #3: Extract real lineage from config.json"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    artifact = _get_artifact(id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    if artifact["type"] != "model":
        raise HTTPException(status_code=400, detail="Lineage only available for model artifacts.")
    
    artifact_url = artifact.get("url", "")
    
    try:
        if HELPERS_AVAILABLE:
            # Use helper function to extract lineage from config.json
            hf_token = os.getenv('HF_TOKEN')
            lineage_graph = extract_lineage_graph(
                artifact_url=artifact_url,
                artifact_id=id,
                hf_token=hf_token
            )
            return lineage_graph
        else:
            # Fallback: basic lineage extraction
            import requests
            from urllib.parse import urlparse
            
            nodes = [{
                "artifact_id": id,
                "name": artifact["name"],
                "source": "registry"
            }]
            edges = []
            
            if "huggingface.co" in artifact_url:
                parsed = urlparse(artifact_url)
                parts = [p for p in parsed.path.split('/') if p]
                if len(parts) >= 2:
                    model_id = f"{parts[-2]}/{parts[-1]}"
                    config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
                    
                    response = requests.get(config_url, timeout=10)
                    if response.status_code == 200:
                        config = response.json()
                        
                        parent_fields = {"base_model": "base_model", "_name_or_path": "name_or_path"}
                        for field, relationship in parent_fields.items():
                            if field in config and config[field]:
                                value = str(config[field]).replace('https://huggingface.co/', '').strip('/')
                                if value and not value.startswith('.') and not value.startswith('/'):
                                    parent_id = f"parent_{abs(hash(value)) % 1000000000}"
                                    nodes.append({
                                        "artifact_id": parent_id,
                                        "name": value,
                                        "source": "config_json"
                                    })
                                    edges.append({
                                        "from_node_artifact_id": parent_id,
                                        "to_node_artifact_id": id,
                                        "relationship": relationship
                                    })
            
            return {"nodes": nodes, "edges": edges}
    
    except Exception as e:
        print(f"Lineage extraction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail="The lineage graph cannot be computed because the artifact metadata is missing or malformed."
        )

@app.post("/artifact/model/{id}/license-check")
def check_license(
    id: str,
    request: SimpleLicenseCheckRequest = Body(...),
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Check license compatibility (BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    artifact = _get_artifact(id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    if artifact["type"] != "model":
        raise HTTPException(status_code=400, detail="License check only available for model artifacts.")
    
    github_url = request.github_url
    artifact_url = artifact.get("url", "")
    
    try:
        if HELPERS_AVAILABLE:
            # Use helper function for comprehensive license checking
            git_token = os.getenv('GITHUB_TOKEN')
            hf_token = os.getenv('HF_TOKEN')
            
            is_compatible = check_license_compatibility(
                github_url=github_url,
                artifact_url=artifact_url,
                git_token=git_token,
                hf_token=hf_token
            )
            
            return is_compatible
        else:
            # Fallback: basic license checking
            import requests
            from urllib.parse import urlparse
            
            # Simplified compatibility matrix
            compatible = {
                ("mit", "apache-2.0"), ("mit", "mit"),
                ("apache-2.0", "apache-2.0"), ("apache-2.0", "mit"),
                ("bsd-2-clause", "mit"), ("bsd-3-clause", "mit"),
                ("bsd-2-clause", "apache-2.0"), ("bsd-3-clause", "apache-2.0"),
            }
            
            # Get GitHub license
            parsed = urlparse(github_url)
            parts = [p for p in parsed.path.split('/') if p]
            if len(parts) < 2:
                raise HTTPException(status_code=404, detail="Invalid GitHub URL.")
            
            owner, repo = parts[0], parts[1]
            gh_response = requests.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=10)
            
            if gh_response.status_code == 404:
                raise HTTPException(status_code=404, detail="The GitHub project could not be found.")
            elif gh_response.status_code != 200:
                raise HTTPException(status_code=502, detail="External license information could not be retrieved.")
            
            gh_data = gh_response.json()
            gh_license = gh_data.get("license", {}).get("key", "").lower()
            
            # Get model license
            parsed = urlparse(artifact_url)
            parts = [p for p in parsed.path.split('/') if p]
            if len(parts) < 2 or "huggingface.co" not in artifact_url:
                raise HTTPException(status_code=404, detail="The artifact could not be found.")
            
            model_id = f"{parts[-2]}/{parts[-1]}"
            hf_response = requests.get(f"https://huggingface.co/api/models/{model_id}", timeout=10)
            
            if hf_response.status_code == 404:
                raise HTTPException(status_code=404, detail="The artifact could not be found.")
            elif hf_response.status_code != 200:
                raise HTTPException(status_code=502, detail="External license information could not be retrieved.")
            
            hf_data = hf_response.json()
            model_license = hf_data.get("license", "").lower().strip()
            
            if not gh_license or not model_license:
                return False
            
            return (gh_license, model_license) in compatible
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"License check error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            detail="External license information could not be retrieved."
        )

@app.put("/authenticate")
def authenticate(auth_request: AuthenticationRequest = Body(...)):
    """Authenticate user (NON-BASELINE)"""
    username = auth_request.user.name
    password = auth_request.secret.password
    
    # Validate user
    user = _get_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="The user or password is invalid.")
    
    # Verify password
    pw_hash = _hash_password(password, user["salt"])
    if pw_hash != user["password_hash"]:
        raise HTTPException(status_code=401, detail="The user or password is invalid.")
    
    # Create JWT token
    payload = {
        'sub': username,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(seconds=SESSION_TTL_SECONDS),
        'is_admin': user.get('is_admin', False)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    return f"bearer {token}"

# Health check at root for compatibility
@app.get("/")
def root():
    return {"status": "ok", "service": "ECE 461 Trustworthy Model Registry"}
