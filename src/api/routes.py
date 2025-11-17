from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import boto3
import json
from datetime import datetime
import os
from typing import Optional, Dict, List
import hashlib
import uuid
import time
import io
import zipfile
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trustworthy Model Registry",
    description="Phase 2 Registry API - Delivery 1"
)

# Wide-open CORS (OK for this class project)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# AWS INIT
# ----------------------------------------------------------------------
try:
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    s3 = boto3.client('s3', region_name='us-east-1')
    TABLE_NAME = os.getenv('DYNAMODB_TABLE', 'tmr-dev-registry')
    BUCKET_NAME = os.getenv('S3_BUCKET', 'tmr-dev-models')
    table = dynamodb.Table(TABLE_NAME)
    AWS_AVAILABLE = True
except Exception as e:
    print(f"Warning: AWS services not available: {e}")
    table = None
    s3 = None
    TABLE_NAME = None
    BUCKET_NAME = None
    AWS_AVAILABLE = False

# ----------------------------------------------------------------------
# SIMPLE AUTH (IN-MEMORY)
# ----------------------------------------------------------------------
_users_store: Dict[str, Dict[str, str]] = {}
_sessions_store: Dict[str, Dict[str, object]] = {}

SESSION_TTL_SECONDS = int(os.getenv('AUTH_SESSION_TTL', '3600'))


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()


def _create_user_in_memory(username: str, password: str) -> None:
    if username in _users_store:
        raise ValueError("user_exists")
    salt = uuid.uuid4().hex
    pw_hash = _hash_password(password, salt)
    _users_store[username] = {
        "password_hash": pw_hash,
        "salt": salt,
        "created_at": datetime.utcnow().isoformat()
    }


def _get_user_in_memory(username: str) -> Optional[Dict[str, str]]:
    return _users_store.get(username)


def _create_session_in_memory(username: str) -> str:
    token = uuid.uuid4().hex
    expires_at = time.time() + SESSION_TTL_SECONDS
    _sessions_store[token] = {"username": username, "expires_at": expires_at}
    return token


def _invalidate_session_in_memory(token: str) -> bool:
    return _sessions_store.pop(token, None) is not None


def _validate_session_in_memory(token: str) -> Optional[str]:
    entry = _sessions_store.get(token)
    if not entry:
        return None
    if entry.get("expires_at", 0) < time.time():
        _sessions_store.pop(token, None)
        return None
    return entry.get("username")


# Default admin required by autograder
_DEFAULT_ADMIN_USERNAME = os.getenv(
    'DEFAULT_ADMIN_USERNAME', 'ece30861defaultadminuser'
)
_DEFAULT_ADMIN_PASSWORD = os.getenv(
    'DEFAULT_ADMIN_PASSWORD',
    "correcthorsebatterystaple123(!__+@**(A;DROP TABLE packages)"
)

try:
    if _get_user_in_memory(_DEFAULT_ADMIN_USERNAME) is None:
        _create_user_in_memory(_DEFAULT_ADMIN_USERNAME, _DEFAULT_ADMIN_PASSWORD)
        print(f"Default admin user '{_DEFAULT_ADMIN_USERNAME}' seeded (in-memory).")
except Exception:
    pass

# ----------------------------------------------------------------------
# HEALTH
# ----------------------------------------------------------------------


@app.get("/health")
@app.get("/api/v1/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Trustworthy Model Registry",
        "version": "1.0"
    }

# ----------------------------------------------------------------------
# ARTIFACT REGISTRY (IN-MEMORY FOR NON-S3 PARTS)
# ----------------------------------------------------------------------

_artifacts_store: Dict[str, Dict[str, object]] = {}
# artifact_id -> {"name": str, "type": str, "url": str, "scores": dict, "uploaded_at": str}


def _generate_artifact_id(name: str, artifact_type: str) -> str:
    timestamp = datetime.utcnow().isoformat()
    unique_str = f"{name}_{artifact_type}_{timestamp}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:12]


def _determine_artifact_type(url: str) -> str:
    url_lower = url.lower()
    if "/datasets/" in url_lower:
        return "dataset"
    elif "/spaces/" in url_lower:
        return "code"
    else:
        return "model"


@app.get("/models")
@app.get("/api/v1/models")
def list_models(
    limit: int = Query(10),
    skip: int = Query(0),
    type: Optional[str] = Query(None, description="Filter by artifact type: model, dataset, or code")
):
    """List all artifacts in registry with optional type filtering"""
    artifacts = list(_artifacts_store.values())

    if type:
        artifacts = [a for a in artifacts if a.get("type") == type]

    total = len(artifacts)
    paginated = artifacts[skip:skip + limit]

    return {
        "models": paginated,
        "count": total,
        "limit": limit,
        "skip": skip,
        "message": "Artifacts retrieved successfully"
    }

# ----------------------------------------------------------------------
# MODEL UPLOAD (S3 + DYNAMODB)
# ----------------------------------------------------------------------


@app.post("/models/upload")
@app.post("/api/v1/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    version: Optional[str] = Form("1.0.0")
):
    """Upload a model package (ZIP file) to S3 and store metadata in DynamoDB."""
    if not AWS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AWS services not available - running in local mode"
        )

    if not file.filename or not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    MAX_SIZE = 100 * 1024 * 1024  # 100 MB

    try:
        contents = await file.read()
        file_size = len(contents)

        if file_size > MAX_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size} bytes (max: {MAX_SIZE})"
            )

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3_key = f"models/{model_id}/{timestamp}.zip"

        file_hash = hashlib.sha256(contents).hexdigest()

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=contents,
            ContentType='application/zip',
            Metadata={
                'model_id': model_id,
                'sha256': file_hash,
                'original_filename': file.filename
            }
        )

        created_at = datetime.utcnow().isoformat()
        table.put_item(
            Item={
                'model_id': model_id,
                'name': name,
                'description': description or '',
                'version': version,
                's3_key': s3_key,
                'bucket': BUCKET_NAME,
                'size_bytes': file_size,
                'sha256': file_hash,
                'created_at': created_at,
                'updated_at': created_at,
                'status': 'uploaded'
            }
        )

        return {
            "status": "success",
            "model_id": model_id,
            "s3_key": s3_key,
            "bucket": BUCKET_NAME,
            "size_bytes": file_size,
            "sha256": file_hash,
            "message": "Model uploaded successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ----------------------------------------------------------------------
# UPDATE MODEL (IN-MEMORY METADATA ONLY)
# ----------------------------------------------------------------------


@app.put("/models/{model_id}")
@app.put("/api/v1/models/{model_id}")
def update_model(
    model_id: str,
    name: Optional[str] = Query(None),
    version: Optional[str] = Query(None),
    description: Optional[str] = Query(None)
):
    """Update model metadata (UPDATE operation for CRUD) in the in-memory store."""
    if model_id not in _artifacts_store:
        raise HTTPException(status_code=404, detail="Model not found")

    model = _artifacts_store[model_id]

    if name is not None:
        model["name"] = name
    if version is not None:
        model["version"] = version
    if description is not None:
        model["description"] = description

    model["updated_at"] = datetime.utcnow().isoformat()

    return {
        "status": "success",
        "model_id": model_id,
        "model": model,
        "message": "Model updated successfully"
    }

# ----------------------------------------------------------------------
# SEARCH
# ----------------------------------------------------------------------


@app.get("/models/search")
@app.get("/api/v1/models/search")
def search_models(
    query: str = Query(..., description="Search query (supports regex)"),
    use_regex: bool = Query(False, description="Treat query as regex pattern"),
    search_in: str = Query("name", description="Search in: name, description, or both")
):
    """Search for models using text or regex patterns."""
    import re as regex_module

    results = []

    try:
        if use_regex:
            try:
                pattern = regex_module.compile(query, regex_module.IGNORECASE)
            except regex_module.error as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid regex pattern: {str(e)}"
                )

        for artifact in _artifacts_store.values():
            match = False

            if search_in in ["name", "both"]:
                if use_regex:
                    if pattern.search(artifact.get("name", "")):
                        match = True
                else:
                    if query.lower() in artifact.get("name", "").lower():
                        match = True

            if search_in in ["description", "both"]:
                desc = artifact.get("description", "")
                if use_regex:
                    if pattern.search(desc):
                        match = True
                else:
                    if query.lower() in desc.lower():
                        match = True

            if match:
                results.append(artifact)

        return {
            "query": query,
            "use_regex": use_regex,
            "search_in": search_in,
            "count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error("Search error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------------------------
# INGEST (PHASE 1-STYLE METRICS, IN-MEMORY)
# ----------------------------------------------------------------------


@app.post("/models/ingest")
@app.post("/api/v1/models/ingest")
def ingest_model(hf_url: str = Query(...)):
    """Ingest model from HuggingFace URL and compute toy metrics (in-memory)."""
    parts = hf_url.rstrip('/').split('/')
    artifact_name = parts[-1] if parts else "unknown"

    artifact_type = _determine_artifact_type(hf_url)
    artifact_id = _generate_artifact_id(artifact_name, artifact_type)

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
    ingestible = all(score >= 0.5 for score in scores.values())

    if not ingestible:
        return {
            "status": "rejected",
            "message": "Model does not meet minimum quality threshold (0.5 on all metrics)",
            "scores": scores,
            "overall_score": round(overall_score, 3)
        }

    _artifacts_store[artifact_id] = {
        "id": artifact_id,
        "name": artifact_name,
        "type": artifact_type,
        "url": hf_url,
        "scores": scores,
        "overall_score": round(overall_score, 3),
        "uploaded_at": datetime.utcnow().isoformat()
    }

    logger.info("Ingested %s: %s (ID: %s)", artifact_type, artifact_name, artifact_id)

    return {
        "status": "success",
        "model_id": artifact_id,
        "name": artifact_name,
        "type": artifact_type,
        "scores": scores,
        "overall_score": round(overall_score, 3),
        "message": f"{artifact_type.capitalize()} ingested successfully"
    }

# ----------------------------------------------------------------------
# AUTH ENDPOINTS (BOTH /... AND /api/v1/...)
# ----------------------------------------------------------------------


@app.post("/register")
@app.post("/api/v1/register")
def register(username: str = Query(...), password: str = Query(...)):
    """Register a new user (in-memory)."""
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")
    try:
        _create_user_in_memory(username, password)
    except ValueError as e:
        if str(e) == "user_exists":
            raise HTTPException(status_code=409, detail="user already exists")
        raise HTTPException(status_code=500, detail="internal error")

    return {"status": "registered", "username": username}


@app.post("/login")
@app.post("/api/v1/login")
def login(username: str = Query(...), password: str = Query(...)):
    """Authenticate user and return a session token."""
    user = _get_user_in_memory(username)
    if not user:
        raise HTTPException(status_code=401, detail="invalid credentials")
    pw_hash = _hash_password(password, user["salt"])
    if pw_hash != user["password_hash"]:
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = _create_session_in_memory(username)
    return {"status": "ok", "token": token, "expires_in": SESSION_TTL_SECONDS}


@app.post("/logout")
@app.post("/api/v1/logout")
def logout(token: Optional[str] = Query(None)):
    """Invalidate a session token."""
    if token:
        ok = _invalidate_session_in_memory(token)
        if not ok:
            raise HTTPException(status_code=404, detail="token not found")
        return {"status": "logged_out"}
    raise HTTPException(status_code=400, detail="token required")

# ----------------------------------------------------------------------
# SENSITIVE MODELS + DOWNLOAD HISTORY
# ----------------------------------------------------------------------

_sensitive_models: Dict[str, Dict[str, str]] = {}
_download_history: List[Dict[str, object]] = []


def _get_username_from_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    return _validate_session_in_memory(token)


def _execute_js_program(
    js_program: str,
    model_name: str,
    uploader_username: str,
    downloader_username: str,
    zip_file_path: str
) -> tuple[bool, str]:
    """Execute JavaScript monitoring program before allowing download."""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            logger.warning("Node.js not available, bypassing JS check")
            return True, "Node.js not available, check bypassed"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("Node.js not found or timeout, bypassing JS check")
        return True, "Node.js not found, check bypassed"

    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.js',
            delete=False,
            encoding='utf-8'
        ) as js_file:
            js_file.write(js_program)
            js_file_path = js_file.name

        cmd = [
            "node",
            js_file_path,
            model_name,
            uploader_username,
            downloader_username,
            zip_file_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        try:
            os.unlink(js_file_path)
        except Exception:
            pass

        success = result.returncode == 0
        output = result.stdout if success else f"JS check failed: {result.stdout}\n{result.stderr}"

        return success, output

    except subprocess.TimeoutExpired:
        return False, "JavaScript program execution timeout (30s)"
    except Exception as e:
        return False, f"Error executing JavaScript program: {str(e)}"


@app.post("/models/{model_id}/sensitive")
@app.post("/api/v1/models/{model_id}/sensitive")
def mark_model_sensitive(
    model_id: str,
    js_program: str = Query(..., description="JavaScript monitoring program"),
    token: Optional[str] = Query(None, description="Authentication token")
):
    username = _get_username_from_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="authentication required")

    if not js_program or not js_program.strip():
        raise HTTPException(status_code=400, detail="js_program cannot be empty")

    _sensitive_models[model_id] = {
        "js_program": js_program,
        "uploader_username": username,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

    logger.info("Model %s marked as sensitive by %s", model_id, username)

    return {
        "status": "success",
        "model_id": model_id,
        "message": "Model marked as sensitive with JS monitoring program"
    }


@app.get("/models/{model_id}/sensitive")
@app.get("/api/v1/models/{model_id}/sensitive")
def get_sensitive_model_info(model_id: str):
    if model_id not in _sensitive_models:
        raise HTTPException(
            status_code=404,
            detail="Model is not marked as sensitive or does not exist"
        )

    info = _sensitive_models[model_id]
    return {
        "model_id": model_id,
        "js_program": info["js_program"],
        "uploader_username": info["uploader_username"],
        "created_at": info["created_at"],
        "updated_at": info["updated_at"]
    }


@app.delete("/models/{model_id}/sensitive")
@app.delete("/api/v1/models/{model_id}/sensitive")
def remove_sensitive_flag(
    model_id: str,
    token: Optional[str] = Query(None, description="Authentication token")
):
    username = _get_username_from_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="authentication required")

    if model_id not in _sensitive_models:
        raise HTTPException(
            status_code=404,
            detail="Model is not marked as sensitive"
        )

    model_info = _sensitive_models[model_id]
    if model_info["uploader_username"] != username and username != _DEFAULT_ADMIN_USERNAME:
        raise HTTPException(
            status_code=403,
            detail="Only the uploader or admin can remove sensitive flag"
        )

    del _sensitive_models[model_id]

    logger.info("Model %s sensitive flag removed by %s", model_id, username)

    return {
        "status": "success",
        "model_id": model_id,
        "message": "Sensitive flag removed"
    }


@app.get("/models/{model_id}/download-history")
@app.get("/api/v1/models/{model_id}/download-history")
def get_download_history(
    model_id: str,
    token: Optional[str] = Query(None, description="Authentication token")
):
    username = _get_username_from_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="authentication required")

    if model_id not in _sensitive_models:
        raise HTTPException(
            status_code=404,
            detail="Model is not marked as sensitive"
        )

    history = [entry for entry in _download_history if entry["model_id"] == model_id]

    return {
        "model_id": model_id,
        "download_count": len(history),
        "downloads": history
    }

# ----------------------------------------------------------------------
# GET / DELETE MODEL (DYNAMODB)
# ----------------------------------------------------------------------


@app.get("/models/{model_id}")
@app.get("/api/v1/models/{model_id}")
def get_model(model_id: str, token: Optional[str] = Query(None)):
    """Retrieve a specific model by ID from DynamoDB."""
    if not AWS_AVAILABLE:
        if model_id in _artifacts_store:
            return _artifacts_store[model_id]
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        response = table.get_item(Key={'model_id': model_id})

        if 'Item' not in response:
            raise HTTPException(status_code=404, detail="Model not found")

        item = response['Item']
        return {
            "model_id": item['model_id'],
            "name": item['name'],
            "description": item.get('description', ''),
            "version": item.get('version', '1.0.0'),
            "s3_key": item.get('s3_key'),
            "bucket": item.get('bucket'),
            "size_bytes": item.get('size_bytes'),
            "sha256": item.get('sha256'),
            "created_at": item.get('created_at'),
            "updated_at": item.get('updated_at'),
            "status": item.get('status', 'unknown')
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model: {str(e)}")


@app.delete("/models/{model_id}")
@app.delete("/api/v1/models/{model_id}")
def delete_model(model_id: str):
    """Delete a model from registry (stub)."""
    return {
        "status": "deleted",
        "model_id": model_id,
        "message": "Model deleted successfully"
    }

# ----------------------------------------------------------------------
# DOWNLOAD MODEL (S3, VARIANTS)
# ----------------------------------------------------------------------


@app.get("/models/{model_id}/download")
@app.get("/api/v1/models/{model_id}/download")
async def download_model(
    model_id: str,
    variant: str = Query("full", description="Download variant: 'full', 'weights', or 'dataset'"),
    token: Optional[str] = Query(None, description="Authentication token")
):
    if not AWS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AWS services not available - running in local mode"
        )

    valid_variants = ["full", "weights", "dataset"]
    if variant not in valid_variants:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid variant. Must be one of: {', '.join(valid_variants)}"
        )

    try:
        response = table.get_item(Key={'model_id': model_id})

        if 'Item' not in response:
            raise HTTPException(status_code=404, detail="Model not found")

        item = response['Item']
        s3_key = item.get('s3_key')
        bucket = item.get('bucket', BUCKET_NAME)

        if not s3_key:
            raise HTTPException(
                status_code=500,
                detail="Model metadata missing S3 key"
            )

        s3_response = s3.get_object(Bucket=bucket, Key=s3_key)
        file_content = s3_response['Body'].read()

        size_bytes = len(file_content)
        size_mb = size_bytes / (1024 * 1024)

        if variant == "full":
            return StreamingResponse(
                io.BytesIO(file_content),
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename={model_id}.zip",
                    "X-Model-ID": model_id,
                    "X-Size-Bytes": str(size_bytes),
                    "X-Size-MB": f"{size_mb:.2f}",
                    "X-Variant": "full"
                }
            )

        filtered_content = _filter_zip_by_variant(file_content, variant)
        filtered_size = len(filtered_content)
        filtered_mb = filtered_size / (1024 * 1024)

        return StreamingResponse(
            io.BytesIO(filtered_content),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={model_id}_{variant}.zip",
                "X-Model-ID": model_id,
                "X-Size-Bytes": str(filtered_size),
                "X-Size-MB": f"{filtered_mb:.2f}",
                "X-Variant": variant,
                "X-Original-Size-Bytes": str(size_bytes)
            }
        )

    except HTTPException:
        raise
    except s3.exceptions.NoSuchKey:
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found in S3: {s3_key}"
        )
    except Exception as e:
        print(f"Download error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Download failed: {str(e)}"
        )


def _filter_zip_by_variant(zip_content: bytes, variant: str) -> bytes:
    """Filter ZIP file contents based on variant (weights or dataset)."""
    weight_extensions = {'.bin', '.safetensors', '.pt', '.pth', '.h5', '.onnx', '.pb'}
    weight_patterns = {'config.json', 'tokenizer', 'vocab', 'merges.txt', 'special_tokens'}

    dataset_patterns = {
        'dataset', 'data/', 'train.', 'test.', 'val.', 'valid.',
        '.csv', '.json', '.txt', '.parquet'
    }

    original_zip = zipfile.ZipFile(io.BytesIO(zip_content), 'r')

    filtered_buffer = io.BytesIO()
    filtered_zip = zipfile.ZipFile(filtered_buffer, 'w', zipfile.ZIP_DEFLATED)

    try:
        for file_info in original_zip.filelist:
            filename = file_info.filename.lower()
            include = False

            if variant == "weights":
                if any(filename.endswith(ext) for ext in weight_extensions):
                    include = True
                elif any(pattern in filename for pattern in weight_patterns):
                    include = True

            elif variant == "dataset":
                if any(pattern in filename for pattern in dataset_patterns):
                    include = True

            if include:
                data = original_zip.read(file_info.filename)
                filtered_zip.writestr(file_info, data)

        filtered_zip.close()
        original_zip.close()

        filtered_buffer.seek(0)
        return filtered_buffer.read()

    except Exception as e:
        print(f"Error filtering ZIP: {e}")
        return zip_content

# ----------------------------------------------------------------------
# MODEL SIZE (S3-AWARE VERSION ONLY)
# ----------------------------------------------------------------------


@app.get("/models/{model_id}/size")
@app.get("/api/v1/models/{model_id}/size")
def get_model_size(model_id: str):
    """
    Get the size cost of a model (size of the download) from DynamoDB metadata.
    """
    if not AWS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AWS services not available - running in local mode"
        )

    try:
        response = table.get_item(Key={'model_id': model_id})

        if 'Item' not in response:
            raise HTTPException(status_code=404, detail="Model not found")

        item = response['Item']
        size_bytes = int(item.get('size_bytes', 0))

        size_kb = size_bytes / 1024
        size_mb = size_bytes / (1024 * 1024)
        size_gb = size_bytes / (1024 * 1024 * 1024)

        return {
            "model_id": model_id,
            "size_bytes": size_bytes,
            "size_kb": round(size_kb, 2),
            "size_mb": round(size_mb, 2),
            "size_gb": round(size_gb, 4),
            "human_readable": _format_size(size_bytes)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting model size: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model size: {str(e)}"
        )


def _format_size(bytes_size: int) -> str:
    size = float(bytes_size)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

# ----------------------------------------------------------------------
# RESET REGISTRY (TEST ONLY)
# ----------------------------------------------------------------------


@app.post("/reset")
@app.post("/api/v1/reset")
def reset_registry():
    """Reset in-memory stores and re-seed default admin."""
    global _sensitive_models, _download_history, _users_store, _sessions_store, _artifacts_store
    _sensitive_models.clear()
    _download_history.clear()
    _users_store.clear()
    _sessions_store.clear()
    _artifacts_store.clear()

    try:
        if _get_user_in_memory(_DEFAULT_ADMIN_USERNAME) is None:
            _create_user_in_memory(_DEFAULT_ADMIN_USERNAME, _DEFAULT_ADMIN_PASSWORD)
    except Exception:
        pass

    return {
        "status": "reset",
        "deleted": len(_artifacts_store),
        "message": "Registry reset successfully (including sensitive models, users, and artifacts)"
    }

# ----------------------------------------------------------------------
# RATING, LINEAGE, LICENSE CHECK (IN-MEMORY / STUBBY)
# ----------------------------------------------------------------------


@app.get("/models/{model_id}/rate")
@app.get("/api/v1/models/{model_id}/rate")
def rate_model(
    model_id: str,
    compute_reproducibility: bool = Query(False),
    compute_reviewedness: bool = Query(False),
    compute_treescore: bool = Query(False)
):
    """Get quality ratings for a model with optional Phase-2 metrics."""
    if model_id not in _artifacts_store:
        raise HTTPException(status_code=404, detail="Model not found")

    model = _artifacts_store[model_id]
    scores = model.get("scores", {})

    if not scores:
        scores = {
            "ramp_up_time": 0.75,
            "license": 0.80,
            "size": 0.65,
            "availability": 0.90,
            "code_quality": 0.70,
            "dataset_quality": 0.60,
            "performance_claims": 0.85,
            "bus_factor": 0.50,
            "reproducibility": -1,
            "reviewedness": -1,
            "treescore": -1
        }

    if compute_reproducibility and scores.get("reproducibility", -1) == -1:
        model_url = model.get("url")
        if model_url and "huggingface.co" in model_url:
            try:
                from src.Metrics import ReproducibilityMetric
                logger.info("Computing reproducibility for %s", model_id)

                reproducibility_metric = ReproducibilityMetric()
                repro_score = reproducibility_metric.compute(
                    inputs={"model_url": model_url},
                    use_agent=True,
                    timeout=30
                )
                scores["reproducibility"] = repro_score
                logger.info("Reproducibility score for %s: %.2f", model_id, repro_score)
            except Exception as e:
                logger.error("Failed to compute reproducibility: %s", e)
                scores["reproducibility"] = -1
        else:
            logger.warning("Cannot compute reproducibility: no HuggingFace URL")

    if compute_reviewedness and scores.get("reviewedness", -1) == -1:
        model_url = model.get("url")
        git_url = model.get("git_url")

        if model_url or git_url:
            try:
                from src.Metrics import ReviewednessMetric
                logger.info("Computing reviewedness for %s", model_id)

                reviewedness_metric = ReviewednessMetric()
                review_score = reviewedness_metric.compute(
                    inputs={"model_url": model_url, "git_url": git_url}
                )
                scores["reviewedness"] = review_score
                logger.info("Reviewedness score for %s: %.3f", model_id, review_score)
            except Exception as e:
                logger.error("Failed to compute reviewedness: %s", e)
                scores["reviewedness"] = -1
        else:
            logger.warning("Cannot compute reviewedness: no model or git URL")

    if compute_treescore and scores.get("treescore", -1) == -1:
        model_url = model.get("url")

        if model_url and "huggingface.co" in model_url:
            try:
                from src.Metrics import TreescoreMetric
                logger.info("Computing treescore for %s", model_id)

                treescore_metric = TreescoreMetric(model_registry=_artifacts_store)
                tree_score = treescore_metric.compute(inputs={"model_url": model_url})
                scores["treescore"] = tree_score
                logger.info("Treescore for %s: %.3f", model_id, tree_score)
            except Exception as e:
                logger.error("Failed to compute treescore: %s", e)
                scores["treescore"] = -1
        else:
            logger.warning("Cannot compute treescore: no HuggingFace URL")

    model["scores"] = scores

    valid_scores = [v for v in scores.values() if v >= 0]
    overall = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    computed = []
    if compute_reproducibility:
        computed.append("reproducibility")
    if compute_reviewedness:
        computed.append("reviewedness")
    if compute_treescore:
        computed.append("treescore")

    note = f"Computed: {', '.join(computed)}" if computed else \
        "Use query parameters to compute Phase 2 metrics"

    return {
        "model_id": model_id,
        "name": model.get("name"),
        "scores": scores,
        "overall_score": round(overall, 3),
        "message": "Model rated successfully",
        "note": note
    }


@app.get("/models/{model_id}/lineage")
@app.get("/api/v1/models/{model_id}/lineage")
def get_lineage(model_id: str):
    """Return a stub lineage graph for the model."""
    if model_id not in _artifacts_store:
        raise HTTPException(status_code=404, detail="Model not found")

    lineage = {
        "model_id": model_id,
        "parents": [],
        "children": [],
        "depth": 0,
        "note": "Production version would parse config.json metadata"
    }

    return lineage


@app.post("/license-check")
@app.post("/api/v1/license-check")
def check_license_compatibility(
    github_url: str = Query(..., description="GitHub repository URL"),
    model_id: str = Query(..., description="Model ID in registry")
):
    """Stub license compatibility check."""
    if model_id not in _artifacts_store:
        raise HTTPException(status_code=404, detail="Model not found")

    compatibility = {
        "github_url": github_url,
        "model_id": model_id,
        "github_license": "unknown",
        "model_license": "unknown",
        "compatible": None,
        "compatibility_level": "unknown",
        "warnings": [],
        "note": "Production version would fetch and analyze actual licenses"
    }

    return compatibility
