"""Tests for model upload and retrieval endpoints with S3 and DynamoDB."""
import io
import os
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
import boto3

# Import after mocking to avoid AWS initialization issues
os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
os.environ['AWS_SECURITY_TOKEN'] = 'testing'
os.environ['AWS_SESSION_TOKEN'] = 'testing'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'


@pytest.fixture
def mock_aws_services():
    """Set up mock AWS services for testing"""
    with mock_aws():
        # Create S3 bucket
        s3_client = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'tmr-dev-models'
        s3_client.create_bucket(Bucket=bucket_name)
        
        # Create DynamoDB table
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        table_name = 'tmr-dev-registry'
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[{'AttributeName': 'model_id', 'KeyType': 'HASH'}],
            AttributeDefinitions=[{'AttributeName': 'model_id', 'AttributeType': 'S'}],
            BillingMode='PAY_PER_REQUEST'
        )
        
        yield {
            's3': s3_client,
            'dynamodb': dynamodb,
            'table': table,
            'bucket_name': bucket_name,
            'table_name': table_name
        }


def test_upload_model_success(mock_aws_services):
    """Test successful model upload to S3 and DynamoDB"""
    # Patch AWS environment variables
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        # Import app after env vars are set
        from src.api.routes import app, _init_aws_clients; _init_aws_clients()
        client = TestClient(app)
        
        # Create a fake ZIP file
        zip_content = b'PK\x03\x04' + b'\x00' * 100  # Minimal ZIP header + padding
        files = {'file': ('test_model.zip', io.BytesIO(zip_content), 'application/zip')}
        data = {
            'model_id': 'test-model-1',
            'name': 'Test Model',
            'description': 'A test model for unit testing',
            'version': '1.0.0'
        }
        
        response = client.post('/api/v1/models/upload', files=files, data=data)
        
        assert response.status_code == 200
        result = response.json()
        assert result['status'] == 'success'
        assert result['model_id'] == 'test-model-1'
        assert 's3_key' in result
        assert result['size_bytes'] == len(zip_content)
        assert 'sha256' in result
        
        # Verify S3 object exists
        s3_objects = mock_aws_services['s3'].list_objects_v2(Bucket=mock_aws_services['bucket_name'])
        assert 'Contents' in s3_objects
        assert len(s3_objects['Contents']) == 1
        
        # Verify DynamoDB item exists
        table = mock_aws_services['table']
        item_response = table.get_item(Key={'model_id': 'test-model-1'})
        assert 'Item' in item_response
        item = item_response['Item']
        assert item['name'] == 'Test Model'
        assert item['version'] == '1.0.0'


def test_upload_model_invalid_file_type(mock_aws_services):
    """Test upload with non-ZIP file"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app, _init_aws_clients; _init_aws_clients()
        client = TestClient(app)
        
        files = {'file': ('test.txt', io.BytesIO(b'not a zip'), 'text/plain')}
        data = {
            'model_id': 'test-model-2',
            'name': 'Test Model 2',
        }
        
        response = client.post('/api/v1/models/upload', files=files, data=data)
        
        assert response.status_code == 400
        assert 'ZIP' in response.json()['detail']


def test_upload_model_too_large(mock_aws_services):
    """Test upload with file exceeding size limit"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app, _init_aws_clients; _init_aws_clients()
        client = TestClient(app)
        
        # Create a file larger than 100MB
        large_content = b'PK\x03\x04' + b'\x00' * (101 * 1024 * 1024)
        files = {'file': ('large_model.zip', io.BytesIO(large_content), 'application/zip')}
        data = {
            'model_id': 'large-model',
            'name': 'Large Model',
        }
        
        response = client.post('/api/v1/models/upload', files=files, data=data)
        
        assert response.status_code == 413
        assert 'too large' in response.json()['detail'].lower()


def test_upload_model_missing_required_fields(mock_aws_services):
    """Test upload with missing required fields"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app, _init_aws_clients; _init_aws_clients()
        client = TestClient(app)
        
        zip_content = b'PK\x03\x04' + b'\x00' * 100
        files = {'file': ('test.zip', io.BytesIO(zip_content), 'application/zip')}
        data = {'model_id': 'test-model-3'}  # Missing 'name'
        
        response = client.post('/api/v1/models/upload', files=files, data=data)
        
        assert response.status_code == 422  # Validation error


def test_get_model_success(mock_aws_services):
    """Test retrieving model metadata from DynamoDB"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app, _init_aws_clients; _init_aws_clients()
        client = TestClient(app)
        
        # First upload a model
        zip_content = b'PK\x03\x04' + b'\x00' * 100
        files = {'file': ('test.zip', io.BytesIO(zip_content), 'application/zip')}
        data = {
            'model_id': 'retrieve-test',
            'name': 'Retrieve Test Model',
            'description': 'Test retrieval',
            'version': '2.0.0'
        }
        
        upload_response = client.post('/api/v1/models/upload', files=files, data=data)
        assert upload_response.status_code == 200
        
        # Now retrieve it
        get_response = client.get('/api/v1/models/retrieve-test')
        assert get_response.status_code == 200
        
        result = get_response.json()
        assert result['model_id'] == 'retrieve-test'
        assert result['name'] == 'Retrieve Test Model'
        assert result['version'] == '2.0.0'
        assert 's3_key' in result
        assert 'size_bytes' in result


def test_get_model_not_found(mock_aws_services):
    """Test retrieving non-existent model"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app, _init_aws_clients; _init_aws_clients()
        client = TestClient(app)
        
        response = client.get('/api/v1/models/nonexistent-model')
        assert response.status_code == 404
        assert 'not found' in response.json()['detail'].lower()


def test_upload_without_aws(monkeypatch):
    """Test upload endpoint when AWS is not available (local mode)"""
    # Set AWS_AVAILABLE to False by not providing credentials
    monkeypatch.delenv('AWS_ACCESS_KEY_ID', raising=False)
    monkeypatch.delenv('AWS_SECRET_ACCESS_KEY', raising=False)
    
    # Force reimport to trigger AWS initialization failure
    import importlib
    import src.api.routes
    importlib.reload(src.api.routes)
    
    from src.api.routes import app, _init_aws_clients; _init_aws_clients()
    client = TestClient(app)
    
    zip_content = b'PK\x03\x04' + b'\x00' * 100
    files = {'file': ('test.zip', io.BytesIO(zip_content), 'application/zip')}
    data = {
        'model_id': 'local-test',
        'name': 'Local Test',
    }
    
    response = client.post('/api/v1/models/upload', files=files, data=data)
    
    assert response.status_code == 503
    assert 'not available' in response.json()['detail'].lower()
