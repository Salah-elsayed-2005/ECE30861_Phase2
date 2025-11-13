"""Tests for model download endpoint with variants and size checking."""
import io
import os
import zipfile
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
import boto3

# Set up test environment
os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
os.environ['AWS_SECURITY_TOKEN'] = 'testing'
os.environ['AWS_SESSION_TOKEN'] = 'testing'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'


@pytest.fixture
def mock_aws_services():
    """Set up mock AWS services with test data"""
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
        
        # Create a test ZIP file with different file types
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add weight files
            zf.writestr('model.bin', b'weight data here')
            zf.writestr('pytorch_model.bin', b'pytorch weights')
            zf.writestr('model.safetensors', b'safetensors data')
            zf.writestr('config.json', b'{"model_type": "bert"}')
            zf.writestr('tokenizer_config.json', b'{"tokenizer": "config"}')
            
            # Add dataset files
            zf.writestr('data/train.csv', b'col1,col2\nval1,val2')
            zf.writestr('data/test.csv', b'col1,col2\nval3,val4')
            zf.writestr('dataset_info.json', b'{"info": "dataset"}')
            
            # Add other files
            zf.writestr('README.md', b'# Test Model')
            zf.writestr('requirements.txt', b'torch>=1.0')
        
        zip_content = zip_buffer.getvalue()
        s3_key = 'models/test-download-model/20251113_120000.zip'
        
        # Upload test ZIP to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=zip_content,
            ContentType='application/zip'
        )
        
        # Add metadata to DynamoDB
        table.put_item(
            Item={
                'model_id': 'test-download-model',
                'name': 'Test Download Model',
                'description': 'Model for download testing',
                'version': '1.0.0',
                's3_key': s3_key,
                'bucket': bucket_name,
                'size_bytes': len(zip_content),
                'sha256': 'abc123',
                'created_at': '2025-11-13T12:00:00',
                'updated_at': '2025-11-13T12:00:00',
                'status': 'uploaded'
            }
        )
        
        yield {
            's3': s3_client,
            'dynamodb': dynamodb,
            'table': table,
            'bucket_name': bucket_name,
            'table_name': table_name,
            'zip_content': zip_content
        }


def test_download_full_model(mock_aws_services):
    """Test downloading full model ZIP"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app
        client = TestClient(app)
        
        response = client.get('/api/v1/models/test-download-model/download?variant=full')
        
        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/zip'
        assert 'test-download-model.zip' in response.headers['content-disposition']
        assert response.headers['x-variant'] == 'full'
        assert 'x-size-bytes' in response.headers
        
        # Verify it's a valid ZIP
        zip_data = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_data, 'r') as zf:
            files = zf.namelist()
            assert 'model.bin' in files
            assert 'data/train.csv' in files
            assert 'README.md' in files


def test_download_weights_only(mock_aws_services):
    """Test downloading weights variant (filters to weight files only)"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app
        client = TestClient(app)
        
        response = client.get('/api/v1/models/test-download-model/download?variant=weights')
        
        assert response.status_code == 200
        assert response.headers['x-variant'] == 'weights'
        
        # Verify it contains only weight files
        zip_data = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_data, 'r') as zf:
            files = zf.namelist()
            # Should include weight files
            assert any('.bin' in f for f in files)
            assert any('config.json' in f for f in files)
            # Should NOT include dataset files
            assert not any('train.csv' in f for f in files)


def test_download_dataset_only(mock_aws_services):
    """Test downloading dataset variant (filters to dataset files only)"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app
        client = TestClient(app)
        
        response = client.get('/api/v1/models/test-download-model/download?variant=dataset')
        
        assert response.status_code == 200
        assert response.headers['x-variant'] == 'dataset'
        
        # Verify it contains only dataset files
        zip_data = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_data, 'r') as zf:
            files = zf.namelist()
            # Should include dataset files
            assert any('train.csv' in f or 'test.csv' in f for f in files)
            # Should NOT include weight files
            assert not any('.bin' in f for f in files)


def test_download_invalid_variant(mock_aws_services):
    """Test download with invalid variant parameter"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app
        client = TestClient(app)
        
        response = client.get('/api/v1/models/test-download-model/download?variant=invalid')
        
        assert response.status_code == 400
        assert 'invalid variant' in response.json()['detail'].lower()


def test_download_nonexistent_model(mock_aws_services):
    """Test downloading a model that doesn't exist"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app
        client = TestClient(app)
        
        response = client.get('/api/v1/models/nonexistent-model/download')
        
        assert response.status_code == 404
        assert 'not found' in response.json()['detail'].lower()


def test_get_model_size(mock_aws_services):
    """Test getting model size cost information"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app
        client = TestClient(app)
        
        response = client.get('/api/v1/models/test-download-model/size')
        
        assert response.status_code == 200
        data = response.json()
        assert data['model_id'] == 'test-download-model'
        assert 'size_bytes' in data
        assert 'size_kb' in data
        assert 'size_mb' in data
        assert 'size_gb' in data
        assert 'human_readable' in data
        assert data['size_bytes'] > 0


def test_get_size_nonexistent_model(mock_aws_services):
    """Test getting size for nonexistent model"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app
        client = TestClient(app)
        
        response = client.get('/api/v1/models/nonexistent/size')
        
        assert response.status_code == 404


def test_download_default_variant_is_full(mock_aws_services):
    """Test that omitting variant parameter defaults to 'full'"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app
        client = TestClient(app)
        
        response = client.get('/api/v1/models/test-download-model/download')
        
        assert response.status_code == 200
        assert response.headers['x-variant'] == 'full'


def test_download_size_headers(mock_aws_services):
    """Test that download response includes size information in headers"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app
        client = TestClient(app)
        
        response = client.get('/api/v1/models/test-download-model/download?variant=full')
        
        assert response.status_code == 200
        assert 'x-size-bytes' in response.headers
        assert 'x-size-mb' in response.headers
        assert 'x-model-id' in response.headers
        assert response.headers['x-model-id'] == 'test-download-model'


def test_download_filtered_is_smaller(mock_aws_services):
    """Test that filtered variants (weights/dataset) are smaller than full"""
    with patch.dict(os.environ, {
        'S3_BUCKET': mock_aws_services['bucket_name'],
        'DYNAMODB_TABLE': mock_aws_services['table_name']
    }):
        from src.api.routes import app
        client = TestClient(app)
        
        # Get full size
        full_response = client.get('/api/v1/models/test-download-model/download?variant=full')
        full_size = int(full_response.headers['x-size-bytes'])
        
        # Get weights size
        weights_response = client.get('/api/v1/models/test-download-model/download?variant=weights')
        weights_size = int(weights_response.headers['x-size-bytes'])
        
        # Get dataset size
        dataset_response = client.get('/api/v1/models/test-download-model/download?variant=dataset')
        dataset_size = int(dataset_response.headers['x-size-bytes'])
        
        # Filtered variants should be smaller than full
        # (In some cases they might be similar size due to ZIP overhead, but should not be larger)
        assert weights_size <= full_size
        assert dataset_size <= full_size
