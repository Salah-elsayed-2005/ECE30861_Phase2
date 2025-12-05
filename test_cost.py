import requests
import json

# Authenticate
auth_response = requests.put(
    'https://2047fz40z1.execute-api.us-east-1.amazonaws.com/authenticate',
    json={
        'User': {'name': 'ece30861defaultadminuser', 'isAdmin': True},
        'Secret': {'password': '''correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;'''}
    }
)
print(f"Auth status: {auth_response.status_code}")
token = auth_response.text.replace('bearer ', '')

# First create a test artifact
headers = {'X-Authorization': f'bearer {token}'}

# Upload a test model
upload_response = requests.post(
    'https://2047fz40z1.execute-api.us-east-1.amazonaws.com/artifact/model',
    headers=headers,
    json={
        'name': 'test-cost-model',
        'url': 'https://example.com/model.zip',
        'download_url': 'https://example.com/download'
    }
)
print(f"\nUpload status: {upload_response.status_code}")
if upload_response.status_code == 201:
    artifact_id = upload_response.json()['id']
    print(f"Artifact ID: {artifact_id}")
    
    # Test cost endpoint
    cost_response = requests.get(
        f'https://2047fz40z1.execute-api.us-east-1.amazonaws.com/artifact/model/{artifact_id}/cost?dependency=false',
        headers=headers
    )
    print(f"\nCost status: {cost_response.status_code}")
    print(f"Cost response: {json.dumps(cost_response.json(), indent=2)}")
