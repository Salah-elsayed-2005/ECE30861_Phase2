import requests
import json

# Get auth token
auth_response = requests.put(
    'https://2047fz40z1.execute-api.us-east-1.amazonaws.com/authenticate',
    json={
        'User': {
            'name': 'ece30861defaultadminuser',
            'isAdmin': True
        },
        'Secret': {
            'password': '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''
        }
    }
)

if auth_response.status_code != 200:
    print(f"Auth failed: {auth_response.status_code}")
    print(auth_response.text)
    exit(1)

token = auth_response.text.strip('"')
headers = {'X-Authorization': f'bearer {token}'}

# Get all artifacts
response = requests.get(
    'https://2047fz40z1.execute-api.us-east-1.amazonaws.com/artifacts',
    headers=headers
)

print(f"Response status: {response.status_code}")
artifacts = response.json()

if isinstance(artifacts, list):
    print(f"Total artifacts: {len(artifacts)}\n")
    print("First 10 artifact names:")
    for i, artifact in enumerate(artifacts[:10]):
        print(f"{i}: {artifact.get('name')} (type: {artifact.get('type')}, id: {artifact.get('id')})")
else:
    print(f"Artifacts response: {artifacts}")
