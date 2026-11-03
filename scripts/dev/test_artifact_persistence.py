import requests
import boto3
import time

BASE_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# 1. Auth
print("1. Authenticating...")
auth_resp = requests.put(f"{BASE_URL}/authenticate", json={
    "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
    "Secret": {"password": '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''}
})
print(f"Auth: {auth_resp.status_code}")
token = auth_resp.text.strip('"')

# 2. Create artifact
print("\n2. Creating artifact...")
create_resp = requests.post(
    f"{BASE_URL}/artifact/model",
    json={"url": "https://github.com/test/test-model"},
    headers={"X-Authorization": token}
)
print(f"Create: {create_resp.status_code}")
if create_resp.status_code == 201:
    artifact = create_resp.json()
    print(f"Created: {artifact['metadata']}")
    artifact_id = artifact['metadata']['id']
    artifact_name = artifact['metadata']['name']
    
    # 3. Wait a moment
    time.sleep(2)
    
    # 4. Check DynamoDB
    print("\n3. Checking DynamoDB...")
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('tmr-dev-registry')
    response = table.scan(FilterExpression='begins_with(model_id, :prefix)',
                          ExpressionAttributeValues={':prefix': 'ARTIFACT#'})
    artifacts = response.get('Items', [])
    print(f"Found {len(artifacts)} artifacts in DynamoDB")
    
    # 5. Try to query it
    print("\n4. Querying via API...")
    query_resp = requests.post(
        f"{BASE_URL}/artifacts",
        json=[{"name": artifact_name}],
        headers={"X-Authorization": token}
    )
    print(f"Query: {query_resp.status_code}")
    print(f"Result: {query_resp.text}")
else:
    print(f"Error: {create_resp.text}")
