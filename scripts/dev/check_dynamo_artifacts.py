import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('tmr-dev-registry')

# Scan for all artifacts
response = table.scan()
artifacts = [item for item in response['Items'] if item['model_id'].startswith('artifact-')]

print(f"Total artifacts in DynamoDB: {len(artifacts)}")
print("\nFirst 10 artifacts:")
for i, art in enumerate(artifacts[:10]):
    print(f"{i+1}. ID: {art['model_id']}, Name: {art.get('name', 'N/A')}, Type: {art.get('type', 'N/A')}")

if not artifacts:
    print("\n⚠️ NO ARTIFACTS FOUND! The autograder is uploading artifacts but they're not persisting.")
