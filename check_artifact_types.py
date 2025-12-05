import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('tmr-dev-registry')

# Scan for all artifacts
response = table.scan()
artifacts = [item for item in response['Items'] if item['PK'].startswith('ARTIFACT#')]

print(f"Total artifacts: {len(artifacts)}\n")

# Count by type
types = {}
for artifact in artifacts:
    artifact_type = artifact.get('type', 'UNKNOWN')
    types[artifact_type] = types.get(artifact_type, 0) + 1
    
print("Artifacts by type:")
for artifact_type, count in sorted(types.items()):
    print(f"  {artifact_type}: {count}")

# Show first few artifacts
print("\nFirst 5 artifacts:")
for artifact in artifacts[:5]:
    print(f"  Name: {artifact.get('name', 'NO_NAME'):40} Type: {artifact.get('type', 'NO_TYPE'):10} ID: {artifact['PK']}")
