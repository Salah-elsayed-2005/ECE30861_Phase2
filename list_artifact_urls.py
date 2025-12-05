import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('tmr-dev-registry')

# Scan for all artifacts
response = table.scan()
artifacts = [item for item in response['Items'] if item['PK'].startswith('ARTIFACT#')]

# Sort by name for easier analysis
artifacts.sort(key=lambda x: x.get('name', ''))

print(f"Found {len(artifacts)} artifacts\n")
for artifact in artifacts:
    name = artifact.get('name', 'NO_NAME')
    url = artifact.get('url', 'NO_URL')
    print(f"{name:50} | {url}")
