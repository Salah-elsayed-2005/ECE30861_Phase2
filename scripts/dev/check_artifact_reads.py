import boto3
import json

# Check artifact naming patterns
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('tmr-dev-registry')

print("=== ALL ARTIFACTS ===")
response = table.scan()
items = response.get('Items', [])

artifacts = []
for item in items:
    if item['pk'].startswith('ARTIFACT#'):
        artifact_id = item['pk'].replace('ARTIFACT#', '')
        artifacts.append({
            'id': artifact_id,
            'name': item.get('name', 'NO_NAME'),
            'type': item.get('type', 'NO_TYPE')
        })

# Sort by name
artifacts.sort(key=lambda x: x['name'])

print(f"\nFound {len(artifacts)} artifacts:")
for i, art in enumerate(artifacts):
    print(f"{i:2d}. {art['type']:7s} | {art['name']:50s} | {art['id']}")

print("\n=== CHECKING FOR NAMING ISSUES ===")
# Look for artifacts with special characters or encoding issues
for art in artifacts:
    name = art['name']
    if '/' in name:
        print(f"SLASH in name: {art['type']:7s} | {name}")
    if name.count('/') > 2:
        print(f"MULTIPLE SLASHES: {art['type']:7s} | {name}")
