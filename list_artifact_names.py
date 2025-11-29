import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('tmr-dev-registry')

response = table.scan(FilterExpression='begins_with(model_id, :prefix)',
                      ExpressionAttributeValues={':prefix': 'ARTIFACT#'})
artifacts = response.get('Items', [])

print(f"Total artifacts: {len(artifacts)}\n")
print("Artifact names:")
for item in sorted(artifacts, key=lambda x: x.get('created_at', '')):
    name = item.get('name', 'N/A')
    artifact_type = item.get('type', 'N/A')
    artifact_id = item['model_id'].replace('ARTIFACT#', '')
    print(f"  - {name} ({artifact_type}, ID: {artifact_id})")
