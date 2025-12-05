"""Check what's in DynamoDB"""
import boto3
from decimal import Decimal

table = boto3.resource('dynamodb', region_name='us-east-1').Table('tmr-dev-registry')

# Scan for all users
response = table.scan(
    FilterExpression='begins_with(model_id, :prefix)',
    ExpressionAttributeValues={':prefix': 'USER#'}
)

print(f"Found {len(response['Items'])} users:")
for item in response['Items']:
    print(f"\nUser: {item['model_id']}")
    print(f"  is_admin: {item.get('is_admin')}")
    print(f"  salt: {item.get('salt')[:16]}...")
    print(f"  password_hash: {item.get('password_hash')[:16]}...")
    print(f"  created_at: {item.get('created_at')}")
