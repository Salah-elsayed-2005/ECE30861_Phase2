import boto3
import hashlib
import uuid
from datetime import datetime

table = boto3.resource('dynamodb', region_name='us-east-1').Table('tmr-dev-registry')
username = 'ece30861defaultadminuser'

# 63-character password from OpenAPI spec
password = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE artifacts;'''

print(f"Creating admin with OpenAPI spec password")
print(f"  Password length: {len(password)}")
print(f"  First 35 chars: {password[:35]}")
print(f"  Last 25 chars: {password[-25:]}")

# Delete existing
table.delete_item(Key={'model_id': f'USER#{username}'})

# Create with SALT+PASSWORD order (matching Lambda _hash_password)
salt = uuid.uuid4().hex
pw_hash = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()

table.put_item(Item={
    'model_id': f'USER#{username}',
    'password_hash': pw_hash,
    'salt': salt,
    'is_admin': True,
    'created_at': datetime.utcnow().isoformat()
})

print(f"\n✓ Created admin user")
print(f"  Salt: {salt[:16]}...")
print(f"  Hash: {pw_hash[:16]}...")
