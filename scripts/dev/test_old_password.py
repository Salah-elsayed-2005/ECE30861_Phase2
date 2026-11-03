import boto3
import hashlib
import uuid

# Test with the OLD password that was working
old_password = 'correcthorsebatterystaple123(!)'

table = boto3.resource('dynamodb', region_name='us-east-1').Table('tmr-dev-registry')
username = 'ece30861defaultadminuser'

# Delete existing
table.delete_item(Key={'model_id': f'USER#{username}'})

# Create with OLD password and SALT+PASSWORD order (like original)
salt = uuid.uuid4().hex
pw_hash = hashlib.sha256((salt + old_password).encode('utf-8')).hexdigest()

from datetime import datetime
table.put_item(Item={
    'model_id': f'USER#{username}',
    'password_hash': pw_hash,
    'salt': salt,
    'is_admin': True,
    'created_at': datetime.utcnow().isoformat()
})

print(f"Created admin with OLD password: {old_password}")
print(f"Length: {len(old_password)}")
print(f"Salt: {salt[:16]}...")
print(f"Hash: {pw_hash[:16]}...")
