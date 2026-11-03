import boto3
import hashlib
import uuid

# Connect to DynamoDB
table = boto3.resource('dynamodb', region_name='us-east-1').Table('tmr-dev-registry')

username = 'ece30861defaultadminuser'
password = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''

# Delete existing user
print(f"Deleting existing user: {username}")
try:
    table.delete_item(Key={'model_id': f'USER#{username}'})
    print("✓ Deleted")
except Exception as e:
    print(f"Delete error (may not exist): {e}")

# Create new user with correct password
print(f"\nCreating user with password:")
print(f"  Length: {len(password)}")
print(f"  First 30 chars: {password[:30]}...")
print(f"  Last 20 chars: ...{password[-20:]}")

salt = uuid.uuid4().hex
pw_hash = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()  # SALT + PASSWORD order to match Lambda!

from datetime import datetime
table.put_item(Item={
    'model_id': f'USER#{username}',
    'password_hash': pw_hash,
    'salt': salt,
    'is_admin': True,
    'created_at': datetime.utcnow().isoformat()
})

print(f"\n✓ User created successfully")
print(f"  Salt: {salt[:16]}...")
print(f"  Hash: {pw_hash[:16]}...")
