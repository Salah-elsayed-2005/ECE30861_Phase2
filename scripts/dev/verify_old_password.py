import boto3
import hashlib

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('tmr-dev-registry')

response = table.get_item(Key={'model_id': 'USER#ece30861defaultadminuser'})
user = response['Item']
stored_salt = user['salt']
stored_hash = user['password_hash']

print(f"Stored hash: {stored_hash}")
print()

# Test old 31-char password
old_pwd = 'correcthorsebatterystaple123(!)'

# Test with SALT+PASSWORD order (what Lambda uses)
test_hash = hashlib.sha256((stored_salt + old_pwd).encode('utf-8')).hexdigest()
print(f"Testing OLD password with SALT+PASSWORD order:")
print(f"  Password: {old_pwd}")
print(f"  Length: {len(old_pwd)}")
print(f"  Computed: {test_hash}")
print(f"  Stored:   {stored_hash}")
print(f"  Match: {test_hash == stored_hash}")
