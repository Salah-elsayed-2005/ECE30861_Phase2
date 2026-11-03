import boto3
import hashlib

table = boto3.resource('dynamodb', region_name='us-east-1').Table('tmr-dev-registry')
resp = table.get_item(Key={'model_id': 'USER#ece30861defaultadminuser'})
item = resp['Item']

print(f"Salt: {item['salt']}")
print(f"Stored Hash: {item['password_hash']}")

# Verify with correct OpenAPI spec password
password = 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE artifacts;'
computed_hash = hashlib.sha256((item['salt'] + password).encode()).hexdigest()
print(f"Computed Hash (salt+password): {computed_hash}")
print(f"Match: {computed_hash == item['password_hash']}")
