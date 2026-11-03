"""Verify the stored password hash is correct"""
import boto3
import hashlib

table = boto3.resource('dynamodb', region_name='us-east-1').Table('tmr-dev-registry')

# Get the user
response = table.get_item(Key={'model_id': 'USER#ece30861defaultadminuser'})
user = response['Item']

salt = user['salt']
stored_hash = user['password_hash']

# Test passwords
passwords = [
    '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE artifacts;''',
    '''correcthorsebatterystaple123(!__+@**(A'"`; DROP TABLE packages;''',
]

print(f"Stored salt: {salt}")
print(f"Stored hash: {stored_hash}\n")

for i, password in enumerate(passwords):
    computed_hash = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
    match = computed_hash == stored_hash
    print(f"Password {i+1}: {password[:30]}...{password[-20:]}")
    print(f"  Computed hash: {computed_hash}")
    print(f"  Match: {match}")
    if match:
        print(f"  ✅ THIS IS THE CORRECT PASSWORD!")
    print()
