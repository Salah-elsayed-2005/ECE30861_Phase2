import boto3
import hashlib

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('tmr-dev-registry')

# Get the admin user from DynamoDB
response = table.get_item(Key={'model_id': 'USER#ece30861defaultadminuser'})

if 'Item' in response:
    user = response['Item']
    stored_salt = user['salt']
    stored_hash = user['password_hash']
    
    print("Admin user found in DynamoDB:")
    print(f"  Salt: {stored_salt[:20]}...")
    print(f"  Hash: {stored_hash}")
    print()
    
    # Test different password possibilities
    passwords = [
        '''correcthorsebatterystaple123(!__+@**(A'"`; DROP TABLE packages;''',  # Old wrong one
        '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE artifacts;''',  # Current one (no space)
        '''correcthorsebatterystaple123(!__+@**(A'"`; DROP TABLE artifacts;''',  # With space
    ]
    
    for i, pwd in enumerate(passwords, 1):
        test_hash = hashlib.sha256((pwd + stored_salt).encode()).hexdigest()
        match = "✓ MATCH" if test_hash == stored_hash else "✗ no match"
        print(f"Password {i} (len={len(pwd)}): {match}")
        print(f"  First 35: {pwd[:35]}")
        print(f"  Last 25: {pwd[-25:]}")
        print(f"  Computed hash: {test_hash}")
        print()
else:
    print("Admin user NOT found in DynamoDB!")
