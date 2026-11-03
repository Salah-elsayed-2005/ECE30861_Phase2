import boto3
import hashlib

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('tmr-dev-registry')

# Get the admin user
response = table.get_item(Key={'model_id': 'USER#ece30861defaultadminuser'})
user = response['Item']
stored_salt = user['salt']
stored_hash = user['password_hash']

print(f"Stored hash: {stored_hash}")
print()

# Extract password from YAML
with open('ece461_fall_2025_openapi_spec (1).yaml', 'r', encoding='utf-8') as f:
    content = f.read()
    
# Find the line with the password
for line in content.split('\n'):
    if 'password:' in line and 'correcthorsebatterystaple' in line:
        # Extract password after "password: "
        pwd = line.split('password: ')[1].strip()
        print(f"Password from OpenAPI spec:")
        print(f"  Length: {len(pwd)}")
        print(f"  Value: {repr(pwd)}")
        print(f"  Bytes: {pwd.encode()}")
        print()
        
        # Compute hash
        test_hash = hashlib.sha256((pwd + stored_salt).encode()).hexdigest()
        print(f"  Computed hash: {test_hash}")
        print(f"  Stored hash:   {stored_hash}")
        print(f"  Match: {test_hash == stored_hash}")
        break
