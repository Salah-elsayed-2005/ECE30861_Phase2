import boto3
import hashlib
import uuid
import os
from datetime import datetime

# Configuration
TABLE_NAME = os.getenv('DYNAMODB_TABLE', 'tmr-dev-registry')
REGION = 'us-east-1'
ADMIN_USERNAME = 'ece30861defaultadminuser'
ADMIN_PASSWORD = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()

def create_admin_user():
    print(f"Connecting to DynamoDB table: {TABLE_NAME} in {REGION}...")
    try:
        dynamodb = boto3.resource('dynamodb', region_name=REGION)
        table = dynamodb.Table(TABLE_NAME)
        
        # Check if user exists
        print(f"Checking if user {ADMIN_USERNAME} exists...")
        response = table.get_item(Key={'model_id': f'USER#{ADMIN_USERNAME}'})
        
        if 'Item' in response:
            print(f"User {ADMIN_USERNAME} already exists. Updating...")
        else:
            print(f"User {ADMIN_USERNAME} does not exist. Creating...")
            
        # Create/Update user
        salt = uuid.uuid4().hex
        pw_hash = _hash_password(ADMIN_PASSWORD, salt)
        
        item = {
            'model_id': f'USER#{ADMIN_USERNAME}',
            'password_hash': pw_hash,
            'salt': salt,
            'is_admin': True,
            'created_at': datetime.utcnow().isoformat()
        }
        
        table.put_item(Item=item)
        print(f"Successfully seeded admin user: {ADMIN_USERNAME}")
        
    except Exception as e:
        print(f"Error seeding admin user: {e}")
        # Print more details if it's a permission issue
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_admin_user()
