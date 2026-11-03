import boto3

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('tmr-dev-registry')

# Delete the admin user
admin_username = 'ece30861defaultadminuser'
key = {'model_id': f'USER#{admin_username}'}

try:
    response = table.delete_item(Key=key)
    print(f"✅ Deleted user: {admin_username}")
    print(f"Response: {response}")
except Exception as e:
    print(f"❌ Error deleting user: {e}")
