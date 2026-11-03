import boto3

table = boto3.resource('dynamodb', region_name='us-east-1').Table('tmr-dev-registry')
resp = table.get_item(Key={'model_id': 'USER#ece30861defaultadminuser'})

if 'Item' in resp:
    item = resp['Item']
    print(f"Admin user EXISTS")
    print(f"  created_at: {item.get('created_at')}")
    print(f"  salt: {item.get('salt')}")
    print(f"  password_hash: {item.get('password_hash')}")
    print(f"  is_admin: {item.get('is_admin')}")
else:
    print("Admin user DOES NOT EXIST")
