import requests
import json

url = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com/authenticate"

# Delete current admin and test with simple password
import boto3
table = boto3.resource('dynamodb', region_name='us-east-1').Table('tmr-dev-registry')
table.delete_item(Key={'model_id': 'USER#ece30861defaultadminuser'})
print("Deleted existing admin user\n")

# Test with simple password
body = {
    "User": {
        "name": "ece30861defaultadminuser",
        "isAdmin": True
    },
    "Secret": {
        "password": "correcthorsebatterystaple123(!)"
    }
}

print(f"Testing with SIMPLE password: correcthorsebatterystaple123(!)")
response = requests.put(url, json=body)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}\n")

# Now test with OpenAPI password (should fail since admin was created with simple password)
body2 = {
    "User": {
        "name": "ece30861defaultadminuser",
        "isAdmin": True
    },
    "Secret": {
        "password": 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE artifacts;'
    }
}

print(f"Testing with OPENAPI password after admin created with simple password:")
response2 = requests.put(url, json=body2)
print(f"Status Code: {response2.status_code}")
print(f"Response: {response2.text}")
