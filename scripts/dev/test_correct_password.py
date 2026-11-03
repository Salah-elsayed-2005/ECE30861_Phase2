import requests
import json

url = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com/authenticate"

password = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''

body = {
    "User": {
        "name": "ece30861defaultadminuser",
        "isAdmin": True
    },
    "Secret": {
        "password": password
    }
}

print(f"Testing authentication with correct password from logs...")
print(f"Password: {repr(password)}")

response = requests.put(url, json=body, headers={"Content-Type": "application/json"})
print(f"\nStatus Code: {response.status_code}")
print(f"Response: {response.text[:100]}...")

if response.status_code == 200:
    print("\n✓ Authentication successful!")
else:
    print(f"\n✗ Authentication failed!")
    print(f"Full response: {response.text}")
