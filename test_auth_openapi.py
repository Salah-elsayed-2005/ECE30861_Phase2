import requests
import json

url = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com/authenticate"
body = {
    "User": {
        "name": "ece30861defaultadminuser",
        "isAdmin": True
    },
    "Secret": {
        "password": 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE artifacts;'
    }
}

print(f"Testing authentication with OpenAPI spec password...")
print(f"Body: {json.dumps(body, indent=2)}")

response = requests.put(url, json=body, headers={"Content-Type": "application/json"})
print(f"\nStatus Code: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code == 200:
    print("\n✓ Authentication successful!")
else:
    print(f"\n✗ Authentication failed!")
