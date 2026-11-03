import requests
import json

url = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com/authenticate"

# Test with BOTH possible interpretations
passwords = [
    'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE artifacts;',  # Current (backtick before DROP)
    'correcthorsebatterystaple123(!__+@**(A\'";DROP TABLE artifacts;',   # No backtick (just regular quotes)
]

for i, pwd in enumerate(passwords):
    body = {
        "User": {
            "name": "ece30861defaultadminuser",
            "isAdmin": True
        },
        "Secret": {
            "password": pwd
        }
    }
    
    print(f"\n=== Test {i+1} ===")
    print(f"Password: {repr(pwd)}")
    
    response = requests.put(url, json=body, headers={"Content-Type": "application/json"})
    print(f"Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"Response: {response.text}")
