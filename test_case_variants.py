import requests
import json

url = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com/authenticate"

# Test both capitalization variants
test_cases = [
    {
        "name": "Capitalized User/Secret",
        "body": {
            "User": {
                "name": "ece30861defaultadminuser",
                "isAdmin": True
            },
            "Secret": {
                "password": 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE artifacts;'
            }
        }
    },
    {
        "name": "Lowercase user/secret",
        "body": {
            "user": {
                "name": "ece30861defaultadminuser",
                "isAdmin": True
            },
            "secret": {
                "password": 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE artifacts;'
            }
        }
    },
    {
        "name": "Lowercase user/secret with is_admin",
        "body": {
            "user": {
                "name": "ece30861defaultadminuser",
                "is_admin": True
            },
            "secret": {
                "password": 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE artifacts;'
            }
        }
    }
]

for test in test_cases:
    print(f"\n=== {test['name']} ===")
    print(f"Body: {json.dumps(test['body'], indent=2)}")
    
    response = requests.put(url, json=test['body'], headers={"Content-Type": "application/json"})
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"✓ Success: {response.text[:80]}...")
    else:
        print(f"✗ Failed: {response.text}")
