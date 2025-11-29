"""Test authentication with various password combinations"""
import requests
import json

API_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

passwords_to_try = [
    '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE artifacts;''',
    '''correcthorsebatterystaple123(!__+@**(A'"`; DROP TABLE packages;''',
    "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;",
    "correcthorsebatterystaple123(!)",
]

for i, password in enumerate(passwords_to_try):
    print(f"\nTrying password {i+1}:")
    print(f"  Length: {len(password)}")
    print(f"  First 30: {password[:30]}")
    print(f"  Last 20: {password[-20:]}")
    
    try:
        response = requests.put(
            f"{API_URL}/authenticate",
            json={
                "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
                "Secret": {"password": password}
            },
            timeout=10
        )
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  ✅ SUCCESS! Token: {response.text[:50]}...")
            break
        else:
            print(f"  Error: {response.text[:100]}")
    except Exception as e:
        print(f"  Exception: {e}")
