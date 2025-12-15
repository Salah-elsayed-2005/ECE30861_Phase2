import requests
import json

API_URL = 'https://9cxpsnmhjl.execute-api.us-east-1.amazonaws.com/prod'

# Get auth token first
auth_resp = requests.put(f'{API_URL}/authenticate', json={
    'User': {'name': 'ece30861defaultadminuser', 'isAdmin': True},
    'Secret': {'password': 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE packages;'}
})
print('Auth status:', auth_resp.status_code)
token = auth_resp.text.strip('"')
print('Token:', token[:50] if len(token) > 50 else token)

headers = {'X-Authorization': f'bearer {token}', 'Content-Type': 'application/json'}

# Test 1: Valid regex
print('\n--- Test 1: Valid regex .* ---')
r = requests.post(f'{API_URL}/artifact/byRegEx', headers=headers, json={'regex': '.*'})
print(f'Status: {r.status_code}')
if r.status_code == 200:
    data = r.json()
    print(f'Found {len(data)} artifacts')
else:
    print(f'Response: {r.text[:300]}')

# Test 2: Empty body
print('\n--- Test 2: Empty body {} ---')
r = requests.post(f'{API_URL}/artifact/byRegEx', headers=headers, json={})
print(f'Status: {r.status_code}')
print(f'Response: {r.text[:300]}')

# Test 3: Invalid regex
print('\n--- Test 3: Invalid regex [invalid( ---')
r = requests.post(f'{API_URL}/artifact/byRegEx', headers=headers, json={'regex': '[invalid('})
print(f'Status: {r.status_code}')
print(f'Response: {r.text[:300]}')

# Test 4: No match regex
print('\n--- Test 4: No match regex xyz123456 ---')
r = requests.post(f'{API_URL}/artifact/byRegEx', headers=headers, json={'regex': 'xyz123456nonexistent'})
print(f'Status: {r.status_code}')
print(f'Response: {r.text[:300]}')

# Test 5: Invalid JSON
print('\n--- Test 5: Invalid JSON ---')
r = requests.post(f'{API_URL}/artifact/byRegEx', headers=headers, data='not json')
print(f'Status: {r.status_code}')
print(f'Response: {r.text[:300]}')

# Test 6: RegEx key (capital E)
print('\n--- Test 6: RegEx key (capital E) ---')
r = requests.post(f'{API_URL}/artifact/byRegEx', headers=headers, json={'RegEx': '.*'})
print(f'Status: {r.status_code}')
if r.status_code == 200:
    print(f'Found {len(r.json())} artifacts')
else:
    print(f'Response: {r.text[:300]}')

# Test 7: Specific name match (from passing tests)
print('\n--- Test 7: Specific artifact name ---')
# First get an artifact name
r = requests.post(f'{API_URL}/artifacts', headers=headers, json=[{'name': '*'}])
if r.status_code == 200:
    artifacts = r.json()
    if 'data' in artifacts and len(artifacts['data']) > 0:
        first_name = artifacts['data'][0].get('name', '')
        print(f'Testing with name: {first_name}')
        r2 = requests.post(f'{API_URL}/artifact/byRegEx', headers=headers, json={'regex': first_name})
        print(f'Status: {r2.status_code}')
        if r2.status_code == 200:
            print(f'Found {len(r2.json())} artifacts')
        else:
            print(f'Response: {r2.text[:300]}')

# Test 8: Empty string regex
print('\n--- Test 8: Empty string regex ---')
r = requests.post(f'{API_URL}/artifact/byRegEx', headers=headers, json={'regex': ''})
print(f'Status: {r.status_code}')
print(f'Response: {r.text[:300]}')

# Test 9: Whitespace regex
print('\n--- Test 9: Whitespace regex ---')
r = requests.post(f'{API_URL}/artifact/byRegEx', headers=headers, json={'regex': '   '})
print(f'Status: {r.status_code}')
print(f'Response: {r.text[:300]}')

print('\n=== SUMMARY ===')
print('Expected for autograder:')
print('- Valid regex: 200 with results')
print('- Empty body: 400')
print('- Invalid regex: 400')
print('- No match: 404')
print('- Invalid JSON: 400')
