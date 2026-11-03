#!/usr/bin/env python3
import hashlib
import sys

# From DynamoDB
salt = "78524982bfda42a8bde7bbac641b1e9a"
expected_hash = "2ba2332be97019b49ab542ba542f4f4f0462c9a6e32d0c3c57b5c516c0807901"

# Test different password interpretations
passwords_to_test = [
    ("OpenAPI YAML (raw)", "correcthorsebatterystaple123(!__+@**(A'\";DROP TABLE artifacts;"),
    ("Requirements doc", "correcthorsebatterystaple123(!__+@**(A;DROP TABLE packages)"),
    ("Python escaped quote", "correcthorsebatterystaple123(!__+@**(A'\\\"DROP TABLE artifacts;"),
    ("Without escaping", "correcthorsebatterystaple123(!__+@**(A'\"DROP TABLE artifacts;"),
]

print("Testing password hashes against DynamoDB value:")
print(f"Salt: {salt}")
print(f"Expected hash: {expected_hash}\n")

for label, pwd in passwords_to_test:
    hash_result = hashlib.sha256((salt + pwd).encode('utf-8')).hexdigest()
    match = "✓ MATCH!" if hash_result == expected_hash else "✗"
    print(f"{label}:")
    print(f"  Password repr: {repr(pwd)}")
    print(f"  Password str:  {pwd}")
    print(f"  Hash: {hash_result}")
    print(f"  {match}\n")
