"""Verify password hash matches what's in DynamoDB"""
import hashlib

password = '''correcthorsebatterystaple123(!__+@**(A'"`; DROP TABLE packages;'''
salt = "7b3e766898ee477a9f686f3ceb624540"
expected_hash = "1fb26a4caff4688dac30d94949e4ddffd19855aff2f56aeadedb14a3859fc7d2"

# Compute hash
computed_hash = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()

print(f"Password: {password[:30]}...{password[-20:]}")
print(f"Salt: {salt}")
print(f"Expected hash: {expected_hash}")
print(f"Computed hash: {computed_hash}")
print(f"Match: {computed_hash == expected_hash}")
