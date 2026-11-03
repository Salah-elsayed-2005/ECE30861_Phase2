import hashlib

# From OpenAPI spec YAML example
openapi_password = """correcthorsebatterystaple123(!__+@**(A'";DROP TABLE artifacts;"""

# Current code password
code_password = """correcthorsebatterystaple123(!__+@**(A'\";DROP TABLE artifacts;"""

salt = "78524982bfda42a8bde7bbac641b1e9a"
expected_hash = "2ba2332be97019b49ab542ba542f4f4f0462c9a6e32d0c3c57b5c516c0807901"

def test_password(pwd, label):
    hash_result = hashlib.sha256((salt + pwd).encode('utf-8')).hexdigest()
    match = "✓ MATCH" if hash_result == expected_hash else "✗ NO MATCH"
    print(f"{label}:")
    print(f"  Password: {repr(pwd)}")
    print(f"  Hash: {hash_result}")
    print(f"  {match}\n")

test_password(openapi_password, "OpenAPI Spec Password")
test_password(code_password, "Current Code Password")
