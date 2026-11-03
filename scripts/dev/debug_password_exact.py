"""Debug the exact password the autograder is sending"""

# From OpenAPI spec example:
openapi_password = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE artifacts;'''

# What we have in code:
our_password = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE artifacts;'''

print("OpenAPI spec password:")
print(f"  Length: {len(openapi_password)}")
print(f"  Repr: {repr(openapi_password)}")
print(f"  Bytes: {openapi_password.encode('utf-8')}")

print("\nOur password:")
print(f"  Length: {len(our_password)}")
print(f"  Repr: {repr(our_password)}")
print(f"  Bytes: {our_password.encode('utf-8')}")

print(f"\nMatch: {openapi_password == our_password}")

# Try the version from the original note
note_password = '''correcthorsebatterystaple123(!__+@**(A'"`; DROP TABLE packages;'''
print(f"\nNote version length: {len(note_password)}")
print(f"Note version repr: {repr(note_password)}")
