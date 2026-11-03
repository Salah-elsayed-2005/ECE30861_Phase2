pwd_in_code = "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;"
pwd_from_yaml = 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE artifacts;'

print(f"Password in code:      {repr(pwd_in_code)}")
print(f"Password from YAML:    {repr(pwd_from_yaml)}")
print(f"Are they equal? {pwd_in_code == pwd_from_yaml}")
print(f"\nActual password: {pwd_in_code}")
print(f"Character breakdown around the tricky part:")
for i, c in enumerate(pwd_in_code[35:45]):
    print(f"  [{35+i}] {repr(c)} (ord={ord(c)})")
