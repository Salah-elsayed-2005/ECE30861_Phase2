import hashlib

salt = '8b63848426ba4df3992a41d969f22e0a'
password = 'correcthorsebatterystaple123(!)'
stored_hash = '4f32aa0bd2c5a158b6e969254f0c086578297a21da79ccaee716441377871059'

computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()

print(f'Computed hash: {computed_hash}')
print(f'Stored hash:   {stored_hash}')
print(f'Match: {computed_hash == stored_hash}')
