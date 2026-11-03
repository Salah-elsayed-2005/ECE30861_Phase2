import boto3
from datetime import datetime

logs = boto3.client('logs', region_name='us-east-1')
log_group = '/aws/lambda/tmr-dev-api'

# Check GET requests around 8:32:01-08 (artifact read tests)
start_time = datetime(2025, 11, 29, 20, 32, 1)
end_time = datetime(2025, 11, 29, 20, 32, 9)

start_ms = int(start_time.timestamp() * 1000)
end_ms = int(end_time.timestamp() * 1000)

print(f"Searching logs from {start_time} to {end_time}")
print("=" * 80)

# Get all log events
try:
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_ms,
        endTime=end_ms,
        limit=200
    )
    
    events = response.get('events', [])
    print(f"\nFound {len(events)} log entries total")
    print("=" * 80)
    
    # Look for 404 or error responses
    errors = [e for e in events if '404' in e['message'] or 'error' in e['message'].lower() or 'not found' in e['message'].lower()]
    
    print(f"\nFound {len(errors)} error/404 entries:")
    for event in errors[:40]:
        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
        message = event['message'].strip()
        print(f"[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] {message}")
        
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("Searching for GET /artifact/byName requests:")
print("=" * 80)

try:
    filter_pattern = '"/artifact/byName"'
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_ms,
        endTime=end_ms,
        filterPattern=filter_pattern,
        limit=100
    )
    
    events = response.get('events', [])
    print(f"\nFound {len(events)} GET byName requests:")
    for event in events[:20]:
        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
        message = event['message'].strip()
        print(f"[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] {message}")
        
except Exception as e:
    print(f"Error: {e}")
