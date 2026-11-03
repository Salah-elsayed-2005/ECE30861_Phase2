import boto3
from datetime import datetime, timedelta

# Check CloudWatch logs for GET by name failures around 1:31 PM
logs = boto3.client('logs', region_name='us-east-1')
log_group = '/aws/lambda/tmr-dev-api'

# Time range: 1:31:44 PM - 1:31:53 PM (when artifact read tests ran)
start_time = datetime(2025, 11, 29, 13, 31, 44)
end_time = datetime(2025, 11, 29, 13, 31, 53)

start_ms = int(start_time.timestamp() * 1000)
end_ms = int(end_time.timestamp() * 1000)

print(f"Searching logs from {start_time} to {end_time}")
print("=" * 80)

# Search for GET requests to /artifact/byName
filter_pattern = '"GET /artifact/byName"'

try:
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_ms,
        endTime=end_ms,
        filterPattern=filter_pattern,
        limit=100
    )
    
    events = response.get('events', [])
    print(f"\nFound {len(events)} GET /artifact/byName requests:")
    print("=" * 80)
    
    for event in events:
        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
        message = event['message'].strip()
        print(f"[{timestamp.strftime('%H:%M:%S')}] {message}")
        
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("Now searching for 404 errors:")
print("=" * 80)

# Search for 404 responses  
filter_pattern = '"404"'

try:
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_ms,
        endTime=end_ms,
        filterPattern=filter_pattern,
        limit=100
    )
    
    events = response.get('events', [])
    print(f"\nFound {len(events)} 404 responses:")
    for event in events[:20]:  # Show first 20
        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
        message = event['message'].strip()
        print(f"[{timestamp.strftime('%H:%M:%S')}] {message}")
        
except Exception as e:
    print(f"Error: {e}")
