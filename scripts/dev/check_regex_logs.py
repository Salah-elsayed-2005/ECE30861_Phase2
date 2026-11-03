import boto3
from datetime import datetime

# Check CloudWatch logs for regex test failures around 8:32 PM
logs = boto3.client('logs', region_name='us-east-1')
log_group = '/aws/lambda/tmr-dev-api'

# Time range: 8:32:00 PM - 8:32:02 PM (when regex tests ran)
start_time = datetime(2025, 11, 29, 20, 32, 0)
end_time = datetime(2025, 11, 29, 20, 32, 2)

start_ms = int(start_time.timestamp() * 1000)
end_ms = int(end_time.timestamp() * 1000)

print(f"Searching logs from {start_time} to {end_time}")
print("=" * 80)

# Search for regex requests
filter_pattern = '"byRegEx"'

try:
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_ms,
        endTime=end_ms,
        filterPattern=filter_pattern,
        limit=100
    )
    
    events = response.get('events', [])
    print(f"\nFound {len(events)} regex-related log entries:")
    print("=" * 80)
    
    for event in events:
        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
        message = event['message'].strip()
        print(f"[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] {message}")
        
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("Searching for all POST requests during this time:")
print("=" * 80)

filter_pattern = '"POST"'

try:
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_ms,
        endTime=end_ms,
        filterPattern=filter_pattern,
        limit=100
    )
    
    events = response.get('events', [])
    print(f"\nFound {len(events)} POST requests:")
    for event in events[:30]:  # Show first 30
        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
        message = event['message'].strip()
        print(f"[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] {message}")
        
except Exception as e:
    print(f"Error: {e}")
