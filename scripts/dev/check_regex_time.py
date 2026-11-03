import boto3
from datetime import datetime

logs = boto3.client('logs', region_name='us-east-1')
log_group = '/aws/lambda/tmr-dev-api'

# Check logs around 8:50 PM - specifically regex test time
start_time = datetime(2025, 11, 29, 20, 50, 28)  # Regex tests should be around here
end_time = datetime(2025, 11, 29, 20, 50, 32)

start_ms = int(start_time.timestamp() * 1000)
end_ms = int(end_time.timestamp() * 1000)

print(f"Searching logs from {start_time} to {end_time}")
print("=" * 80)

# Get ALL log events for this narrow window
try:
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_ms,
        endTime=end_ms,
        limit=200
    )
    
    events = response.get('events', [])
    print(f"\nFound {len(events)} total log entries")
    print("=" * 80)
    
    # Show all logs
    for event in events[:100]:
        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
        message = event['message'].strip()
        if 'RequestId' not in message or 'START' in message:
            print(f"[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] {message}")
        
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("Searching for 'POST' requests:")
print("=" * 80)

try:
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_ms,
        endTime=end_ms,
        filterPattern='"POST"',
        limit=50
    )
    
    events = response.get('events', [])
    print(f"Found {len(events)} POST requests")
    for event in events:
        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
        message = event['message'].strip()
        print(f"[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] {message}")
        
except Exception as e:
    print(f"Error: {e}")
