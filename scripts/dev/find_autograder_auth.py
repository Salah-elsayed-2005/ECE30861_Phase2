import boto3
from datetime import datetime, timedelta

logs_client = boto3.client('logs', region_name='us-east-1')
log_group_name = '/aws/lambda/tmr-dev-api'

# Last 30 minutes to catch the autograder run
start_time = int((datetime.utcnow() - timedelta(minutes=30)).timestamp() * 1000)
end_time = int(datetime.utcnow().timestamp() * 1000)

print("Searching for autograder authentication attempts...")
print("="*80)

response = logs_client.filter_log_events(
    logGroupName=log_group_name,
    startTime=start_time,
    endTime=end_time,
)

# Find autograder IP requests (128.46.3.1)
autograder_events = []
for event in response.get('events', []):
    message = event['message'].strip()
    if '128.46.3.1' in message or 'AUTH' in message or 'Password hash mismatch' in message or 'Authentication' in message:
        autograder_events.append(event)

print(f"Found {len(autograder_events)} autograder-related events\n")

for event in autograder_events[-30:]:  # Last 30 events
    timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
    message = event['message'].strip()
    print(f"{timestamp.strftime('%H:%M:%S')} | {message}")
