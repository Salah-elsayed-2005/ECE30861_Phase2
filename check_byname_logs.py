import boto3
from datetime import datetime, timedelta

logs_client = boto3.client('logs', region_name='us-east-1')
log_group_name = '/aws/lambda/tmr-dev-api'

# Get logs from the most recent autograder run (around 10:52 AM)
start_time = int((datetime.utcnow() - timedelta(minutes=10)).timestamp() * 1000)
end_time = int(datetime.utcnow().timestamp() * 1000)

response = logs_client.filter_log_events(
    logGroupName=log_group_name,
    startTime=start_time,
    endTime=end_time,
)

# Find requests to /artifact/byName
byname_requests = []
for event in response.get('events', []):
    message = event['message'].strip()
    if '/artifact/byName' in message or 'byName' in message:
        byname_requests.append(event)

print(f"Found {len(byname_requests)} requests to /artifact/byName")
print("=" * 80)

for event in byname_requests[:20]:  # Show first 20
    timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
    message = event['message'].strip()
    print(f"{timestamp.strftime('%H:%M:%S')} | {message[:150]}")
