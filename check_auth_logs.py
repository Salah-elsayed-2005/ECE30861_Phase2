import boto3
from datetime import datetime, timedelta

logs_client = boto3.client('logs', region_name='us-east-1')

log_group_name = '/aws/lambda/tmr-dev-api'

# Look at last 5 minutes to catch the recent test
start_time = int((datetime.utcnow() - timedelta(minutes=5)).timestamp() * 1000)
end_time = int(datetime.utcnow().timestamp() * 1000)

print("Fetching Lambda logs from last 5 minutes...")
print(f"Looking for [INIT] and [AUTH] messages...")

response = logs_client.filter_log_events(
    logGroupName=log_group_name,
    startTime=start_time,
    endTime=end_time,
)

events = response.get('events', [])
print(f"\nFound {len(events)} log events\n")

for event in events:
    message = event['message'].strip()
    timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
    
    # Only show relevant messages
    if any(keyword in message for keyword in ['[INIT]', '[AUTH]', 'Error', 'Exception', 'admin']):
        print(f"{timestamp.strftime('%H:%M:%S')} | {message}")
