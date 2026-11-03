import boto3
from datetime import datetime

logs_client = boto3.client('logs', region_name='us-east-1')

# Check logs around the wildcard query tests (11:56:49 and 11:56:51)
start_time = int(datetime(2025, 11, 29, 11, 56, 48).timestamp() * 1000)
end_time = int(datetime(2025, 11, 29, 11, 56, 52).timestamp() * 1000)

print("Checking wildcard query logs:")
print("="*80)

response = logs_client.filter_log_events(
    logGroupName='/aws/lambda/tmr-dev-api',
    startTime=start_time,
    endTime=end_time,
    filterPattern='[QUERY]'
)

for event in response.get('events', []):
    message = event['message'].strip()
    if '[QUERY]' in message:
        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000).strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")
