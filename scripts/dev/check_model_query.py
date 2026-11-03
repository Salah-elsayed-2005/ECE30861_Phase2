import boto3
from datetime import datetime

# Get CloudWatch logs for the ALL MODEL query test
logs_client = boto3.client('logs', region_name='us-east-1')

# Time of "All Model Query Test" (11:53:21, BEFORE All Artifacts at 11:53:23)
start_time = int(datetime(2025, 11, 29, 11, 53, 20).timestamp() * 1000)
end_time = int(datetime(2025, 11, 29, 11, 53, 22).timestamp() * 1000)

print(f"Checking logs for 'All Model Query Test' from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
print("="*80)
print()

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
