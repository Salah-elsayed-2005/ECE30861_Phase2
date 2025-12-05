import boto3
from datetime import datetime, timedelta

logs = boto3.client('logs', region_name='us-east-1')
log_group = '/aws/lambda/tmr-dev-api'

# Get logs from test time (11:15 AM)
start_time = int(datetime(2025, 11, 29, 11, 15, 0).timestamp() * 1000)
end_time = int(datetime(2025, 11, 29, 11, 16, 0).timestamp() * 1000)

print(f"Searching logs from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")

try:
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_time,
        endTime=end_time,
        filterPattern='GET',
        limit=100
    )
    
    print(f"\nFound {len(response['events'])} GET events:")
    for event in response['events'][-30:]:
        timestamp = datetime.fromtimestamp(event['timestamp']/1000)
        msg = event['message'].strip()
        if 'artifact' in msg.lower() or 'GET' in msg:
            print(f"[{timestamp.strftime('%H:%M:%S')}] {msg[:150]}")
        
except Exception as e:
    print(f"Error: {e}")
