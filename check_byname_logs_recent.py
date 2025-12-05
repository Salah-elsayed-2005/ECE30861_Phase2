import boto3
from datetime import datetime, timedelta

logs = boto3.client('logs', region_name='us-east-1')
log_group = '/aws/lambda/tmr-dev-api'

# Get logs from last 10 minutes
start_time = int((datetime.now() - timedelta(minutes=10)).timestamp() * 1000)
end_time = int(datetime.now().timestamp() * 1000)

print(f"Searching logs from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")

try:
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_time,
        endTime=end_time,
        filterPattern='byName',
        limit=50
    )
    
    print(f"\nFound {len(response['events'])} byName events:")
    for event in response['events'][-20:]:  # Last 20 events
        timestamp = datetime.fromtimestamp(event['timestamp']/1000)
        print(f"\n[{timestamp}]")
        print(event['message'])
        
except Exception as e:
    print(f"Error: {e}")
