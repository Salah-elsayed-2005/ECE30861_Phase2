import boto3
from datetime import datetime

logs = boto3.client('logs', region_name='us-east-1')
log_group = '/aws/lambda/tmr-dev-api'

# Get logs during All Artifacts Query Test (11:53:21-23)
start_time = int(datetime(2025, 11, 29, 11, 53, 21).timestamp() * 1000)
end_time = int(datetime(2025, 11, 29, 11, 53, 23).timestamp() * 1000)

print(f"Checking logs from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}\n")

try:
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_time,
        endTime=end_time,
        filterPattern='QUERY',
        limit=200
    )
    
    print(f"Found {len(response['events'])} [QUERY] events:\n")
    
    for event in response['events']:
        timestamp = datetime.fromtimestamp(event['timestamp']/1000)
        msg = event['message'].strip()
        print(f"[{timestamp.strftime('%H:%M:%S')}] {msg}")
        
except Exception as e:
    print(f"Error: {e}")
