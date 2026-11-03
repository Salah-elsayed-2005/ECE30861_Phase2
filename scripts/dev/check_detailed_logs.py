import boto3
from datetime import datetime
import json

logs = boto3.client('logs', region_name='us-east-1')
log_group = '/aws/lambda/tmr-dev-api'

# Get logs during byName test time
start_time = int(datetime(2025, 11, 29, 11, 15, 28).timestamp() * 1000)
end_time = int(datetime(2025, 11, 29, 11, 15, 33).timestamp() * 1000)

print(f"Checking logs from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")

try:
    # Get all events, no filter
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_time,
        endTime=end_time,
        limit=200
    )
    
    print(f"\nFound {len(response['events'])} events\n")
    
    for event in response['events']:
        msg = event['message'].strip()
        
        # Look for API Gateway access logs or Lambda logs mentioning paths
        if any(keyword in msg for keyword in ['byName', 'artifact', 'path', 'GET', 'status']):
            timestamp = datetime.fromtimestamp(event['timestamp']/1000)
            print(f"[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] {msg[:200]}")
            
except Exception as e:
    print(f"Error: {e}")
