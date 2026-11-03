import boto3
from datetime import datetime, timedelta

logs_client = boto3.client('logs', region_name='us-east-1')

log_group_name = '/aws/lambda/tmr-dev-api'

# Check the entire day to catch autograder runs
start_time = int((datetime.utcnow() - timedelta(hours=12)).timestamp() * 1000)
end_time = int(datetime.utcnow().timestamp() * 1000)

print("Fetching ALL Lambda logs from last 12 hours...")
print(f"Time range: {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
print("="*80)

try:
    response = logs_client.filter_log_events(
        logGroupName=log_group_name,
        startTime=start_time,
        endTime=end_time,
    )
    
    events = response.get('events', [])
    print(f"\nFound {len(events)} total log events")
    
    if len(events) == 0:
        print("\nNo logs found. Checking log streams...")
        streams = logs_client.describe_log_streams(
            logGroupName=log_group_name,
            orderBy='LastEventTime',
            descending=True,
            limit=5
        )
        print(f"\nRecent log streams:")
        for stream in streams['logStreams']:
            print(f"  {stream['logStreamName']}")
            print(f"    Last event: {datetime.fromtimestamp(stream['lastEventTime']/1000)}")
    else:
        print("\nAll log messages:")
        print("="*80)
        for event in events:
            timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
            message = event['message'].strip()
            print(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {message}")
            
except Exception as e:
    print(f"Error: {e}")
