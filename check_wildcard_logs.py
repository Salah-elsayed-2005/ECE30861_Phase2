import boto3
from datetime import datetime, timedelta

logs = boto3.client('logs', region_name='us-east-1')
log_group = '/aws/lambda/tmr-dev-api'

# Get logs from 12:06:56 to 12:07:05 (when wildcard tests run)
start_time = int(datetime(2025, 11, 29, 12, 6, 56).timestamp() * 1000)
end_time = int(datetime(2025, 11, 29, 12, 7, 5).timestamp() * 1000)

print("=== SEARCHING FOR WILDCARD QUERY LOGS ===")
print(f"Timeframe: 12:06:56 PM - 12:07:05 PM\n")

try:
    # Search for wildcard query logs
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_time,
        endTime=end_time,
        filterPattern='"[QUERY] Wildcard"'
    )
    
    if response['events']:
        print(f"Found {len(response['events'])} wildcard query log entries:\n")
        for event in response['events']:
            timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
            print(f"[{timestamp.strftime('%H:%M:%S')}] {event['message']}")
    else:
        print("NO WILDCARD QUERY LOGS FOUND!")
        print("\nThis means either:")
        print("1. The deployment didn't complete successfully")
        print("2. The wildcard query endpoint wasn't called")
        print("3. The logging code has a bug\n")
        
        # Try broader search
        print("=== SEARCHING FOR ANY [QUERY] LOGS ===")
        response2 = logs.filter_log_events(
            logGroupName=log_group,
            startTime=start_time,
            endTime=end_time,
            filterPattern='"[QUERY]"'
        )
        
        if response2['events']:
            print(f"\nFound {len(response2['events'])} [QUERY] log entries:")
            for event in response2['events'][:20]:  # Show first 20
                timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
                print(f"[{timestamp.strftime('%H:%M:%S')}] {event['message']}")
        else:
            print("NO [QUERY] LOGS FOUND AT ALL!")
            print("\nThe deployment likely failed or didn't include the new code.")
            
except Exception as e:
    print(f'Error: {e}')
