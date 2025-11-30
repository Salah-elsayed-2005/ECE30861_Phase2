import boto3
from datetime import datetime

logs = boto3.client('logs', region_name='us-east-1')
log_group = '/aws/lambda/tmr-dev-api'

# Check logs around 8:50 PM test run
start_time = datetime(2025, 11, 29, 20, 50, 12)  # Test start
end_time = datetime(2025, 11, 29, 20, 50, 38)    # Test end

start_ms = int(start_time.timestamp() * 1000)
end_ms = int(end_time.timestamp() * 1000)

print(f"Searching logs from {start_time} to {end_time}")
print("=" * 80)

# Get all log events
try:
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_ms,
        endTime=end_ms,
        limit=500
    )
    
    events = response.get('events', [])
    print(f"\nFound {len(events)} log entries")
    
    # Filter for our custom logging
    regex_logs = [e for e in events if '[REGEX]' in e['message']]
    get_name_logs = [e for e in events if '[GET_BY_NAME]' in e['message']]
    get_id_logs = [e for e in events if '[GET_BY_ID]' in e['message']]
    
    print(f"\n{'='*80}")
    print(f"REGEX LOGS ({len(regex_logs)} entries):")
    print(f"{'='*80}")
    for event in regex_logs[:50]:
        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
        message = event['message'].strip()
        if 'START RequestId' not in message and 'END RequestId' not in message:
            print(f"[{timestamp.strftime('%H:%M:%S')}] {message}")
    
    print(f"\n{'='*80}")
    print(f"GET BY NAME LOGS ({len(get_name_logs)} entries):")
    print(f"{'='*80}")
    for event in get_name_logs[:50]:
        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
        message = event['message'].strip()
        if 'START RequestId' not in message and 'END RequestId' not in message:
            print(f"[{timestamp.strftime('%H:%M:%S')}] {message}")
    
    print(f"\n{'='*80}")
    print(f"GET BY ID LOGS ({len(get_id_logs)} entries - showing TYPE MISMATCH only):")
    print(f"{'='*80}")
    type_mismatch = [e for e in get_id_logs if 'TYPE MISMATCH' in e['message'] or 'NOT FOUND' in e['message']]
    for event in type_mismatch[:50]:
        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
        message = event['message'].strip()
        if 'START RequestId' not in message and 'END RequestId' not in message:
            print(f"[{timestamp.strftime('%H:%M:%S')}] {message}")
    
    if not type_mismatch:
        print("No type mismatches or NOT FOUND errors in GET_BY_ID")
        
except Exception as e:
    print(f"Error: {e}")
