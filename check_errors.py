import boto3
from datetime import datetime, timedelta

logs_client = boto3.client('logs', region_name='us-east-1')
log_group_name = '/aws/lambda/tmr-dev-api'

# Last 2 minutes
start_time = int((datetime.utcnow() - timedelta(minutes=2)).timestamp() * 1000)
end_time = int(datetime.utcnow().timestamp() * 1000)

response = logs_client.filter_log_events(
    logGroupName=log_group_name,
    startTime=start_time,
    endTime=end_time,
)

for event in response.get('events', []):
    message = event['message'].strip()
    if any(word in message for word in ['Error', 'Traceback', 'Exception', '500', 'REPORT']):
        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
        print(f"{timestamp.strftime('%H:%M:%S')} | {message}")
