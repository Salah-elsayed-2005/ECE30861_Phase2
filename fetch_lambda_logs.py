"""Fetch and display Lambda logs"""
import boto3
from datetime import datetime, timedelta

logs = boto3.client('logs', region_name='us-east-1')

# Get logs from last 15 minutes
start_time = int((datetime.utcnow() - timedelta(minutes=15)).timestamp() * 1000)

response = logs.filter_log_events(
    logGroupName='/aws/lambda/tmr-dev-api',
    startTime=start_time
)

print(f"Found {len(response['events'])} log events:\n")

for event in response['events'][-30:]:  # Last 30 events
    timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
    message = event['message'].strip()
    print(f"[{timestamp}] {message}")
