# Deploy and check wildcard query behavior
Write-Host "=== DEPLOYING WITH ENHANCED LOGGING ===" -ForegroundColor Cyan

# Recreate admin
python recreate_admin.py

# Package and deploy
Start-Sleep -Seconds 1
cd lambda_check
Remove-Item -Recurse -Force package -ErrorAction SilentlyContinue
pip install -r ../requirements-lambda.txt -t package --upgrade
Copy-Item -Recurse api package/
Compress-Archive -Path package/* -DestinationPath ../lambda_package.zip -Force
cd ..

aws lambda update-function-code --function-name tmr-dev-api --zip-file fileb://lambda_package.zip --region us-east-1
Start-Sleep -Seconds 5
aws lambda wait function-updated --function-name tmr-dev-api --region us-east-1

Write-Host "`n=== DEPLOYMENT COMPLETE ===" -ForegroundColor Green
Write-Host "Lambda last modified:" -ForegroundColor Yellow
aws lambda get-function --function-name tmr-dev-api --region us-east-1 --query 'Configuration.LastModified' --output text

Write-Host "`n=== WAITING FOR AUTOGRADER ===" -ForegroundColor Cyan
Write-Host "Please run the autograder now, then press Enter when tests complete..." -ForegroundColor Yellow
Read-Host

Write-Host "`n=== CHECKING CLOUDWATCH LOGS ===" -ForegroundColor Cyan
python -c "
import boto3
from datetime import datetime, timedelta

logs = boto3.client('logs', region_name='us-east-1')
log_group = '/aws/lambda/tmr-dev-api'

# Get logs from last 10 minutes
start_time = int((datetime.utcnow() - timedelta(minutes=10)).timestamp() * 1000)

try:
    response = logs.filter_log_events(
        logGroupName=log_group,
        startTime=start_time,
        filterPattern='[QUERY] Wildcard query'
    )
    
    print('\\n=== WILDCARD QUERY LOGS ===')
    for event in response['events']:
        print(event['message'])
        
except Exception as e:
    print(f'Error: {e}')
"
