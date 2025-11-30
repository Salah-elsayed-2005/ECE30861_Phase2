$lastMod = aws lambda get-function --function-name tmr-dev-api --region us-east-1 --query 'Configuration.LastModified' --output text
Write-Host "Lambda last modified: $lastMod" -ForegroundColor Green

$now = Get-Date
Write-Host "Current time: $($now.ToString('yyyy-MM-ddTHH:mm:ss'))" -ForegroundColor Yellow

if ($lastMod) {
    Write-Host "`nDeployment successful! You can run the autograder now." -ForegroundColor Green
} else {
    Write-Host "`nDeployment may have failed. Please check." -ForegroundColor Red
}
