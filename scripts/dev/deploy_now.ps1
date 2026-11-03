# Quick deployment script
Write-Host "Creating deployment package..."
if (Test-Path lambda_deployment.zip) {
    Remove-Item lambda_deployment.zip
}

Set-Location lambda_pkg
Compress-Archive -Path * -DestinationPath ..\lambda_deployment.zip
Set-Location ..

Write-Host "Deploying to Lambda..."
aws lambda update-function-code `
    --function-name tmr-dev-api `
    --zip-file fileb://lambda_deployment.zip `
    --region us-east-1

Write-Host "Done!"
