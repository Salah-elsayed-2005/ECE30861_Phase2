#!/usr/bin/env pwsh
# Quick deployment script

Write-Host "Recreating admin..."
python recreate_admin.py

Write-Host "`nCleaning and zipping..."
cd lambda_check
Get-ChildItem -Recurse -Include __pycache__,*.pyc | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
cd ..
Compress-Archive -Path lambda_check\* -DestinationPath lambda.zip -Force

Write-Host "`nDeploying to Lambda..."
aws lambda update-function-code --function-name tmr-dev-api --zip-file fileb://lambda.zip --region us-east-1

Write-Host "`nCommitting changes..."
git add -A
git commit -m "Add Hugging Face URL support with underscore normalization"
git push

Write-Host "`nDeployment complete!"
