@echo off
echo Recreating admin...
python recreate_admin.py

echo.
echo Cleaning...
cd lambda_check
del /s /q __pycache__ >nul 2>&1
del /s /q *.pyc >nul 2>&1
cd ..

echo.
echo Zipping...
powershell -Command "Compress-Archive -Path lambda_check\* -DestinationPath lambda.zip -Force"

echo.
echo Deploying...
aws lambda update-function-code --function-name tmr-dev-api --zip-file fileb://lambda.zip --region us-east-1

echo.
echo Committing...
git add -A
git commit -m "REVERT: Back to GitHub-only URL parsing (restore 237 score)"
git push

echo.
echo Done!
