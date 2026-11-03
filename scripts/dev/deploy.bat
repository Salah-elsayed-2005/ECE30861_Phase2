@echo off
echo === DEPLOYMENT ===
echo.
python recreate_admin.py
if errorlevel 1 goto :error

echo Cleaning Python cache...
cd lambda_check
if exist __pycache__ rd /s /q __pycache__
for /r %%i in (*.pyc) do del "%%i"
cd ..

echo Creating deployment package...
powershell -Command "Compress-Archive -Path lambda_check\* -DestinationPath lambda.zip -Force"
if errorlevel 1 goto :error

echo Deploying to Lambda...
aws lambda update-function-code --function-name tmr-dev-api --zip-file fileb://lambda.zip --region us-east-1
if errorlevel 1 goto :error

echo Waiting for deployment...
timeout /t 3 /nobreak >nul
aws lambda wait function-updated --function-name tmr-dev-api --region us-east-1
if errorlevel 1 goto :error

git add -A
git commit -m "Fix: Return all results for wildcard queries (no pagination)"
git push

echo.
echo === SUCCESS ===
goto :end

:error
echo.
echo === ERROR OCCURRED ===
exit /b 1

:end
