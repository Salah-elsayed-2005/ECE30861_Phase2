$API_BASE = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# Test health
Write-Host "Testing /health..." -ForegroundColor Cyan
$health = Invoke-RestMethod -Uri "$API_BASE/health" -Method Get
Write-Host "✅ Health check passed" -ForegroundColor Green

# Test authentication
Write-Host "`nTesting /authenticate..." -ForegroundColor Cyan
$password = 'correcthorsebatterystaple123(!__+@**(A''"``;DROP TABLE packages;'
$body = @{
    User = @{
        name = 'ece30861defaultadminuser'
        isAdmin = $true
    }
    Secret = @{
        password = $password
    }
} | ConvertTo-Json -Depth 10

try {
    $response = Invoke-WebRequest -Uri "$API_BASE/authenticate" -Method PUT -Body $body -ContentType "application/json"
    Write-Host "✅ AUTHENTICATION SUCCESSFUL!" -ForegroundColor Green
    Write-Host "Status: $($response.StatusCode)" -ForegroundColor Yellow
    Write-Host "Token: $($response.Content.Substring(0, [Math]::Min(50, $response.Content.Length)))..." -ForegroundColor Yellow
} catch {
    Write-Host "❌ AUTHENTICATION FAILED!" -ForegroundColor Red
    Write-Host "Status: $($_.Exception.Response.StatusCode.value__)" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}
