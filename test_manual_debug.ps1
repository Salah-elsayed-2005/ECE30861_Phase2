# Manual test - upload artifact and query for it
$apiUrl = "https://bje51swfu7.execute-api.us-east-1.amazonaws.com"

# Get token
$authBody = @{
    user = @{
        name = "ece30861defaultadminuser"
        is_admin = $true
    }
    secret = @{
        password = "correcthorsebatterystaple123(!__+@**(A'`"`;DROP TABLE packages;"
    }
} | ConvertTo-Json

Write-Host "=== Getting auth token ===" -ForegroundColor Cyan
$authResponse = Invoke-RestMethod -Uri "$apiUrl/authenticate" -Method Put -Body $authBody -ContentType "application/json"
$token = $authResponse
Write-Host "Token: $token" -ForegroundColor Green

# Upload a test dataset
$uploadBody = @{
    url = "https://huggingface.co/datasets/bookcorpus"
} | ConvertTo-Json

Write-Host "`n=== Uploading test dataset ===" -ForegroundColor Cyan
try {
    $uploadResponse = Invoke-RestMethod -Uri "$apiUrl/artifact/dataset" -Method Post -Body $uploadBody -ContentType "application/json" -Headers @{"X-Authorization" = $token}
    Write-Host "Upload success!" -ForegroundColor Green
    Write-Host ($uploadResponse | ConvertTo-Json -Depth 5)
    $artifactId = $uploadResponse.metadata.id
    $artifactName = $uploadResponse.metadata.name
} catch {
    Write-Host "Upload failed: $($_.Exception.Message)" -ForegroundColor Red
    exit
}

# Query for datasets using wildcard + type filter
$queryBody = @(
    @{
        name = "*"
        types = @("dataset")
    }
) | ConvertTo-Json

Write-Host "`n=== Querying for datasets ===" -ForegroundColor Cyan
try {
    $queryResponse = Invoke-RestMethod -Uri "$apiUrl/artifacts" -Method Post -Body $queryBody -ContentType "application/json" -Headers @{"X-Authorization" = $token}
    Write-Host "Query success! Found $($queryResponse.Count) results" -ForegroundColor Green
    Write-Host ($queryResponse | ConvertTo-Json -Depth 5)
    
    # Check if our artifact is in the results
    $found = $queryResponse | Where-Object { $_.id -eq $artifactId }
    if ($found) {
        Write-Host "`n✓ Our artifact was found in query results!" -ForegroundColor Green
    } else {
        Write-Host "`n✗ Our artifact was NOT found in query results!" -ForegroundColor Red
    }
} catch {
    Write-Host "Query failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Get artifact by name
Write-Host "`n=== Getting artifact by name: $artifactName ===" -ForegroundColor Cyan
try {
    $byNameResponse = Invoke-RestMethod -Uri "$apiUrl/artifact/byName/$artifactName" -Method Get -Headers @{"X-Authorization" = $token}
    Write-Host "Get by name success! Found $($byNameResponse.Count) results" -ForegroundColor Green
    Write-Host ($byNameResponse | ConvertTo-Json -Depth 5)
} catch {
    Write-Host "Get by name failed: $($_.Exception.Message)" -ForegroundColor Red
    $statusCode = $_.Exception.Response.StatusCode.value__
    Write-Host "Status: $statusCode" -ForegroundColor Red
}
