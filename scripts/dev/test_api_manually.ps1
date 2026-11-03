# Manual API test script
# This script tests the API endpoints step by step

$apiUrl = "https://5rxpypozba.execute-api.us-east-1.amazonaws.com/dev"

# 1. Authenticate
Write-Host "=== Step 1: Authenticate ===" -ForegroundColor Cyan
$authBody = @{
    User = @{
        name = "ece30861defaultadminuser"
        isAdmin = $true
    }
    Secret = @{
        password = 'correcthorsebatterystaple123(!__+@**(A''"`;DROP TABLE packages;'
    }
} | ConvertTo-Json

$authResponse = Invoke-RestMethod -Uri "$apiUrl/authenticate" -Method Put -Body $authBody -ContentType "application/json"
$token = $authResponse
Write-Host "Token received: $($token.Substring(0, [Math]::Min(50, $token.Length)))..." -ForegroundColor Green

# 2. Reset system
Write-Host "`n=== Step 2: Reset System ===" -ForegroundColor Cyan
try {
    $resetResponse = Invoke-RestMethod -Uri "$apiUrl/reset" -Method Delete -Headers @{"X-Authorization" = $token} -ContentType "application/json"
    Write-Host "Reset successful" -ForegroundColor Green
} catch {
    Write-Host "Reset failed: $_" -ForegroundColor Red
}

# 3. Upload a test model
Write-Host "`n=== Step 3: Upload Model ===" -ForegroundColor Cyan
$modelBody = @{
    url = "https://huggingface.co/google-bert/bert-base-uncased"
} | ConvertTo-Json

try {
    $modelResponse = Invoke-RestMethod -Uri "$apiUrl/artifact/model" -Method Post -Body $modelBody -Headers @{"X-Authorization" = $token} -ContentType "application/json"
    Write-Host "Model uploaded successfully:" -ForegroundColor Green
    Write-Host ($modelResponse | ConvertTo-Json -Depth 5)
    $modelName = $modelResponse.metadata.name
    $modelId = $modelResponse.metadata.id
    Write-Host "`nExtracted name: $modelName" -ForegroundColor Yellow
    Write-Host "Extracted ID: $modelId" -ForegroundColor Yellow
} catch {
    Write-Host "Model upload failed: $_" -ForegroundColor Red
    exit
}

# 4. Query for the model by name using POST /artifacts
Write-Host "`n=== Step 4: Query for Model by Name ===" -ForegroundColor Cyan
$queryBody = @(
    @{
        name = $modelName
        types = @("model")
    }
) | ConvertTo-Json

try {
    $queryResponse = Invoke-RestMethod -Uri "$apiUrl/artifacts" -Method Post -Body $queryBody -Headers @{"X-Authorization" = $token} -ContentType "application/json"
    Write-Host "Query results:" -ForegroundColor Green
    Write-Host ($queryResponse | ConvertTo-Json -Depth 5)
    
    # Check if our model is in the results
    $found = $queryResponse | Where-Object { $_.id -eq $modelId }
    if ($found) {
        Write-Host "`nModel FOUND in query results!" -ForegroundColor Green
    } else {
        Write-Host "`nModel NOT FOUND in query results!" -ForegroundColor Red
        Write-Host "Expected ID: $modelId" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Query failed: $_" -ForegroundColor Red
}

# 5. Get artifact by name using GET /artifact/byName/{name}
Write-Host "`n=== Step 5: Get Artifact by Name ===" -ForegroundColor Cyan
try {
    $byNameResponse = Invoke-RestMethod -Uri "$apiUrl/artifact/byName/$modelName" -Method Get -Headers @{"X-Authorization" = $token}
    Write-Host "Get by name results:" -ForegroundColor Green
    Write-Host ($byNameResponse | ConvertTo-Json -Depth 5)
} catch {
    Write-Host "Get by name failed: $_" -ForegroundColor Red
    Write-Host "Error details: $($_.Exception.Message)"
}

# 6. Upload a dataset
Write-Host "`n=== Step 6: Upload Dataset ===" -ForegroundColor Cyan
$datasetBody = @{
    url = "https://huggingface.co/datasets/bookcorpus"
} | ConvertTo-Json

try {
    $datasetResponse = Invoke-RestMethod -Uri "$apiUrl/artifact/dataset" -Method Post -Body $datasetBody -Headers @{"X-Authorization" = $token} -ContentType "application/json"
    Write-Host "Dataset uploaded successfully:" -ForegroundColor Green
    Write-Host ($datasetResponse | ConvertTo-Json -Depth 5)
    $datasetName = $datasetResponse.metadata.name
    $datasetId = $datasetResponse.metadata.id
    Write-Host "`nExtracted name: $datasetName" -ForegroundColor Yellow
    Write-Host "Extracted ID: $datasetId" -ForegroundColor Yellow
} catch {
    Write-Host "Dataset upload failed: $_" -ForegroundColor Red
}

# 7. Query for the dataset by name
Write-Host "`n=== Step 7: Query for Dataset by Name ===" -ForegroundColor Cyan
$datasetQueryBody = @(
    @{
        name = $datasetName
        types = @("dataset")
    }
) | ConvertTo-Json

try {
    $datasetQueryResponse = Invoke-RestMethod -Uri "$apiUrl/artifacts" -Method Post -Body $datasetQueryBody -Headers @{"X-Authorization" = $token} -ContentType "application/json"
    Write-Host "Query results:" -ForegroundColor Green
    Write-Host ($datasetQueryResponse | ConvertTo-Json -Depth 5)
    
    # Check if our dataset is in the results
    $found = $datasetQueryResponse | Where-Object { $_.id -eq $datasetId }
    if ($found) {
        Write-Host "`nDataset FOUND in query results!" -ForegroundColor Green
    } else {
        Write-Host "`nDataset NOT FOUND in query results!" -ForegroundColor Red
        Write-Host "Expected ID: $datasetId" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Query failed: $_" -ForegroundColor Red
}
