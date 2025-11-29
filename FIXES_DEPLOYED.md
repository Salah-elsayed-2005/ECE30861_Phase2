# Fixes Deployed - Session 1

## Date: November 29, 2025

### Critical Fixes Applied:

#### 1. **Password Correction** ✅
- **Issue**: Password was incorrect (`correcthorsebatterystaple123(!__+@**(A'"\`; DROP TABLE packages;`)
- **Fix**: Updated to match OpenAPI spec exactly: `correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE artifacts;`
- **Files Modified**:
  - `lambda_check/api/autograder_routes.py`
  - `recreate_admin.py`
  - `test_artifact_read_fix.py`

#### 2. **Artifact Read Endpoints** ✅
- **Issue**: Artifacts with forward slashes in names (e.g., `facebook/opt-350m`) were not being retrieved correctly
- **Fix**: Changed route from `/artifact/byName/{name}` to `/artifact/byName/{name:path}` to allow path-like parameters
- **Additional Improvements**:
  - Added `.get()` method calls to safely access dictionary keys
  - Consistent use of `JSONResponse` for all responses
  - Added type validation in GET and DELETE endpoints

#### 3. **Delete Endpoint Type Validation** ✅  
- **Issue**: Delete endpoint wasn't validating artifact type before deletion
- **Fix**: Added type check to ensure artifact type matches before deleting

### Test Results:
- ✅ Authentication working
- ✅ GET by name with URL-encoded names (including `/` chars)
- ✅ GET by ID working
- ✅ All artifact types (model, dataset, code) tested successfully

### Expected Improvements:
- **Artifact Read Test Group**: 23/61 → Expected ~55-60/61
- **Delete Test Group**: 7/10 → Expected ~10/10
- **Overall Score**: 228/318 → Expected ~260-270/318

### Deployment Status:
- Lambda function: `tmr-dev-api` (us-east-1)
- Deployed: November 29, 2025, 14:52 UTC
- DynamoDB: Admin user recreated with correct password
- Status: ✅ Ready for autograder submission

### Next Steps:
1. Submit to autograder to verify improvements
2. If Artifact Read tests improve, move to next failing endpoint group
3. Focus on either Regex Tests (0/7) or Rating improvements (124/156)
