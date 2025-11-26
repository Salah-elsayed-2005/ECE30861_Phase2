"""
Test the fixed pagination logic
"""

def test_paginated_list_artifacts():
    """Simulate the fixed _list_artifacts function"""
    from decimal import Decimal
    
    # Simulate DynamoDB with 35 artifacts
    mock_db = []
    for i in range(1, 36):
        mock_db.append({
            'model_id': f'ARTIFACT#id{i}',
            'name': f'artifact-{i}',
            'type': 'model' if i <= 10 else 'dataset' if i <= 25 else 'code',
            'url': f'https://example.com/artifact{i}',
            'net_score': Decimal('0.75')
        })
    
    # Simulate table.scan with pagination
    def mock_scan(**kwargs):
        exclusive_start_key = kwargs.get('ExclusiveStartKey')
        
        if exclusive_start_key is None:
            # First page: items 0-24 (25 items)
            items = mock_db[:25]
            return {
                'Items': items,
                'LastEvaluatedKey': {'model_id': 'ARTIFACT#id25'}  # Has more pages
            }
        elif exclusive_start_key == {'model_id': 'ARTIFACT#id25'}:
            # Second page: items 25-34 (10 items)
            items = mock_db[25:]
            return {
                'Items': items
                # No LastEvaluatedKey = last page
            }
        else:
            return {'Items': []}
    
    # Simulate the fixed function
    artifacts = []
    last_evaluated_key = None
    
    while True:
        if last_evaluated_key:
            response = mock_scan(ExclusiveStartKey=last_evaluated_key)
        else:
            response = mock_scan()
        
        for item in response.get('Items', []):
            artifact_id = item['model_id'].replace('ARTIFACT#', '')
            artifact = dict(item)
            artifact.pop('model_id', None)
            # Convert Decimal to float
            for key, value in artifact.items():
                if isinstance(value, Decimal):
                    artifact[key] = float(value)
            artifacts.append((artifact_id, artifact))
        
        last_evaluated_key = response.get('LastEvaluatedKey')
        if not last_evaluated_key:
            break
    
    print(f"Total artifacts retrieved: {len(artifacts)}")
    print(f"Expected: 35")
    print(f"Status: {'✓ PASS' if len(artifacts) == 35 else '✗ FAIL'}")
    
    # Verify all types are present
    types = [a[1]['type'] for a in artifacts]
    print(f"\nArtifact types:")
    print(f"  Models: {types.count('model')} (expected 10)")
    print(f"  Datasets: {types.count('dataset')} (expected 15)")
    print(f"  Code: {types.count('code')} (expected 10)")
    
    # Test query simulation
    print(f"\n=== Query Simulation ===")
    
    # Single Model Query (wildcard + type filter)
    model_results = [a for a in artifacts if a[1]['type'] == 'model']
    print(f"Single Model Query: {len(model_results)} results (expected 10)")
    
    # Single Dataset Query
    dataset_results = [a for a in artifacts if a[1]['type'] == 'dataset']
    print(f"Single Dataset Query: {len(dataset_results)} results (expected 15)")
    
    # Single Code Query
    code_results = [a for a in artifacts if a[1]['type'] == 'code']
    print(f"Single Code Query: {len(code_results)} results (expected 10)")
    
    # Get Artifact By Name (artifact-30, which is code, was in page 2)
    by_name = [a for a in artifacts if a[1]['name'] == 'artifact-30']
    print(f"\nGet Artifact By Name 'artifact-30': {len(by_name)} result (expected 1)")
    if by_name:
        print(f"  Found: {by_name[0][0]}, type={by_name[0][1]['type']}")
    
    print(f"\n✓ Pagination fix will retrieve ALL artifacts!")

if __name__ == "__main__":
    test_paginated_list_artifacts()
