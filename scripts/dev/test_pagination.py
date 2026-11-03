"""
Test if the DynamoDB scan has pagination issues
"""

# Simulate DynamoDB scan with pagination
def simulate_paginated_scan():
    # DynamoDB limits scan to 1MB or 1000 items per page
    # If response has 'LastEvaluatedKey', there are more pages
    
    # Scenario: 30 artifacts uploaded, but scan only returns first page
    all_items = [f"ARTIFACT#id{i}" for i in range(1, 36)]  # 35 artifacts
    
    # First scan (no pagination handling)
    first_page = all_items[:25]  # DynamoDB might limit to ~25 items
    print(f"First scan returns: {len(first_page)} items")
    print(f"Remaining: {len(all_items) - len(first_page)} items")
    
    # Without pagination, we miss the rest!
    print(f"\n❌ BUG: Code doesn't handle LastEvaluatedKey")
    print(f"Result: Only {len(first_page)}/35 artifacts returned")
    print(f"This would cause queries for artifacts 26-35 to fail with 404!")
    
    return first_page

# The fix: handle pagination
def fixed_scan():
    all_items = [f"ARTIFACT#id{i}" for i in range(1, 36)]
    
    results = []
    last_key = None
    page = 0
    
    while True:
        page += 1
        # Simulate paginated responses
        if page == 1:
            items = all_items[:25]
            last_key = "some_key" if len(all_items) > 25 else None
        elif page == 2:
            items = all_items[25:]
            last_key = None
        else:
            break
            
        results.extend(items)
        print(f"Page {page}: {len(items)} items, LastEvaluatedKey: {last_key}")
        
        if not last_key:
            break
    
    print(f"\n✓ FIX: Handled pagination correctly")
    print(f"Result: {len(results)}/35 artifacts returned")
    return results

print("=== Simulating DynamoDB Scan Issue ===\n")
print("Without pagination handling:")
simulate_paginated_scan()

print("\n\n=== With Pagination Fix ===\n")
fixed_scan()

print("\n\n=== Key Insight ===")
print("If _list_artifacts() doesn't handle pagination, it will only return")
print("the first page of results. This could explain why some queries work")
print("(for early-uploaded artifacts) but others fail (for later uploads).")
print("\nThis would particularly affect 'Get Artifact By Name' tests if they're")
print("testing artifacts that were uploaded later in the sequence.")
