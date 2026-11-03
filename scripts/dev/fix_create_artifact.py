import re

# Read the file
with open('temp_lambda/api/autograder_routes.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the old name extraction with the improved one
old_pattern = r'# Extract name from URL\s+parts = artifact_data\.url\.rstrip\(\'/\'\)\.split\(\'/\'\)\s+name = parts\[-1\] if parts else "unknown"'
new_code = '# Extract name from URL using improved extraction\n    name = _extract_artifact_name(artifact_data.url)'

content = re.sub(old_pattern, new_code, content)

# Write back
with open('temp_lambda/api/autograder_routes.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed! Now create_artifact() uses _extract_artifact_name()")
