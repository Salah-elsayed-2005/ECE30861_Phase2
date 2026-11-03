import sys
print(sys.executable)
try:
    import boto3
    print("boto3 imported successfully")
    print(boto3.__version__)
except ImportError as e:
    print(f"Failed to import boto3: {e}")
