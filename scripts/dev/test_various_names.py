import requests

BASE_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

test_names = [
    "bert",
    "audience_classifier_model",  
    "distilbert-base-uncased-distilled-squad",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
    "ptm-recommendation-with-transformers.git"
]

for name in test_names:
    resp = requests.get(f"{BASE_URL}/artifact/byName/{name}")
    result_count = len(resp.json()) if resp.status_code == 200 else 0
    print(f"{resp.status_code} - {name[:50]}: {result_count} results")
