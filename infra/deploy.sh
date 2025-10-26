#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/terraform"
: "${AWS_PROFILE:=default}"

terraform init -upgrade
terraform apply -auto-approve \
  -var "project=tmr" \
  -var "env=dev" \
  -var "region=us-east-1"