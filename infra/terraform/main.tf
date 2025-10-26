locals {
  name_prefix = "${var.project}-${var.env}"
}

# Unique suffix for S3 bucket name (S3 bucket names must be globally unique)
resource "random_id" "suffix" {
  byte_length = 2
}

# 1) S3 bucket for model artifacts
resource "aws_s3_bucket" "models" {
  bucket = "${local.name_prefix}-models-${random_id.suffix.hex}"

  tags = {
    Project = var.project
    Env     = var.env
  }
}

# Optional: versioning + block public access
resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# 2) DynamoDB table for model metadata
resource "aws_dynamodb_table" "registry" {
  name         = "${local.name_prefix}-registry"
  billing_mode = "PAY_PER_REQUEST"

  hash_key = "model_id"

  attribute {
    name = "model_id"
    type = "S"
  }

  tags = {
    Project = var.project
    Env     = var.env
  }
}

# 3) CloudWatch Log Group (placeholder for future Lambda/API)
resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${local.name_prefix}-api"
  retention_in_days = 7
}
