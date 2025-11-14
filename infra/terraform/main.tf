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

# 3) CloudWatch Log Group
resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${local.name_prefix}-api"
  retention_in_days = 7
}

# 4) IAM Role for Lambda
resource "aws_iam_role" "lambda_exec" {
  name = "${local.name_prefix}-lambda-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = {
    Project = var.project
    Env     = var.env
  }
}

# Attach basic execution policy
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Custom policy for S3 and DynamoDB access
resource "aws_iam_role_policy" "lambda_access" {
  name = "${local.name_prefix}-lambda-access"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = aws_dynamodb_table.registry.arn
      }
    ]
  })
}

# 5) Lambda Function
resource "aws_lambda_function" "api" {
  filename         = "lambda_deployment.zip"
  function_name    = "${local.name_prefix}-api"
  role            = aws_iam_role.lambda_exec.arn
  handler         = "lambda_handler.handler"
  source_code_hash = filebase64sha256("lambda_deployment.zip")
  runtime         = "python3.10"
  timeout         = 30
  memory_size     = 512

  environment {
    variables = {
      S3_BUCKET      = aws_s3_bucket.models.bucket
      DYNAMODB_TABLE = aws_dynamodb_table.registry.name
    }
  }

  tags = {
    Project = var.project
    Env     = var.env
  }
}

# 6) API Gateway HTTP API
resource "aws_apigatewayv2_api" "http_api" {
  name          = "${local.name_prefix}-api"
  protocol_type = "HTTP"
  
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers = ["*"]
    max_age       = 300
  }

  tags = {
    Project = var.project
    Env     = var.env
  }
}

# API Gateway Stage
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http_api.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }
}

# Lambda Integration
resource "aws_apigatewayv2_integration" "lambda" {
  api_id             = aws_apigatewayv2_api.http_api.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.api.invoke_arn
  integration_method = "POST"
  payload_format_version = "2.0"
}

# Catch-all route
resource "aws_apigatewayv2_route" "proxy" {
  api_id    = aws_apigatewayv2_api.http_api.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

# Lambda permission for API Gateway
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http_api.execution_arn}/*/*"
}
