output "bucket_name" {
  value = aws_s3_bucket.models.bucket
}

output "table_name" {
  value = aws_dynamodb_table.registry.name
}

output "log_group" {
  value = aws_cloudwatch_log_group.api.name
}
