# terraform/outputs.tf
output "ecr_repository_url" {
  description = "URI of the ECR repository"
  value       = aws_ecr_repository.neuroforge.repository_url
}