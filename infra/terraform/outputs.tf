# terraform/outputs.tf
output "ecr_repository_url" {
  description = "URL of the NeuroForge ECR repository"
  value       = aws_ecr_repository.neuroforge.repository_url
}