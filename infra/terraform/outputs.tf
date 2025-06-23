// terraform/outputs.tf
output "ecr_repository_url" {
  description = "URI of the ECR repository"
  value       = aws_ecr_repository.neuroforge.repository_url
}

output "codebuild_role_name" {
  description = "Name of CodeBuild role"
  value       = aws_iam_role.codebuild.name
}

output "sagemaker_role_arn" {
  description = "ARN of SageMaker role"
  value       = aws_iam_role.sagemaker.arn
}