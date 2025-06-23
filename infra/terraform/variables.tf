// terraform/variables.tf
variable "project_prefix" {
  description = "Prefix for resource names"
  type        = string
  default     = "neuroforge"
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "sagemaker_image" {
  description = "ECR URI for SageMaker container image"
  type        = string
  default     = ""
}

variable "training_instance_type" {
  description = "Instance type for SageMaker training"
  type        = string
  default     = "ml.m5.large"
}

variable "endpoint_instance_type" {
  description = "Instance type for SageMaker endpoint"
  type        = string
  default     = "ml.t2.medium"
}