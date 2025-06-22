# terraform/main.tf
terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Create ECR repository for NeuroForge
resource "aws_ecr_repository" "neuroforge" {
  name                 = "neuroforge"
  image_scanning_configuration {
    scan_on_push = true
  }
}

# S3 buckets for data storage
resource "aws_s3_bucket" "data_bucket" {
  bucket = "${var.project_prefix}-data"
  acl    = "private"
  versioning {
    enabled = true
  }
}

resource "aws_s3_bucket" "model_bucket" {
  bucket = "${var.project_prefix}-models"
  acl    = "private"
  versioning {
    enabled = true
  }
}

# IAM role for CodeBuild
resource "aws_iam_role" "codebuild_role" {
  name = "${var.project_prefix}-codebuild-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "codebuild.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

# Attach policies to CodeBuild role
resource "aws_iam_role_policy_attachment" "codebuild_ecr" {
  role       = aws_iam_role.codebuild_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
}

resource "aws_iam_role_policy_attachment" "codebuild_s3" {
  role       = aws_iam_role.codebuild_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# SageMaker execution role
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_prefix}-sagemaker-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action   = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_policy" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# SageMaker training and endpoint resources
resource "aws_sagemaker_model" "neuroforge_model" {
  name          = "${var.project_prefix}-model"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {
    image = var.sagemaker_image_url
    model_data_url = "s3://${aws_s3_bucket.model_bucket.bucket}/models/model.tar.gz"
  }
}

resource "aws_sagemaker_endpoint_configuration" "neuroforge_endpoint_config" {
  name = "${var.project_prefix}-endpoint-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.neuroforge_model.name
    initial_instance_count = 1
    instance_type          = var.endpoint_instance_type
  }
}

resource "aws_sagemaker_endpoint" "neuroforge_endpoint" {
  name = "${var.project_prefix}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.neuroforge_endpoint_config.name
}

# CodeBuild project for build/test/push
resource "aws_codebuild_project" "neuroforge" {
  name         = "${var.project_prefix}-build"
  service_role = aws_iam_role.codebuild_role.arn

  artifacts { type = "NO_ARTIFACTS" }

  environment {
    compute_type    = "BUILD_GENERAL1_SMALL"
    image           = "aws/codebuild/standard:6.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true

    environment_variable {
      name  = "IMAGE_REPO_NAME"
      value = aws_ecr_repository.neuroforge.name
    }
    environment_variable {
      name  = "AWS_REGION"
      value = var.aws_region
    }
  }

  source {
    type      = "GITHUB"
    location  = "https://github.com/${var.github_owner}/${var.github_repo}.git"
    buildspec = file("buildspec.yml")
  }
}

variable "project_prefix" {
  description = "Prefix for resource names"
  type        = string
  default     = "neuroforge"
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "github_owner" {
  description = "GitHub owner or organization"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
}

variable "sagemaker_image_url" {
  description = "Docker image URI for SageMaker"
  type        = string
}

variable "endpoint_instance_type" {
  description = "Instance type for SageMaker endpoint"
  type        = string
  default     = "ml.t2.medium"
}