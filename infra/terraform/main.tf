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

# IAM role for CodeBuild
resource "aws_iam_role" "codebuild_role" {
  name = "neuroforge-codebuild-role"
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
data
resource "aws_iam_role_policy_attachment" "codebuild_ecr" {
  role       = aws_iam_role.codebuild_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
}

resource "aws_iam_role_policy_attachment" "codebuild_s3" {
  role       = aws_iam_role.codebuild_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# CodeBuild project for build/test/push
resource "aws_codebuild_project" "neuroforge" {
  name         = "neuroforge-build"
  service_role = aws_iam_role.codebuild_role.arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type                = "BUILD_GENERAL1_SMALL"
    image                       = "aws/codebuild/standard:6.0"
    type                        = "LINUX_CONTAINER"
    privileged_mode             = true
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