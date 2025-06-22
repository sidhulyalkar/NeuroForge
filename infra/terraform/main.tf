# terraform/main.tf
terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = { source = "hashicorp/aws"; version = "~> 4.0" }
  }
}
provider "aws" { region = var.aws_region }

# ECR repository\ nresource "aws_ecr_repository" "neuroforge" {
  name = "${var.project_prefix}-ecr"
  image_scanning_configuration { scan_on_push = true }
}

# S3 buckets
resource "aws_s3_bucket" "data" {
  bucket = "${var.project_prefix}-data"
  acl    = "private"
  versioning { enabled = true }
}
resource "aws_s3_bucket" "models" {
  bucket = "${var.project_prefix}-models"
  acl    = "private"
  versioning { enabled = true }
}

# IAM Policies for least-privilege
data "aws_iam_policy_document" "sagemaker_s3_access" {
  statement {
    actions = ["s3:GetObject", "s3:PutObject"]
    resources = [
      "${aws_s3_bucket.data.arn}/*",
      "${aws_s3_bucket.models.arn}/*"
    ]
  }
}
resource "aws_iam_policy" "sagemaker_s3_policy" {
  name   = "${var.project_prefix}-sagemaker-s3"
  policy = data.aws_iam_policy_document.sagemaker_s3_access.json
}

data "aws_iam_policy_document" "codebuild_ecr_access" {
  statement {
    actions   = ["ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:PutImage", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload"]
    resources = [aws_ecr_repository.neuroforge.arn]
  }
}
resource "aws_iam_policy" "codebuild_ecr_policy" {
  name   = "${var.project_prefix}-codebuild-ecr"
  policy = data.aws_iam_policy_document.codebuild_ecr_access.json
}

data "aws_iam_policy_document" "codebuild_s3_access" {
  statement {
    actions = ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/*"]
  }
}
resource "aws_iam_policy" "codebuild_s3_policy" {
  name   = "${var.project_prefix}-codebuild-s3"
  policy = data.aws_iam_policy_document.codebuild_s3_access.json
}

# CodeBuild role\ nresource "aws_iam_role" "codebuild" {
  name               = "${var.project_prefix}-codebuild-role"
  assume_role_policy = data.aws_iam_policy_document.codebuild_assume_role.json
}
data "aws_iam_policy_document" "codebuild_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals { type = "Service"; identifiers = ["codebuild.amazonaws.com"] }
  }
}
resource "aws_iam_role_policy_attachment" "codebuild_ecr_attach" {
  role       = aws_iam_role.codebuild.name
  policy_arn = aws_iam_policy.codebuild_ecr_policy.arn
}
resource "aws_iam_role_policy_attachment" "codebuild_s3_attach" {
  role       = aws_iam_role.codebuild.name
  policy_arn = aws_iam_policy.codebuild_s3_policy.arn
}

# SageMaker execution role and attachment
resource "aws_iam_role" "sagemaker" {
  name               = "${var.project_prefix}-sagemaker-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}
data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals { type = "Service"; identifiers = ["sagemaker.amazonaws.com"] }
  }
}
resource "aws_iam_role_policy_attachment" "sagemaker_s3_attach" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = aws_iam_policy.sagemaker_s3_policy.arn
}

# SageMaker model and training job\ nresource "aws_sagemaker_model" "model" {
  name               = "${var.project_prefix}-model"
  execution_role_arn = aws_iam_role.sagemaker.arn
  primary_container {
    image          = var.sagemaker_image
    model_data_url = "s3://${aws_s3_bucket.models.bucket}/model.tar.gz"
  }
}
resource "aws_sagemaker_training_job" "train" {
  name                  = "${var.project_prefix}-training"
  role_arn              = aws_iam_role.sagemaker.arn
  algorithm_specification {
    training_image     = var.sagemaker_image
    training_input_mode = "File"
  }
  input_data_config {
    channel_name = "training"
    data_source {
      s3_data_source {
        s3_uri             = "s3://${aws_s3_bucket.data.bucket}/training/"
        s3_data_type       = "S3Prefix"
        s3_data_distribution_type = "FullyReplicated"
      }
    }
  }
  output_data_config {
    s3_output_path = "s3://${aws_s3_bucket.models.bucket}/"
  }
  resource_config {
    instance_count = 1
    instance_type  = var.training_instance_type
    volume_size    = 50
  }
  stopping_condition {
    max_runtime_in_seconds = 3600
  }
}

variable "project_prefix" { default = "neuroforge" }
variable "aws_region"     { default = "us-west-2" }
variable "sagemaker_image" { description = "Container image URI" }
variable "training_instance_type" { default = "ml.m5.large" }
variable "endpoint_instance_type" { default = "ml.t2.medium" }
