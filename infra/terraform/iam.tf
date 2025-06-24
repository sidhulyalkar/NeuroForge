// terraform/iam.tf

// Variables (define in variables.tf)
// variable "project_prefix" {}
// variable "github_owner" {}
// variable "github_repo" {}

// 1. Reference existing GitHub Actions OIDC Provider
data "aws_iam_openid_connect_provider" "github_actions" {
  url = "https://token.actions.githubusercontent.com"
}

// 2. CodeBuild Role Trust Policy
data "aws_iam_policy_document" "codebuild_assume" {
  statement {
    effect    = "Allow"
    actions   = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github_actions.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_owner}/${var.github_repo}:ref:refs/heads/main"]
    }
  }
}

// 3. CodeBuild Role
resource "aws_iam_role" "codebuild" {
  name               = "${var.project_prefix}-codebuild-role"
  assume_role_policy = data.aws_iam_policy_document.codebuild_assume.json
}

// 4. CodeBuild Permissions
data "aws_iam_policy_document" "codebuild_policy_doc" {
  statement {
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:PutImage",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload"
    ]
    resources = ["*"]
  }
  statement {
    effect = "Allow"
    actions = ["s3:ListBucket"]
    resources = [
      aws_s3_bucket.data.arn,
      aws_s3_bucket.models.arn
    ]
  }
  statement {
    effect = "Allow"
    actions = ["s3:GetObject", "s3:PutObject"]
    resources = [
      format("%s/*", aws_s3_bucket.data.arn),
      format("%s/*", aws_s3_bucket.models.arn)
    ]
  }
}

resource "aws_iam_policy" "codebuild_policy" {
  name   = "${var.project_prefix}-codebuild-policy"
  policy = data.aws_iam_policy_document.codebuild_policy_doc.json
}

resource "aws_iam_role_policy_attachment" "codebuild_attach" {
  role       = aws_iam_role.codebuild.name
  policy_arn = aws_iam_policy.codebuild_policy.arn
}

// 5. SageMaker Role Trust Policy
data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    effect    = "Allow"
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
    actions   = ["sts:AssumeRole"]
  }
}

// 6. SageMaker Execution Role
resource "aws_iam_role" "sagemaker" {
  name               = "${var.project_prefix}-sagemaker-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

// 7. SageMaker Permissions
data "aws_iam_policy_document" "sagemaker_policy_doc" {
  statement {
    effect = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [
      aws_s3_bucket.data.arn,
      aws_s3_bucket.models.arn
    ]
  }
  statement {
    effect = "Allow"
    actions   = ["s3:GetObject"]
    resources = [
      format("%s/*", aws_s3_bucket.data.arn),
      format("%s/*", aws_s3_bucket.models.arn)
    ]
  }
  statement {
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage"
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "sagemaker_policy" {
  name   = "${var.project_prefix}-sagemaker-policy"
  policy = data.aws_iam_policy_document.sagemaker_policy_doc.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_attach" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = aws_iam_policy.sagemaker_policy.arn
}