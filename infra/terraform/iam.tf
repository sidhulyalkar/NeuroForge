// terraform/iam.tf

// 1. OIDC Provider for GitHub Actions
resource "aws_iam_openid_connect_provider" "github_actions" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

// 2. CodeBuild Role with OIDC Trust
data "aws_iam_policy_document" "codebuild_trust" {
  statement {
    effect    = "Allow"
    actions   = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github_actions.arn]
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

resource "aws_iam_role" "codebuild" {
  name               = "${var.project_prefix}-codebuild-role"
  assume_role_policy = data.aws_iam_policy_document.codebuild_trust.json
}

// 3. CodeBuild Permissions: ECR & S3
resource "aws_iam_policy" "codebuild_policy" {
  name        = "${var.project_prefix}-codebuild-policy"
  description = "ECR push/pull and S3 access for CodeBuild"
  policy      = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ],
        Resource = aws_ecr_repository.neuroforge.arn
      },
      {
        Effect   = "Allow",
        Action   = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject"
        ],
        Resource = [
          aws_s3_bucket.data.arn,
          format("%s/*", aws_s3_bucket.data.arn),
          aws_s3_bucket.models.arn,
          format("%s/*", aws_s3_bucket.models.arn)
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "codebuild_attach" {
  role       = aws_iam_role.codebuild.name
  policy_arn = aws_iam_policy.codebuild_policy.arn
}

// 4. SageMaker Execution Role
resource "aws_iam_role" "sagemaker" {
  name               = "${var.project_prefix}-sagemaker-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "sagemaker.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })
}

// 5. SageMaker Permissions: S3 & ECR Pull
resource "aws_iam_policy" "sagemaker_policy" {
  name        = "${var.project_prefix}-sagemaker-policy"
  description = "S3 and ECR pull access for SageMaker"
  policy      = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = ["s3:ListBucket"],
        Resource = [
          aws_s3_bucket.data.arn,
          aws_s3_bucket.models.arn
        ]
      },
      {
        Effect   = "Allow",
        Action   = ["s3:GetObject"],
        Resource = [
          format("%s/*", aws_s3_bucket.data.arn),
          format("%s/*", aws_s3_bucket.models.arn)
        ]
      },
      {
        Effect   = "Allow",
        Action   = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ],
        Resource = aws_ecr_repository.neuroforge.arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_attach" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = aws_iam_policy.sagemaker_policy.arn
}