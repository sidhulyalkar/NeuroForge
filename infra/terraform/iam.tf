// terraform/iam.tf

# OIDC trust for GitHub Actions

data "aws_iam_policy_document" "codebuild_assume" {
  statement {
    effect    = "Allow"
    actions   = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.project_prefix}/*:ref:refs/heads/*"]
    }
  }
}

resource "aws_iam_role" "codebuild" {
  name               = "${var.project_prefix}-codebuild-role"
  assume_role_policy = data.aws_iam_policy_document.codebuild_assume.json
}

resource "aws_iam_policy" "codebuild_ecr_s3" {
  name        = "${var.project_prefix}-codebuild-policy"
  description = "Access to ECR and S3 for CodeBuild"
  policy      = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ],
        Resource = aws_ecr_repository.neuroforge.arn
      },
      {
        Effect   = "Allow",
        Action   = ["s3:GetObject", "s3:PutObject"],
        Resource = [
          "${aws_s3_bucket.data.arn}/*",
          "${aws_s3_bucket.models.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "codebuild_attach" {
  role       = aws_iam_role.codebuild.name
  policy_arn = aws_iam_policy.codebuild_ecr_s3.arn
}

# SageMaker execution role trust

data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    effect    = "Allow"
    actions   = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker" {
  name               = "${var.project_prefix}-sagemaker-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

resource "aws_iam_policy" "sagemaker_access" {
  name        = "${var.project_prefix}-sagemaker-policy"
  description = "Access to S3 and ECR for SageMaker"
  policy      = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = ["s3:ListBucket"],
        Resource = [aws_s3_bucket.data.arn, aws_s3_bucket.models.arn]
      },
      {
        Effect   = "Allow",
        Action   = ["s3:GetObject", "s3:PutObject"],
        Resource = [
          "${aws_s3_bucket.data.arn}/*",
          "${aws_s3_bucket.models.arn}/*"
        ]
      },
      {
        Effect   = "Allow",
        Action   = ["ecr:GetAuthorizationToken"],
        Resource = "*"
      },
      {
        Effect   = "Allow",
        Action   = [
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
  policy_arn = aws_iam_policy.sagemaker_access.arn
}