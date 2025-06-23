# terraform/s3.tf
resource "aws_s3_bucket" "data" {
  bucket = "${var.project_prefix}-data"
  acl    = "private"

  versioning {
    enabled = true
  }
}

resource "aws_s3_bucket" "models" {
  bucket = "${var.project_prefix}-models"
  acl    = "private"

  versioning {
    enabled = true
  }
}