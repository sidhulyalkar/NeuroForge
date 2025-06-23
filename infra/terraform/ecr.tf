# terraform/ecr.tf
resource "aws_ecr_repository" "neuroforge" {
  name = "${var.project_prefix}-ecr"
  image_scanning_configuration {
    scan_on_push = true
  }
}