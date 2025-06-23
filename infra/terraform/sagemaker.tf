// terraform/sagemaker.tf
resource "aws_sagemaker_model" "neuroforge_model" {
  count              = length(var.sagemaker_image) > 0 ? 1 : 0
  name               = "${var.project_prefix}-model"
  execution_role_arn = aws_iam_role.sagemaker.arn

  primary_container {
    image          = var.sagemaker_image
    model_data_url = "s3://${aws_s3_bucket.models.bucket}/model.tar.gz"
  }
}

resource "aws_sagemaker_endpoint_configuration" "neuroforge_config" {
  count = length(var.sagemaker_image) > 0 ? 1 : 0
  name  = "${var.project_prefix}-endpoint-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.neuroforge_model[0].name
    initial_instance_count = 1
    instance_type          = var.endpoint_instance_type
  }
}

resource "aws_sagemaker_endpoint" "neuroforge_endpoint" {
  count                = length(var.sagemaker_image) > 0 ? 1 : 0
  name                 = "${var.project_prefix}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.neuroforge_config[0].name
}