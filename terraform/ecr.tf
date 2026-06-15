# ECR repository for the API image. EKS pulls from here at deploy time.
#
# Lifecycle rule keeps the repo tidy: keep the last 10 images, expire the rest.
# Without this a busy CI pipeline fills the repo and you pay for old layers.

resource "aws_ecr_repository" "api" {
  name                 = "${local.name}/predictive-maintenance-api"
  image_tag_mutability = "MUTABLE" # so `:latest` can be overwritten in CI
  force_delete         = true      # so `terraform destroy` works without manual cleanup

  image_scanning_configuration {
    scan_on_push = true # automatic CVE scan -- free with ECR
  }

  tags = local.tags
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      }
    ]
  })
}
