# Local values + supporting data sources.

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  # Use the first three AZs for subnets. EKS requires at least 2.
  azs = slice(data.aws_availability_zones.available.names, 0, 3)

  name = "${var.project}-${terraform.workspace == "default" ? "demo" : terraform.workspace}"

  tags = {
    Project = var.project
    Owner   = var.owner
  }
}
