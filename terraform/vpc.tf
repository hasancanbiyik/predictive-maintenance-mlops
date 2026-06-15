# VPC for the EKS cluster.
#
# Why we use the upstream module instead of hand-rolled aws_vpc resources:
#   - It correctly tags subnets for the EKS load-balancer controller.
#   - It handles private DNS hostname options EKS needs.
#   - Hand-rolled VPCs are where 80% of EKS networking issues come from.

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.13"

  name = "${local.name}-vpc"
  cidr = var.vpc_cidr

  azs             = local.azs
  private_subnets = [for i, az in local.azs : cidrsubnet(var.vpc_cidr, 4, i)]
  public_subnets  = [for i, az in local.azs : cidrsubnet(var.vpc_cidr, 8, i + 48)]

  enable_nat_gateway     = true
  single_nat_gateway     = var.single_nat_gateway
  one_nat_gateway_per_az = false
  enable_dns_hostnames   = true

  # Standard EKS subnet tags. The aws-load-balancer-controller uses these
  # to discover where to put ELBs.
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }

  tags = local.tags
}
