# EKS cluster via the upstream module. Same reasoning as for the VPC --
# rolling your own EKS in raw resources is a 2,000-line mess; the module
# bundles the IAM, OIDC, and add-on plumbing that EKS demands.

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.24"

  cluster_name    = "${local.name}-eks"
  cluster_version = var.cluster_version

  # Endpoint access: public so kubectl from your laptop works. In production
  # this would be private + a VPN / Session Manager bastion.
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Cluster add-ons keep the OIDC + CoreDNS + kube-proxy versions in sync
  # with the cluster version. Without these, version skew is a real issue.
  cluster_addons = {
    coredns                = {}
    eks-pod-identity-agent = {}
    kube-proxy             = {}
    vpc-cni                = {}
  }

  eks_managed_node_groups = {
    workers = {
      ami_type       = "AL2023_x86_64_STANDARD"
      instance_types = [var.node_instance_type]
      min_size       = 1
      max_size       = 3
      desired_size   = var.node_desired_size

      # Small disk -- containers are < 200MB; we don't need 80GB defaults.
      disk_size = 20
    }
  }

  # Grant your IAM identity admin access to the cluster so kubectl works after
  # apply. The data source pulls your current caller identity automatically.
  enable_cluster_creator_admin_permissions = true

  tags = local.tags
}
